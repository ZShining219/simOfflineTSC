"""Phase-level Double-DQN trainer for the scene-guided signal agent.

Single-layer counterpart of ``train_router``: episodes are full S2 runs
with a sampled congestion (surge) injection schedule; each ``interval``
-second decision tick yields one (s, a, r, s') transition whose gradient
flows through the scene encoder — the condensed representation is shaped
by control value directly, with no controller-selection layer in between.

Auxiliary loss (binding alignment): while an event report is active the
cross-attention lane weights are encouraged toward the report's bound
lanes — same as the router trainer, skipped for state-only arms.

Usage:
    python -m scene_attention.train --variant sga \
        --episodes 100 --out output_data/dev_takeover_probe/sig_sga
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scene_attention.probe.takeover import (  # noqa: E402
    build_controllers, build_world, _bind_all,
)
from scene_attention.dispatch_state import FeatureBuilder  # noqa: E402
from scene_attention.signal_model import (  # noqa: E402
    STATE_ONLY, SceneJointPolicy, SceneResidualPolicy,
    SceneSignalAgent,
)
from scene_attention.env import (  # noqa: E402
    run_signal_episode,
)
from scene_attention.events import (  # noqa: E402
    DISCHARGE_OF, EDGE_TARGETS, FEEDER_OF, TRAIN_DURS, TRAIN_EDGES, TRAIN_ONSETS,
    TRAIN_PERIODS,
)
from hitl.live.injects import parse_injects  # noqa: E402


def sample_injects(rng, allow_none_p=0.25, p_corridor=0.0):
    """Single congestion event on a random injectable edge.

    Reports are pure-phenomenon ('道路拥堵') — no objective flag is ever
    set (the broadcast names a phenomenon, not a governance directive).

    ``p_corridor``: with that probability a second, UNREPORTED surge is
    placed on the event edge's upstream feeder link, routed into the
    event edge — the corridor carries real inflow pressure so watching
    the feeder (the coordination edge the intent names) has a physical
    payoff.  Starts earlier so the feeder is already loading when the
    primary report arrives.
    """
    if rng.random() < allow_none_p:
        return []
    edge = TRAIN_EDGES[int(rng.integers(len(TRAIN_EDGES)))]
    onset = int(TRAIN_ONSETS[rng.integers(len(TRAIN_ONSETS))])
    dur = int(TRAIN_DURS[rng.integers(len(TRAIN_DURS))])
    period = int(TRAIN_PERIODS[rng.integers(len(TRAIN_PERIODS))])
    specs = [f'surge:{edge}:{EDGE_TARGETS[edge]}:{onset}:{dur}:{period}']
    if rng.random() < p_corridor:
        feeder = FEEDER_OF[edge]
        specs.append(
            f'surge:{feeder}:{edge}:{max(0, onset - 120)}:'
            f'{dur + 120}:{period + 3}:global')
        discharge = DISCHARGE_OF.get(edge)
        if discharge:
            specs.append(
                f'throttle:{discharge}:{max(0, onset - 60)}:'
                f'{dur + 60}:3.0:global')
    return parse_injects(specs)


def train(args):
    import torch

    rng = np.random.default_rng(args.rng_seed)
    torch.manual_seed(args.rng_seed)
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    world = build_world(args.scene, args.traffic_seed_base, Path(
        args.runtime_dir or '/tmp/takeover_signal'), out / 'logger',
        {'t_fixed': args.t_fixed, 't_min': args.t_min})
    ranks = list(range(len(world.intersection_ids)))
    n_phases = len(world.intersections[0].phases)

    fb0 = FeatureBuilder(
        world, [], report_delay=args.report_delay, layout='identity',
        phys=args.phys, include_forecast=not args.no_forecast,
        phase_age=True, intent_coord_w=args.intent_coord_w)
    state_only = args.variant in STATE_ONLY
    residual_mode = args.residual_dqn is not None
    joint_mode = args.joint_dqn is not None
    if (residual_mode or joint_mode) \
            and args.variant not in ('sga', 'attn'):
        raise ValueError(
            '--residual-dqn/--joint-dqn need a binding variant '
            f'(sga/attn), got {args.variant!r}')
    if residual_mode and joint_mode:
        raise ValueError('--residual-dqn and --joint-dqn are exclusive')
    if args.distill_eps > 0 and not args.distill_encoder:
        raise ValueError('--distill-eps needs --distill-encoder '
                         '(a trained agent.pt whose encoder drives '
                         'the teacher adapter)')
    if joint_mode:
        # 'controller借助注意力优化' form: the base DQN lives in-graph,
        # initialised from the deployed checkpoint and fine-tuned with
        # a small lr — TD lands in the controller's own weights.
        from scene_attention.scene_mp import bound_dqn_agent
        from hitl.live.controllers import (
            load_qnet_payload,
        )
        probe = bound_dqn_agent(world, args.joint_dqn, rank=0)
        flat_in_lanes = [l for g in probe.ob_generator.lanes
                         for l in g]
        payload = load_qnet_payload(args.joint_dqn)
        if not payload['phase']:
            raise ValueError('joint mode assumes phase-in-feature DQN '
                             '(payload phase=False unsupported)')
        agent = SceneJointPolicy(
            lane_dim=fb0.lane_tokens(0.0).shape[1],
            text_dim=fb0.text_dim, global_dim=fb0.global_dim,
            n_phases=n_phases, d_model=args.d_model,
            variant=args.variant, residual_pool=args.residual_pool,
            geom=fb0.model_geom(), n_edges=fb0.n_edges,
            edge_idx=fb0.lane_edge_idx,
            dose_frac=args.res_dose, bind_tau=args.res_bind_tau,
            base_state_dict=payload['state_dict'],
            base_arch=payload['algorithm_id'],
            base_in_lanes=flat_in_lanes, lane_names=fb0.lanes,
            base_input_dim=payload['input_dim'])
        target = SceneJointPolicy(
            lane_dim=fb0.lane_tokens(0.0).shape[1],
            text_dim=fb0.text_dim, global_dim=fb0.global_dim,
            n_phases=n_phases, d_model=args.d_model,
            variant=args.variant, residual_pool=args.residual_pool,
            geom=fb0.model_geom(), n_edges=fb0.n_edges,
            edge_idx=fb0.lane_edge_idx,
            dose_frac=args.res_dose, bind_tau=args.res_bind_tau,
            base_state_dict=payload['state_dict'],
            base_arch=payload['algorithm_id'],
            base_in_lanes=flat_in_lanes, lane_names=fb0.lanes,
            base_input_dim=payload['input_dim'])
        base_init = {k: v.detach().clone()
                     for k, v in agent.base.state_dict().items()}
    elif residual_mode:
        # guided补课 form: frozen base DQN supplies q_base per tick;
        # the scene module only learns a bounded gated correction.
        # BC is pointless here — the base already drives.
        agent = SceneResidualPolicy(
            lane_dim=fb0.lane_tokens(0.0).shape[1],
            text_dim=fb0.text_dim, global_dim=fb0.global_dim,
            n_phases=n_phases, d_model=args.d_model,
            variant=args.variant, residual_pool=args.residual_pool,
            geom=fb0.model_geom(), n_edges=fb0.n_edges,
            edge_idx=fb0.lane_edge_idx,
            dose_frac=args.res_dose, bind_tau=args.res_bind_tau)
        target = SceneResidualPolicy(
            lane_dim=fb0.lane_tokens(0.0).shape[1],
            text_dim=fb0.text_dim, global_dim=fb0.global_dim,
            n_phases=n_phases, d_model=args.d_model,
            variant=args.variant, residual_pool=args.residual_pool,
            geom=fb0.model_geom(), n_edges=fb0.n_edges,
            edge_idx=fb0.lane_edge_idx,
            dose_frac=args.res_dose, bind_tau=args.res_bind_tau)
    else:
        agent = SceneSignalAgent(
            lane_dim=fb0.lane_tokens(0.0).shape[1],
            text_dim=fb0.text_dim, global_dim=fb0.global_dim,
            n_phases=n_phases, d_model=args.d_model,
            variant=args.variant, residual_pool=args.residual_pool,
            geom=fb0.model_geom() if not state_only else None,
            n_lanes=fb0.n_lanes)
        target = SceneSignalAgent(
            lane_dim=fb0.lane_tokens(0.0).shape[1],
            text_dim=fb0.text_dim, global_dim=fb0.global_dim,
            n_phases=n_phases, d_model=args.d_model,
            variant=args.variant, residual_pool=args.residual_pool,
            geom=fb0.model_geom() if not state_only else None,
            n_lanes=fb0.n_lanes)
    if args.init_from:
        # warm-start from a prior checkpoint (e.g. the BC student)
        # before continuing with TD — joint base_init was snapshotted
        # earlier so drift still measures cumulative movement from the
        # deployed dqn_s1 weights.
        payload0 = torch.load(args.init_from, map_location='cpu',
                              weights_only=False)
        agent.load_state_dict(payload0['state_dict'])
        print(json.dumps({'init_from': args.init_from}), flush=True)
    target.load_state_dict(agent.state_dict())
    target.eval()
    if joint_mode:
        # base gets a small lr (fine-tune a converged controller);
        # the scene branch gets the normal lr — the controller is
        # the object being optimised, attention is the channel
        opt = torch.optim.Adam([
            {'params': agent.base.parameters(), 'lr': args.base_lr},
            {'params': list(agent.encoder.parameters())
             + list(agent.delta.parameters()), 'lr': args.lr},
        ])
    else:
        opt = torch.optim.Adam(agent.parameters(), lr=args.lr)
    replay = deque(maxlen=args.buffer)

    meta = dict(
        lane_dim=fb0.lane_tokens(0.0).shape[1], text_dim=fb0.text_dim,
        global_dim=fb0.global_dim, n_lanes=fb0.n_lanes,
        d_model=args.d_model, residual_pool=args.residual_pool,
        **fb0.feature_meta())
    if residual_mode:
        meta.update(kind='residual',
                    residual_dqn=str(Path(args.residual_dqn).name),
                    res_dose=args.res_dose,
                    res_bind_tau=args.res_bind_tau)
    if joint_mode:
        meta.update(kind='joint',
                    joint_dqn=str(Path(args.joint_dqn).name),
                    base_lr=args.base_lr,
                    res_dose=args.res_dose,
                    res_bind_tau=args.res_bind_tau)

    # ---- phase 0: teacher demonstrations + BC warm-start ---------------
    if args.teacher_eps > 0 and not (residual_mode or joint_mode):
        demos = []
        for tep in range(args.teacher_eps):
            injects = sample_injects(rng, allow_none_p=args.allow_none_p,
                                 p_corridor=args.p_corridor)
            world.close()
            world = build_world(
                args.scene, args.traffic_seed_base - 1 - tep,
                Path(args.runtime_dir or '/tmp/takeover_signal'),
                out / 'logger',
                {'t_fixed': args.t_fixed, 't_min': args.t_min,
                 'min_green_vehicle': 3, 'max_red_vehicle': 6})
            ctrls = build_controllers()
            _bind_all(world, ctrls, ranks, needed={args.teacher})
            fb_t = FeatureBuilder(
                world, injects, report_delay=args.report_delay,
                layout='identity', phys=args.phys,
                include_forecast=not args.no_forecast, phase_age=True,
                intent_coord_w=args.intent_coord_w)
            transitions, _ = run_signal_episode(
                world, fb_t, ranks, injects=injects,
                controller=ctrls[args.teacher], interval=args.interval,
                episode_seconds=args.episode_seconds, record_snaps=True,
                switch_penalty=args.switch_penalty)
            replay.extend(transitions)
            demos += [(t['s'], t['a']) for t in transitions]
        bc_loss = bc_updates(agent, opt, demos, args, rng)
        target.load_state_dict(agent.state_dict())
        print(json.dumps({'teacher': args.teacher,
                          'demo_pairs': len(demos),
                          'bc_loss': round(bc_loss, 4)}), flush=True)
        if args.episodes <= 0:
            agent.save(out / 'agent.pt', meta={**meta, 'episode': 0,
                                               'bc_only': True})
    if (residual_mode or joint_mode) and args.episodes <= 0 \
            and not args.distill_eps:
        agent.save(out / 'agent.pt',
                   meta={**meta, 'episode': 0, 'untrained': True})

    # ---- phase 0.5: adapter-teacher distillation ---------------------
    # 'controller借助注意力优化' via imitation: the SceneGuidedDQN
    # adapter's per-tick actions (attention already inside them) are
    # the labels; the joint student's weights update toward them —
    # the controller changes, guided by attention, inside its own
    # behaviour manifold instead of being dragged by a foreign reward.
    if joint_mode and args.distill_eps > 0:
        import copy
        from scene_attention.scene_mp import (
            SceneGuidedDQN, bound_dqn_agent,
        )
        from hitl.live.controllers import (
            inference_q_values,
        )
        enc_agent, _ = SceneSignalAgent.load(args.distill_encoder)
        vanilla = build_controllers(dqn_s1_path=args.joint_dqn)['dqn_s1'] \
            if args.distill_filter else None
        demos, kept_eps = [], 0
        for i in range(args.distill_eps):
            injects = sample_injects(rng, allow_none_p=args.allow_none_p,
                                 p_corridor=args.p_corridor)
            world.close()
            world = build_world(
                args.scene, args.traffic_seed_base - 100 - i,
                Path(args.runtime_dir or '/tmp/takeover_signal'),
                out / 'logger',
                {'t_fixed': args.t_fixed, 't_min': args.t_min,
                 'min_green_vehicle': 3, 'max_red_vehicle': 6})
            ranks = list(range(len(world.intersection_ids)))
            fb_d = FeatureBuilder(
                world, injects, report_delay=args.report_delay,
                layout='identity', phys=args.phys,
                include_forecast=not args.no_forecast, phase_age=True,
                intent_coord_w=args.intent_coord_w)
            dqn_d = bound_dqn_agent(world, args.joint_dqn,
                                    rank=ranks[0])
            adapter = SceneGuidedDQN(
                enc_agent, fb_d, world, dqn_d,
                beta=args.distill_beta, t_min=0.0,
                gate=args.distill_gate, fade='in',
                direction='auto', bind_tau=0.4, dose_frac=0.35)
            transitions, records = run_signal_episode(
                world, fb_d, ranks,
                injects=copy.deepcopy(injects), agent=adapter,
                interval=args.interval,
                episode_seconds=args.episode_seconds,
                q_base_fn=lambda d=dqn_d: inference_q_values(d))
            if vanilla is not None:
                world.reset()
                vanilla.bind(world, ranks=ranks)
                fb_v = FeatureBuilder(world, [],
                                      report_delay=args.report_delay)
                _, records_v = run_signal_episode(
                    world, fb_v, ranks,
                    injects=copy.deepcopy(injects),
                    controller=vanilla, interval=args.interval,
                    episode_seconds=args.episode_seconds)
                q_adv = sum(r['total_queue'] for r in records)
                q_van = sum(r['total_queue'] for r in records_v)
                if q_adv >= q_van:
                    continue                    # loser episode dropped
            kept_eps += 1
            replay.extend(transitions)          # teacher data in replay
            demos += [(t['s'], t['a'],
                       int(t['a'] != int(np.argmax(t['qb']))))
                      for t in transitions]
        dev = float(np.mean([d[2] for d in demos])) if demos else 0.0
        bc_loss, dev_recall = bc_updates_joint(
            agent, opt, demos, args, rng)
        target.load_state_dict(agent.state_dict())
        print(json.dumps({
            'distill_encoder': args.distill_encoder,
            'demo_eps': kept_eps, 'demo_pairs': len(demos),
            'dev_frac': round(dev, 4), 'bc_loss': round(bc_loss, 4),
            'dev_recall': round(dev_recall, 4)}), flush=True)
        if args.episodes <= 0:
            agent.save(out / 'agent.pt',
                       meta={**meta, 'episode': 0,
                             'bc_only': True, 'distill': True})

    log = []
    for ep in range(1, args.episodes + 1):
        injects = sample_injects(rng, allow_none_p=args.allow_none_p,
                                 p_corridor=args.p_corridor)
        # fresh SUMO world per episode -> independent traffic seed
        world.close()
        world = build_world(args.scene, args.traffic_seed_base + ep,
                            Path(args.runtime_dir or '/tmp/takeover_signal'),
                            out / 'logger',
                            {'t_fixed': args.t_fixed, 't_min': args.t_min,
                             'min_green_vehicle': 3,
                             'max_red_vehicle': 6})
        ranks = list(range(len(world.intersection_ids)))
        fb = FeatureBuilder(
            world, injects, report_delay=args.report_delay,
            layout='identity', phys=args.phys,
            include_forecast=not args.no_forecast, phase_age=True,
            intent_coord_w=args.intent_coord_w)
        q_base_fn = None
        if residual_mode:
            from scene_attention.scene_mp import bound_dqn_agent
            from hitl.live.controllers import (
                inference_q_values,
            )
            dqn = bound_dqn_agent(world, args.residual_dqn,
                                  rank=ranks[0])
            q_base_fn = lambda dqn=dqn: inference_q_values(dqn)
        eps = max(args.eps_final,
                  args.eps_start - (args.eps_start - args.eps_final)
                  * ep / max(1, args.eps_decay_episodes))
        transitions, records = run_signal_episode(
            world, fb, ranks, injects=injects, agent=agent,
            interval=args.interval, episode_seconds=args.episode_seconds,
            epsilon=eps, rng=rng, switch_penalty=args.switch_penalty,
            q_base_fn=q_base_fn, orig_reward=args.orig_reward,
            scene_w=args.scene_reward_w)
        replay.extend(transitions)
        if residual_mode:
            stats = update_residual(agent, target, opt, replay, args, rng)
        elif joint_mode:
            stats = update_joint(agent, target, opt, replay, args, rng)
            drift = float(np.mean([
                float((p - base_init[k]).abs().mean())
                for k, p in agent.base.state_dict().items()]))
            stats['base_drift'] = round(drift, 6)
        else:
            stats = update(agent, target, opt, replay, args, rng)
        if ep % args.target_sync == 0:
            target.load_state_dict(agent.state_dict())
        ret = float(sum(t['r'] for t in transitions))
        mean_q = float(np.mean([r['total_queue'] for r in records])) \
            if records else 0.0
        ev_edge = (injects[0]['edge_from'] if injects else '')
        ev_lanes = set(fb.edge_to_laneidx.get(ev_edge, []))
        ev_q = float(sum(
            sum(rec['lane_queue'].get(fb.lanes[i], 0) for i in ev_lanes)
            for rec in records)) if ev_lanes else 0.0
        row = {
            'ep': ep, 'eps': round(eps, 3), 'return': round(ret, 3),
            'mean_queue': round(mean_q, 2), 'ev_queue_int': round(ev_q, 1),
            'event': (f"{injects[0]['kind']}:{ev_edge}" if injects
                      else 'none'),
            **stats,
        }
        log.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if ep % args.ckpt_every == 0 or ep == args.episodes:
            agent.save(out / 'agent.pt', meta={**meta, 'episode': ep})
    (out / 'train_log.json').write_text(
        json.dumps(log, ensure_ascii=False, indent=1))


def bc_updates_joint(policy, opt, demos, args, rng):
    """Weighted-CE imitation of adapter actions on the joint student.

    demos: (snap, teacher_action, dev) where dev flags the rare ticks
    the adapter deviated from the frozen base's argmax — those carry
    the scene-conditioned signal and get ``1 + distill_dev_weight``
    weight so the majority "agree with base" labels don't drown them.
    Returns (mean_loss, deviation_recall)."""
    import torch
    import torch.nn.functional as F
    from scene_attention.dispatch_model import collate
    if not demos:
        return 0.0, 0.0
    losses, hits, n_dev = [], 0, 0
    for _ in range(args.distill_updates):
        idx = rng.integers(0, len(demos), size=args.batch)
        s = collate([demos[i][0] for i in idx])
        a = torch.as_tensor([demos[i][1] for i in idx])
        wt = torch.as_tensor(
            [1.0 + args.distill_dev_weight * demos[i][2] for i in idx],
            dtype=torch.float32)
        q, _ = policy.q_values(s)
        loss = (F.cross_entropy(q, a, reduction='none') * wt).mean()
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(policy.parameters(),
                                           args.grad_clip)
        opt.step()
        opt.zero_grad()
        losses.append(loss.item())
        dev_mask = torch.as_tensor(
            [demos[i][2] for i in idx], dtype=torch.bool)
        if dev_mask.any():
            n_dev += 1
            hits += int((q.argmax(-1)[dev_mask] == a[dev_mask]).sum())
    dev_total = sum(d[2] for d in demos)
    # final pass: recall over all deviation samples (informative ticks)
    recall = 0.0
    if dev_total:
        import torch as _t
        correct = 0
        idx_all = _t.arange(0, len(demos))
        for chunk in idx_all.split(512):
            s = collate([demos[i][0] for i in chunk])
            a = _t.as_tensor([demos[i][1] for i in chunk])
            dv = _t.as_tensor([demos[i][2] for i in chunk],
                              dtype=_t.bool)
            if dv.any():
                with _t.no_grad():
                    q, _ = policy.q_values(s)
                correct += int((q.argmax(-1)[dv] == a[dv]).sum())
        recall = correct / float(dev_total)
    return float(np.mean(losses)), recall


def bc_updates(agent, opt, demos, args, rng):
    """Warm-start: imitate the teacher's phase choices (CE over Q logits)."""
    import torch
    from scene_attention.dispatch_model import collate
    if not demos:
        return 0.0
    losses = []
    for _ in range(args.bc_updates):
        idx = rng.integers(0, len(demos), size=args.batch)
        s = collate([demos[i][0] for i in idx])
        a = torch.as_tensor([demos[i][1] for i in idx])
        q, _ = agent.q_values(s)
        loss = torch.nn.functional.cross_entropy(q, a)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(agent.parameters(),
                                           args.grad_clip)
        opt.step()
        opt.zero_grad()
        losses.append(loss.item())
    return float(np.mean(losses))


def update_residual(policy, target, opt, replay, args, rng):
    """Double-DQN TD on the residual policy's biased Q surface.

    q_base enters as stored per-tick values ('qb'/'qbp') — the frozen
    base stays outside the graph; gradients reach only the encoder and
    the zero-init delta head.  ``res_reg`` optionally penalises the
    normalised intervention magnitude (|Q-Q_base|/cap), a sparsity
    prior for "engage only where it pays".
    """
    import torch
    from scene_attention.dispatch_model import collate
    if len(replay) < args.batch * 2:
        return {'loss': 0.0, 'aux': 0.0, 'gate': 0.0}
    losses, auxs, gates = [], [], []
    idxs = rng.integers(0, len(replay),
                        size=(args.updates_per_ep, args.batch))
    rlist = list(replay)
    for idx_row in idxs:
        batch_tr = [rlist[i] for i in idx_row]
        s = collate([t['s'] for t in batch_tr])
        sp = collate([t['sp'] for t in batch_tr])
        qb = torch.as_tensor(
            np.stack([t['qb'] for t in batch_tr]), dtype=torch.float32)
        qbp = torch.as_tensor(
            np.stack([t['qbp'] for t in batch_tr]), dtype=torch.float32)
        a = torch.as_tensor([t['a'] for t in batch_tr])
        r = torch.as_tensor([t['r'] for t in batch_tr],
                            dtype=torch.float32)
        done = torch.as_tensor([t['done'] for t in batch_tr],
                               dtype=torch.float32)
        q, w = policy(s, qb)
        with torch.no_grad():
            a_star = policy(sp, qbp)[0].argmax(dim=-1)
            q_next = target(sp, qbp)[0]
            td = r + args.gamma * (1 - done) * q_next.gather(
                1, a_star.unsqueeze(1)).squeeze(1)
        loss = torch.nn.functional.smooth_l1_loss(
            q.gather(1, a.unsqueeze(1)).squeeze(1), td)
        aux = torch.zeros(())
        terms = []
        for i, t in enumerate(batch_tr):
            tw = t['s'].get('intent_lane_w')
            if tw is not None and float(np.asarray(tw).sum()) > 0:
                twt = torch.as_tensor(np.asarray(tw, dtype=np.float32))
                terms.append(-(twt * torch.log(
                    w[i].clamp_min(1e-8))).sum())
                continue
            tgt = (t['s'].get('intent_lane_idx')
                     or t['s'].get('event_lane_idx') or [])
            if tgt:
                p = w[i, torch.as_tensor(tgt)].sum()
                terms.append(-torch.log(p.clamp_min(1e-6)))
        if terms:
            aux = torch.stack(terms).mean() * args.aux_weight
        reg = torch.zeros(())
        if args.res_reg > 0:
            spread = (qb.max(dim=-1).values
                      - qb.min(dim=-1).values).clamp_min(1e-6)
            cap = args.res_dose * spread.unsqueeze(-1)
            reg = args.res_reg * ((q - qb) / cap).pow(2).mean()
        (loss + aux + reg).backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(policy.parameters(),
                                           args.grad_clip)
        opt.step()
        opt.zero_grad()
        losses.append(loss.item())
        auxs.append(float(aux))
        gates.append(float(policy.bind_gate(w).mean()))
    return {'loss': round(float(np.mean(losses)), 4),
            'aux': round(float(np.mean(auxs)), 4),
            'gate': round(float(np.mean(gates)), 4)}


def update_joint(policy, target, opt, replay, args, rng):
    """Double-DQN TD for the joint policy — the base Q network is in
    the graph, so gradients reach BOTH the controller's own weights
    (small lr) and the scene branch.  No stored qb needed."""
    import torch
    from scene_attention.dispatch_model import collate
    if len(replay) < args.batch * 2:
        return {'loss': 0.0, 'aux': 0.0, 'gate': 0.0}
    losses, auxs, gates = [], [], []
    idxs = rng.integers(0, len(replay),
                        size=(args.updates_per_ep, args.batch))
    rlist = list(replay)
    for idx_row in idxs:
        batch_tr = [rlist[i] for i in idx_row]
        s = collate([t['s'] for t in batch_tr])
        sp = collate([t['sp'] for t in batch_tr])
        a = torch.as_tensor([t['a'] for t in batch_tr])
        r = torch.as_tensor([t['r'] for t in batch_tr],
                            dtype=torch.float32)
        done = torch.as_tensor([t['done'] for t in batch_tr],
                               dtype=torch.float32)
        q, w = policy(s)
        with torch.no_grad():
            a_star = policy(sp)[0].argmax(dim=-1)
            q_next = target(sp)[0]
            td = r + args.gamma * (1 - done) * q_next.gather(
                1, a_star.unsqueeze(1)).squeeze(1)
        loss = torch.nn.functional.smooth_l1_loss(
            q.gather(1, a.unsqueeze(1)).squeeze(1), td)
        aux = torch.zeros(())
        terms = []
        for i, t in enumerate(batch_tr):
            tw = t['s'].get('intent_lane_w')
            if tw is not None and float(np.asarray(tw).sum()) > 0:
                twt = torch.as_tensor(np.asarray(tw, dtype=np.float32))
                terms.append(-(twt * torch.log(
                    w[i].clamp_min(1e-8))).sum())
                continue
            tgt = (t['s'].get('intent_lane_idx')
                     or t['s'].get('event_lane_idx') or [])
            if tgt:
                p = w[i, torch.as_tensor(tgt)].sum()
                terms.append(-torch.log(p.clamp_min(1e-6)))
        if terms:
            aux = torch.stack(terms).mean() * args.aux_weight
        (loss + aux).backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(policy.parameters(),
                                           args.grad_clip)
        opt.step()
        opt.zero_grad()
        losses.append(loss.item())
        auxs.append(float(aux))
        gates.append(float(policy.bind_gate(w).mean()))
    return {'loss': round(float(np.mean(losses)), 4),
            'aux': round(float(np.mean(auxs)), 4),
            'gate': round(float(np.mean(gates)), 4)}


def update(agent, target, opt, replay, args, rng):
    import torch
    from scene_attention.dispatch_model import collate
    if len(replay) < args.batch * 2:
        return {'loss': 0.0, 'aux': 0.0}
    losses, auxs = [], []
    idxs = rng.integers(0, len(replay),
                        size=(args.updates_per_ep, args.batch))
    rlist = list(replay)
    for idx_row in idxs:
        batch_tr = [rlist[i] for i in idx_row]
        s = collate([t['s'] for t in batch_tr])
        sp = collate([t['sp'] for t in batch_tr])
        a = torch.as_tensor([t['a'] for t in batch_tr])
        r = torch.as_tensor([t['r'] for t in batch_tr],
                            dtype=torch.float32)
        done = torch.as_tensor([t['done'] for t in batch_tr],
                               dtype=torch.float32)
        q, w = agent.q_values(s)
        with torch.no_grad():
            a_star = agent.q_values(sp)[0].argmax(dim=-1)
            q_next = target.q_values(sp)[0]
            td = r + args.gamma * (1 - done) * q_next.gather(
                1, a_star.unsqueeze(1)).squeeze(1)
        loss = torch.nn.functional.smooth_l1_loss(
            q.gather(1, a.unsqueeze(1)).squeeze(1), td)
        aux = torch.zeros(())
        if agent.variant in ('attn', 'sga'):
            terms = []
            for i, t in enumerate(batch_tr):
                tw = t['s'].get('intent_lane_w')
                if tw is not None and float(np.asarray(tw).sum()) > 0:
                    twt = torch.as_tensor(np.asarray(tw, dtype=np.float32))
                    terms.append(-(twt * torch.log(
                        w[i].clamp_min(1e-8))).sum())
                    continue
                tgt = (t['s'].get('intent_lane_idx')
                     or t['s'].get('event_lane_idx') or [])
                if tgt:
                    p = w[i, torch.as_tensor(tgt)].sum()
                    terms.append(-torch.log(p.clamp_min(1e-6)))
            if terms:
                aux = torch.stack(terms).mean() * args.aux_weight
        (loss + aux).backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(agent.parameters(),
                                           args.grad_clip)
        opt.step()
        opt.zero_grad()
        losses.append(loss.item())
        auxs.append(float(aux))
    return {'loss': round(float(np.mean(losses)), 4),
            'aux': round(float(np.mean(auxs)), 4)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene', default='S2')
    p.add_argument('--variant', default='sga',
                   choices=['attn', 'sga', 'selfattn', 'concat',
                            'selfattn_st', 'concat_st', 'mlp_st'])
    p.add_argument('--phys', choices=['full', 'queue'], default='full')
    p.add_argument('--no-forecast', action='store_true',
                   help='zero the est_remaining text slot (report = '
                        'phenomenon only, duration is a prediction target)')
    p.add_argument('--report-delay', type=float, default=0.0,
                   help='seconds between event onset and the broadcast '
                        'reaching the model (0 = immediate report)')
    p.add_argument('--residual-pool', choices=['mean', 'mms'],
                   default='mean')
    p.add_argument('--d-model', type=int, default=64)
    p.add_argument('--episodes', type=int, default=100)
    p.add_argument('--episode-seconds', type=int, default=3600)
    p.add_argument('--interval', type=int, default=10)
    p.add_argument('--allow-none-p', type=float, default=0.25)
    p.add_argument('--residual-dqn', default=None,
                   help='PATH to frozen DQN weights — train a '
                        'SceneResidualPolicy (bounded gated ΔQ on the '
                        'live Q surface) instead of a from-scratch '
                        'agent; variant must be sga or attn')
    p.add_argument('--res-dose', type=float, default=0.35,
                   help='residual tilt budget as a fraction of the live '
                        'Q spread (same cap as the adapter)')
    p.add_argument('--res-bind-tau', type=float, default=0.4,
                   help='edge bind mass below this -> gate exactly 0 -> '
                        'policy ≡ frozen base')
    p.add_argument('--res-reg', type=float, default=0.0,
                   help='weight of the normalised-intervention penalty '
                        '(engage-only-where-it-pays sparsity)')
    p.add_argument('--joint-dqn', default=None,
                   help='PATH to deployed DQN weights — train a '
                        'SceneJointPolicy: base in-graph, fine-tuned '
                        'with --base-lr; TD lands in the controller '
                        'itself (exclusive with --residual-dqn)')
    p.add_argument('--init-from', default=None,
                   help='warm-start agent+target from a prior checkpoint '
                        '(e.g. the BC student) before TD continues')
    p.add_argument('--p-corridor', type=float, default=0.0,
                   help='prob. of an unreported feeder-load surge paired '
                        'with each event — makes watching the corridor '
                        'edge pay off physically')
    p.add_argument('--intent-coord-w', type=float, default=0.0,
                   help='governance-corridor intent: feeder+discharge '
                        'edges of the reported edge carry this fraction '
                        'of its intent weight (0 = v1 single-edge)')
    p.add_argument('--scene-reward-w', type=float, default=0.0,
                   help='SIGMA-style intent utility weight — while a '
                        'report is live, -w*mean(queue on the intent '
                        'edges) is added to the reward, making the text '
                        'channel load-bearing instead of optional')
    p.add_argument('--orig-reward', action='store_true',
                   help='use the deployed DQN original objective '
                        '(-12 x mean incoming-lane waiting count) instead '
                        'of the probe shaping — keeps the TD target '
                        'consistent with what the base converged on')
    p.add_argument('--base-lr', type=float, default=1e-5,
                   help='fine-tune lr for the in-graph base DQN '
                        '(joint mode)')
    p.add_argument('--distill-encoder', default=None,
                   help='PATH to a trained agent.pt — its encoder '
                        'drives the SceneGuidedDQN teacher that emits '
                        'the demo labels')
    p.add_argument('--distill-eps', type=int, default=0,
                   help='adapter-teacher demo episodes before RL '
                        '(joint mode; 0 = off)')
    p.add_argument('--distill-updates', type=int, default=1500)
    p.add_argument('--distill-dev-weight', type=float, default=2.0,
                   help='extra CE weight on deviation ticks '
                        '(adapter != base argmax)')
    p.add_argument('--distill-filter', action='store_true',
                   help='keep only demo episodes where the adapter '
                        'beat vanilla DQN on the same seed')
    p.add_argument('--distill-beta', type=float, default=40.0)
    p.add_argument('--distill-gate', type=float, default=8.0)
    p.add_argument('--teacher', default='maxpressure',
                   help='teacher controller for demonstration episodes '
                        '(0 episodes = BC off)')
    p.add_argument('--teacher-eps', type=int, default=0)
    p.add_argument('--bc-updates', type=int, default=500)
    p.add_argument('--switch-penalty', type=float, default=0.0,
                   help='reward cost per commanded phase change — the '
                        'yellow/all-red transition is real capacity loss; '
                        'without it greedy policies collapse to toggling')
    p.add_argument('--eps-start', type=float, default=1.0)
    p.add_argument('--eps-final', type=float, default=0.05)
    p.add_argument('--eps-decay-episodes', type=int, default=60)
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--buffer', type=int, default=50000)
    p.add_argument('--updates-per-ep', type=int, default=60)
    p.add_argument('--gamma', type=float, default=0.95)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--grad-clip', type=float, default=10.0)
    p.add_argument('--aux-weight', type=float, default=0.05)
    p.add_argument('--target-sync', type=int, default=5)
    p.add_argument('--ckpt-every', type=int, default=25)
    p.add_argument('--rng-seed', type=int, default=0)
    p.add_argument('--traffic-seed-base', type=int, default=100)
    p.add_argument('--t-fixed', type=int, default=30)
    p.add_argument('--t-min', type=int, default=10)
    p.add_argument('--runtime-dir', default=None)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    train(args)


if __name__ == '__main__':
    main()
