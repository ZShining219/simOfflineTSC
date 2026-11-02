"""Build the scen_cov_v1 pilot queue manifest + per-run overlays.

Spec: docs/experiments/2026-11-02_scen_cov_v1.md (P4 card, dsh/design@e9eae61).
Protocol mirrors dev/cl_hz1x1/build_queue.py (online clear arm) and
build_queue_so.py (semi-offline arm), shrunk to the pilot budget:

  order o2 = [sumohz1x1, sumohz1x1_config4, sumohz1x1_config2, sumohz1x1_config3]
  4 stages x <stage_episodes> ep, stage>1 resumes via model.stage_checkpoint
  lower-triangle frozen eval after each stage (10 cells x 3 eval sumo seeds)
  final coverage eval: L1 4 cells + L2 4 unseen cells x 3 eval sumo seeds
  maxpressure baseline: 8 cells x 3 eval sumo seeds x 1 ep (same seeds as eval)

Outputs (under --out-dir, default <run-root>/artifacts/scen_cov_v1):
  overlays/<run_id>.yml      per-run experiment overlay consumed by run.py
  run_queue_<tag>.json       QueueRunner manifest (depends_on chains)
  ledger_rows_<tag>.jsonl    REGISTERED rows to append into ledger/runs.jsonl

The script only writes files; it never launches runs.

Usage:
    python dev/scen_cov/build_pilot.py --run-root <run-worktree>
    python dev/scen_cov/build_pilot.py --run-root <run-worktree> \
        --stage-episodes 2 --train-seeds 0 --arms ddqn_clear \
        --out-dir /tmp/scv1_smoke --tag smoke
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ORDER = ['sumohz1x1', 'sumohz1x1_config4',
         'sumohz1x1_config2', 'sumohz1x1_config3']
NET_TAG = {'sumohz1x1': 's1', 'sumohz1x1_config4': 'c4',
           'sumohz1x1_config2': 'c2', 'sumohz1x1_config3': 'c3'}
L2_NETS = {'sumohz1x1_sparse_ultra': 'sparse_ultra',
           'sumohz1x1_dense_extreme': 'dense_extreme',
           'sumohz1x1_lt_extreme': 'lt_extreme',
           'sumohz1x1_asym_wonly': 'asym_wonly'}
ALL_NETS = {**NET_TAG, **L2_NETS}
OFFLINE_RATIO = 0.25
SO_BASE = {
    'offline_sampling_strategy': 'scene_intersection_balanced',
    'history_access_mode': 'causal',
    'collect_history': True,
}
ARMS = {
    'ddqn_clear': {
        'agent': 'dqn', 'condition': 'online',
        'model_extra': {'double_dqn': True},
    },
    'soddqn': {
        'agent': 'dqn', 'condition': 'semi_offline',
        'model_extra': {'double_dqn': True, **SO_BASE},
    },
}
ENV = {
    'CUDA_VISIBLE_DEVICES': '-1',
    'OMP_NUM_THREADS': '1',
    'MKL_NUM_THREADS': '1',
    'OPENBLAS_NUM_THREADS': '1',
    'NUMEXPR_NUM_THREADS': '1',
    'VECLIB_MAXIMUM_THREADS': '1',
}


def yaml_value(value):
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        return '[' + ', '.join(yaml_value(v) for v in value) + ']'
    return str(value)


def write_overlay(path, model_extra, trainer_extra):
    lines = []
    if model_extra:
        lines.append('model:')
        lines.extend(f'  {k}: {yaml_value(v)}' for k, v in model_extra.items())
        lines.append('')
    lines.append('trainer:')
    lines.extend(f'  {k}: {yaml_value(v)}' for k, v in trainer_extra.items())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-root', required=True,
                    help='worktree whose run.py + data/ are used (cwd of runs)')
    ap.add_argument('--out-dir', default=None,
                    help='default: <run-root>/artifacts/scen_cov_v1')
    ap.add_argument('--tag', default='scv1', help='queue/ledger file tag')
    ap.add_argument('--stage-episodes', type=int, default=25)
    ap.add_argument('--train-seeds', default='0,1')
    ap.add_argument('--eval-seeds', default='101,102,103')
    ap.add_argument('--arms', default='ddqn_clear,soddqn')
    ap.add_argument('--offline-ratio', type=float, default=OFFLINE_RATIO)
    ap.add_argument('--skip-mp', action='store_true',
                    help='do not emit maxpressure baseline tasks')
    ap.add_argument('--extra-args', default='',
                    help='extra run.py args appended to every command (debug)')
    args = ap.parse_args()

    run_root = Path(args.run_root).expanduser().resolve()
    if not (run_root / 'run.py').is_file():
        raise SystemExit(f'run-root has no run.py: {run_root}')
    out_dir = (Path(args.out_dir).expanduser().resolve() if args.out_dir
               else run_root / 'artifacts' / 'scen_cov_v1')
    overlay_dir = out_dir / 'overlays'
    out_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    script_root = Path(__file__).resolve().parents[2]

    def _git(*cmd):
        try:
            return subprocess.run(
                ['git', '-C', str(script_root), *cmd],
                capture_output=True, text=True, timeout=10,
            ).stdout.strip()
        except Exception:
            return ''

    builder_commit = _git('rev-parse', '--short', 'HEAD') or 'unknown'
    builder_branch = _git('rev-parse', '--abbrev-ref', 'HEAD') or 'unknown'
    builder_dirty = bool(_git('status', '--porcelain'))
    script_hashes = {}
    for name in ('build_pilot.py', 'harvest_coverage.py'):
        path = script_root / 'dev' / 'scen_cov' / name
        script_hashes[name] = (
            hashlib.sha256(path.read_bytes()).hexdigest()
            if path.is_file() else 'missing')
    scripts_sha256 = ','.join(
        f'{name}:{digest}' for name, digest in script_hashes.items())
    # r2 audit fields (P5 card ad66a8b): the three items ledger notes must
    # carry so "which code ran" stays reconstructable without merging main.
    audit_note = (f'scen_cov_scripts={builder_branch}@{builder_commit}; '
                  f'scripts_sha256={scripts_sha256}; '
                  f'workspace_dirty={str(builder_dirty).lower()}')

    py = sys.executable
    train_seeds = [int(x) for x in args.train_seeds.split(',') if x.strip()]
    eval_seeds = [int(x) for x in args.eval_seeds.split(',') if x.strip()]
    arm_names = [x.strip() for x in args.arms.split(',') if x.strip()]
    for arm in arm_names:
        if arm not in ARMS:
            raise SystemExit(f'unknown arm {arm!r}; choose {list(ARMS)}')
    stage_ep = args.stage_episodes
    extra = args.extra_args.split() if args.extra_args else []
    out_root = run_root / 'data' / 'output_data' / 'tsc'

    def out_dir_for(agent, net, run_id):
        return out_root / f'sumo_{agent}' / net / run_id

    def run_cmd(agent, net, prefix, overlay, seed, sumo_seed):
        return [py, 'run.py', '-w', 'sumo', '-a', agent, '-n', net,
                '--seed', str(seed), '--sumo_seed', str(sumo_seed),
                '--interface', 'libsumo', '--prefix', prefix,
                '--experiment-config', str(overlay)] + extra

    def completion(run_dir):
        return {'path': str(run_dir / 'run_status.json'),
                'json_field': 'status', 'equals': '已完成'}

    def train_overlay(run_id, arm, stage_idx, history_paths, prev_ckpt):
        me = dict(ARMS[arm]['model_extra'])
        me['double_dqn'] = True
        me['scene_id'] = ORDER[stage_idx]
        me['roadnet_id'] = 'hz1x1'
        me['training_stage'] = stage_idx
        me['scene_order'] = list(ORDER)
        me['offline_ratio'] = (
            0.0 if stage_idx == 0 else args.offline_ratio)
        if history_paths:
            me['history_paths'] = [str(p) for p in history_paths]
        if stage_idx > 0:
            me['stage_checkpoint'] = str(prev_ckpt)
            me['replay_policy'] = 'clear'
            me['epsilon_mode'] = 'reset_schedule'
            me['stage_load_optimizer'] = 'true'
        te = {
            'episodes': stage_ep,
            'evaluation_episodes': [0, stage_ep],
            'resumable_checkpoint_episodes': [0, stage_ep],
            'test_when_train': False,
        }
        path = overlay_dir / f'{run_id}.yml'
        write_overlay(path, me, te)
        return path

    def eval_overlay(run_id, arm, ckpt):
        me = dict(ARMS[arm]['model_extra'])
        for key in SO_BASE:
            me.pop(key, None)
        me.update({
            'double_dqn': True,
            'stage_checkpoint': str(ckpt),
            'replay_policy': 'clear',
            'epsilon_mode': 'fixed_low',
            'stage_load_optimizer': 'true',
        })
        te = {
            'episodes': 0,
            'evaluation_episodes': [0],
            'resumable_checkpoint_episodes': [0],
            'test_when_train': False,
        }
        path = overlay_dir / f'{run_id}.yml'
        write_overlay(path, me, te)
        return path

    tasks = []
    ledger = []

    def add_ledger(run_id, arm, net, seed, kind, stage, episodes,
                   eval_mode=None, sumo_seed=None, behavior=None,
                   condition=None, episodes_total=None, notes=''):
        ledger.append({
            'run_id': run_id, 'line': 'scen_cov_v1',
            'campaign': 'scen_cov_v1', 'arm': arm, 'alias_of': None,
            'net': net, 'seed': seed,
            'episodes_total': episodes if episodes_total is None
            else episodes_total,
            'episodes_increment': episodes, 'resume_from': None, 'ngpu': -1,
            'status': 'REGISTERED', 'plan_id': 'scen_cov_v1_pilot',
            'code_commit': None, 'dirty': None,
            'queue_file': f'artifacts/scen_cov_v1/run_queue_{args.tag}.json',
            'run_dir': None, 'tmux': None, 'started': None,
            'eval_mode': eval_mode, 'behavior_source': behavior,
            'metrics': {}, 'abort_reason': None, 'error': None,
            'evidence_lost': False,
            'notes': notes + f' | {audit_note}',
            'run_kind': kind, 'config_sha': None,
            'sumo_seed_mode': 'explicit' if sumo_seed is not None else None,
            'sumo_seed': sumo_seed, 'eval_seed': sumo_seed,
            'condition': condition, 'superseded_by': None,
            'runtime_env': 'colight', 'end_time': None,
        })

    def add_task(task):
        tasks.append(task)

    for seed in train_seeds:
        for arm in arm_names:
            spec = ARMS[arm]
            agent = spec['agent']
            cond = spec['condition']
            is_semi = cond == 'semi_offline'
            prev_ckpt = None
            prev_rid = None
            history_paths = []
            for st, net in enumerate(ORDER):
                rid = f'scv1_{arm}_o2_sd{seed}_st{st + 1}'
                odir = out_dir_for(agent, net, rid)
                ov = train_overlay(rid, arm, st, history_paths, prev_ckpt)
                ckpt = (odir / 'checkpoints' / 'resumable' /
                        f'episode_{stage_ep:04d}.pt')
                task = {
                    'run_id': rid,
                    'command': run_cmd(agent, net, rid, ov, seed, 0),
                    'cwd': str(run_root), 'env': dict(ENV),
                    'output_path': str(odir),
                    'completion': completion(odir),
                    'checkpoint_path': str(ckpt),
                    'campaign': 'scen_cov_v1', 'arm': arm,
                    'order': 'o2', 'stage': st + 1, 'scene': net,
                    'scene_tag': NET_TAG[net],
                    'training_seed': seed,
                    'replay_condition': cond,
                    'episode_budget': stage_ep,
                }
                requires = []
                if prev_rid:
                    task['depends_on'] = [prev_rid]
                    requires.append({'path': str(prev_ckpt)})
                    # p5-04 fix: only the semi-offline arm archives
                    # history; the clear arm chains via
                    # stage_checkpoint only (cl1 protocol) and must
                    # not require history_archive/manifest.json.
                    if is_semi:
                        requires.extend(
                            {'path': str(p / 'manifest.json')}
                            for p in history_paths)
                if requires:
                    task['requires'] = requires
                add_task(task)
                add_ledger(
                    rid, arm, net, seed, 'train', f'o2/st{st + 1}',
                    stage_ep, condition=cond,
                    notes=(f'REGISTERED: scen_cov_v1 {cond} train '
                           f'stage={st + 1} offline_ratio='
                           f'{0.0 if st == 0 else args.offline_ratio}'))
                for j in range(st + 1):
                    enet = ORDER[j]
                    etag = NET_TAG[enet]
                    for es in eval_seeds:
                        eid = (f'scv1_{arm}_o2_sd{seed}_ev{st + 1}_'
                               f'{etag}_es{es}')
                        eodir = out_dir_for(agent, enet, eid)
                        eov = eval_overlay(eid, arm, ckpt)
                        add_task({
                            'run_id': eid,
                            'command': run_cmd(agent, enet, eid, eov,
                                               seed, es),
                            'cwd': str(run_root), 'env': dict(ENV),
                            'output_path': str(eodir),
                            'completion': completion(eodir),
                            'depends_on': [rid],
                            'requires': [{'path': str(ckpt)}],
                            'campaign': 'scen_cov_v1', 'arm': arm,
                            'order': 'o2', 'stage': st + 1,
                            'scene': enet, 'scene_tag': etag,
                            'training_seed': seed, 'eval_sumo_seed': es,
                            'checkpoint_path': str(ckpt),
                            'run_kind': 'eval',
                        })
                        add_ledger(
                            eid, arm, enet, seed, 'eval',
                            f'o2/ev{st + 1}', 0,
                            eval_mode='frozen_stage_ckpt', sumo_seed=es,
                            behavior=rid, condition=cond,
                            notes=('REGISTERED: scen_cov_v1 lower-triangle '
                                   f'eval stage={st + 1} sumo_seed={es}'))
                if is_semi:
                    history_paths.append(odir / 'history_archive')
                prev_ckpt, prev_rid = ckpt, rid

            # final coverage eval: L1 4 cells + L2 4 unseen cells
            if prev_rid is None:
                continue
            for net, tag in ALL_NETS.items():
                for es in eval_seeds:
                    cid = f'scv1_{arm}_o2_sd{seed}_cov_{tag}_es{es}'
                    codir = out_dir_for(agent, net, cid)
                    cov = eval_overlay(cid, arm, prev_ckpt)
                    add_task({
                        'run_id': cid,
                        'command': run_cmd(agent, net, cid, cov, seed, es),
                        'cwd': str(run_root), 'env': dict(ENV),
                        'output_path': str(codir),
                        'completion': completion(codir),
                        'depends_on': [prev_rid],
                        'requires': [{'path': str(prev_ckpt)}],
                        'campaign': 'scen_cov_v1', 'arm': arm,
                        'order': 'o2', 'stage': len(ORDER),
                        'scene': net, 'scene_tag': tag,
                        'training_seed': seed, 'eval_sumo_seed': es,
                        'checkpoint_path': str(prev_ckpt),
                        'run_kind': 'eval', 'eval_layer': (
                            'L1' if net in NET_TAG else 'L2'),
                    })
                    add_ledger(
                        cid, arm, net, seed, 'eval', 'o2/cov', 0,
                        eval_mode='frozen_stage_ckpt', sumo_seed=es,
                        behavior=prev_rid, condition=cond,
                        notes=('REGISTERED: scen_cov_v1 coverage eval '
                               f'layer={"L1" if net in NET_TAG else "L2"} '
                               f'sumo_seed={es}'))

    if not args.skip_mp:
        for net, tag in ALL_NETS.items():
            for es in eval_seeds:
                rid = f'scv1_mp_{tag}_es{es}'
                odir = out_dir_for('maxpressure', net, rid)
                add_task({
                    'run_id': rid,
                    'command': [py, 'run.py', '-w', 'sumo', '-a',
                                'maxpressure', '-n', net, '--seed', '0',
                                '--sumo_seed', str(es), '--interface',
                                'libsumo', '--prefix', rid],
                    'cwd': str(run_root), 'env': dict(ENV),
                    'output_path': str(odir),
                    'completion': completion(odir),
                    'campaign': 'scen_cov_v1', 'arm': 'maxpressure',
                    'order': 'o2', 'stage': 0, 'scene': net,
                    'scene_tag': tag, 'training_seed': 0,
                    'eval_sumo_seed': es, 'run_kind': 'eval',
                    'eval_layer': 'L1' if net in NET_TAG else 'L2',
                })
                add_ledger(
                    rid, 'maxpressure', net, 0, 'eval', 'baseline', 1,
                    eval_mode='single_episode', sumo_seed=es,
                    condition='baseline', episodes_total=1,
                    notes=('REGISTERED: scen_cov_v1 maxpressure baseline '
                           f'sumo_seed={es} (1 ep)'))

    run_ids = [t['run_id'] for t in tasks]
    dup = {r for r in run_ids if run_ids.count(r) > 1}
    if dup:
        raise SystemExit(f'duplicate run_id: {sorted(dup)[:5]}')
    required = {'run_id', 'command', 'cwd', 'env', 'output_path',
                'completion', 'campaign'}
    for t in tasks:
        missing = required - set(t)
        if missing:
            raise SystemExit(f'{t["run_id"]} missing {missing}')

    n_train = sum(1 for t in tasks if t.get('run_kind') != 'eval')
    n_eval = len(tasks) - n_train
    train_episodes = sum(int(t.get('episode_budget') or 0)
                         for t in tasks if 'episode_budget' in t)
    manifest = {
        'schema_version': 1,
        'plan': 'scen_cov_v1_pilot',
        'spec': 'docs/experiments/2026-11-02_scen_cov_v1.md',
        'run_root': str(run_root),
        'code_commit_at_build': None,
        'builder_commit': builder_commit,
        'builder_branch': builder_branch,
        'builder_dirty': builder_dirty,
        'scripts_sha256': script_hashes,
        'order': ORDER,
        'l1_cells': NET_TAG,
        'l2_cells': L2_NETS,
        'arms': {a: ARMS[a]['condition'] for a in arm_names},
        'train_seeds': train_seeds,
        'eval_sumo_seeds': eval_seeds,
        'stage_episodes': stage_ep,
        'offline_ratio': args.offline_ratio,
        'expected_tasks': len(tasks),
        'train_tasks': n_train,
        'eval_tasks': n_eval,
        'train_episodes_total': train_episodes,
        'budget_limits': {'train_episodes_max': 600,
                          'eval_attempts_max': 324},
        'tasks': tasks,
    }
    queue_path = out_dir / f'run_queue_{args.tag}.json'
    queue_path.write_text(
        json.dumps(manifest, indent=1, ensure_ascii=False), encoding='utf-8')
    ledger_path = out_dir / f'ledger_rows_{args.tag}.jsonl'
    ledger_path.write_text(
        '\n'.join(json.dumps(r, ensure_ascii=False) for r in ledger) + '\n',
        encoding='utf-8')
    print(f'tasks={len(tasks)} train={n_train} eval={n_eval} '
          f'train_episodes={train_episodes} ledger_rows={len(ledger)} '
          f'builder={builder_branch}@{builder_commit} '
          f'dirty={builder_dirty}')
    print(f'queue -> {queue_path}')
    print(f'ledger rows -> {ledger_path}')
    if train_episodes > 600:
        raise SystemExit('S6 violated: train episode budget > 600')
    if n_eval > 324:
        raise SystemExit('S6 violated: eval attempt budget > 324')


if __name__ == '__main__':
    main()
