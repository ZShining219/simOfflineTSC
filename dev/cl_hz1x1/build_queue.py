"""Build the cl_hz1x1 continual-learning queue manifest + per-run overlays.

Design (frozen in EXPERIMENTS/<campaign> / conversation 2026-10-11):
  - Group: homogeneous hz1x1 SUMO single-intersection packages only.
  - Orders (two fixed sequences over the four formal hz1x1 nets):
      O2 = [sumohz1x1, sumohz1x1_config4, sumohz1x1_config2, sumohz1x1_config3]
      O3 = [sumohz1x1_config2, sumohz1x1_config3, sumohz1x1, sumohz1x1_config4]
  - Arms: ddqn / duel(ing_ddqn) / frap run under replay conditions
    {clear, fifo}; ppo_pfrl runs only under clear (on-policy).
  - Stages: A->B->C->D, 100 episodes each; stages 2-4 continue via
    model.stage_checkpoint (run.py load_shared_stage_checkpoint path).
  - Freeze-eval matrix: after stage k, the resumable ckpt is evaluated
    frozen on nets[:k] x 3 eval sumo seeds (101/102/103) -> lower triangle.
  - Single-task baselines: each arm x each net x 3 seeds, fresh 100ep run.

Outputs (all under gitignored artifacts/cl_hz1x1/):
  overlays/<run_id>.yml   per-run experiment overlay consumed by run.py
  run_queue_cl1.json      QueueRunner manifest (depends_on chains)
  ledger_rows.jsonl       REGISTERED rows to append into ledger/runs.jsonl
"""
import json
import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / 'artifacts' / 'cl_hz1x1'
OVERLAYS = ART / 'overlays'
PY = sys.executable
LD_PRELOAD = str(ROOT / 'data' / 'output_data' / 'cross_algorithm' /
                 'plan5_b100' / 'environment' / 'micromamba' / 'envs' /
                 'colight' / 'lib' / 'libstdc++.so.6')
ENV = {
    'LD_PRELOAD': LD_PRELOAD,
    'CUDA_VISIBLE_DEVICES': '-1',
    'OMP_NUM_THREADS': '1',
    'MKL_NUM_THREADS': '1',
    'OPENBLAS_NUM_THREADS': '1',
    'NUMEXPR_NUM_THREADS': '1',
    'VECLIB_MAXIMUM_THREADS': '1',
}

NETS = {
    's1': 'sumohz1x1',
    'c4': 'sumohz1x1_config4',
    'c2': 'sumohz1x1_config2',
    'c3': 'sumohz1x1_config3',
}
NET_TAG = {v: k for k, v in NETS.items()}
ORDERS = {
    'o2': [NETS['s1'], NETS['c4'], NETS['c2'], NETS['c3']],
    'o3': [NETS['c2'], NETS['c3'], NETS['s1'], NETS['c4']],
}
# PPO arm is the plan5-calibrated variant (v2): entropy_coef 0.1 keeps the
# policy from collapsing to a subset of actions (v1 with entropy 0.001
# produced a flat, non-learning curve); reward_scale x12 + minibatch 90 +
# clip 0.2 + gamma 0.95 match Plan5PPOAdapter's working configuration.
ARMS = {
    'ddqn': {'agent': 'dqn', 'model_extra': {'double_dqn': True}},
    'duel': {'agent': 'dqn',
             'model_extra': {'double_dqn': True, 'dueling': True}},
    'frap': {'agent': 'frap', 'model_extra': {}},
    'ppov2': {'agent': 'ppo_pfrl', 'model_extra': {
        'entropy_coef': 0.1, 'minibatch_size': 90, 'clip_eps': 0.2,
        'ppo_epochs': 4, 'reward_scale': 12.0, 'gamma': 0.95}},
}
CONDS = {  # replay condition -> arms that run it
    'clear': ['ddqn', 'duel', 'frap', 'ppov2'],
    'fifo': ['ddqn', 'duel', 'frap'],
}
SEEDS = [0, 1, 2]
EVAL_SEEDS = [101, 102, 103]
STAGE_EPISODES = 100
EVAL_EPISODES_AT = [0, 50, 100]

OUT = ROOT / 'data' / 'output_data' / 'tsc'


def out_dir(agent, net, prefix):
    return OUT / f'sumo_{agent}' / net / prefix


def overlay_path(run_id):
    return OVERLAYS / f'{run_id}.yml'


def write_overlay(run_id, model_extra, trainer_extra):
    lines = []
    if model_extra:
        lines.append('model:')
        for k, v in model_extra.items():
            lines.append(f'  {k}: {v}' if not isinstance(v, str)
                         else f'  {k}: {v}')
        lines.append('')
    lines.append('trainer:')
    for k, v in trainer_extra.items():
        if isinstance(v, list):
            v = '[' + ', '.join(str(x) for x in v) + ']'
        lines.append(f'  {k}: {v}')
    path = overlay_path(run_id)
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return path


def train_overlay(run_id, arm, stage_idx, prev_ckpt, cond):
    me = dict(ARMS[arm]['model_extra'])
    if stage_idx > 0:
        me['stage_checkpoint'] = prev_ckpt
        me['replay_policy'] = cond
        me['epsilon_mode'] = 'reset_schedule'
        me['stage_load_optimizer'] = 'true'
    te = {
        'episodes': STAGE_EPISODES,
        'evaluation_episodes': EVAL_EPISODES_AT,
        'resumable_checkpoint_episodes': EVAL_EPISODES_AT,
        'test_when_train': 'false',
    }
    return write_overlay(run_id, me, te)


def eval_overlay(run_id, arm, ckpt):
    me = dict(ARMS[arm]['model_extra'])
    me.update({
        'stage_checkpoint': ckpt,
        'replay_policy': 'clear',
        'epsilon_mode': 'fixed_low',
        'stage_load_optimizer': 'true',
    })
    te = {
        'episodes': 0,
        'evaluation_episodes': [0],
        'resumable_checkpoint_episodes': [0],
        'test_when_train': 'false',
    }
    return write_overlay(run_id, me, te)


def cmd(agent, net, prefix, overlay, seed, sumo_seed):
    return [
        PY, 'run.py', '-w', 'sumo', '-a', agent, '-n', net,
        '--seed', str(seed), '--sumo_seed', str(sumo_seed),
        '--interface', 'libsumo', '--prefix', prefix,
        '--experiment-config', str(overlay),
    ]


def completion(prefix_out):
    return {
        'path': str(prefix_out / 'run_status.json'),
        'json_field': 'status',
        'equals': '已完成',
    }


def main():
    OVERLAYS.mkdir(parents=True, exist_ok=True)
    tasks = []
    ledger = []

    def add_ledger(run_id, arm, cond, net, seed, kind, stage, episodes):
        ledger.append({
            'run_id': run_id, 'line': 'cl_hz1x1', 'campaign': 'cl_hz1x1',
            'arm': f'{arm}_{cond}' if cond else arm, 'alias_of': None,
            'net': net, 'seed': seed, 'episodes_total': episodes,
            'episodes_increment': episodes, 'resume_from': None, 'ngpu': -1,
            'status': 'REGISTERED', 'plan_id': 'cl_hz1x1_v1',
            'code_commit': None, 'dirty': None,
            'queue_file': 'artifacts/cl_hz1x1/run_queue_cl1.json',
            'run_dir': None, 'tmux': None, 'started': None,
            'eval_mode': 'frozen_stage_ckpt' if kind == 'eval' else None,
            'behavior_source': None, 'metrics': {}, 'abort_reason': None,
            'error': None, 'evidence_lost': False,
            'notes': f'REGISTERED: cl_hz1x1 continual-learning {kind} '
                     f'order_stage={stage}',
            'run_kind': kind, 'config_sha': None, 'sumo_seed_mode': None,
            'sumo_seed': None, 'eval_seed': None, 'condition': cond,
            'superseded_by': None, 'runtime_env': 'colight',
            'end_time': None,
        })

    # ---- sequential chains ----
    for order_name, nets in ORDERS.items():
        for seed in SEEDS:
            for cond, arms in CONDS.items():
                for arm in arms:
                    agent = ARMS[arm]['agent']
                    prev_ckpt = None
                    prev_run_id = None
                    for st, net in enumerate(nets):
                        rid = (f'cl1_{arm}_{cond}_{order_name}_sd{seed}'
                               f'_st{st + 1}')
                        odir = out_dir(agent, net, rid)
                        ov = train_overlay(
                            rid, arm, st, str(prev_ckpt) if prev_ckpt else '',
                            cond)
                        # stage-1 has no stage_checkpoint; drop empty key
                        if st == 0:
                            txt = ov.read_text()
                            ov.write_text(txt.replace(
                                '  stage_checkpoint: \n', ''), encoding='utf-8')
                        ckpt = (odir / 'checkpoints' / 'resumable' /
                                f'episode_{STAGE_EPISODES:04d}.pt')
                        task = {
                            'run_id': rid,
                            'command': cmd(agent, net, rid, ov, seed, 0),
                            'cwd': str(ROOT), 'env': dict(ENV),
                            'output_path': str(odir),
                            'completion': completion(odir),
                            'checkpoint_path': str(ckpt),
                            'campaign': 'cl_hz1x1', 'arm': f'{arm}_{cond}',
                            'order': order_name, 'stage': st + 1,
                            'scene': net, 'training_seed': seed,
                            'replay_condition': cond,
                            'episode_budget': STAGE_EPISODES,
                        }
                        if prev_run_id:
                            task['depends_on'] = [prev_run_id]
                            task['requires'] = [{
                                'path': str(prev_ckpt)}]
                        tasks.append(task)
                        add_ledger(rid, arm, cond, net, seed, 'train',
                                   f'{order_name}/st{st + 1}', STAGE_EPISODES)
                        # eval cells: nets[:st+1] x eval seeds
                        for enet in nets[:st + 1]:
                            for es in EVAL_SEEDS:
                                eid = (f'cl1_{arm}_{cond}_{order_name}_sd{seed}'
                                       f'_ev{st + 1}_{NET_TAG[enet]}_es{es}')
                                eodir = out_dir(agent, enet, eid)
                                eov = eval_overlay(eid, arm, str(ckpt))
                                tasks.append({
                                    'run_id': eid,
                                    'command': cmd(agent, enet, eid, eov,
                                                   seed, es),
                                    'cwd': str(ROOT), 'env': dict(ENV),
                                    'output_path': str(eodir),
                                    'completion': completion(eodir),
                                    'depends_on': [rid],
                                    'requires': [{'path': str(ckpt)}],
                                    'campaign': 'cl_hz1x1',
                                    'arm': f'{arm}_{cond}',
                                    'order': order_name,
                                    'stage': st + 1, 'scene': enet,
                                    'training_seed': seed,
                                    'eval_sumo_seed': es,
                                    'checkpoint_path': str(ckpt),
                                    'run_kind': 'eval',
                                })
                                add_ledger(eid, arm, cond, enet, seed, 'eval',
                                           f'{order_name}/ev{st + 1}', 0)
                        prev_ckpt, prev_run_id = ckpt, rid

    # ---- single-task baselines (fresh 100ep per arm x net x seed) ----
    for arm, spec in ARMS.items():
        agent = spec['agent']
        for tag, net in NETS.items():
            for seed in SEEDS:
                rid = f'cl1_{arm}_base_{tag}_sd{seed}'
                odir = out_dir(agent, net, rid)
                ov = write_overlay(rid, dict(spec['model_extra']), {
                    'episodes': STAGE_EPISODES,
                    'evaluation_episodes': EVAL_EPISODES_AT,
                    'resumable_checkpoint_episodes': EVAL_EPISODES_AT,
                    'test_when_train': 'false',
                })
                tasks.append({
                    'run_id': rid,
                    'command': cmd(agent, net, rid, ov, seed, 0),
                    'cwd': str(ROOT), 'env': dict(ENV),
                    'output_path': str(odir),
                    'completion': completion(odir),
                    'campaign': 'cl_hz1x1', 'arm': f'{arm}_base',
                    'order': 'single_task', 'stage': 1, 'scene': net,
                    'training_seed': seed, 'episode_budget': STAGE_EPISODES,
                })
                add_ledger(rid, arm, 'base', net, seed, 'train',
                           f'single/{tag}', STAGE_EPISODES)

    manifest = {
        'schema_version': 1,
        'plan': 'cl_hz1x1_continual_learning_v1',
        'expected_tasks': len(tasks),
        'orders': ORDERS,
        'arms': list(ARMS),
        'replay_conditions': {k: v for k, v in CONDS.items()},
        'training_seeds': SEEDS,
        'eval_sumo_seeds': EVAL_SEEDS,
        'stage_episodes': STAGE_EPISODES,
        'sumo_seed_policy': 'train=0 fixed; eval=101/102/103',
        'tasks': tasks,
    }
    (ART / 'run_queue_cl1.json').write_text(
        json.dumps(manifest, indent=1, ensure_ascii=False), encoding='utf-8')
    (ART / 'ledger_rows.jsonl').write_text(
        '\n'.join(json.dumps(r, ensure_ascii=False) for r in ledger) + '\n',
        encoding='utf-8')
    n_train = sum(1 for t in tasks if 'stage' in t and t.get('run_kind') != 'eval')
    print(f'tasks={len(tasks)} train={n_train} '
          f'eval={len(tasks) - n_train} ledger_rows={len(ledger)}')


if __name__ == '__main__':
    main()
