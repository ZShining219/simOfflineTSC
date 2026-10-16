#!/usr/bin/env python3
"""seq8_anchor 队列生成件——seq8 三序列（N1/R1/D1 语义需求包）上的
半离线+policy-anchor 补充臂（cl_hz1x1 链路）。

设计（2026-10-16 与 Z 对齐）：
  臂     soddqnp_r25 = dqn + double_dqn + collect_history + causal +
         scene_intersection_balanced + offline_ratio=0.25 +
         anchor_lambda_policy=1.0 (T=1.0)，λq=0 纯 policy KL。
  序列   seq8 三序（需求包即 configs/sim/sumohz1x1_<pack>.cfg，路网同 hz1x1）。
  seeds  {0,1}（与已完成的 sequential/ CONT/DHOA-R50 配对）。
  超参   对齐 seq8 协议：100ep/stage、3600s、interval10、lr0.001、γ0.95、
         ε_decay0.995、batch64、buffer5000、target_update10
         （与 configs/tsc/dqn_backup.yml 默认值一致，仍显式覆写）。
  评估   ① 训练 run 内 evaluation_episodes=0..100 逐 ep 当前场景评估（波动）；
         ② 每 stage 末对已见全部场景发独立 eval 格（sumo_seed=101，下三角）。
  基线   CONT 与 DHOA-R50 数据来自 sequential/ runtime 既有完成 run，
         跨 runtime 对比为已知 caveat（runtime/训练循环/rng 差异），
         记录于 docs/experiments/2026-10-16_seq8_anchor.md。

用法：python dev/seq8_anchor/build_queue.py [smoke|formal]
  smoke   N1 × sd0 × 1 链，5 ep/stage，机制自检（前缀 s8q_）
  formal  N1/R1/D1 × sd{0,1} 全批：48 stage-run + 216 stage末eval格
         （前缀 s8f_——正式/冒烟命名空间分离，避免产物目录撞名）

Outputs（gitignored artifacts/seq8_anchor/）：
  overlays/<run_id>.yml        per-run overlay 供 run.py --experiment-config
  run_queue_seq8_<phase>.json  QueueRunner manifest（depends_on 链）
  ledger_rows_seq8_<phase>.jsonl  REGISTERED 台账行（由治理脚本并入 ledger/runs.jsonl）
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / 'artifacts' / 'seq8_anchor'
OVERLAYS = ART / 'overlays'
# 固定 colight 解释器：队列由独立进程拉起 run.py，不能依赖构建时的
# sys.executable（builder 在非 colight shell 下运行会记错路径）。
PY = '/home/dev/miniforge3/envs/colight/bin/python'
LD_PRELOAD = '/home/dev/miniforge3/envs/colight/lib/libstdc++.so.6'
ENV = {
    'LD_PRELOAD': LD_PRELOAD,
    'CUDA_VISIBLE_DEVICES': '-1',
    'OMP_NUM_THREADS': '1',
    'MKL_NUM_THREADS': '1',
    'OPENBLAS_NUM_THREADS': '1',
    'NUMEXPR_NUM_THREADS': '1',
    'VECLIB_MAXIMUM_THREADS': '1',
}

ORDERS = {
    'N1': ['sumohz1x1_sparse_ultra', 'sumohz1x1_fluct_burst',
           'sumohz1x1_lt_extreme', 'sumohz1x1_asym_wonly',
           'sumohz1x1_dense_extreme', 'sumohz1x1_st_dom',
           'sumohz1x1_asym_we', 'sumohz1x1_fluct_ramp'],
    'R1': ['sumohz1x1_sparse_low', 'sumohz1x1_lt_high',
           'sumohz1x1_fluct_burst', 'sumohz1x1_dense_high'] * 2,
    'D1': ['sumohz1x1_st_dom', 'sumohz1x1', 'sumohz1x1_lt_high',
           'sumohz1x1_lt_extreme', 'sumohz1x1_lt_wsing',
           'sumohz1x1_asym_wonly', 'sumohz1x1_asym_we', 'sumohz1x1_asym_ns'],
}

OFFLINE_RATIO = 0.25
EVAL_SUMO_SEED = 101          # stage 末 eval 格固定评估种子
SEEDS = [0, 1]

SO_BASE = {
    'offline_sampling_strategy': 'scene_intersection_balanced',
    'history_access_mode': 'causal',
    'collect_history': True,
}
ANCHOR_KEYS = ('anchor_lambda_policy', 'anchor_lambda_value',
               'anchor_temperature', 'teacher_paths')

# 与 sequential/seq8 trainer 协议一致的显式覆写
ARM_MODEL = {
    'double_dqn': True,
    'learning_rate': 0.001, 'batch_size': 64, 'gamma': 0.95,
    'epsilon': 0.1, 'epsilon_decay': 0.995, 'epsilon_min': 0.01,
    'grad_clip': 5.0,
    **SO_BASE,
    'offline_ratio': OFFLINE_RATIO,
    'anchor_lambda_policy': 1.0, 'anchor_temperature': 1.0,
}
TRAINER_BASE = {
    'learning_start': 1000, 'buffer_size': 5000,
    'steps': 3600, 'test_steps': 3600, 'action_interval': 10,
    'update_model_rate': 1, 'update_target_rate': 10,
    'test_when_train': False,
}

OUT = ROOT / 'data' / 'output_data' / 'tsc'


def out_dir(agent, net, prefix):
    return OUT / f'sumo_{agent}' / net / prefix


def overlay_path(run_id):
    return OVERLAYS / f'{run_id}.yml'


def yaml_value(v):
    if isinstance(v, list):
        return '[' + ', '.join(
            f"'{x}'" if isinstance(x, str) else str(x) for x in v) + ']'
    if isinstance(v, bool):
        return 'true' if v else 'false'
    return str(v)


def write_overlay(run_id, model_extra, trainer_extra):
    lines = []
    if model_extra:
        lines.append('model:')
        for k, v in model_extra.items():
            if isinstance(v, dict):
                lines.append(f'  {k}:')
                for sub_k, sub_v in v.items():
                    lines.append(f'    {sub_k}: {sub_v}')
            else:
                lines.append(f'  {k}: {yaml_value(v)}')
        lines.append('')
    lines.append('trainer:')
    for k, v in trainer_extra.items():
        lines.append(f'  {k}: {yaml_value(v)}')
    path = overlay_path(run_id)
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return path


def train_overlay(run_id, stage_idx, nets, history_paths, teacher_ckpts,
                  prev_ckpt, stage_episodes):
    me = dict(ARM_MODEL)
    me['scene_id'] = nets[stage_idx]
    me['roadnet_id'] = 'hz1x1'
    me['training_stage'] = stage_idx
    me['scene_order'] = list(nets)
    if stage_idx == 0:
        me['offline_ratio'] = 0.0    # stage1 无可见历史，纯在线
    if history_paths:
        me['history_paths'] = history_paths
    if stage_idx > 0:
        me['stage_checkpoint'] = str(prev_ckpt)
        me['replay_policy'] = 'clear'
        me['epsilon_mode'] = 'reset_schedule'
        me['stage_load_optimizer'] = 'true'
        me['teacher_paths'] = {
            scene: str(teacher_ckpts[scene]) for scene in nets[:stage_idx]}
    te = dict(TRAINER_BASE)
    te.update({
        'episodes': stage_episodes,
        'evaluation_episodes': list(range(stage_episodes + 1)),
        'resumable_checkpoint_episodes': [0, stage_episodes],
    })
    return write_overlay(run_id, me, te)


def eval_overlay(run_id, ckpt):
    me = dict(ARM_MODEL)
    for key in (*SO_BASE, *ANCHOR_KEYS, 'offline_ratio'):
        me.pop(key, None)
    me.update({
        'stage_checkpoint': str(ckpt),
        'replay_policy': 'clear',
        'epsilon_mode': 'fixed_low',
        'stage_load_optimizer': 'true',
    })
    te = dict(TRAINER_BASE)
    te.update({
        'episodes': 0,
        'evaluation_episodes': [0],
        'resumable_checkpoint_episodes': [0],
    })
    return write_overlay(run_id, me, te)


def cmd(prefix, net, overlay, seed, sumo_seed):
    return [
        PY, 'run.py', '-w', 'sumo', '-a', 'dqn', '-n', net,
        '--seed', str(seed), '--sumo_seed', str(sumo_seed),
        '--interface', 'libsumo', '--prefix', prefix,
        '--experiment-config', str(overlay),
    ]


def completion(odir):
    return {'path': str(odir / 'run_status.json'),
            'json_field': 'status', 'equals': '已完成'}


def build(phase, orders, seeds, stage_episodes, prefix):
    OVERLAYS.mkdir(parents=True, exist_ok=True)
    tasks, ledger = [], []
    for order_name, nets in orders.items():
        for seed in seeds:
            prev_ckpt = prev_run_id = None
            history_paths, teacher_ckpts = [], {}
            for st, net in enumerate(nets):
                rid = f'{prefix}_soddqnp_{order_name}_sd{seed}_st{st + 1}'
                odir = out_dir('dqn', net, rid)
                ov = train_overlay(rid, st, nets, history_paths,
                                   teacher_ckpts, prev_ckpt, stage_episodes)
                ckpt = (odir / 'checkpoints' / 'resumable' /
                        f'episode_{stage_episodes:04d}.pt')
                task = {
                    'run_id': rid,
                    'command': cmd(rid, net, ov, seed, 0),
                    'cwd': str(ROOT), 'env': dict(ENV),
                    'output_path': str(odir),
                    'completion': completion(odir),
                    'checkpoint_path': str(ckpt),
                    'campaign': 'seq8_anchor', 'arm': 'soddqnp_r25',
                    'order': order_name, 'stage': st + 1, 'scene': net,
                    'training_seed': seed,
                    'replay_condition': 'semi_offline_anchor',
                    'episode_budget': stage_episodes,
                }
                requires = []
                if prev_run_id:
                    task['depends_on'] = [prev_run_id]
                    requires.append({'path': str(prev_ckpt)})
                    requires.extend({'path': str(p / 'manifest.json')}
                                    for p in history_paths)
                    requires.extend({'path': str(p)}
                                    for p in teacher_ckpts.values())
                if requires:
                    task['requires'] = requires
                tasks.append(task)
                ledger.append({
                    'run_id': rid, 'line': 'cl_hz1x1', 'campaign': 'seq8_anchor',
                    'arm': 'soddqnp_r25', 'alias_of': None, 'net': net,
                    'seed': seed, 'episodes_total': stage_episodes,
                    'episodes_increment': stage_episodes, 'resume_from': None,
                    'ngpu': -1, 'status': 'REGISTERED',
                    'plan_id': f'seq8_anchor_{phase}', 'code_commit': None,
                    'dirty': None,
                    'queue_file': f'artifacts/seq8_anchor/run_queue_seq8_{phase}.json',
                    'run_dir': None, 'tmux': None, 'started': None,
                    'eval_mode': None, 'behavior_source': None, 'metrics': {},
                    'abort_reason': None, 'error': None, 'evidence_lost': False,
                    'notes': f'REGISTERED: seq8_anchor {phase} train '
                             f'order={order_name} stage={st + 1} '
                             f'ratio={0.0 if st == 0 else OFFLINE_RATIO} '
                             f'anchor_lambda_policy=1.0',
                    'run_kind': 'train', 'config_sha': None,
                    'sumo_seed_mode': 'fixed', 'sumo_seed': 0,
                    'eval_seed': None, 'condition': 'semi_offline_anchor',
                    'superseded_by': None, 'runtime_env': 'colight',
                    'end_time': None,
                })
                # stage 末下三角 eval 格：对 1..k 全部已见场景
                for enet in nets[:st + 1]:
                    tag = net_tag(enet)
                    eid = f'{prefix}_soddqnp_{order_name}_sd{seed}_ev{st + 1}_{tag}'
                    eodir = out_dir('dqn', enet, eid)
                    eov = eval_overlay(eid, ckpt)
                    tasks.append({
                        'run_id': eid,
                        'command': cmd(eid, enet, eov, seed, EVAL_SUMO_SEED),
                        'cwd': str(ROOT), 'env': dict(ENV),
                        'output_path': str(eodir),
                        'completion': completion(eodir),
                        'depends_on': [rid],
                        'requires': [{'path': str(ckpt)}],
                        'campaign': 'seq8_anchor', 'arm': 'soddqnp_r25',
                        'order': order_name, 'stage': st + 1, 'scene': enet,
                        'training_seed': seed, 'eval_sumo_seed': EVAL_SUMO_SEED,
                        'checkpoint_path': str(ckpt), 'run_kind': 'eval',
                    })
                    ledger.append({
                        'run_id': eid, 'line': 'cl_hz1x1',
                        'campaign': 'seq8_anchor', 'arm': 'soddqnp_r25',
                        'alias_of': None, 'net': enet, 'seed': seed,
                        'episodes_total': 0, 'episodes_increment': 0,
                        'resume_from': None, 'ngpu': -1, 'status': 'REGISTERED',
                        'plan_id': f'seq8_anchor_{phase}', 'code_commit': None,
                        'dirty': None,
                        'queue_file': f'artifacts/seq8_anchor/run_queue_seq8_{phase}.json',
                        'run_dir': None, 'tmux': None, 'started': None,
                        'eval_mode': 'frozen_stage_ckpt',
                        'behavior_source': None, 'metrics': {},
                        'abort_reason': None, 'error': None,
                        'evidence_lost': False,
                        'notes': f'REGISTERED: seq8_anchor {phase} eval '
                                 f'order={order_name} after_stage={st + 1} '
                                 f'scene={enet}',
                        'run_kind': 'eval', 'config_sha': None,
                        'sumo_seed_mode': 'fixed',
                        'sumo_seed': EVAL_SUMO_SEED,
                        'eval_seed': EVAL_SUMO_SEED,
                        'condition': 'semi_offline_anchor',
                        'superseded_by': None, 'runtime_env': 'colight',
                        'end_time': None,
                    })
                history_paths.append(odir / 'history_archive')
                teacher_ckpts[net] = ckpt
                prev_ckpt, prev_run_id = ckpt, rid
    manifest = {
        'schema_version': 1,
        'plan': f'seq8_anchor_{phase}',
        'expected_tasks': len(tasks),
        'orders': orders,
        'arms': ['soddqnp_r25'],
        'offline_ratio': OFFLINE_RATIO,
        'history_access_mode': 'causal',
        'replay_condition': 'clear_online_buffer',
        'anchor_loss': 'policy KL softmax(Q/T) vs per-scene frozen '
                       'stage-end teachers, lambda_pi=1.0, T=1.0',
        'training_seeds': seeds,
        'eval_sumo_seeds': [EVAL_SUMO_SEED],
        'stage_episodes': stage_episodes,
        'sumo_seed_policy': 'train=0 fixed; eval=101',
        'baseline_note': 'CONT/DHOA-R50 baselines from sequential runtime '
                         'runs (cross-runtime comparison caveat)',
        'tasks': tasks,
    }
    qp = ART / f'run_queue_seq8_{phase}.json'
    qp.write_text(json.dumps(manifest, indent=1, ensure_ascii=False),
                  encoding='utf-8')
    (ART / f'ledger_rows_seq8_{phase}.jsonl').write_text(
        '\n'.join(json.dumps(r, ensure_ascii=False) for r in ledger) + '\n',
        encoding='utf-8')
    n_train = sum(1 for t in tasks if t.get('run_kind') != 'eval')
    print(f'phase={phase} tasks={len(tasks)} train={n_train} '
          f'eval={len(tasks) - n_train} ledger_rows={len(ledger)}')
    return qp


def net_tag(net):
    return net.replace('sumohz1x1_', '').replace('sumohz1x1', 'base')


if __name__ == '__main__':
    phase = sys.argv[1] if len(sys.argv) > 1 else 'smoke'
    if phase == 'smoke':
        qp = build('smoke', {'N1': ORDERS['N1'][:4]}, [0], 5, 's8q')
    elif phase == 'formal':
        qp = build('formal', ORDERS, SEEDS, 100, 's8f')
    else:
        raise SystemExit(f'unknown phase: {phase}')
    print(qp)
