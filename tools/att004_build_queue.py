#!/usr/bin/env python3
"""ATT-ENTITY-004 queue builder.

Generates per-seed model configs (stage_checkpoint paths differ per seed) and
run-queue manifests for:

  wave1  A2 stage1 (freeze traffic_mlp+state_projection, <=200ep)
         A2 compute-matched control (same ckpt, normal continue 100ep)
         A3 SGA-GB mechanism pilot (2 seeds, 100ep)
  wave2  A2 stage2 (unfreeze, traffic_lr_scale=0.1, 100ep) -- needs stage1 done

Stage2's stage_checkpoint points at the stage1 run's own episode_0200
resumable checkpoint, so wave2 must only start after wave1 stage1 tasks
succeed.  All runs keep the stage5 protocol: sumo hz4x4, random-E plan,
canonical text, condition normal, same seeds/eval cadence.
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = (ROOT / 'data/output_data/cross_algorithm/plan5_b100/environment/'
      'micromamba/envs/colight/bin/python3.10')
GEN = ROOT / 'configs/tsc/att_entity_004/generated'
OUT_ATT = ROOT / 'data/output_data/tsc/sumo_tarl_attention/hz4x4'
ART = ROOT / 'artifacts/att_entity_004'
SEEDS = (7, 17, 27, 37, 47)
PILOT_SEEDS = (7, 17)

BASE_MODEL = """model:
  name: tarl_attention
  train_model: True
  tarl_impl: paper
  one_hot: True
  phase: True
  batch_size: 128
  gamma: 0.99
  learning_rate: 0.0001
  epsilon: 0.8
  epsilon_decay: 0.99995
  epsilon_min: 0.05
  grad_clip: 5.0
"""

BASE_TRAINER_HEAD = """trainer:
  steps: 3600
  test_steps: 3600
  learning_start: 1000
  buffer_size: 50000
  action_interval: 10
  update_model_rate: 1
  update_target_rate: 200
  test_when_train: false
  event_schedule_plan: configs/events/plans/hz4x4_random_v1.yml

logger:
  save_rate: 25
"""

ENV = {
    'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1',
    'OPENBLAS_NUM_THREADS': '1', 'VECLIB_MAXIMUM_THREADS': '1',
    'CUDA_VISIBLE_DEVICES': '-1',
    'TARL_EVENT_CONDITION': 'normal', 'TARL_TEXT_CONDITION': 'canonical',
}

EP200 = 'checkpoints/resumable/episode_0200.pt'


def write_config(name, model_extra, episodes, eval_eps):
    eps = ', '.join(str(e) for e in eval_eps)
    text = (f'# ATT-ENTITY-004 generated config ({name})\n'
            + BASE_MODEL + model_extra
            + BASE_TRAINER_HEAD.replace(
                'trainer:\n', f'trainer:\n  episodes: {episodes}\n')
            .replace('  test_when_train: false\n',
                     f'  test_when_train: false\n'
                     f'  evaluation_episodes: [{eps}]\n'
                     f'  resumable_checkpoint_episodes: [{eps}]\n'))
    path = GEN / f'{name}.yml'
    path.write_text(text)
    return path


def task(run_id, prefix, cfg_path, priority, experiment='ATT-ENTITY-004'):
    out = OUT_ATT / prefix
    return {
        'run_id': run_id,
        'command': [str(PY), 'run.py', '-w', 'sumo', '-a', 'tarl_attention',
                    '-n', 'hz4x4', '--seed', str(prefix_seed(prefix)),
                    '--interface', 'libsumo', '--prefix', prefix,
                    '--experiment-config', str(cfg_path.relative_to(ROOT))],
        'cwd': str(ROOT), 'env': ENV, 'priority': priority,
        'completion': {'path': str(out / 'run_status.json'),
                       'json_field': 'status', 'equals': '已完成'},
        'experiment_id': experiment,
        'agent': 'tarl_attention', 'config_path': str(cfg_path),
        'output_path': str(out),
    }


def prefix_seed(prefix):
    return int(prefix.rsplit('_s', 1)[1])


def stage1_cfg(seed):
    return write_config(
        f'a2_stage1_s{seed}',
        f"""  # A2 stage1: text-conditioned decision pathway only; the traffic
  # encoder stays frozen at the att_ep200 solution so TD updates flow into
  # text_projection + ATT/fusion + GAT + q_head exclusively.
  stage_checkpoint: {OUT_ATT}/tarlp_att_s{seed}/{EP200}
  replay_policy: fifo
  epsilon_mode: continue_schedule
  freeze_params: [traffic_mlp, state_projection]
  grad_ledger: true
""", 200, (0, 50, 100, 150, 200))


def stage2_cfg(seed):
    return write_config(
        f'a2_stage2_s{seed}',
        f"""  # A2 stage2: unfreeze the traffic encoder but keep its learning rate
  # far below the text pathway's (traffic_lr_scale applied to lr).
  # stage_load_optimizer=false because the two-group RMSprop layout differs
  # from the single-group state saved in the stage1 checkpoint.
  stage_checkpoint: {OUT_ATT}/att004_stg1_s{seed}/{EP200}
  replay_policy: fifo
  epsilon_mode: continue_schedule
  stage_load_optimizer: false
  traffic_lr_scale: 0.1
  grad_ledger: true
""", 100, (0, 50, 100))


def control_cfg(seed):
    return write_config(
        f'a2_control_s{seed}',
        f"""  # A2 compute-matched control: identical ckpt/replay/epsilon
  # semantics as stage1 but the whole network keeps training normally.
  stage_checkpoint: {OUT_ATT}/tarlp_att_s{seed}/{EP200}
  replay_policy: fifo
  epsilon_mode: continue_schedule
  grad_ledger: true
""", 100, (0, 50, 100))


def pilot_cfg(seed):
    return write_config(
        f'a3_pilot_s{seed}',
        f"""  # A3 SGA-GB mechanism pilot: event-conditioned per-sample traffic
  # gradient damping driven by the EMA-smoothed fusion-input ratio R_t.
  # alpha_s = clip(R*/R_ema, alpha_min, 1) applied only to the event samples'
  # contribution to theta_traffic = {{traffic_mlp, state_projection}}.
  grad_ledger: true
  sga_gb:
    enabled: true
    r_star: 3.0
    alpha_min: 0.1
    ema_beta: 0.98
    warmup_steps: 500
""", 100, (0, 50, 100))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=list(SEEDS))
    ap.add_argument('--pilot-seeds', type=int, nargs='+',
                    default=list(PILOT_SEEDS))
    args = ap.parse_args()
    GEN.mkdir(parents=True, exist_ok=True)
    ART.mkdir(parents=True, exist_ok=True)

    wave1 = []
    for s in args.seeds:
        wave1.append(task(f'att004/a2_stg1_s{s}', f'att004_stg1_s{s}',
                          stage1_cfg(s), priority=1))
        wave1.append(task(f'att004/a2_ctl_s{s}', f'att004_ctl_s{s}',
                          control_cfg(s), priority=2))
    for s in args.pilot_seeds:
        wave1.append(task(f'att004/a3_pilot_s{s}', f'att004_sgagb_s{s}',
                          pilot_cfg(s), priority=0))
    (ART / 'run_queue_att004_wave1.json').write_text(
        json.dumps({'tasks': wave1}, indent=1) + '\n')

    wave2 = [task(f'att004/a2_stg2_s{s}', f'att004_stg2_s{s}',
                  stage2_cfg(s), priority=0) for s in args.seeds]
    (ART / 'run_queue_att004_stage2.json').write_text(
        json.dumps({'tasks': wave2}, indent=1) + '\n')

    print(f'wave1: {len(wave1)} tasks -> {ART}/run_queue_att004_wave1.json')
    print(f'wave2: {len(wave2)} tasks -> {ART}/run_queue_att004_stage2.json')
    print(f'configs -> {GEN}')


if __name__ == '__main__':
    main()
