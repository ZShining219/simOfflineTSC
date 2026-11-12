#!/usr/bin/env python3
"""跨语料「场景×控制器」数据集单元收割（覆盖/质量统计）。

数据集单元 = 一个控制器在某语料中一段采集的全部 transition。
语料：
  seq8    seq8_{norep,rep,drift}_formal CONT/SELF-DHOA-R50 +
          seq8_anchor s8f_ soddqnp_r25（8 段 × 需求包场景）
  ha      ha_sodqn/formal_e7705f7_20260726 全部 DONE train run
          (CONT-FIFO / P1C·P1F×{DHOA,RAND,COV,CQ,CQA}×R{25,50,75})，4 段
  plan34  sequential/plan34_{b100,b400}_formal_60 的 clear/fifo/
          fifo_matched_wait，O1–O4 × seed0–4（stage_01 轨迹未持久化，
          实际贡献 st2–4）
  p1      tsc/sumo_dqn/<net>/p1_formal_*_400ep 按 PLAN2_STAGES 切 Q1–Q4
  hitl    hitl_probe/{pilot,pilot5,multiscene/*,warmloop/p*}/runs/* 人类接管

覆盖口径 = ha_coverage_v1（semi_offline/sequential/owp.py）去 network/seed 维：
  cell = (total_load_bin, max_lane_bin, imbalance_bin, phase_argmax, action)
  bins = np.digitize(v, [0,1,3,7,15,31]) -> 0..6
可比性：每单元固定取 ≤CAP_EPISODES 个 episode（等距 stride），使各单元
样本量一致（~9k transitions）；hitl 单元本身较小（~2.4k）。

产物：cache/units.json + cache/marginals.npz（7x8 load×action 计数矩阵）
用法：python dev/coverage_figs/harvest.py [--jobs N]
"""
import argparse
import glob
import json
import os
import re
import sys
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, 'data/output_data')
CACHE = os.path.join(ROOT, 'dev/coverage_figs/cache')

CAP_EPISODES = 25
BINS = np.asarray([0, 1, 3, 7, 15, 31])
N_BIN, N_ACT, N_PHASE = 7, 8, 8
CELL_UNIVERSE = (N_BIN ** 3) * N_PHASE * N_ACT

SEQ8_BATCHES = {
    'N1': 'sequential/seq8_norep_formal_20261015',
    'R1': 'sequential/seq8_rep_formal_20261015',
    'D1': 'sequential/seq8_drift_formal_20261015',
}
SEQ8_ORDERS = {
    'N1': ['sparse_ultra', 'fluct_burst', 'lt_extreme', 'asym_wonly',
           'dense_extreme', 'st_dom', 'asym_we', 'fluct_ramp'],
    'R1': ['sparse_low', 'lt_high', 'fluct_burst', 'dense_high',
           'sparse_low', 'lt_high', 'fluct_burst', 'dense_high'],
    'D1': ['st_dom', 'sumohz1x1', 'lt_high', 'lt_extreme',
           'lt_wsing', 'asym_wonly', 'asym_we', 'asym_ns'],
}
CONFIG_SCENES = ['sumohz1x1', 'sumohz1x1_config2', 'sumohz1x1_config3',
                 'sumohz1x1_config4']
HITL_TAG = {'QC': 'sumohz1x1_config2', 'KN': 'sumohz1x1_config3',
            'SB': 'sumohz1x1_config4'}
HA_DIR = 'ha_sodqn/formal_e7705f7_20260726'
P34_DIRS = {'plan34_b100': 'sequential/plan34_b100_formal_60_20260723',
            'plan34_b400': 'sequential/plan34_b400_formal_60_20260723'}


def scene_tag(net):
    return 'base' if net == 'sumohz1x1' else net.replace('sumohz1x1_', '')


def net_name(tag):
    return 'sumohz1x1' if tag == 'base' else f'sumohz1x1_{tag}'


def _bin(v):
    return np.digitize(v, BINS)


def stride(files, cap=CAP_EPISODES):
    if len(files) <= cap:
        return files
    step = len(files) / cap
    return [files[int(i * step)] for i in range(cap)]


def best_attempt(run_dir):
    """返回含最多 trajectory npz 的 attempt 目录。"""
    best, best_n = None, -1
    for att in glob.glob(os.path.join(run_dir, 'attempt*', '*')) \
            + [run_dir]:
        if not os.path.isdir(att):
            continue
        n = len(glob.glob(os.path.join(att, 'trajectory/episodes/*.npz')))
        if n > best_n:
            best, best_n = att, n
    return best if best_n > 0 else None


def read_networks(att_dir):
    for name in ('child_run_manifest.json', 'logical_run_manifest.json'):
        p = os.path.join(att_dir, name)
        if os.path.isfile(p):
            try:
                nets = json.load(open(p)).get('networks')
                if nets:
                    return nets
            except Exception:
                pass
    return None


def episode_stats(path):
    with np.load(path, allow_pickle=False) as d:
        state = np.asarray(d['state']).reshape(-1, 8)
        action = np.asarray(d['action']).reshape(-1).astype(np.int64)
        phase = np.asarray(
            d['phase'] if 'phase' in d.files else d['current_phase']
        ).reshape(-1).astype(np.int64)
        reward = np.asarray(d['reward']).reshape(-1)
    load_bin = _bin(state.sum(axis=1))
    max_bin = _bin(state.max(axis=1))
    imb_bin = _bin(state.max(axis=1) - state.min(axis=1))
    cell = (((load_bin * N_BIN + max_bin) * N_BIN + imb_bin) * N_PHASE
            + phase) * N_ACT + action
    marginal = np.zeros((N_BIN, N_ACT), dtype=np.int64)
    np.add.at(marginal, (load_bin, action), 1)
    return np.unique(cell), marginal, len(action), float(reward.sum())


def unit_stats(args):
    uid, files = args
    cells = np.empty(0, dtype=np.int64)
    marginal = np.zeros((N_BIN, N_ACT), dtype=np.int64)
    n = rsum = 0.0
    n_ep = 0
    for f in files:
        try:
            c, m, k, r = episode_stats(f)
        except Exception as e:
            print(f'  skip {f}: {e}', file=sys.stderr)
            continue
        cells = np.union1d(cells, c)
        marginal += m
        n += k
        rsum += r
        n_ep += 1
    return uid, cells, marginal, int(n), rsum, n_ep


def add_unit(units, corpus, group, order, arm, seed, stage, scene,
             run, files):
    uid = f'{corpus}__{group}__{order or "-"}__{seed}__st{stage}'
    units.append(dict(uid=uid, corpus=corpus, group=group, order=order,
                      arm=arm, seed=seed, stage=stage,
                      scene=scene_tag(scene), run=run,
                      files=stride(sorted(files))))


def build_units():
    units = []

    # ---------- corpus seq8 ----------
    for order, batch in SEQ8_BATCHES.items():
        for arm_dir, arm in {'CONT': 'CONT', 'SELF-DHOA-R50': 'DHOA'}.items():
            for sd in ('SD0', 'SD1'):
                run = f'{arm_dir}-{order}-{sd}'
                rd = os.path.join(OUT, batch, run)
                for stage in range(1, 9):
                    files = glob.glob(os.path.join(
                        rd, 'attempts/attempt_*/trajectory/episodes',
                        f'stage_{stage:02d}_episode_*.npz'))
                    add_unit(units, 'seq8', arm, order, arm, sd, stage,
                             SEQ8_ORDERS[order][stage - 1], run, files)
        for sd in ('SD0', 'SD1'):
            for stage in range(1, 9):
                scene = SEQ8_ORDERS[order][stage - 1]
                rd = glob.glob(os.path.join(
                    OUT, 'tsc/sumo_dqn', net_name(scene_tag(scene)),
                    f's8f_soddqnp_{order}_{sd.lower()}_st{stage}'))
                files = glob.glob(os.path.join(
                    rd[0], 'trajectory/episodes/episode_*.npz')) if rd else []
                add_unit(units, 'seq8', 'ANCHOR', order, 'ANCHOR', sd,
                         stage, scene, os.path.basename(rd[0]) if rd else
                         f's8f_{order}_{sd}_st{stage}', files)

    # ---------- corpus ha (ha_sodqn formal) ----------
    ha_root = os.path.join(OUT, HA_DIR)
    for run in sorted(os.listdir(ha_root)):
        m = re.match(r'(.+)-O(\d)-SD(\d)$', run)
        rd = os.path.join(ha_root, run)
        if not m or not os.path.isdir(os.path.join(rd, 'attempts')):
            continue
        cond, order, sd = m.group(1), f'O{m.group(2)}', f'SD{m.group(3)}'
        att = best_attempt(rd)
        if not att:
            continue
        nets = read_networks(att)
        if not nets:
            continue
        group = ('CONT-FIFO' if cond == 'CONT-FIFO'
                 else 'ha-' + re.sub(r'^P1[CF]-|-R\d+$', '', cond))
        for stage in range(1, len(nets) + 1):
            files = glob.glob(os.path.join(
                att, 'trajectory/episodes', f'stage_{stage:02d}_*.npz'))
            add_unit(units, 'ha', group, order, cond, sd, stage,
                     nets[stage - 1], run, files)

    # ---------- corpus plan34 ----------
    for corpus, batch in P34_DIRS.items():
        broot = os.path.join(OUT, batch)
        for run in sorted(os.listdir(broot)):
            m = re.match(r'plan34_b\d+_O(\d)_seed(\d)_(.+)$', run)
            if not m:
                continue
            order, sd, mode = (f'O{m.group(1)}', f'SD{m.group(2)}',
                               m.group(3))
            att = best_attempt(os.path.join(broot, run))
            if not att:
                continue
            nets = read_networks(att)
            if not nets:
                continue
            for stage in range(1, len(nets) + 1):
                files = glob.glob(os.path.join(
                    att, 'trajectory/episodes', f'stage_{stage:02d}_*.npz'))
                add_unit(units, corpus, f'p34-{mode}', order, mode, sd,
                         stage, nets[stage - 1], run, files)

    # ---------- corpus p1 (Plan1 formal, quartile slices) ----------
    # 同 (net,seed) 可能存在多次批 run（如 _r2），只保留 episode 最全者
    for net in CONFIG_SCENES:
        cand = {}
        for rd in glob.glob(os.path.join(
                OUT, 'tsc/sumo_dqn', net,
                'p1_formal_dqn_*_seed*_400ep_*')):
            run = os.path.basename(rd)
            m = re.search(r'_seed(\d)', run)
            if not m:
                continue
            n_ep = len(glob.glob(os.path.join(
                rd, 'trajectory/episodes/episode_*.npz')))
            key = int(m.group(1))
            if key not in cand or n_ep > cand[key][1]:
                cand[key] = (rd, n_ep)
        for sd_i, (rd, _) in sorted(cand.items()):
            run = os.path.basename(rd)
            sd = f'SD{sd_i}'
            eps = sorted(glob.glob(os.path.join(
                rd, 'trajectory/episodes/episode_*.npz')))
            for qi, (lo, hi) in enumerate(
                    [(1, 100), (101, 200), (201, 300), (301, 400)], 1):
                files = [f for f in eps
                         if lo <= int(re.search(r'episode_(\d+)', f).group(1))
                         <= hi]
                add_unit(units, 'p1', f'P1-Q{qi}', None, 'dqn_p1', sd, qi,
                         net, run, files)

    # ---------- corpus hitl ----------
    hitl_root = os.path.join(OUT, 'hitl_probe')
    run_dirs = set()
    for pat in ('pilot/runs/*', 'pilot5/runs/*', 'multiscene/*/runs/*',
                'warmloop/p*/runs/*'):
        run_dirs.update(glob.glob(os.path.join(hitl_root, pat)))
    for rd in sorted(run_dirs):
        if not os.path.isdir(rd):
            continue
        files = glob.glob(os.path.join(
            rd, 'trajectory/episodes/*.npz')) or glob.glob(os.path.join(
            rd, 'attempt*/trajectory/episodes/*.npz')) or glob.glob(
            os.path.join(rd, 'attempts/*/trajectory/episodes/*.npz'))
        if not files:
            continue
        run = os.path.basename(rd)
        att = best_attempt(rd)
        nets = read_networks(att) if att else None
        if not nets:
            nets = None
            for name in ('child_run_manifest.json', 'run_manifest.json'):
                p = os.path.join(rd, name)
                if os.path.isfile(p):
                    try:
                        nets = json.load(open(p)).get('networks') or [
                            json.load(open(p)).get('network')]
                    except Exception:
                        pass
        scene = (nets[0] if nets else None)
        if scene is None:
            for tag, netv in HITL_TAG.items():
                if f'-{tag}-' in run:
                    scene = netv
                    break
        if scene is None:
            scene = 'sumohz1x1'
        add_unit(units, 'hitl', 'HITL', None, 'hitl',
                 (re.search(r'SD(\d)', run) or [None, '?']).group(1)
                 if re.search(r'SD(\d)', run) else 'SD?',
                 1, scene, run, files)

    return units


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jobs', type=int, default=16)
    args = ap.parse_args()
    os.makedirs(CACHE, exist_ok=True)
    units = build_units()
    work = [(u['uid'], u['files']) for u in units if u['files']]
    for u in units:
        if not u['files']:
            print(f'  NO FILES: {u["uid"]} ({u["run"]})', file=sys.stderr)
    results = {}
    with Pool(args.jobs) as pool:
        for i, (uid, cells, marginal, n, rsum, n_ep) in enumerate(
                pool.imap_unordered(unit_stats, work)):
            results[uid] = dict(cells=cells, marginal=marginal, n=n,
                                rsum=rsum, n_ep=n_ep)
            if (i + 1) % 200 == 0:
                print(f'  {i + 1}/{len(work)} units', file=sys.stderr)
    marginals = {}
    for u in units:
        r = results.get(u['uid'])
        if r is None or r['n'] == 0:
            u.update(n_transitions=0, n_episodes=0, cells_occupied=0,
                     saco_frac=0.0, cell_entropy_norm=0.0,
                     tq_mean_reward=None)
            continue
        counts = np.bincount(r['cells'],
                             minlength=CELL_UNIVERSE).astype(np.float64)
        probs = counts[counts > 0] / counts.sum()
        ent = float(-(probs * np.log(probs)).sum())
        u.update(n_transitions=r['n'], n_episodes=r['n_ep'],
                 cells_occupied=int(len(r['cells'])),
                 saco_frac=float(len(r['cells']) / CELL_UNIVERSE),
                 cell_entropy_norm=float(ent / np.log(CELL_UNIVERSE)),
                 tq_mean_reward=float(r['rsum'] / r['n']))
        marginals[u['uid']] = r['marginal']
        u.pop('files')
    for u in units:
        u.pop('files', None)
    with open(os.path.join(CACHE, 'units.json'), 'w') as fh:
        json.dump(units, fh, indent=1, ensure_ascii=False)
    np.savez_compressed(os.path.join(CACHE, 'marginals.npz'), **marginals)
    print(f'wrote {len(units)} units -> {CACHE}')


if __name__ == '__main__':
    main()
