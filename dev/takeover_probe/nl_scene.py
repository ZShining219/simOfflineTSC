"""Natural-language scene reports -> structured fields -> the existing
attention/dispatch pipeline, unchanged.

The structured report channel (``FeatureBuilder.fields_text_vec``)
assumes the report arrives as typed fields.  Real TMC reports are
natural language.  This module adds the missing front-end:

    NL sentence -> NLSceneParser -> {kind, edge, ...fields}
                -> fb.fields_text_vec -> batch['text']
                -> attention binding + key dispatch (untouched)

Two pieces:

* ``nl_render``     report-side: fields -> one of several paraphrased
                    NL realisations (simulates heterogeneous reporters —
                    the canonical ``injects.active_event`` sentence is
                    just one template).
* ``NLSceneParser`` system-side: learned hashed-ngram linear classifier
                    over EVENT_KINDS + literal edge-name extraction +
                    light scalar heuristics.  Trained on a held-out-
                    template split so accuracy reflects generalisation,
                    not memorisation.

``NLDispatchAgent`` is a drop-in ``KeyDispatchAgent`` that performs the
NL->fields step inside act(); ``eval_signal``/``run_signal_episode``
need no changes.

Demo:
    python -m dev.takeover_probe.nl_scene \
        --out output_data/dev_takeover_probe/nl_scene_v1
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import zlib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dev.takeover_probe.probe import (  # noqa: E402
    DEFAULT_DQN_S1, build_controllers, build_world, _bind_all,
)
from dev.takeover_probe.dispatch_state import (  # noqa: E402
    FeatureBuilder, fb_kwargs_from_meta,
)
from dev.takeover_probe.signal_model import SceneJointPolicy  # noqa: E402
from dev.takeover_probe.eval_signal import eval_episode  # noqa: E402
from dev.takeover_probe.dispatch_eval import SIG9, _EncoderShim  # noqa: E402
from dev.takeover_probe.dispatch_specialist import (  # noqa: E402
    KeyDispatchAgent, wrap_controller,
)
from hitl.live.injects import (  # noqa: E402
    EVENT_KIND_LABEL, EVENT_KINDS, parse_injects,
)

# ---------------------------------------------------------------------
# report-side: fields -> natural-language realisation
# ---------------------------------------------------------------------

_SEV_WORD = {0: '', 1: '轻微', 2: '中等', 3: '严重'}

# {edge} {age}s {sev} {sev_word} {dur}s slots; a template may omit any.
# Held-out split = template index % 3 == 2, so each kind's list is
# ordered to keep every phrasing family (CN/EN, edge-mentioning,
# hard negatives) represented in BOTH train and held-out pools.
NL_TEMPLATES = {
    'congest': [
        # 0 = the canonical injects.py phrasing
        '道路拥堵：{edge} 进口，已持续 {age}s，强度等级 {sev}',
        '接报{edge}进口方向发生交通拥堵，已排队约{age}秒，{sev_word}拥堵',
        '路况通报：{edge} 出现持续性排队，拥堵等级{sev}',
        '市民来电反映：{edge} 车流量过大，通行缓慢',
        'Congestion reported on {edge}, severity {sev}, ongoing {age}s',
        'Major delays on {edge}; heavy congestion reported',
        '{edge} 路段拥堵加剧，请信号机配合管控',
        'Traffic jam at the {edge} approach, level {sev}',
        '拥堵播报：{edge} 方向车辆积压严重，建议启用拥堵预案',
        '{edge} 上游来流突增，路口出现拥堵迹象',
        'Vehicles piling up on {edge}; congestion developing, est {dur}s',
    ],
    'lane_close': [
        '车道封闭：{edge} 内侧车道施工封闭',
        'Lane closure on {edge} due to roadwork',
        '{edge} 发生交通事故，一条车道封闭',
        '{edge} 外侧车道临时封闭，请绕行',
        'A lane is blocked on {edge} after a crash',
        '管制通知：{edge} 车道封闭，通行能力下降',
        '{edge} 因管线抢修封闭一条车道',
        'One lane shut on {edge} for emergency repairs',
    ],
    'none': [
        '正常交通：无事件播报',
        '路况正常，各路口通行顺畅',
        # hard negatives: mention an edge / the word 拥堵, label none
        '道路养护预告：{edge} 将于明日夜间施工',
        'Roadwork advisory: {edge} maintenance scheduled tomorrow',
        '天气晴，能见度良好，无异常路况',
        '{edge} 上周拥堵投诉已办结，今日通行正常',
        'No incidents reported in the network',
        '昨日{edge}拥堵缓行已解除，目前通行顺畅',
        '信号系统巡检正常，无故障',
    ],
    # report lifecycle: kind retained while queues drain (per injects.py)
    'recovering': [
        '{label}已结束：{edge}，残余排队消散中',
        '{edge} 事件解除，交通秩序恢复中',
        'Recovery notice: {edge} congestion clearing',
    ],
}


def nl_render(fields, variant=0, rng=None):
    """Structured report fields -> one NL realisation.

    ``variant`` selects the template deterministically (per-episode
    phrasing); edges/scalars are taken from ``fields``.
    """
    rng = rng or np.random.default_rng(0)
    kind = fields.get('kind', 'none')
    if kind == 'none':
        cands = [t for t in NL_TEMPLATES['none']
                 if '{edge}' not in t or fields.get('edge')]
        t = cands[variant % len(cands)]
        edge = fields.get('edge') or 'road_0_1_0'
    elif fields.get('recovering'):
        t = NL_TEMPLATES['recovering'][variant
                                       % len(NL_TEMPLATES['recovering'])]
        edge = fields.get('edge', '')
    else:
        cands = NL_TEMPLATES.get(kind) or NL_TEMPLATES['none']
        t = cands[variant % len(cands)]
        edge = fields.get('edge', '')
    return t.format(
        edge=edge,
        label=EVENT_KIND_LABEL.get(kind, '事件'),
        age=int(fields.get('onset_age_s', 0)),
        sev=int(fields.get('severity', 0)),
        sev_word=_SEV_WORD.get(int(fields.get('severity', 0)), ''),
        dur=int(fields.get('est_remaining_s', 0)),
    )


# ---------------------------------------------------------------------
# system-side: NL -> structured fields
# ---------------------------------------------------------------------

def _tokens(s):
    """Word tokens for ASCII runs, char uni+bigrams for CJK runs."""
    toks = []
    for m in re.finditer(r'[a-z0-9_]+|[一-鿿]+', s.lower()):
        t = m.group(0)
        if t.isascii():
            toks.append('w:' + t)
        else:
            toks += ['c:' + c for c in t]
            toks += ['b:' + t[i:i + 2] for i in range(len(t) - 1)]
    return toks


class NLSceneParser(nn.Module):
    """Hashed-ngram linear kind classifier + literal edge extraction.

    Deliberately small: the demo claim is the *interface* (NL -> the
    report field schema), not open-domain NLU.  ``parse`` returns a
    fields dict in the same schema ``active_event`` produces.
    """

    def __init__(self, edges, kinds=EVENT_KINDS, dim=2048):
        super().__init__()
        self.edges = sorted(edges, key=len, reverse=True)  # longest first
        self.kinds = list(kinds)
        self.dim = int(dim)
        self.head = nn.Linear(self.dim, len(self.kinds))

    def _feats(self, s):
        x = torch.zeros(self.dim)
        for t in _tokens(s):
            x[zlib.crc32(t.encode()) % self.dim] += 1.0
        return x

    def parse(self, sentence):
        """NL string -> report fields dict (same schema as injects)."""
        with torch.no_grad():
            probs = torch.softmax(
                self.head(self._feats(sentence)), dim=-1).numpy()
        ki = int(probs.argmax())
        kind = self.kinds[ki]
        edge = next((e for e in self.edges if e in sentence), '')
        m = re.search(r'(\d+)\s*(?:s|秒)', sentence)
        age = float(m.group(1)) if m else 0.0
        m = re.search(r'(?:等级|severity|level)\s*(\d)', sentence)
        sev = int(m.group(1)) if m else (
            3 if re.search(r'严重|severe', sentence, re.I)
            else 2 if re.search(r'中等|轻微|moderate|minor', sentence)
            else (2 if kind != 'none' else 0))
        recovering = bool(re.search(
            r'消散中|恢复中|已结束|recovering|cleared', sentence, re.I))
        m = re.search(r'(?:est|预计还需?)\s*(\d+)\s*s?', sentence)
        dur = float(m.group(1)) if m else 0.0
        active = kind != 'none'
        if kind == 'none':
            return {'kind': 'none', 'edge': '', 'active': False,
                    'recovering': False, 'onset_age_s': 0.0,
                    'severity': 0, 'est_remaining_s': 0.0,
                    'objective': 'global',
                    'nl_conf': round(float(probs[ki]), 4)}
        return {'kind': kind, 'edge': edge, 'active': active,
                'recovering': recovering, 'onset_age_s': age,
                'severity': sev, 'est_remaining_s': dur,
                'objective': 'global',
                'nl_conf': round(float(probs[ki]), 4)}


def _corpus(edges, rng):
    """All (kind, template) x sampled slots -> (sentence, kind_label)."""
    rows = []
    edges = list(edges)
    for kind in EVENT_KINDS:
        for ti, t in enumerate(NL_TEMPLATES[kind]):
            n = 24 if kind != 'none' else 16
            for _ in range(n):
                edge = str(rng.choice(edges)) if '{edge}' in t else ''
                sev = int(rng.integers(1, 4))
                s = t.format(
                    edge=edge, age=int(rng.integers(10, 900)),
                    sev=sev, sev_word=_SEV_WORD[sev],
                    dur=int(rng.integers(60, 600)),
                    label=EVENT_KIND_LABEL[kind])
                rows.append((s, kind, ti))
    for ti, t in enumerate(NL_TEMPLATES['recovering']):
        for kind in ('congest', 'lane_close'):
            for _ in range(10):
                s = t.format(edge=str(rng.choice(edges)),
                             label=EVENT_KIND_LABEL[kind])
                # ti+40: recovering template idx1 lands in held-out
                rows.append((s, kind, ti + 40))
    return rows


def train_parser(parser, edges, seed=0, iters=600, lr=3e-3):
    """Template-split train/test: held-out phrasing, not held-out words."""
    rng = np.random.default_rng(seed)
    rows = _corpus(edges, rng)
    test_mask = np.array([ti % 3 == 2 for _, _, ti in rows])
    X = torch.stack([parser._feats(s) for s, _, _ in rows])
    y = torch.tensor([parser.kinds.index(k) for _, k, _ in rows])
    opt = torch.optim.Adam(parser.parameters(), lr=lr)
    for _ in range(iters):
        loss = nn.functional.cross_entropy(parser.head(X[~test_mask]),
                                           y[~test_mask])
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        p = parser.head(X).argmax(dim=-1)
    ev = {
        'train_acc': float((p[~test_mask] == y[~test_mask]).float().mean()),
        'test_acc': float((p[test_mask] == y[test_mask]).float().mean()),
        'n_train': int((~test_mask).sum()), 'n_test': int(test_mask.sum()),
    }
    # per-held-out-template accuracy: generalisation to unseen phrasing
    per_t = {}
    for i, (s, k, ti) in enumerate(rows):
        if test_mask[i]:
            per_t.setdefault((k, ti), []).append(bool((p == y)[i]))
    ev['heldout'] = {f'{k}#{ti}': round(float(np.mean(v)), 3)
                     for (k, ti), v in sorted(per_t.items())}
    return ev


# ---------------------------------------------------------------------
# dispatch agent: NL report in place of the structured text channel
# ---------------------------------------------------------------------

class NLDispatchAgent(KeyDispatchAgent):
    """KeyDispatchAgent whose report channel is natural language.

    Each tick the canonical report fields are realised as an NL string
    (variant-selected phrasing), parsed back to fields, and the text
    vector rebuilt — the attention binder, key rule and registry see an
    ordinary batch.  Audit rows record the raw sentence and parse.
    """

    def __init__(self, parser, *args, variant=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.parser = parser
        self.variant = int(variant)

    def act(self, batch, epsilon=0.0, rng=None, q_base=None):
        fields, _ = self.fb.event_fields(
            self.world.get_current_time())
        nl = nl_render(fields, variant=self.variant)
        pf = self.parser.parse(nl)
        vec = self.fb.fields_text_vec(pf)
        batch = dict(batch)
        batch['text'] = torch.as_tensor(
            vec, dtype=torch.float32).unsqueeze(0)
        out = super().act(batch, epsilon=epsilon, rng=rng, q_base=q_base)
        self.audit[-1].update({
            'nl': nl,
            'nl_kind': pf['kind'], 'nl_edge': pf.get('edge', ''),
            'nl_conf': pf.get('nl_conf'),
        })
        return out


# ---------------------------------------------------------------------
# demo: parser eval + NL-driven takeover vs structured reference
# ---------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene', default='S2')
    p.add_argument('--event',
                   default='surge:road_2_1_2:road_1_2_1:300:600:4')
    p.add_argument('--specialist', default=SIG9)
    p.add_argument('--dqn-s1', default=str(DEFAULT_DQN_S1))
    p.add_argument('--bind-tau', type=float, default=0.4)
    p.add_argument('--nl-variants', default='3,6,1',
                   help='template variants per NL arm; v1 hits the '
                        'held-out recovering phrasing — graceful-failure '
                        'check')
    p.add_argument('--report-delay', type=float, default=0.0)
    p.add_argument('--episode-seconds', type=int, default=1400)
    p.add_argument('--interval', type=int, default=10)
    p.add_argument('--seed', type=int, default=9009)
    p.add_argument('--t-fixed', type=int, default=30)
    p.add_argument('--t-min', type=int, default=10)
    p.add_argument('--runtime-dir', default=None)
    p.add_argument('--out', required=True)
    args = p.parse_args()

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    world = build_world(args.scene, args.seed, Path(
        args.runtime_dir or '/tmp/takeover_nl_scene'), out / 'logger',
        {'t_fixed': args.t_fixed, 't_min': args.t_min,
         'min_green_vehicle': 3, 'max_red_vehicle': 6})
    ranks = list(range(len(world.intersection_ids)))
    injects = parse_injects([args.event])

    specialist, s_meta = SceneJointPolicy.load(args.specialist)
    binder = _EncoderShim(specialist)
    fb_meta = fb_kwargs_from_meta(s_meta.get('meta'))

    controllers = build_controllers(dqn_s1_path=args.dqn_s1)
    _bind_all(world, controllers, ranks,
              needed=set(controllers) - {'dqn_s1'})
    controllers['dqn_s1'] = controllers['dqn_s1'].bind(world, ranks=ranks)
    fb = FeatureBuilder(world, injects,
                        report_delay=args.report_delay, **fb_meta)

    # --- NL front-end: train on template split ------------------------
    parser = NLSceneParser(fb.edges)
    parse_eval = train_parser(parser, fb.edges)
    examples = []
    for kind, v in (('congest', 3), ('congest', 9), ('lane_close', 4),
                    ('none', 5), ('none', 2)):
        t = NL_TEMPLATES[kind][v % len(NL_TEMPLATES[kind])]
        s = t.format(edge=fb.edges[1], age=120, sev=3,
                     sev_word='严重', dur=300,
                     label=EVENT_KIND_LABEL[kind])
        pf = parser.parse(s)
        examples.append({'nl': s, 'kind': pf['kind'],
                         'edge': pf.get('edge', ''),
                         'conf': pf.get('nl_conf')})
    parse_eval['examples'] = examples
    print(json.dumps(parse_eval, indent=1, ensure_ascii=False))

    # --- episodes: structured reference + NL arms ---------------------
    results = {}
    arms = [('dispatch_attn', None)] + [
        (f'dispatch_nl_v{v}', int(v))
        for v in args.nl_variants.split(',') if v != '']
    for name, variant in arms:
        if variant is None:
            agent = KeyDispatchAgent(
                binder, fb, world,
                base_act=wrap_controller(controllers['dqn_s1']),
                registry={'congest': specialist},
                bind_tau=args.bind_tau)
        else:
            agent = NLDispatchAgent(
                parser, binder, fb, world,
                base_act=wrap_controller(controllers['dqn_s1']),
                registry={'congest': specialist},
                bind_tau=args.bind_tau, variant=variant)
        m = eval_episode(world, fb, ranks, injects, agent=agent,
                         episode_seconds=args.episode_seconds,
                         interval=args.interval)
        tk = [r['tick'] for r in agent.audit if r['choice'] == 'specialist']
        results[name] = {
            'metrics': {k: (round(v, 3) if isinstance(v, float) else v)
                        for k, v in m.items()},
            'takeover_ticks': tk,
            'audit': agent.audit,
        }
        world.reset()
        print(f'{name}: takeover {len(tk)} ticks, '
              f'ev_queue_int {m["ev_queue_int"]:.0f}', flush=True)
    world.close()

    ref = results['dispatch_attn']['takeover_ticks']
    for name in results:
        if name == 'dispatch_attn':
            continue
        tk = results[name]['takeover_ticks']
        results[name]['takeover_match'] = {
            'shared': len(set(tk) & set(ref)),
            'nl_only': sorted(set(tk) - set(ref)),
            'ref_only': sorted(set(ref) - set(tk)),
        }
    (out / 'nl_demo.json').write_text(json.dumps(
        {'parse_eval': parse_eval, 'episodes': results},
        indent=1, ensure_ascii=False))
    print('saved ->', out / 'nl_demo.json')


if __name__ == '__main__':
    main()
