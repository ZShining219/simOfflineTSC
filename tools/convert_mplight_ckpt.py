#!/usr/bin/env python3
"""Convert legacy MPLight model/*.pt snapshots into audited checkpoint files.

MPLightAgent is PFRL-based and sets ``self.target_model = None``, so the
trainer's ``save_checkpoint`` early-returns and produces no audited
checkpoints.  Its legacy ``model/{ep}_{rank}.pt`` files still carry the full
FRAP ``model_state_dict``/``optimizer_state_dict``.  This converter wraps the
legacy weights into an ``evaluation``-type checkpoint payload (schema v1) so
the frozen-evaluation collection path can load them like every other agent.

The resumable role is intentionally NOT fabricated: optimizer/epsilon/replay
state cannot be reconstructed after the fact, so MPLight manifests must use
``checkpoint_role='best'`` (summary.json already points at this file).
"""
import argparse
import json
from pathlib import Path

import torch


def convert(run_dir: Path, episode: int) -> Path:
    legacy = run_dir / 'model' / f'{episode}_0.pt'
    if not legacy.is_file():
        raise FileNotFoundError(f'{run_dir.name}: missing {legacy}')
    payload_src = torch.load(legacy, map_location='cpu')
    manifest = json.loads((run_dir / 'run_manifest.json').read_text())
    status_path = run_dir / 'run_status.json'
    status = json.loads(status_path.read_text()) if status_path.is_file() else {}
    gstep = int(status.get('global_decision_step') or episode * 360)
    payload = {
        'schema_version': 1,
        'checkpoint_type': 'evaluation',
        'episode': episode,
        'global_decision_step': gstep,
        'gradient_updates': int(status.get('gradient_updates') or gstep),
        'config_hash': manifest['config_hash'],
        'agents': [{
            'rank': 0,
            'online_model_state_dict': payload_src['model_state_dict'],
        }],
    }
    out = run_dir / 'checkpoints' / 'evaluation' / f'episode_{episode:04d}.pt'
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    (out.parent / f'episode_{episode:04d}.provenance.json').write_text(
        json.dumps({
            'converted_from': str(legacy),
            'legacy_keys': sorted(payload_src),
            'checkpoint_type': 'evaluation',
            'note': ('Legacy MPLight snapshot wrapped into audited evaluation '
                     'checkpoint; target/optimizer/replay state not preserved '
                     '(resumable role unavailable by design).'),
        }, indent=2) + '\n')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_dirs', nargs='+', type=Path)
    ap.add_argument('--episode', type=int, default=200)
    args = ap.parse_args()
    for run_dir in args.run_dirs:
        print(convert(run_dir.resolve(), args.episode))


if __name__ == '__main__':
    main()
