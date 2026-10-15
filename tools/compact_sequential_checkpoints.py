#!/usr/bin/env python3
"""Lossless re-encode of Sequential checkpoints to the packed reuse_counts
wire format.

Walks ``*.pt`` files under ``--root`` and rewrites only payloads whose
``agent_state.ha_sodqn.historical_sampler_state`` still stores
``reuse_counts`` as a plain dict.  The canonical state digest is verified
before and after every rewrite; each file is replaced atomically through
``atomic_torch_save`` and its mtime is preserved.  Files already in the
packed wire format are skipped, so the tool is safe to re-run after an
interruption.  Non-checkpoint payloads (snapshots, ``agents[]`` files,
Plan 2 checkpoints) are never touched.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from sequential.checkpoint import (  # noqa: E402
    atomic_torch_save, decode_checkpoint_payload, validate_full_checkpoint,
)
from sequential.owp import OWP_SCHEMA_VERSION  # noqa: E402


def _sampler_state(payload):
    agent_state = payload.get('agent_state') if isinstance(payload, dict) else None
    if not isinstance(agent_state, dict):
        return None
    ha_sodqn = agent_state.get('ha_sodqn')
    if not isinstance(ha_sodqn, dict):
        return None
    sampler = ha_sodqn.get('historical_sampler_state')
    return sampler if isinstance(sampler, dict) else None


def process_file(path):
    """Rewrite one checkpoint if needed.  Returns a per-file record."""
    record = {'path': str(path), 'status': None}
    try:
        before = os.stat(path)
        try:
            raw = torch.load(path, map_location='cpu')
        except Exception as error:
            record.update(status='load_error', error=str(error))
            return record
        sampler = _sampler_state(raw)
        if sampler is None:
            record['status'] = 'no_sampler_state'
            return record
        reuse_counts = sampler.get('reuse_counts')
        if not isinstance(reuse_counts, dict):
            record['status'] = 'no_reuse_counts'
            return record
        if reuse_counts.get('encoding') is not None:
            record['status'] = 'already_packed'
            return record
        if not reuse_counts:
            record['status'] = 'empty_reuse_counts'
            return record
        if raw.get('canonical_state_digest') is None:
            record['status'] = 'no_canonical_digest'
            return record
        if int(sampler.get('schema_version', -1)) != OWP_SCHEMA_VERSION:
            record.update(status='unexpected_sampler_schema',
                          schema_version=sampler.get('schema_version'))
            return record

        decoded = decode_checkpoint_payload(raw)
        validate_full_checkpoint(decoded)
        expected_digest = decoded['canonical_state_digest']

        atomic_torch_save(decoded, path)

        reloaded = torch.load(path, map_location='cpu')
        verified = decode_checkpoint_payload(reloaded)
        validate_full_checkpoint(verified)
        if verified['canonical_state_digest'] != expected_digest:
            raise RuntimeError('post-rewrite canonical digest changed')
        after_counts = _sampler_state(verified)['reuse_counts']
        if after_counts != reuse_counts:
            raise RuntimeError('post-rewrite reuse_counts differs')
        os.utime(path, (before.st_atime, before.st_mtime))
        after = os.stat(path)
        record.update(
            status='rewritten',
            before_bytes=before.st_size,
            after_bytes=after.st_size,
        )
        return record
    except Exception as error:
        record.update(status='error', error=f'{type(error).__name__}: {error}')
        return record


def _iter_checkpoint_paths(root):
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if name.endswith('.pt'):
                yield os.path.join(dirpath, name)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Losslessly re-encode Sequential HA checkpoints '
                    '(reuse_counts dict -> packed columns)'
    )
    parser.add_argument('--root', required=True,
                        help='directory tree to scan for *.pt checkpoints')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--limit', type=int, default=0,
                        help='process at most N files (smoke runs)')
    parser.add_argument('--dry-run', action='store_true',
                        help='scan and classify files without rewriting')
    parser.add_argument('--report', default=None,
                        help='JSON report output path')
    args = parser.parse_args(argv)

    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        parser.error(f'not a directory: {root}')
    report_path = args.report or os.path.join(
        root, f'checkpoint_compaction_report_{int(time.time())}.json'
    )

    paths = sorted(_iter_checkpoint_paths(root))
    if args.limit:
        paths = paths[:args.limit]
    total = len(paths)
    print(f'scanning {total} .pt files under {root}', flush=True)

    started = time.time()
    records = []
    counts = {}
    reclaimed = 0
    if args.dry_run:
        for index, path in enumerate(paths, 1):
            try:
                raw = torch.load(path, map_location='cpu')
            except Exception as error:
                counts['load_error'] = counts.get('load_error', 0) + 1
                records.append({
                    'path': path, 'status': 'load_error',
                    'error': str(error),
                })
                continue
            sampler = _sampler_state(raw)
            reuse = sampler.get('reuse_counts') if sampler else None
            if sampler is None:
                status = 'no_sampler_state'
            elif not isinstance(reuse, dict):
                status = 'no_reuse_counts'
            elif reuse.get('encoding') is not None:
                status = 'already_packed'
            elif not reuse:
                status = 'empty_reuse_counts'
            else:
                status = 'rewritable'
            counts[status] = counts.get(status, 0) + 1
            if status == 'rewritable':
                reclaimed += os.path.getsize(path)
            if index % 2000 == 0:
                print(f'  scanned {index}/{total} ({counts})', flush=True)
        print(f'dry-run complete in {time.time() - started:.0f}s: {counts}')
        print(f'rewritable bytes (pre-compression): {reclaimed / 1e9:.1f} GB')
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(process_file, path): path for path in paths}
            for index, future in enumerate(as_completed(futures), 1):
                record = future.result()
                records.append(record)
                status = record['status']
                counts[status] = counts.get(status, 0) + 1
                if status == 'rewritten':
                    reclaimed += record['before_bytes'] - record['after_bytes']
                if index % 500 == 0 or status == 'error':
                    print(f'  [{index}/{total}] {counts} '
                          f'reclaimed={reclaimed / 1e9:.2f} GB', flush=True)
        print(f'done in {time.time() - started:.0f}s: {counts}')
        print(f'reclaimed: {reclaimed / 1e9:.2f} GB')

    report = {
        'root': root,
        'dry_run': bool(args.dry_run),
        'files_scanned': total,
        'status_counts': counts,
        'bytes_reclaimed': reclaimed,
        'elapsed_seconds': time.time() - started,
        'records': records,
    }
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    temporary = report_path + '.tmp'
    with open(temporary, 'w', encoding='utf-8') as handle:
        json.dump(report, handle, indent=1)
    os.replace(temporary, report_path)
    print(f'report: {report_path}')
    return 0 if not counts.get('error') else 1


if __name__ == '__main__':
    raise SystemExit(main())
