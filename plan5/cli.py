"""Command line entry points for non-destructive Plan5 Phase0 audits."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from .config import load_config
from .manifest import (
    build_run_manifest, canonical_reference_manifest, write_run_manifest,
)
from .provenance import collect_source_provenance, freeze_environment
from .schema import write_schema_index
from .resources import resource_snapshot
from sequential.core import canonical_digest
from sequential.io import atomic_json, read_json, sha256_file
from .schema import RUN_MANIFEST_COLUMNS
from .validation import validate_fixedtime_evidence
from .probe import validate_probe_manifest


def _latest_report(root, group, algorithm, filename):
    candidates = sorted(
        (Path(root) / group / algorithm).glob(f'attempt_*/{filename}'),
        key=lambda path: int(path.parent.name.split('_')[-1]),
    )
    for path in reversed(candidates):
        try:
            payload = read_json(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if payload.get('valid') is True:
            return path, payload
    return None, None


def _phase0_evidence(root, env, source, references, resource):
    root = Path(root)
    smoke = {}
    isolation = True
    for algorithm in ('DDQN', 'CTXDDQN', 'PPO'):
        path, payload = _latest_report(root, 'smoke', algorithm,
                                       'smoke_report.json')
        if path is None:
            isolation = False
            continue
        checks = (payload.get('evaluation_isolation') or {}).get('checks', {})
        isolation = isolation and all(checks.get(name) is True for name in (
            'separate_process', 'training_state_unchanged',
            'snapshot_unchanged', 'parent_checkpoint_unchanged',
            'deterministic_action'))
        smoke[algorithm] = {
            'report': str(path.resolve()),
            'report_sha256': sha256_file(path),
            'attempt_id': path.parent.name,
            'valid': payload.get('valid') is True,
            'episodes': payload.get('episodes'),
            'decisions': payload.get('counters', {}).get('global_decision_step'),
            'gradient_updates': payload.get('counters', {}).get('gradient_updates'),
            'rollout_updates': payload.get('counters', {}).get('rollout_updates'),
            'isolation_checks': checks,
        }

    resume = {}
    resume_ready = True
    for algorithm in ('DDQN', 'CTXDDQN', 'PPO'):
        path, payload = _latest_report(
            root, 'resume_equivalence', algorithm,
            'resume_equivalence_report.json',
        )
        if path is None:
            resume_ready = False
            continue
        valid = all(payload.get(name) is True for name in (
            'state_equal', 'metrics_equal', 'counters_equal'))
        resume_ready = resume_ready and valid
        resume[algorithm] = {
            'report': str(path.resolve()),
            'report_sha256': sha256_file(path),
            'attempt_id': path.parent.name,
            'valid': payload.get('valid') is True and valid,
            'split_episode': payload.get('split_episode'),
            'final_episode': payload.get('final_episode'),
            'state_equal': payload.get('state_equal'),
            'metrics_equal': payload.get('metrics_equal'),
            'counters_equal': payload.get('counters_equal'),
        }

    fixedtime = root / 'plan5_fixedtime_reference.json'
    fixedtime_evidence = root / 'fixedtime'
    try:
        fixedtime_result = validate_fixedtime_evidence(fixedtime, fixedtime_evidence)
        fixedtime_ready = fixedtime_result['valid']
    except (OSError, ValueError):
        fixedtime_result = None
        fixedtime_ready = False
    probe_path = root / 'probe/plan5_fixed_probe_v1/probe_manifest.json'
    try:
        probe_result = validate_probe_manifest(probe_path)
        probe_ready = bool(probe_result.get('all_causal_gates_valid'))
    except (OSError, ValueError):
        probe_result = None
        probe_ready = False

    report_path = root / 'plan5_resource_report_attempt_3.json'
    resource_ready = False
    resource_report = None
    if report_path.is_file():
        resource_report = read_json(report_path)
        resource_ready = resource_report.get('valid') is True and all(
            resource_report.get('levels', {}).get(str(level), {}).get('passed') is True
            for level in (1, 4, 8)
        )
    test_command = [sys.executable, '-m', 'pytest', '-q',
                    'tests/test_plan5_infrastructure.py',
                    'tests/test_plan5_runtime.py']
    test_run = subprocess.run(test_command, capture_output=True, text=True,
                              check=False)
    test_output = (test_run.stdout + test_run.stderr).strip()
    tests_passed = test_run.returncode == 0 and '38 passed' in test_output
    validation = {
        'schema_version': 1,
        'phase': 'Phase0',
        'phase0_engineering_ready': bool(
            env.get('formal_ready') and references['available_counts']['available'] == 20
            and fixedtime_ready and probe_ready and tests_passed
            and bool(smoke) and resume_ready and isolation and resource_ready
        ),
        'environment_ready': bool(env.get('formal_ready')),
        'source_ready': bool(source.get('formal_source_ready')),
        'canonical_reference_ready': references['available_counts']['available'] == 20,
        'fixedtime_ready': fixedtime_ready,
        'probe_ready': probe_ready,
        'tests_passed': tests_passed,
        'smoke_ready': len(smoke) == 3 and all(item['valid'] for item in smoke.values()),
        'resume_ready': resume_ready and len(resume) == 3,
        'evaluation_isolation_ready': isolation,
        'resource_gate_ready': resource_ready,
        'selected_formal_concurrency': resource_report.get('selected_formal_concurrency') if resource_report else None,
        'formal_training': 'HOLD',
        'reason': (
            'Phase0 engineering and source-readiness gates pass; formal training '
            'remains HOLD pending explicit Phase1 confirmation.'
            if all((env.get('formal_ready'), fixedtime_ready, probe_ready,
                    tests_passed, bool(smoke), resume_ready, isolation,
                    resource_ready, source.get('formal_source_ready')))
            else 'One or more Phase0 engineering or source-readiness gates '
                 'remain incomplete; formal training remains HOLD.'
        ),
        'tests': {'command': ' '.join(test_command), 'returncode': test_run.returncode,
                  'output_tail': test_output[-500:]},
        'smokes': smoke,
        'resume_equivalence': resume,
        'fixedtime': fixedtime_result,
        'probe': probe_result,
        'resource_gate': {
            'report': str(report_path.resolve()),
            'report_sha256': sha256_file(report_path) if report_path.is_file() else None,
            'selected_formal_concurrency': resource_report.get('selected_formal_concurrency') if resource_report else None,
            'levels': ({str(level): {key: resource_report['levels'][str(level)].get(key)
                                     for key in ('passed', 'median_slowdown',
                                                 'peak_host_ram_bytes',
                                                 'swap_growth_bytes', 'disk_ok')}
                        for level in (1, 4, 8)} if resource_report else {}),
        },
        'source': source,
        'environment': {'formal_ready': env.get('formal_ready'),
                        'manifest': str((root / 'provenance/environment/environment_manifest.json').resolve())},
    }
    return validation


def run_manifest_destination(root, source, config_sha256, *, formal=False):
    root = Path(root)
    if formal:
        if not source.get('formal_source_ready'):
            raise ValueError('Formal Plan5 manifest requires clean source')
        return root / 'plan5_run_manifest.csv'
    source_digest = source.get('source_scope_digest', source['head'])
    return root / (
        'plan5_run_manifest_draft_'
        f"{source['head'][:12]}_{config_sha256[:12]}_"
        f"{canonical_digest(RUN_MANIFEST_COLUMNS)[:12]}_"
        f"{source_digest[:12]}.csv"
    )


def phase0_audit(config_path, whitelist_path, output_root):
    config=load_config(config_path); root=Path(output_root); root.mkdir(parents=True,exist_ok=True)
    provenance=root/"provenance"; provenance.mkdir(exist_ok=True)
    env=freeze_environment(provenance / "environment")
    source=collect_source_provenance(provenance/"source.json")
    references=canonical_reference_manifest(whitelist_path, root/"canonical_reference_manifest.json")
    rows=build_run_manifest(config, source["head"])
    # Phase0 validates the complete identity matrix, but it cannot freeze the
    # final manifest yet: PPO rows still need the selected calibration hash and
    # transition rows later need exact episode-100 source checkpoint hashes.
    manifest_path = run_manifest_destination(
        root, source, config.sha256(), formal=False,
    )
    run_manifest = write_run_manifest(rows, manifest_path)
    run_manifest['formal'] = False
    write_schema_index(root/"result_schema.json")
    resource = resource_snapshot(root)
    validation = _phase0_evidence(root, env, source, references, resource)
    atomic_json(root/"plan5_validation_report.json",validation)
    return {"config_sha256":config.sha256(),"environment":env,"source":source,"reference":references["available_counts"],"run_manifest":run_manifest,"run_count":len(rows),"validation":validation}


def main(argv=None):
    parser=argparse.ArgumentParser(prog="python -m plan5")
    sub=parser.add_subparsers(dest="command",required=True)
    audit=sub.add_parser("phase0-audit",help="collect provenance and validate frozen identities")
    audit.add_argument("--config",default="configs/sequential/plan5_cross_algorithm_b100.yml")
    audit.add_argument("--whitelist",default="data/output_data/analysis/plan1/p1_formal_20_trajectory_whitelist_20260722.csv")
    audit.add_argument("--output",default="data/output_data/cross_algorithm/plan5_b100")
    fixed = sub.add_parser("fixedtime", help="build repeatable Plan5 FixedTime references")
    fixed.add_argument("--output", default="data/output_data/cross_algorithm/plan5_b100")
    fixed.add_argument("--interface", choices=("traci", "libsumo"), default="traci")
    probe = sub.add_parser("generate-probe", help="generate immutable plan5_fixed_probe_v1")
    probe.add_argument("--reference-manifest", default="data/output_data/cross_algorithm/plan5_b100/canonical_reference_manifest.json")
    probe.add_argument("--output", default="data/output_data/cross_algorithm/plan5_b100")
    probe.add_argument("--interface", choices=("traci", "libsumo"), default="traci")
    smoke = sub.add_parser("smoke", help="run one real-SUMO algorithm smoke")
    smoke.add_argument("--algorithm", required=True, choices=("DDQN", "CTXDDQN", "PPO"))
    smoke.add_argument("--output", required=True)
    smoke.add_argument("--seed", type=int, default=999)
    smoke.add_argument("--episodes", type=int, default=None)
    smoke.add_argument("--interface", choices=("traci", "libsumo"), default="traci")
    smoke.add_argument(
        "--skip-evaluation-isolation", action="store_true",
        help=argparse.SUPPRESS,
    )
    resume = sub.add_parser("resume-equivalence", help="run real-SUMO checkpoint equivalence")
    resume.add_argument("--algorithm", required=True, choices=("DDQN", "CTXDDQN", "PPO"))
    resume.add_argument("--output", required=True)
    resume.add_argument("--seed", type=int, default=998)
    resume.add_argument("--interface", choices=("traci", "libsumo"), default="traci")
    concurrency = sub.add_parser(
        "concurrency-gate", help="run the DDQN 1/4/8 resource gate",
    )
    concurrency.add_argument(
        "--output", default="data/output_data/cross_algorithm/plan5_b100",
    )
    concurrency.add_argument(
        "--interface", choices=("traci", "libsumo"), default="traci",
    )
    formal = sub.add_parser(
        'formal-child', help='run one pre-authorized formal/calibration child',
    )
    formal.add_argument('--row', required=True)
    formal.add_argument('--attempt', required=True)
    formal.add_argument('--probe', default=None)
    formal.add_argument('--resume-from', default=None)
    formal.add_argument('--ppo-config', default=None)
    formal.add_argument(
        '--interface', choices=('traci', 'libsumo'), default='traci',
    )
    args=parser.parse_args(argv)
    if args.command=="phase0-audit":
        result=phase0_audit(args.config,args.whitelist,args.output)
    elif args.command == "fixedtime":
        from .config import SCENES
        from .evaluator import build_fixedtime_reference
        from .sumo_runtime import FixedTimePolicy, evaluate_policy
        counters = {scene: 0 for scene in SCENES}
        def evaluate(scene):
            counters[scene] += 1
            repeat_root = (
                Path(args.output) / "fixedtime" / scene
                / f"repeat_{counters[scene]}"
            )
            attempt = 1
            while (repeat_root / f'attempt_{attempt}').exists():
                attempt += 1
            attempt_dir = repeat_root / f'attempt_{attempt}'
            run = evaluate_policy(
                SCENES[scene],
                attempt_dir,
                FixedTimePolicy(), interface=args.interface,
            )
            atomic_json(
                repeat_root / f"attempt_{attempt}_summary.json",
                run["summary"],
            )
            return run["summary"]
        result = build_fixedtime_reference(
            SCENES, evaluate,
            Path(args.output) / "plan5_fixedtime_reference.json",
        )
    elif args.command == "generate-probe":
        from .probe import generate_probe
        reference = read_json(args.reference_manifest)
        result = generate_probe(
            reference,
            Path(args.output) / "probe" / "plan5_fixed_probe_v1",
            Path(args.output) / "probe" / "evaluation_runs",
            interface=args.interface,
        )
    elif args.command == "smoke":
        from .smoke import run_smoke
        result = run_smoke(
            args.algorithm, args.output, seed=args.seed,
            episodes=args.episodes, interface=args.interface,
            verify_evaluation_isolation=not args.skip_evaluation_isolation,
        )
    elif args.command == "resume-equivalence":
        from .smoke import resume_equivalence
        result = resume_equivalence(
            args.algorithm, args.output, seed=args.seed,
            interface=args.interface,
        )
    elif args.command == 'concurrency-gate':
        from .resources import run_concurrency_gate
        result = run_concurrency_gate(
            Path(args.output) / 'resource_gate',
            Path(args.output) / 'plan5_resource_report.json',
            interface=args.interface,
        )
    elif args.command == 'formal-child':
        from .runtime import run_formal_child
        row = read_json(args.row)
        result = run_formal_child(
            row, args.attempt, probe_manifest=args.probe,
            interface=args.interface, resume_from=args.resume_from,
            ppo_config=args.ppo_config,
        )
    print(result)


if __name__=="__main__": main()
