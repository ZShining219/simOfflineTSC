"""Environment/source provenance probes used by the Phase0 gate."""
import importlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from sequential.core import canonical_digest
from sequential.io import atomic_json, sha256_file


SOURCE_SUFFIXES = {
    '.py', '.pyi', '.yml', '.yaml', '.toml', '.json', '.cfg', '.sumocfg',
    '.xml', '.sh', '.ini', '.txt',
}


def is_source_scope_path(path):
    path = Path(path)
    if path.parts[:2] == ('data', 'output_data'):
        return False
    if path.parts and path.parts[0] == 'docs':
        return False
    return path.suffix.lower() in SOURCE_SUFFIXES


def collect_environment(output_path=None, sumo_binary="sumo"):
    packages={}
    distributions = {
        'torch': 'torch', 'gym': 'gym', 'numpy': 'numpy', 'pfrl': 'pfrl',
        'traci': 'traci', 'libsumo': 'libsumo', 'sumolib': 'sumolib',
        'yaml': 'PyYAML', 'lmdb': 'lmdb',
        'eclipse-sumo': 'eclipse-sumo',
    }
    modules = {
        'torch': 'torch', 'gym': 'gym', 'numpy': 'numpy', 'pfrl': 'pfrl',
        'traci': 'traci', 'libsumo': 'libsumo', 'sumolib': 'sumolib',
        'yaml': 'yaml', 'lmdb': 'lmdb', 'eclipse-sumo': 'sumo',
    }
    for name, module_name in modules.items():
        try:
            module=importlib.import_module(module_name)
            packages[name]={
                "version": importlib.metadata.version(distributions[name]),
                "path": getattr(module,"__file__", None),
                "module_file_sha256": (
                    sha256_file(module.__file__)
                    if getattr(module, '__file__', None)
                    and os.path.isfile(module.__file__) else None
                ),
            }
        except Exception as error:
            packages[name]={"status":"missing","error":f"{type(error).__name__}: {error}"}
    binary=shutil.which(sumo_binary)
    sumo_version=None
    binary_target = None
    if binary:
        try: sumo_version=subprocess.run([binary,"--version"],capture_output=True,text=True,check=False).stdout.strip()
        except OSError as error: sumo_version=f"error: {error}"
        try:
            sumo_module = importlib.import_module('sumo')
            candidate = Path(sumo_module.__file__).parent / 'bin' / 'sumo'
            if candidate.is_file():
                binary_target = str(candidate.resolve())
        except Exception:
            binary_target = None
    api_probe = {}
    for module_name, attributes in {
        "traci": ("start", "getConnection"),
        "libsumo": ("start", "simulation", "vehicle", "trafficlight"),
        "sumolib": ("checkBinary", "net"),
    }.items():
        try:
            module = importlib.import_module(module_name)
            api_probe[module_name] = {
                "available": all(hasattr(module, item) for item in attributes),
                "attributes": {item: hasattr(module, item) for item in attributes},
                "path": getattr(module, "__file__", None),
            }
        except Exception as error:
            api_probe[module_name] = {
                "available": False,
                "error": f"{type(error).__name__}: {error}",
            }
    expected = {
        "torch": "1.13.1+cu116", "gym": "0.26.2",
        "numpy": "1.26.4", "pfrl": "0.4.0", "lmdb": "1.3.0",
        "traci": "1.27.1", "libsumo": "1.27.1",
        "sumolib": "1.27.1", "eclipse-sumo": "1.27.1",
    }
    isolated_script = """
import importlib
import json
import os
import sys

modules = %r
loaded = {}
for name, module_name in modules.items():
    module = importlib.import_module(module_name)
    loaded[name] = os.path.realpath(module.__file__)
prefix = os.path.realpath(sys.prefix)
inside_prefix = all(
    os.path.commonpath((prefix, path)) == prefix
    for path in loaded.values()
)
print(json.dumps({
    'inside_prefix': inside_prefix,
    'modules': loaded,
    'prefix': prefix,
}))
""" % modules
    isolated_process = subprocess.run(
        [sys.executable, '-c', isolated_script],
        capture_output=True, text=True, check=False,
        env={**os.environ, 'PYTHONNOUSERSITE': '1'},
    )
    try:
        isolated_probe = json.loads(isolated_process.stdout)
    except json.JSONDecodeError:
        isolated_probe = {}
    isolated_probe.update({
        'exit_code': isolated_process.returncode,
        'stderr': isolated_process.stderr,
        'user_site_disabled': True,
    })
    isolated_ready = bool(
        isolated_process.returncode == 0
        and isolated_probe.get('inside_prefix')
        and set(isolated_probe.get('modules', {})) == set(modules)
    )
    checks = {
        "python": platform.python_version() == "3.10.18",
        **{
            name: isinstance(packages.get(name), dict)
            and packages[name].get("version") == version
            for name, version in expected.items()
        },
        "sumo_binary": binary is not None
        and "sumo 1.27.1" in (sumo_version or ""),
        "sumo_python_apis": all(
            item.get("available") for item in api_probe.values()
        ),
        "isolated_runtime_import": isolated_ready,
        "cpu_device": True,
    }
    payload={
        "schema_version":1,"python":sys.version,
        "python_version": platform.python_version(),
        "python_executable":sys.executable,
        "python_executable_sha256": (
            sha256_file(sys.executable) if os.path.isfile(sys.executable) else None
        ),
        "platform":platform.platform(),"packages":packages,
        "sumo":{"binary":binary,"version":sumo_version,
                "binary_sha256":sha256_file(binary) if binary else None,
                "binary_target": binary_target,
                "binary_target_sha256": (
                    sha256_file(binary_target) if binary_target else None
                )},
        "sumo_python_api_probe": api_probe,
        "isolated_runtime_probe": isolated_probe,
        "formal_device":"cpu","expected_versions":expected,
        "checks": checks,
    }
    payload["formal_ready"]=all(checks.values())
    if output_path: atomic_json(output_path,payload)
    return payload


def freeze_environment(output_dir):
    """Write reproducible package inventory and hashes into provenance."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    probe = collect_environment(output_dir / "environment_probe.json")
    freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze", "--all"],
        capture_output=True, text=True, check=True,
        env={**os.environ, "PYTHONNOUSERSITE": "1"},
    ).stdout
    freeze_path = output_dir / "requirements_frozen.txt"
    with open(freeze_path, "w", encoding="utf-8") as handle:
        handle.write(freeze)
        if freeze and not freeze.endswith("\n"):
            handle.write("\n")
    conda_explicit_path = output_dir / 'conda_explicit.txt'
    mamba_executable = os.environ.get('MAMBA_EXE')
    conda_prefix = os.environ.get('CONDA_PREFIX')
    if mamba_executable and conda_prefix:
        conda_explicit = subprocess.run(
            [mamba_executable, 'list', '-p', conda_prefix, '--explicit'],
            capture_output=True, text=True, check=True,
        ).stdout
    else:
        conda_explicit = '# unavailable: run freeze inside activated colight\n'
    with open(conda_explicit_path, 'w', encoding='utf-8') as handle:
        handle.write(conda_explicit)
        if conda_explicit and not conda_explicit.endswith('\n'):
            handle.write('\n')
    activation_scripts = {}
    if conda_prefix:
        for relative in (
            'etc/conda/activate.d/plan5-sumo.sh',
            'etc/conda/deactivate.d/plan5-sumo.sh',
        ):
            path = Path(conda_prefix) / relative
            if path.is_file():
                activation_scripts[relative] = sha256_file(path)
    environment_patches = {}
    patch_root = Path(conda_prefix).parents[2] / 'patches' if conda_prefix else None
    if patch_root and patch_root.is_dir():
        environment_patches = {
            path.name: sha256_file(path)
            for path in sorted(patch_root.iterdir()) if path.is_file()
        }
    manifest = {
        "schema_version": 1, "formal_ready": probe["formal_ready"],
        "environment_probe": str((output_dir / "environment_probe.json").resolve()),
        "environment_probe_sha256": sha256_file(
            output_dir / "environment_probe.json"
        ),
        "requirements_frozen": str(freeze_path.resolve()),
        "requirements_frozen_sha256": sha256_file(freeze_path),
        'conda_explicit': str(conda_explicit_path.resolve()),
        'conda_explicit_sha256': sha256_file(conda_explicit_path),
        'activation_script_sha256': activation_scripts,
        'environment_patch_sha256': environment_patches,
    }
    atomic_json(output_dir / "environment_manifest.json", manifest)
    return manifest


def collect_source_provenance(output_path=None):
    def command(*args):
        result=subprocess.run(args,capture_output=True,text=True,check=False)
        return result.stdout.strip()
    status=command("git","status","--short","--untracked-files=all")
    tracked_diff=command("git","diff","HEAD","--name-only")
    tracked_changes = tracked_diff.splitlines() if tracked_diff else []
    untracked_changes = [
        line[3:] for line in status.splitlines() if line.startswith("?? ")
    ]
    untracked_source_changes = [
        path for path in untracked_changes if is_source_scope_path(path)
    ]
    head = command("git", "rev-parse", "HEAD")
    source_paths = sorted(set(tracked_changes) | set(untracked_source_changes))
    source_files = []
    for relative in source_paths:
        path = Path(relative)
        source_files.append({
            'path': relative,
            'sha256': sha256_file(path) if path.is_file() else None,
            'kind': (
                'file' if path.is_file() else
                'directory' if path.is_dir() else 'missing'
            ),
        })
    payload={"branch":command("git","branch","--show-current"),"head":head,"status":status,"tracked_changes":tracked_changes,"untracked_changes":untracked_changes,"untracked_source_changes":untracked_source_changes,"source_files":source_files,"remote_tracking":command("git","rev-parse","--abbrev-ref","--symbolic-full-name","@{u}")}
    payload["worktree_clean"]=not bool(status)
    payload["tracked_worktree_clean"]=not bool(tracked_changes)
    payload["source_scope_clean"] = not (
        tracked_changes or untracked_source_changes
    )
    payload['source_scope_digest'] = canonical_digest({
        'head': head, 'source_files': source_files,
    })
    payload["formal_source_ready"]=payload["source_scope_clean"]
    if output_path: atomic_json(output_path,payload)
    return payload


def freeze_source_bundle(output_dir, expected_commit):
    """Create the authorized immutable Git bundle for a clean source commit."""
    source = collect_source_provenance()
    if not source['formal_source_ready']:
        raise ValueError('Plan5 Git bundle requires clean formal source')
    if source['head'] != str(expected_commit):
        raise ValueError('Plan5 Git bundle commit does not match HEAD')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = output_dir / f'plan5_source_{source["head"][:12]}.bundle'
    manifest_path = output_dir / 'source_bundle_manifest.json'
    if bundle_path.exists() or manifest_path.exists():
        raise FileExistsError('Plan5 source bundle is immutable')
    descriptor, temporary = tempfile.mkstemp(
        prefix='.tmp-plan5-source-', suffix='.bundle', dir=output_dir,
    )
    os.close(descriptor)
    os.unlink(temporary)
    try:
        created = subprocess.run(
            ['git', 'bundle', 'create', temporary, 'HEAD'],
            capture_output=True, text=True, check=False,
        )
        if created.returncode != 0:
            raise RuntimeError('Plan5 Git bundle creation failed: ' + created.stderr)
        verified = subprocess.run(
            ['git', 'bundle', 'verify', temporary],
            capture_output=True, text=True, check=False,
        )
        if verified.returncode != 0:
            raise RuntimeError('Plan5 Git bundle verification failed: ' + verified.stderr)
        os.replace(temporary, bundle_path)
        temporary = None
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)
    manifest = {
        'schema_version': 1, 'branch': source['branch'],
        'head': source['head'], 'source_scope_digest': source[
            'source_scope_digest'
        ],
        'bundle_path': str(bundle_path.resolve()),
        'bundle_sha256': sha256_file(bundle_path),
        'bundle_verify_stdout': verified.stdout,
        'bundle_verify_stderr': verified.stderr,
        'valid': True,
    }
    atomic_json(manifest_path, manifest)
    return manifest
