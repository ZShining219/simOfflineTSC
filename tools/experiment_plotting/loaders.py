import csv
import json
from dataclasses import dataclass
from pathlib import Path


REQUIRED_RUN_LIST_FIELDS = (
    "role", "agent", "network", "training_seed", "run_dir", "include",
)
TRUE_VALUES = {"1", "true", "yes", "y"}
FALSE_VALUES = {"0", "false", "no", "n"}


@dataclass(frozen=True)
class RunSpec:
    role: str
    agent: str
    network: str
    training_seed: int
    run_dir: Path
    include: bool

    @property
    def run_key(self):
        return f"{self.role}:{self.agent}:{self.network}:seed{self.training_seed}:{self.run_dir.name}"


def _parse_bool(value, line_number):
    normalized = value.strip().lower()
    if normalized in TRUE_VALUES:
        return True
    if normalized in FALSE_VALUES:
        return False
    raise ValueError(
        f"Invalid include value at run-list line {line_number}: {value!r}"
    )


def load_run_list(path):
    run_list_path = Path(path).expanduser().resolve()
    if not run_list_path.is_file():
        raise FileNotFoundError(f"Run-list does not exist: {run_list_path}")
    with run_list_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("Run-list is empty")
        missing = [field for field in REQUIRED_RUN_LIST_FIELDS if field not in reader.fieldnames]
        if missing:
            raise ValueError(f"Run-list is missing required field(s): {', '.join(missing)}")
        specs = []
        seen_paths = set()
        for line_number, row in enumerate(reader, start=2):
            if not any((value or "").strip() for value in row.values()):
                continue
            try:
                seed = int(row["training_seed"])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid training_seed at run-list line {line_number}"
                ) from error
            run_dir = Path(row["run_dir"]).expanduser().resolve()
            if run_dir in seen_paths:
                raise ValueError(f"Duplicate run_dir in run-list: {run_dir}")
            seen_paths.add(run_dir)
            specs.append(RunSpec(
                role=row["role"].strip().lower(),
                agent=row["agent"].strip().lower(),
                network=row["network"].strip(),
                training_seed=seed,
                run_dir=run_dir,
                include=_parse_bool(row["include"], line_number),
            ))
    included = [spec for spec in specs if spec.include]
    if not included:
        raise ValueError("Run-list has no included runs")
    return run_list_path, specs, included


def load_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def load_metric_records(run_dir):
    path = Path(run_dir) / "metrics" / "records.jsonl"
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"Blank metric record at {path}:{line_number}")
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON metric record at {path}:{line_number}"
                ) from error
    return records
