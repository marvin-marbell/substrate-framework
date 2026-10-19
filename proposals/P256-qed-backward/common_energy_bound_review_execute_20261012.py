"""Archive declared review inputs before a NEW control process; retain natural exit."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PYTHON = "/home/administrator/substrate-framework/.venv/bin/python"


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def save(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    prefix = request["epoch_prefix"]
    assert prefix.startswith("common_energy_bound_review_")
    inputs = {name: (ROOT / name).read_bytes() for name in request["pins"]}
    before = {name: sha(raw) for name, raw in inputs.items()}
    assert before == request["pins"]
    pre = ROOT / f"{prefix}_preexecution.zip"
    executed = ROOT / f"{prefix}_executed.zip"
    output = ROOT / f"{prefix}_receipt.json"
    stdout = ROOT / f"{prefix}_stdout.txt"
    stderr = ROOT / f"{prefix}_stderr.txt"
    exit_path = ROOT / f"{prefix}_exit.json"
    capsule = ROOT / f"{prefix}_capsule.json"
    assert not any(path.exists() for path in (pre, executed, output, stdout, stderr, exit_path, capsule))
    with zipfile.ZipFile(pre, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, raw in inputs.items():
            archive.writestr(Path(name).name, raw)
        archive.writestr(args.request.name, args.request.read_bytes())
    pre_hash = sha(pre.read_bytes())
    command = [PYTHON, str(ROOT / request["consumer"]), "--request", str(args.request), "--output", str(output)]
    started_at = dt.datetime.now(dt.timezone.utc).isoformat()
    started = time.monotonic()
    process = subprocess.run(command, capture_output=True, check=False)
    elapsed = time.monotonic() - started
    completed_at = dt.datetime.now(dt.timezone.utc).isoformat()
    stdout.write_bytes(process.stdout)
    stderr.write_bytes(process.stderr)
    runtime = {"command": command, "returncode": process.returncode,
               "natural_exit": process.returncode >= 0, "started_at": started_at,
               "completed_at": completed_at, "elapsed_seconds": elapsed,
               "timeout_or_signal_exit_used": False}
    save(exit_path, runtime)
    after = {name: sha((ROOT / name).read_bytes()) for name in inputs}
    outputs = [args.request, stdout, stderr, exit_path]
    if output.exists():
        outputs.append(output)
    with zipfile.ZipFile(executed, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, raw in inputs.items():
            archive.writestr(Path(name).name, raw)
        for path in outputs:
            archive.writestr(path.name, path.read_bytes())
    result = {"kind": "INDEPENDENT_REVIEW_NEW_CONTROLS_IMMUTABLE_EXECUTED_CAPSULE",
              "pins_before": before, "pins_after": after, "inputs_unchanged": before == after,
              "preexecution": {"path": pre.name, "sha256": pre_hash, "created_before_child_launch": True},
              "executed": {"path": executed.name, "sha256": sha(executed.read_bytes())},
              "outputs": {path.name: sha(path.read_bytes()) for path in outputs}, "runtime": runtime,
              "failed_epoch_preserved": process.returncode != 0, "original_producer_rerun": False,
              "native_ground_phase_IR_atom": False, "preexecution_backfill": False}
    save(capsule, result)
    print(json.dumps({"capsule": capsule.name, "returncode": process.returncode,
                      "natural_exit": runtime["natural_exit"], "inputs_unchanged": before == after,
                      "elapsed_seconds": elapsed}))
    if process.returncode != 0:
        raise SystemExit(process.returncode)
    assert before == after


if __name__ == "__main__":
    main()
