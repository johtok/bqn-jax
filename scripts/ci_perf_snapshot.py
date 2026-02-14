"""Collect quick benchmark snapshots for CI artifact upload."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path


def _run(command: list[str]) -> dict[str, object]:
    proc = subprocess.run(command, check=False, text=True, capture_output=True)
    return {
        "command": command,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    outdir = root / "benchmarks" / "output" / "perf_ci" / stamp
    outdir.mkdir(parents=True, exist_ok=True)

    py = sys.executable
    runs = [
        _run(
            [
                py,
                "benchmarks/perf_benchmarks.py",
                "--profile",
                "quick",
                "--json-out",
                str(outdir / "perf.json"),
            ]
        ),
        _run(
            [
                py,
                "benchmarks/jax_workloads.py",
                "--profile",
                "quick",
                "--json-out",
                str(outdir / "workloads.json"),
            ]
        ),
        _run(
            [
                py,
                "benchmarks/valence_benchmarks.py",
                "--profile",
                "quick",
                "--ns",
                "10,1000,10000",
                "--samples",
                "1",
                "--warmup",
                "0",
                "--repeat-scale",
                "0.03",
                "--json-out",
                str(outdir / "valence.json"),
            ]
        ),
    ]

    summary = {
        "timestamp_utc": stamp,
        "runs": runs,
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote perf snapshot: {outdir}")
    for row in runs:
        status = "ok" if row["returncode"] == 0 else "error"
        cmd = " ".join(row["command"])
        print(f"[{status}] {cmd}")

    # Snapshot collection should not hard-fail CI; diagnostics are in artifacts.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
