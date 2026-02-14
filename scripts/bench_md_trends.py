"""Per-op trend charts for monad/dyad benchmark runs."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

SPARK_CHARS = " .:-=+*#%@"
SECTION_ORDER = ("monadic", "dyadic", "modifier")


@dataclass(frozen=True)
class TrendCase:
    section: str
    name: str
    n: int
    baseline_ms: float
    latest_ms: float
    delta_pct: float
    sample_points: int
    sparkline: str
    series_ms: tuple[float, ...]
    run_labels: tuple[str, ...]


def _discover_result_files(root: Path) -> list[Path]:
    files = [path for path in root.glob("*/results.json") if path.is_file()]
    files.extend(path for path in root.glob("results-*.json") if path.is_file())
    files.sort(key=lambda path: path.parent.name if path.name == "results.json" else path.name)
    return files


def _load_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must decode to an object payload")
    return payload


def _run_label(path: Path, payload: dict[str, Any]) -> str:
    label = payload.get("timestamp_utc")
    if isinstance(label, str) and label:
        return label
    if path.name == "results.json":
        return path.parent.name
    return path.stem


def _sparkline(values: list[float]) -> str:
    if not values:
        return ""
    if len(values) == 1:
        return "="
    lo = min(values)
    hi = max(values)
    if hi <= lo:
        return "=" * len(values)
    scale = len(SPARK_CHARS) - 1
    chars: list[str] = []
    for value in values:
        idx = int(round(((value - lo) / (hi - lo)) * scale))
        idx = max(0, min(scale, idx))
        chars.append(SPARK_CHARS[idx])
    return "".join(chars)


def build_trend_cases(files: list[Path], *, n: int, min_points: int = 2) -> tuple[list[str], list[TrendCase]]:
    run_labels: list[str] = []
    table: dict[tuple[str, str], list[tuple[str, float]]] = {}
    for path in files:
        payload = _load_payload(path)
        rows = payload.get("rows")
        if not isinstance(rows, list):
            continue
        label = _run_label(path, payload)
        run_labels.append(label)
        for row in rows:
            try:
                if int(row["n"]) != n:
                    continue
                section = str(row["section"])
                name = str(row["name"])
                mean_ms = float(row["mean_ms"])
            except Exception:
                continue
            table.setdefault((section, name), []).append((label, mean_ms))

    out: list[TrendCase] = []
    for (section, name), series in table.items():
        if len(series) < min_points:
            continue
        labels = tuple(item[0] for item in series)
        values = tuple(float(item[1]) for item in series)
        baseline = values[0]
        latest = values[-1]
        delta_pct = ((latest / baseline) - 1.0) * 100.0 if baseline > 0 else float("inf")
        out.append(
            TrendCase(
                section=section,
                name=name,
                n=n,
                baseline_ms=baseline,
                latest_ms=latest,
                delta_pct=delta_pct,
                sample_points=len(values),
                sparkline=_sparkline(list(values)),
                series_ms=values,
                run_labels=labels,
            )
        )
    return run_labels, out


def _render_markdown(run_labels: list[str], cases: list[TrendCase], *, n: int) -> str:
    lines: list[str] = []
    lines.append("# Bench MD Per-Op Trends")
    lines.append("")
    lines.append(f"- target_n: `{n}`")
    lines.append(f"- runs_considered: `{len(run_labels)}`")
    if run_labels:
        lines.append(f"- run_labels: `{', '.join(run_labels)}`")
    lines.append("")
    for section in SECTION_ORDER:
        section_rows = [row for row in cases if row.section == section]
        if not section_rows:
            continue
        section_rows.sort(key=lambda row: row.delta_pct, reverse=True)
        lines.append(f"## {section}")
        lines.append("")
        lines.append("| case | latest_ms | delta_vs_first | points | trend |")
        lines.append("| --- | ---: | ---: | ---: | --- |")
        for row in section_rows:
            lines.append(
                f"| `{row.name}` | `{row.latest_ms:.6f}` | `{row.delta_pct:+.2f}%` | `{row.sample_points}` | `{row.sparkline}` |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        default="benchmarks/output/monad_dyad",
        help="root containing bench-md outputs (timestamp dirs with results.json)",
    )
    parser.add_argument("--n", type=int, default=10000, help="array size row to chart")
    parser.add_argument("--window", type=int, default=10, help="number of latest runs to include")
    parser.add_argument("--min-points", type=int, default=2, help="minimum points required per op")
    parser.add_argument("--json-out", default="", help="optional machine-readable output path")
    parser.add_argument("--md-out", default="", help="optional markdown output path")
    args = parser.parse_args()

    root = Path(args.results_root)
    files = _discover_result_files(root)
    if not files:
        raise SystemExit(f"No benchmark result files found under {root}")
    files = files[-max(1, int(args.window)) :]

    run_labels, cases = build_trend_cases(files, n=int(args.n), min_points=max(1, int(args.min_points)))
    markdown = _render_markdown(run_labels, cases, n=int(args.n))
    print(markdown)

    if args.md_out:
        md_path = Path(args.md_out)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(markdown, encoding="utf-8")
        print(f"\nWrote markdown: {md_path}")

    if args.json_out:
        payload = {
            "target_n": int(args.n),
            "runs_considered": len(run_labels),
            "run_labels": run_labels,
            "cases": [asdict(case) for case in cases],
        }
        json_path = Path(args.json_out)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote JSON: {json_path}")


if __name__ == "__main__":
    main()
