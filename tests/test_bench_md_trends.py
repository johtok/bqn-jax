from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "bench_md_trends.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("bench_md_trends", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load bench_md_trends module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class BenchMdTrendScriptTests(unittest.TestCase):
    def test_build_trend_cases_from_synthetic_runs(self) -> None:
        module = _load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_a = root / "20260216T000000Z"
            run_b = root / "20260216T000100Z"
            run_a.mkdir(parents=True, exist_ok=True)
            run_b.mkdir(parents=True, exist_ok=True)

            payload_a = {
                "timestamp_utc": "20260216T000000Z",
                "rows": [
                    {"section": "monadic", "name": "plus", "n": 10000, "mean_ms": 0.010},
                    {"section": "dyadic", "name": "minus", "n": 10000, "mean_ms": 0.020},
                ],
            }
            payload_b = {
                "timestamp_utc": "20260216T000100Z",
                "rows": [
                    {"section": "monadic", "name": "plus", "n": 10000, "mean_ms": 0.012},
                    {"section": "dyadic", "name": "minus", "n": 10000, "mean_ms": 0.018},
                ],
            }
            (run_a / "results.json").write_text(json.dumps(payload_a), encoding="utf-8")
            (run_b / "results.json").write_text(json.dumps(payload_b), encoding="utf-8")

            files = module._discover_result_files(root)
            self.assertEqual(2, len(files))

            run_labels, cases = module.build_trend_cases(files, n=10000, min_points=2)
            self.assertEqual(2, len(run_labels))
            self.assertEqual(2, len(cases))

            by_key = {(case.section, case.name): case for case in cases}
            plus = by_key[("monadic", "plus")]
            minus = by_key[("dyadic", "minus")]

            self.assertAlmostEqual(plus.baseline_ms, 0.010, places=9)
            self.assertAlmostEqual(plus.latest_ms, 0.012, places=9)
            self.assertAlmostEqual(plus.delta_pct, 20.0, places=6)

            self.assertAlmostEqual(minus.baseline_ms, 0.020, places=9)
            self.assertAlmostEqual(minus.latest_ms, 0.018, places=9)
            self.assertAlmostEqual(minus.delta_pct, -10.0, places=6)


if __name__ == "__main__":
    unittest.main()
