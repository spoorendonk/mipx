import argparse
import importlib.util
from pathlib import Path
import tempfile
import sys
import unittest


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "tests"
    / "perf"
    / "run_barrier_lp_regression_gate.py"
)
SPEC = importlib.util.spec_from_file_location("barrier_gate", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
gate = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = gate
SPEC.loader.exec_module(gate)


class BarrierRegressionGateArgsTest(unittest.TestCase):
    def make_args(self, **overrides):
        args = argparse.Namespace(
            instances="",
            max_instances=0,
            all_available_instances=False,
        )
        for key, value in overrides.items():
            setattr(args, key, value)
        return args

    def test_default_uses_curated_gpu_stable_set(self):
        filt = gate.resolve_instance_filter(self.make_args())
        self.assertEqual(
            filt,
            "adlittle,afiro,blend,sc50b,share2b,stocfor1",
        )

    def test_explicit_instances_override_curated_default(self):
        filt = gate.resolve_instance_filter(self.make_args(instances="afiro,blend"))
        self.assertEqual(filt, "afiro,blend")

    def test_all_available_instances_disables_curated_default(self):
        filt = gate.resolve_instance_filter(
            self.make_args(all_available_instances=True)
        )
        self.assertEqual(filt, "")

    def test_max_instances_disables_curated_default(self):
        filt = gate.resolve_instance_filter(self.make_args(max_instances=5))
        self.assertEqual(filt, "")


class BarrierRegressionGateCsvTest(unittest.TestCase):
    def test_write_lane_metric_csv_supports_split_baseline_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            in_csv = Path(tmp) / "baseline.csv"
            out_csv = Path(tmp) / "metric.csv"
            in_csv.write_text(
                "\n".join(
                    [
                        "instance,time_seconds,work_units,status",
                        "afiro,0.24,8622,optimal",
                        "blend,0.00,0,error",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            stats = gate.write_lane_metric_csv(
                in_csv, out_csv, solver=None, metric="work_units"
            )
            self.assertEqual(stats.total_solver_rows, 2)
            self.assertEqual(stats.kept_rows, 1)
            self.assertEqual(stats.dropped_non_optimal, 1)
            self.assertEqual(
                out_csv.read_text(encoding="utf-8").strip(),
                "instance,work_units\nafiro,8622",
            )

    def test_write_lane_metric_csv_filters_solver_when_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            in_csv = Path(tmp) / "compare.csv"
            out_csv = Path(tmp) / "metric.csv"
            in_csv.write_text(
                "\n".join(
                    [
                        "instance,solver,time_seconds,work_units,status",
                        "afiro,mipx_barrier_cpu,0.01,100,error",
                        "afiro,mipx_barrier_gpu,0.24,8622,optimal",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            stats = gate.write_lane_metric_csv(
                in_csv, out_csv, solver="mipx_barrier_gpu", metric="work_units"
            )
            self.assertEqual(stats.total_solver_rows, 1)
            self.assertEqual(stats.kept_rows, 1)
            self.assertEqual(stats.dropped_non_optimal, 0)
            self.assertEqual(
                out_csv.read_text(encoding="utf-8").strip(),
                "instance,work_units\nafiro,8622",
            )


if __name__ == "__main__":
    unittest.main()
