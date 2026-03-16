import argparse
import importlib.util
from pathlib import Path
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


if __name__ == "__main__":
    unittest.main()
