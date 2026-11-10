"""Regression tests: first-batch Dice loss and failed-run recovery guard."""
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))

from experiment_planner import archive_failed_preflight, save_decision


class TrainLossAndRecoveryTests(unittest.TestCase):
    def test_masked_dice_loss_without_main_scope(self):
        import torch
        spec = importlib.util.spec_from_file_location("deeplab_v5_train_regression", ROOT / "training" / "train.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        outputs = torch.randn(2, 6, 8, 8, requires_grad=True)
        targets = torch.randint(0, 6, (2, 8, 8))
        targets[0, 0, 0] = 255
        loss = module.masked_dice_loss(outputs, targets)
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        self.assertTrue(torch.isfinite(outputs.grad).all().item())

    def test_recover_failed_pre_epoch_archives_without_deleting(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in ("control_for_shrub_weight_down_pilot", "shrub_weight_down_pilot"):
                folder = root / name
                folder.mkdir()
                (folder / "class_weights.png").write_bytes(b"keep this file")
            save_decision(root / "monitor" / "experiment_decision.json", {
                "status": "failed", "experiment_id": "shrub_weight_down",
                "control_output": str(root / "control_for_shrub_weight_down_pilot"),
                "output": str(root / "shrub_weight_down_pilot"),
            })
            archive_failed_preflight(root)
            archives = list((root / "failed_runs").iterdir())
            self.assertEqual(len(archives), 2)
            for archive in archives:
                self.assertEqual((archive / "class_weights.png").read_bytes(), b"keep this file")

    def test_recovery_refuses_existing_model_artifacts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first = root / "control_for_shrub_weight_down_pilot"
            first.mkdir()
            (first / "latest_state.pth").write_bytes(b"must preserve")
            save_decision(root / "monitor" / "experiment_decision.json", {
                "status": "failed", "experiment_id": "shrub_weight_down",
                "control_output": str(first),
                "output": str(root / "shrub_weight_down_pilot"),
            })
            with self.assertRaises(RuntimeError):
                archive_failed_preflight(root)
            self.assertTrue((first / "latest_state.pth").exists())


if __name__ == "__main__":
    unittest.main()
