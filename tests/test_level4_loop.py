"""Offline Level4 autonomous search tests without Codex CLI or GPU."""
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
import level4_loop as l4


def baseline_confusion():
    return {"classes": [
        {"class": "Shrub", "iou": .30, "gt_pixels": 1000, "predicted_pixels": 1250},
        {"class": "Cultivated", "iou": .30, "gt_pixels": 1000, "predicted_pixels": 650},
        {"class": "Trees", "iou": .72, "gt_pixels": 1000, "predicted_pixels": 950},
        {"class": "Herbaceous", "iou": .62, "gt_pixels": 1000, "predicted_pixels": 1005},
    ]}


class Level4Tests(unittest.TestCase):
    def test_directionally_generated_options_and_lr_always_available(self):
        items = l4.candidates_for(baseline_confusion(), set(), [])
        ids = {x["id"] for x in items}
        self.assertIn("shrub_weight_down_soft", ids)
        self.assertIn("shrub_weight_down_more", ids)
        self.assertIn("cultivated_weight_up_more", ids)
        self.assertIn("low_lr", ids)
        self.assertNotIn("shrub_weight", ids)
        self.assertNotIn("trees_weight_up", ids)

    def test_attempted_not_repeated(self):
        items = l4.candidates_for(baseline_confusion(), {"shrub_weight_down", "low_lr"}, [])
        self.assertNotIn("shrub_weight_down", {x["id"] for x in items})
        self.assertNotIn("low_lr", {x["id"] for x in items})

    def test_no_class_direction_can_still_test_lr(self):
        items = l4.candidates_for({"classes": []}, set(), [])
        self.assertEqual({x["id"] for x in items}, {"low_lr", "very_low_lr"})

    def test_training_refuses_existing_destination(self):
        with tempfile.TemporaryDirectory() as temp:
            dest = Path(temp) / "l4_trial"
            dest.mkdir()
            with self.assertRaises(FileExistsError):
                l4.run_training(dest, "shrub_weight_down", Path(temp) / "best_model.pth", 8, 42)

    def test_disk_budget_and_overheated_gpu_stop(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            args = SimpleNamespace(max_rounds=12, max_stall=5,
                                   target_miou=.80, min_free_gb=5)
            state = {"rounds": [], "stall": 0, "champion_miou": .63}
            with patch.object(l4, "OUTPUT", folder), patch.object(l4, "gpu_snapshot", return_value={
                "devices": [{"temperature_c": 82}]
            }):
                self.assertIn("GPU", l4.ensure_budget(args, state, float("inf")))
            state["stall"] = 5
            with patch.object(l4, "OUTPUT", folder):
                self.assertIn("未改善", l4.ensure_budget(args, state, float("inf")))

    def test_resume_preview_and_reject_incomplete_without_optin(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            marker = folder / "level4_state.json"
            champion = folder / "best.pth"
            champion.write_bytes(b"placeholder")
            args = SimpleNamespace(resume_search=False, skip_incomplete=False,
                                   baseline=folder, assume_baseline_stopped=True)
            original = l4.STATE_PATH
            try:
                l4.STATE_PATH = marker
                l4.save_decision(marker, {
                    "status": "preview", "rounds": [], "attempted": [],
                    "champion_checkpoint": str(champion)
                })
                result = l4.load_or_initialize(args)
                self.assertEqual(result["status"], "preview")
                data = dict(result, status="interrupted")
                l4.save_decision(marker, data)
                args.resume_search = True
                with self.assertRaises(RuntimeError):
                    l4.load_or_initialize(args)
            finally:
                l4.STATE_PATH = original


if __name__ == "__main__":
    unittest.main()
