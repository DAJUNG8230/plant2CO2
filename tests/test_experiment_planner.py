"""Offline tests for isolated Codex pilot experiment planner."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
from experiment_templates import get_experiment, BASE_CLASS_WEIGHTS, EXPERIMENTS
from experiment_planner import choose_candidates, plan, comparison, reusable_empty_dir


class ExperimentsTests(unittest.TestCase):
    def evidence(self, with_matrix=True):
        matrix = None
        if with_matrix:
            matrix = {"epoch": 42, "classes": [
                {"class": "Shrub", "iou": 0.29, "gt_pixels": 100, "predicted_pixels": 150,
                 "top_misclassifications": [{"predicted_as": "Trees", "pixels": 20}]},
                {"class": "Cultivated", "iou": 0.23, "gt_pixels": 200, "predicted_pixels": 120,
                 "top_misclassifications": [{"predicted_as": "Shrub", "pixels": 48}]},
            ]}
        return {"history": {"best_epoch": 12, "best_vegetation_miou": .6326},
                "best_confusion": matrix, "latest_confusion": matrix,
                "fixed_images": {"aggregates": {}}, "limitations": []}

    def test_each_template_changes_single_class_weight(self):
        for name, template in EXPERIMENTS.items():
            diffs = [i for i, (x, y) in enumerate(zip(BASE_CLASS_WEIGHTS,
                                                      template["class_weights"])) if x != y]
            self.assertEqual(len(diffs), 0 if name == "baseline_control" else 1, name)
            self.assertEqual(get_experiment(name), template)
        with self.assertRaises(ValueError):
            get_experiment("change_LUT")

    def test_no_confusion_means_no_candidates(self):
        self.assertEqual(choose_candidates(self.evidence(False)), [])

    def test_weak_class_candidates_generated(self):
        chosen = choose_candidates(self.evidence())
        self.assertEqual({item["id"] for item in chosen},
                         {"shrub_weight_down", "cultivated_weight"})

    def test_directional_candidate_threshold(self):
        e = self.evidence()
        e["best_confusion"]["classes"][0]["predicted_pixels"] = 105
        ids = {v["id"] for v in choose_candidates(e)}
        self.assertNotIn("shrub_weight", ids)
        self.assertNotIn("shrub_weight_down", ids)

    def test_underpredict_shrub_enables_increase(self):
        e = self.evidence()
        e["best_confusion"]["classes"][0]["predicted_pixels"] = 70
        ids = {v["id"] for v in choose_candidates(e)}
        self.assertIn("shrub_weight", ids)
        self.assertNotIn("shrub_weight_down", ids)

    def test_overpredicted_cultivated_decreases_weight(self):
        e = self.evidence()
        e["best_confusion"]["classes"][1]["gt_pixels"] = 100
        e["best_confusion"]["classes"][1]["predicted_pixels"] = 140
        ids = {x["id"] for x in choose_candidates(e)}
        self.assertIn("cultivated_weight_down", ids)
        self.assertNotIn("cultivated_weight", ids)

    def test_only_empty_leftover_directories_are_reusable(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "control_for_shrub_weight_down_pilot"
            self.assertTrue(reusable_empty_dir(path))
            path.mkdir()
            self.assertTrue(reusable_empty_dir(path))
            (path / "history.csv").write_text("epoch,loss\\n", encoding="utf-8")
            self.assertFalse(reusable_empty_dir(path))

    def test_all_registry_ids_are_accepted_by_train_argparse(self):
        source = (Path(__file__).resolve().parents[1] / "training" / "train.py").read_text(encoding="utf-8")
        self.assertIn('choices=tuple(EXPERIMENTS)', source)
        for name in ("baseline_control", "shrub_weight_down", "cultivated_weight_down"):
            self.assertIn(name, EXPERIMENTS)

    def test_skip_does_not_spawn_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch("experiment_planner.check_baseline", return_value=[{"epoch": 42}]), \
                 patch("experiment_planner.collect_evidence", return_value=self.evidence()), \
                 patch("experiment_planner.choose", return_value={
                     "decision": "skip", "experiment_id": None,
                     "reason": "not enough evidence", "risk": "no training",
                 }), patch("experiment_planner.subprocess.run") as spawn:
                decision = plan(root, root / "outputs_training_v5_experiments",
                                dry_run=False, assume_stopped=True, epochs=12, timeout=30)
            self.assertEqual(decision["status"], "skipped")
            spawn.assert_not_called()

    def test_preview_does_not_spawn_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch("experiment_planner.check_baseline", return_value=[{"epoch": 42}]), \
                 patch("experiment_planner.collect_evidence", return_value=self.evidence()), \
                 patch("experiment_planner.choose", return_value={
                     "decision": "execute", "experiment_id": "shrub_weight_down",
                     "reason": "test only", "risk": "degraded results",
                 }), patch("experiment_planner.subprocess.run") as spawn:
                decision = plan(root, root / "outputs_training_v5_experiments",
                                dry_run=True, assume_stopped=True, epochs=12, timeout=30)
            self.assertEqual(decision["status"], "selected")
            spawn.assert_not_called()

    def test_comparison_marks_coverage_mae_unavailable(self):
        import csv
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            fields = ["epoch", "train_loss", "val_loss", "miou_all",
                      "miou_vegetation", "binary_vegetation_iou",
                      "learning_rate", "shrub_iou", "cultivated_iou"]
            pilot = folder / "history.csv"
            with pilot.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerow({
                    "epoch": 2, "train_loss": .3, "val_loss": .5,
                    "miou_all": .64, "miou_vegetation": .65,
                    "binary_vegetation_iou": .85, "learning_rate": 1e-5,
                    "shrub_iou": .42, "cultivated_iou": .3,
                })
            baseline = [{"epoch": 12, "val_loss": .5,
                         "miou_vegetation": .6326, "shrub_iou": .3}]
            result = comparison(baseline, folder, "shrub_weight")
            self.assertTrue(result["available"])
            self.assertAlmostEqual(result["vegetation_miou_change_pp"], 1.74)
            self.assertIsNone(result["coverage_mae"])

    def test_unsupported_epoch_count_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                plan(Path(tmp), Path(tmp), True, True, 51, 30)


if __name__ == "__main__":
    unittest.main()
