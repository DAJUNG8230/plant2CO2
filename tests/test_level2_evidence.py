"""Evidence and review approval integration tests.

Run python -m unittest discover -s tests -p test_level2_evidence.py
"""
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
from level2_evidence import (CLASSES, checkpoint_info, collect_evidence,
                             evaluate_fixed_pairs, read_confusions, review_tasks)
from level2_review import approve, evidence_review_proposal, write_json
from level2_checkpoint import read_approved


class EvidenceTests(unittest.TestCase):
    def make_confusion(self, folder, epoch, shrub_to_trees=20):
        p = folder / ("confusion_matrix_epoch_%03d.csv" % epoch)
        matrix = [[0] * 6 for _ in range(6)]
        for i in range(6):
            matrix[i][i] = 100
        matrix[3][3] = 80
        matrix[3][4] = shrub_to_trees
        with p.open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.writer(stream)
            writer.writerow(["GT / Pred", *CLASSES])
            for i in range(6):
                writer.writerow([CLASSES[i], *matrix[i]])
        return p

    def test_confusion_direction(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            self.make_confusion(folder, 38, 20)
            evidence = read_confusions(folder)
            self.assertTrue(evidence["available"])
            self.assertEqual(evidence["epochs"][0]["classes"][3]
                             ["top_misclassifications"][0]["predicted_as"], "Trees")
            self.assertEqual(evidence["epochs"][0]["classes"][3]["gt_pixels"], 100)

    def test_bad_confusion_is_reported_not_invented(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / "confusion_matrix_epoch_002.csv"
            p.write_text("bad,data\n", encoding="utf-8")
            result = read_confusions(Path(temp))
            self.assertFalse(result["available"])
            self.assertTrue(result["errors"])

    def test_fixed_mask_coverage_mae_and_shapes(self):
        import numpy as np
        from PIL import Image
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            gt = np.array([[2, 0], [5, 255]], dtype=np.uint8)
            best = np.array([[2, 0], [5, 0]], dtype=np.uint8)
            latest = np.array([[0, 0], [5, 0]], dtype=np.uint8)
            for name, data in (("gt.png", gt), ("best.png", best), ("last.png", latest)):
                Image.fromarray(data, mode="L").save(folder / name)
            manifest = folder / "fixed_predictions.csv"
            with manifest.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(["image_id", "gt_mask", "pred_best_mask", "pred_latest_mask"])
                writer.writerow(["sample", "gt.png", "best.png", "last.png"])
            result = evaluate_fixed_pairs(manifest)
            self.assertTrue(result["available"])
            self.assertEqual(result["aggregates"]["best"]["coverage_mae_pp"], 0.0)
            self.assertAlmostEqual(result["aggregates"]["latest"]["coverage_mae_pp"], 33.3333)
            self.assertAlmostEqual(result["aggregates"]["latest"]["coverage_signed_bias_pp"], -33.3333)

    def test_full_evidence_and_readonly_task_approval(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            self.make_confusion(folder, 10)
            self.make_confusion(folder, 38)
            rows = [
                {"epoch": 10, "miou_vegetation": .63, "val_loss": .5},
                {"epoch": 38, "miou_vegetation": .42, "val_loss": .6},
            ]
            evidence = collect_evidence(rows, folder)
            self.assertEqual(evidence["history"]["best_epoch"], 10)
            self.assertEqual(evidence["latest_confusion"]["epoch"], 38)
            self.assertFalse(evidence["fixed_images"]["available"])
            tasks = review_tasks(evidence)
            self.assertTrue(any(x["task_code"] == "collect_fixed_prediction_pairs" for x in tasks))
            task = next(x for x in tasks if x["task_code"] == "collect_fixed_prediction_pairs")
            proposal = evidence_review_proposal(task, {"latest_epoch": 38})
            approvals = folder / "approvals"
            write_json(approvals / ("proposal_" + proposal["id"] + ".json"), proposal)
            approve(proposal["id"], "APPROVE", approvals)
            approved = approvals / ("approved_" + proposal["id"] + ".json")
            self.assertEqual(json.loads(approved.read_text(encoding="utf-8"))["kind"], "review_task")
            with self.assertRaises(ValueError):
                read_approved(approved)

    def test_checkpoint_metadata_missing_is_explicit(self):
        with tempfile.TemporaryDirectory() as temp:
            result = checkpoint_info(Path(temp) / "best_model.pth")
            self.assertFalse(result["available"])


if __name__ == "__main__":
    unittest.main()
