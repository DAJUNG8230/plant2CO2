"""Offline tests: python -m unittest discover -s tests -p test_monitor_v5.py"""
import csv
import tempfile
import unittest
from pathlib import Path
from sys import path as sys_path

sys_path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
import monitor_v5 as m


def row(epoch, train, val, miou, shrub=None):
    r = {"epoch": epoch, "train_loss": train, "val_loss": val,
         "miou_all": miou, "miou_vegetation": miou,
         "binary_vegetation_iou": miou, "learning_rate": 3e-5}
    r.update({c + "_iou": 0.65 for c in m.CLASSES})
    r["shrub_iou"] = shrub
    return r


class MonitorTests(unittest.TestCase):
    def test_nan_validation_skipped_and_missing_shrub(self):
        rows = [row(1, .7, .6, .45, None),
                row(2, .6, None, None, None),
                row(3, .5, .55, .50, None)]
        result = m.analyze(rows, {"available": False}, [])
        self.assertEqual(result["latest_validation"]["epoch"], 3)
        self.assertTrue(any(a["code"] == "missing_class_iou"
                            for a in result["alerts"]))

    def test_overfitting(self):
        rows = [row(i, .5 - .05 * i, .3 + .05 * i, .6, .5)
                for i in range(1, 5)]
        result = m.analyze(rows, {"available": True, "devices": []}, [])
        self.assertIn("possible_overfitting", [a["code"] for a in result["alerts"]])

    def test_gpu_temperature(self):
        result = m.analyze([], {"available": True, "devices": [
            {"index": "0", "name": "GPU", "temperature_c": 83,
             "utilization_pct": 90, "memory_used_mb": 2000,
             "memory_total_mb": 8000, "power_w": 100}]}, [])
        self.assertIn("gpu_temperature", [a["code"] for a in result["alerts"]])

    def test_partial_csv_and_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            history = Path(tmp) / "history.csv"
            fields = list(m.EXPECTED) + [c + "_iou" for c in m.CLASSES]
            with history.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                for ep in (1, 2, 1):
                    d = row(ep, .4, .4, .5, .5)
                    writer.writerow(d)
                f.write("2,0.5\n")
            rows, errors = m.read_history(history)
            self.assertFalse(errors)
            self.assertEqual([r["epoch"] for r in rows], [1])


if __name__ == "__main__":
    unittest.main()
