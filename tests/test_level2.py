"""Tests: python -m unittest discover -s tests -p test_level2.py"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
from level2_checkpoint import (
    apply_approved, read_approved, restore_full, save_full,
    save_best_protected, validate_changes,
)
from level2_review import approve, propose_from_status, propose


class ApprovalTests(unittest.TestCase):
    def test_unapproved_is_rejected_and_bounds_enforced(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "approved.json"
            p.write_text(json.dumps({"status": "pending", "changes": {"learning_rate": 1e-5}}),
                         encoding="utf-8")
            with self.assertRaises(ValueError):
                read_approved(p)
        with self.assertRaises(ValueError):
            validate_changes({"dataset_mapping": 0})
        with self.assertRaises(ValueError):
            validate_changes({"learning_rate": 100})
        with self.assertRaises(ValueError):
            validate_changes({"scheduler_patience": 2.5})

    def test_propose_approve_then_apply(self):
        status = {"alerts": [{"code": "possible_overfitting"}],
                  "latest_validation": {"epoch": 8, "learning_rate": 3e-5}}
        p = propose_from_status(status)
        self.assertEqual(p["changes"]["learning_rate"], 1.5e-5)
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            data = base / "status.json"
            data.write_text(json.dumps(status), encoding="utf-8")
            propose(data, base)
            with self.assertRaises(ValueError):
                approve(p["id"], "NO", base)
            approve(p["id"], "APPROVE", base)
            approved = read_approved(base / ("approved_" + p["id"] + ".json"))
            model = torch.nn.Linear(2, 1)
            opt = torch.optim.AdamW(model.parameters(), lr=3e-5)
            sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt)
            apply_approved(approved, opt, sch)
            self.assertAlmostEqual(opt.param_groups[0]["lr"], 1.5e-5)


class ResumeTests(unittest.TestCase):
    def test_full_state_roundtrip_and_mismatch(self):
        model = torch.nn.Linear(2, 1)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt)
        scaler = torch.amp.GradScaler("cpu", enabled=False)
        loss = model(torch.randn(2, 2)).sum()
        loss.backward()
        opt.step()
        opt.zero_grad()
        sch.step(0.5)
        expected = [x.detach().clone() for x in model.parameters()]
        meta = {"classes": ["soil", "vegetation"], "num_train": 20}

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "full.pth"
            save_full(path, model=model, optimizer=opt, scheduler=sch,
                      scaler=scaler, epoch=7, best_miou=0.71,
                      no_improve=2, metadata=meta)
            with torch.no_grad():
                for x in model.parameters():
                    x.zero_()
            epoch, best, no_improve = restore_full(
                path, model=model, optimizer=opt, scheduler=sch,
                scaler=scaler, expected_metadata=meta)
            self.assertEqual((epoch, no_improve), (7, 2))
            self.assertAlmostEqual(best, 0.71)
            for a, b in zip(model.parameters(), expected):
                self.assertTrue(torch.allclose(a, b))
            with self.assertRaises(ValueError):
                restore_full(path, model=model, optimizer=opt, scheduler=sch,
                             scaler=scaler, expected_metadata={"num_train": 99})

    def test_best_previous_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "best_model.pth"
            save_best_protected(path, {"step": 1})
            save_best_protected(path, {"step": 2})
            self.assertEqual(torch.load(path, weights_only=True)["step"], 2)
            self.assertEqual(torch.load(path.parent / "best_model_previous.pth",
                                        weights_only=True)["step"], 1)


if __name__ == "__main__":
    unittest.main()
