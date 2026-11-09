"""Offline tests for Codex autonomous control, no live Codex or GPU needed."""
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
from auto_controller import candidates_at, process, save_json
from level2_review import canonical
from level2_checkpoint import read_approved


class AutonomousControllerTests(unittest.TestCase):
    def add_proposal(self, folder, base_epoch=38, kind="training_parameters",
                     changes=None):
        if changes is None:
            changes = {"scheduler_patience": 4}
        raw = {
            "status": "pending", "kind": kind, "base_epoch": base_epoch,
            "changes": changes, "source": "rule_based_monitor",
            "evidence": "early_stop_patience_exceeded", "reason": "test candidate",
        }
        key = hashlib.sha256(canonical(raw).encode("utf-8")).hexdigest()[:16]
        raw["id"] = key
        save_json(folder / ("proposal_" + key + ".json"), raw)
        return key

    def test_only_valid_current_epoch_training_proposal(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            desired = self.add_proposal(folder)
            self.add_proposal(folder, base_epoch=36)
            self.add_proposal(folder, kind="review_task", changes={})
            self.add_proposal(folder, base_epoch=38, changes={"dataset_mapping": 42})
            found = candidates_at(folder, 38)
            self.assertEqual([p["id"] for p in found], [desired])

    def test_tampered_proposal_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            key = self.add_proposal(folder)
            path = folder / ("proposal_" + key + ".json")
            data = json.loads(path.read_text(encoding="utf-8"))
            data["changes"] = {"learning_rate": 1e-4}
            save_json(path, data)
            self.assertEqual(candidates_at(folder, 38), [])

    def test_automated_approval_requires_explicit_opt_in(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "approved.json"
            save_json(path, {
                "status": "approved", "kind": "training_parameters",
                "approval": "autonomous-codex-guarded",
                "proposal_id": "1234567890abcdef",
                "changes": {"scheduler_patience": 4},
            })
            with self.assertRaises(ValueError):
                read_approved(path)
            self.assertEqual(read_approved(path, allow_autonomous=True),
                             {"scheduler_patience": 4})

    def test_codex_skip_never_executes(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            approvals = folder / "approvals"
            approvals.mkdir()
            self.add_proposal(approvals)
            dummy = {
                "history": {"full": [], "best_epoch": 8},
                "best_confusion": None, "latest_confusion": None,
                "fixed_images": {"available": False},
                "limitations": [],
            }
            with patch("auto_controller.safe_state", return_value=([{"epoch": 38}], 38)), \
                 patch("auto_controller.collect_evidence", return_value=dummy), \
                 patch("auto_controller.choose_with_codex", return_value={
                     "decision": "skip", "proposal_id": None,
                     "reason": "lack of evidence", "risk": "none",
                 }), patch("auto_controller.execute") as launch:
                decided = process(folder, True, 30, False)
            self.assertTrue(decided)
            launch.assert_not_called()
            state = json.loads((folder / "monitor" / "auto_decision.json").read_text(encoding="utf-8"))
            self.assertEqual(state["status"], "skipped")


if __name__ == "__main__":
    unittest.main()
