"""Dashboard tests with fake rollout snapshots; no network or GPU required."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
from level2_dashboard import codex_usage, proposals_in


class DashboardTests(unittest.TestCase):
    def test_missing_sessions_not_fabricated(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = codex_usage(tmp)
            self.assertNotIn("short", result)
            self.assertIn("message", result)

    def test_tokens_and_rate_limit_windows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sessions" / "2026" / "11" / "08" / "rollout-test.jsonl"
            path.parent.mkdir(parents=True)
            event = {
                "timestamp": "2026-11-08T12:00:00Z",
                "type": "event_msg",
                "payload": {
                    "type": "token_count",
                    "info": {
                        "last_token_usage": {
                            "total_tokens": 120, "input_tokens": 90,
                            "output_tokens": 30,
                        },
                        "total_token_usage": {"total_tokens": 380},
                    },
                    "rate_limits": {
                        "primary": {"used_percent": 35.5, "window_minutes": 300},
                        "secondary": {"used_percent": 66, "window_minutes": 10080},
                    },
                },
            }
            path.write_text(json.dumps(event) + "\n", encoding="utf-8")
            result = codex_usage(tmp)
            self.assertEqual(result["tokens"]["last_total_tokens"], 120)
            self.assertEqual(result["tokens"]["session_total_tokens"], 380)
            self.assertEqual(result["short"]["used_percent"], 35.5)
            self.assertEqual(result["long"]["used_percent"], 66)

    def test_missing_percent_not_estimated(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sessions" / "2026" / "11" / "08" / "rollout-solo.jsonl"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({
                "timestamp": "2026-11-08T12:00:00Z", "type": "event_msg",
                "payload": {"type": "token_count", "info": {
                    "total_token_usage": {"total_tokens": 100},
                    "last_token_usage": {"total_tokens": 10},
                }},
            }) + "\n", encoding="utf-8")
            result = codex_usage(tmp)
            self.assertIsNone(result["short"])
            self.assertIsNone(result["long"])

    def test_pending_proposals_hide_approved(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            identity = "0123456789abcdef"
            (folder / ("proposal_" + identity + ".json")).write_text(
                json.dumps({"id": identity, "status": "pending",
                            "changes": {"learning_rate": 1e-5}}),
                encoding="utf-8")
            self.assertEqual(len(proposals_in(folder)), 1)
            (folder / ("approved_" + identity + ".json")).write_text("{}", encoding="utf-8")
            self.assertEqual(len(proposals_in(folder)), 0)


if __name__ == "__main__":
    unittest.main()
