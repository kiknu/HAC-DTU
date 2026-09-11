"""Detached launch and truthful campaign liveness reporting."""
import fcntl
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

from hac_asteroids.campaign_runner import campaign_command, heartbeat, main, write_json


class CampaignRunnerTests(unittest.TestCase):
    def test_survey_dispatch_preserves_legacy_command(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for profile in ("octantoid", "quick", "full", "l4_survey"):
                write_json(root / "study.json", {"profile": profile})
                command = campaign_command(root)
                self.assertEqual(command[-2:], ["--output-dir", str(root)])
                if profile == "l4_survey":
                    self.assertIn("hac_asteroids.l4_survey", command)
                    self.assertNotIn("--profile", command)
                else:
                    self.assertIn("hac_asteroids.representation_study", command)
                    self.assertEqual(command[command.index("--profile") + 1], profile)

    def test_launch_detaches_and_redirects_all_standard_streams(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_json(root / "frozen_configuration.json", {})
            with patch("hac_asteroids.campaign_runner.subprocess.Popen", return_value=Mock(pid=123)) as launch:
                self.assertEqual(main(["--output-dir", str(root)]), 0)
            self.assertTrue(launch.call_args.kwargs["start_new_session"])
            self.assertTrue(launch.call_args.kwargs["close_fds"])
            self.assertEqual(launch.call_args.kwargs["stdin"], -3)
            self.assertIn("--worker", launch.call_args.args[0])

    def test_live_campaign_lock_prevents_duplicate_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_json(root / "frozen_configuration.json", {})
            with (root / ".campaign.lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with patch("hac_asteroids.campaign_runner.subprocess.Popen") as launch:
                    with self.assertRaises(BlockingIOError):
                        main(["--output-dir", str(root)])
                    launch.assert_not_called()

    def test_completed_campaign_is_not_restarted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_json(root / "frozen_configuration.json", {})
            write_json(root / "status.json", {"state": "complete"})
            with patch("hac_asteroids.campaign_runner.subprocess.Popen") as launch:
                self.assertEqual(main(["--output-dir", str(root)]), 0)
                launch.assert_not_called()

    def test_heartbeat_reports_exit_and_requires_completed_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            child = Mock(pid=999999999)
            write_json(root / "status.json", {"state": "fitting"})
            for code, expected in ((None, "running"), (-9, "failed"), (0, "failed")):
                child.poll.return_value = code
                self.assertEqual(heartbeat(root, child, time.time()), expected)
                saved = json.loads((root / "runner_status.json").read_text())
                self.assertEqual(saved["exit_code"], code)
                self.assertIn("heartbeat_utc", saved)
            write_json(root / "status.json", {"state": "complete"})
            self.assertEqual(heartbeat(root, child, time.time()), "complete")


if __name__ == "__main__":
    unittest.main()
