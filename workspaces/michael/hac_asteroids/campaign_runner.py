"""Detached supervision for a frozen representation campaign.

python -m hac_asteroids.campaign_runner --output-dir artifacts/<campaign>
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def read_status(root):
    path = root / "status.json"
    return json.loads(path.read_text()) if path.exists() else {}


def process_memory(pid):
    try:
        fields = Path(f"/proc/{pid}/status").read_text().splitlines()
        return {line.split(":")[0]: int(line.split()[1])
                for line in fields if line.startswith(("VmRSS:", "VmHWM:"))}
    except FileNotFoundError:
        return {}


def heartbeat(root, child, started):
    result = child.poll()
    campaign = read_status(root)
    state = "running" if result is None else (
        "complete" if result == 0 and campaign.get("state") == "complete" else "failed")
    value = dict(state=state, heartbeat_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                 supervisor_pid=os.getpid(), campaign_pid=child.pid, exit_code=result,
                 elapsed_seconds=time.time()-started, campaign=campaign,
                 process_memory_kib=process_memory(child.pid),
                 completed_fits=len(list(root.glob("asteroid*/*/run.json"))))
    log = root / "campaign.log"
    value["last_log_update_unix"] = log.stat().st_mtime if log.exists() else None
    write_json(root / "runner_status.json", value)
    return state


def campaign_command(root):
    description = json.loads((root / "study.json").read_text())
    if description.get("profile") == "l4_survey":
        return [sys.executable, "-u", "-m", "hac_asteroids.l4_survey", "run",
                "--output-dir", str(root)]
    return [sys.executable, "-u", "-m", "hac_asteroids.representation_study", "run",
            "--profile", description["profile"], "--output-dir", str(root)]


def supervise(root):
    # The lifetime lock is also inherited by the fitter, preventing a second
    # supervisor from launching another GPU process if its parent disappears.
    with (root / ".supervisor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (root / ".campaign.lock").open("a") as campaign_lock:
            fcntl.flock(campaign_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if read_status(root).get("state") == "complete":
            return 0
        command = campaign_command(root)
        started = time.time()
        with (root / "campaign.log").open("a", buffering=1) as log:
            child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT, pass_fds=(lock.fileno(),))

            def stop(_signal, _frame):
                if child.poll() is None:
                    child.send_signal(signal.SIGTERM)

            signal.signal(signal.SIGTERM, stop)
            signal.signal(signal.SIGINT, stop)
            while True:
                heartbeat(root, child, started)
                try:
                    code = child.wait(timeout=30)
                    heartbeat(root, child, started)
                    return code
                except subprocess.TimeoutExpired:
                    continue


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    root = args.output_dir.resolve()
    if not (root / "frozen_configuration.json").exists():
        parser.error("expected an existing frozen campaign directory")
    if args.worker:
        return supervise(root)
    if read_status(root).get("state") == "complete":
        print("Campaign already complete; no fit launched.")
        return 0
    # Refuse a duplicate before detaching; the worker repeats these checks.
    for name in (".supervisor.lock", ".campaign.lock"):
        with (root / name).open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    with (root / "runner.log").open("a") as log:
        child = subprocess.Popen([sys.executable, "-u", "-m", "hac_asteroids.campaign_runner",
                                  "--worker", "--output-dir", str(root)], cwd=ROOT,
                                 stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True, close_fds=True)
    print(json.dumps(dict(supervisor_pid=child.pid, status=str(root / "runner_status.json"))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
