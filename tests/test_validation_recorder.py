"""Recorder script sanity: node syntax check + package/docs presence.

No npm install, no network: `node --check` only parses the file.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
VALIDATION_DIR = REPO / "tools" / "validation"
RECORDER = VALIDATION_DIR / "recorder" / "record_duel.js"
PACKAGE = VALIDATION_DIR / "recorder" / "package.json"

NODE = shutil.which("node")


@pytest.mark.skipif(NODE is None, reason="node not installed")
def test_record_duel_js_passes_node_check():
    proc = subprocess.run(
        [NODE, "--check", str(RECORDER)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr.decode()


def test_package_json_declares_mineflayer():
    pkg = json.loads(PACKAGE.read_text())
    assert "mineflayer" in pkg.get("dependencies", {})
    assert pkg.get("private") is True


def test_recorder_reads_same_schedule_fields_harness_validates():
    # the JS consumes segments[].{bot,tick_start,tick_end,inputs} and
    # arena.spawn[bot].{pos,yaw_mc_deg}; make sure those exact key names
    # appear in the script so a schedule-schema rename cannot silently
    # desync the two replayers
    src = RECORDER.read_text()
    for key in ("tick_start", "tick_end", "yaw_mc_deg", "segments", "spawn", "bots"):
        assert key in src, "record_duel.js no longer references %r" % key


def test_docs_present():
    assert (VALIDATION_DIR / "RUNBOOK.md").is_file()
    readme = (VALIDATION_DIR / "schedules" / "README.md").read_text()
    # the convention note is the load-bearing part of the docs
    assert "mc_yaw + 90" in readme


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
