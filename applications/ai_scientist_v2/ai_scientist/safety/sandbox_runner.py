import json
import subprocess
import time
from pathlib import Path
from typing import Optional
from .config import SafetyConfig, SafetyReport, SafetyIssue


def run_in_sandbox(script_path: str | Path, cfg: SafetyConfig) -> SafetyReport:
    script_path = Path(script_path).resolve()

    # Sandbox dir: sibling "sandbox" under the experiment dir
    sandbox_dir = script_path.parent / "sandbox"
    sandbox_dir.mkdir(exist_ok=True)

    start = time.time()
    try:
        # IMPORTANT CHANGE:
        # Use the absolute script_path, not just script_path.name,
        # while cwd is sandbox_dir.
        proc = subprocess.run(
            ["python", str(script_path)],
            cwd=str(sandbox_dir),
            capture_output=True,
            text=True,
            timeout=cfg.max_runtime_sec,
        )
        runtime = time.time() - start
    except subprocess.TimeoutExpired:
        issue = SafetyIssue(
            severity="error",
            code="RUNTIME_TIMEOUT",
            detail=f"Script exceeded max_runtime_sec={cfg.max_runtime_sec}",
            location=None,
        )
        return SafetyReport(passed=False, issues=[issue], risk_score=1.0)

    issues = []
    if proc.returncode != 0:
        issues.append(
            SafetyIssue(
                severity="warning",
                code="NONZERO_EXIT",
                detail=f"Script exited with code {proc.returncode}. stderr: {proc.stderr[:500]}",
                location=None,
            )
        )

    # Log outputs inside sandbox
    (sandbox_dir / "stdout.txt").write_text(proc.stdout or "")
    (sandbox_dir / "stderr.txt").write_text(proc.stderr or "")

    risk_score = 0.0 if not issues else 0.3
    return SafetyReport(passed=True, issues=issues, risk_score=risk_score)
