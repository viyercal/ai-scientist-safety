from __future__ import annotations
import json
from pathlib import Path
from .config import SafetyConfig, SafetyReport, SafetyIssue, load_safety_config
from .static_analyzer import analyze_file
from .sandbox_runner import run_in_sandbox


def run_with_guardian(script_path: str | Path, cfg: SafetyConfig) -> SafetyReport:
    script_path = Path(script_path)
    static_report = analyze_file(script_path, cfg)

    if not static_report.passed:
        # Just return static report; do not execute
        return static_report

    runtime_report = run_in_sandbox(script_path, cfg)

    # Merge reports
    issues = list(static_report.issues) + list(runtime_report.issues)
    risk_score = max(static_report.risk_score, runtime_report.risk_score)
    passed = static_report.passed and runtime_report.passed
    return SafetyReport(passed=passed, issues=issues, risk_score=risk_score)


def report_to_dict(report: SafetyReport) -> dict:
    return {
        "passed": report.passed,
        "risk_score": report.risk_score,
        "issues": [
            {
                "severity": i.severity,
                "code": i.code,
                "detail": i.detail,
                "location": i.location,
            }
            for i in report.issues
        ],
    }


def main_cli(script_path: str, cfg_path: str = "ai_scientist/safety/safety_config.yaml"):
    cfg = load_safety_config(cfg_path)
    report = run_with_guardian(script_path, cfg)
    print(json.dumps(report_to_dict(report), indent=2))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--script", required=True, help="Path to experiment script")
    parser.add_argument(
        "--config",
        default="ai_scientist/safety/safety_config.yaml",
        help="Path to safety config YAML",
    )
    args = parser.parse_args()
    main_cli(args.script, args.config)
