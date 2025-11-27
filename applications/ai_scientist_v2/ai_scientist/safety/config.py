from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Literal
import yaml
from pathlib import Path


@dataclass
class SafetyIssue:
    severity: Literal["info", "warning", "error"]
    code: str
    detail: str
    location: str | None = None


@dataclass
class SafetyReport:
    passed: bool
    issues: List[SafetyIssue] = field(default_factory=list)
    risk_score: float = 0.0  # 0–1, higher = riskier


@dataclass
class SafetyConfig:
    # Optional allowlist, in case you want to override blocked_modules later
    allowed_modules: List[str] = field(default_factory=list)

    # NOTE: os is intentionally NOT blocked
    blocked_modules: List[str] = field(
        default_factory=lambda: [
            #"os",   # allowed
            "subprocess",
            "socket",
            "shutil",
            "sys",
            "requests",
            "httpx",
            "urllib",
            "urllib3",
            "aiohttp",
            "websocket",
            "paramiko",
        ]
    )

    blocked_functions: List[str] = field(
        default_factory=lambda: [
            "exec",
            "compile",
            "open",
        ]
    )

    allow_network: bool = False
    max_runtime_sec: int = 600
    max_ram_mb: int = 4096
    allow_fs_outside_sandbox: bool = False

    # Interactive confirmation gate
    require_user_confirm: bool = True
    confirm_timeout_sec: int = 15


def load_safety_config(path: str | Path) -> SafetyConfig:
    path = Path(path)
    if not path.exists():
        return SafetyConfig()

    cfg_dict = yaml.safe_load(path.read_text()) or {}
    return SafetyConfig(**cfg_dict)
