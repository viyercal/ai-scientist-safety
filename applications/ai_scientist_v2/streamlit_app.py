import os
import sys
import json
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st
import importlib

# ---------------------------
# Page setup
# ---------------------------
st.set_page_config(page_title="AI Scientist v2 — Safety Demo", layout="wide")
st.markdown("# 🧪 AI Scientist v2 — Safety Layer Demo")



# ---------------------------
this_file = Path(__file__).resolve()
#   parents[0] = applications/ai_scientist_v2
#   parents[1] = applications
#   parents[2] = <repo_root>
repo_root = this_file.parents[2]

if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

user_root = st.sidebar.text_input("Override repo root (optional)", value=str(repo_root))
if user_root and Path(user_root).exists():
    if user_root not in sys.path:
        sys.path.insert(0, user_root)

Interpreter = None
SafetyConfig = None
build_default_safety = None

def try_import(module_path: str):
    try:
        return importlib.import_module(module_path)
    except Exception:
        return None

interp_mod = try_import("applications.ai_scientist_v2.ai_scientist.treesearch.interpreter") or \
             try_import("ai_scientist_v2.ai_scientist.treesearch.interpreter") or \
             try_import("ai_scientist.treesearch.interpreter")

safety_mod = try_import("applications.ai_scientist_v2.ai_scientist.safety.config") or \
             try_import("ai_scientist_v2.ai_scientist.safety.config") or \
             try_import("ai_scientist.safety.config")

if interp_mod is not None:
    Interpreter = getattr(interp_mod, "Interpreter", None) or getattr(interp_mod, "SafeInterpreter", None)
if safety_mod is not None:
    SafetyConfig = getattr(safety_mod, "SafetyConfig", None)
    build_default_safety = getattr(safety_mod, "build_default_safety", None)

has_repo = Interpreter is not None and (SafetyConfig is not None or build_default_safety is not None)

# Sidebar — Safety Controls
st.sidebar.header("⚙️ Safety Controls")

enforce_hard_gate = st.sidebar.checkbox("Enforce hard safety gate", value=True)
enable_static_checks = st.sidebar.checkbox("Enable static checks (AST / lints)", value=True)
max_exec_time = st.sidebar.number_input("Max exec time (s)", min_value=1, max_value=120, value=15)
allowed_imports = st.sidebar.text_area("Allowed imports (comma-separated)", value="math,random,json,time")
blocked_calls = st.sidebar.text_area("Blocked calls (comma-separated)", value="os.system,subprocess.Popen,subprocess.call,eval,exec")

st.sidebar.markdown("---")
st.sidebar.subheader("Quick Tests")
col_a, col_b = st.sidebar.columns(2)
run_safe = col_a.button("Run Safe Test")
run_unsafe = col_b.button("Run Unsafe Test")

# Summary of key changes
with st.expander("🔑 Key Changes to the Sakana AI Repository (summary)", expanded=True):
    st.markdown("""
**1) Safety Module** — `ai_scientist/safety/` provides configurations for runtime safety checks.

**2) Modified Interpreter** — `ai_scientist/treesearch/interpreter.py` integrates safety for model-generated code.

**3) Tests** — `test_safe.py` should pass; `test_unsafe.py` should trigger safety flags.

**4) Experiments** — under `experiments/` showcase researcher prompts and pipeline behavior.

**5) Colab** — ready notebook to run experiments and connect Drive.
""")

# Helpers
def compute_static_issues(code: str, blocked_calls_list: List[str]) -> List[str]:
    issues = []
    lowered = code.lower()
    for pattern in blocked_calls_list:
        p = pattern.strip()
        if not p:
            continue
        if p.lower() in lowered:
            issues.append(f"Pattern '{p}' detected")
    if "import os" in code or "import subprocess" in code:
        issues.append("Blocked import detected (os / subprocess)")
    return issues

def build_safety_payload() -> Dict[str, Any]:
    allow = [x.strip() for x in allowed_imports.split(",") if x.strip()]
    block = [x.strip() for x in blocked_calls.split(",") if x.strip()]
    return {
        "enforce_hard_gate": enforce_hard_gate,
        "enable_static_checks": enable_static_checks,
        "max_exec_time": int(max_exec_time),
        "allowed_imports": allow,
        "blocked_calls": block,
    }

def run_with_interpreter(code: str) -> Dict[str, Any]:
    safety_payload = build_safety_payload()
    result: Dict[str, Any] = {
        "ok": False,
        "term_out": "",
        "exception": None,
        "safety_report": {
            "passed": True,
            "issues": [],
            "config": safety_payload,
        },
        "raw_exc_info": None,
        "exec_time": None,
    }

    if safety_payload["enable_static_checks"]:
        static_issues = compute_static_issues(code, safety_payload["blocked_calls"])
        if static_issues:
            result["safety_report"]["passed"] = False
            result["safety_report"]["issues"].extend(static_issues)
            if safety_payload["enforce_hard_gate"]:
                result["exception"] = "Hard gate blocked execution due to static safety issues."
                return result

    if has_repo and Interpreter is not None:
        try:
            safety_cfg = None
            if SafetyConfig is not None:
                try:
                    safety_cfg = SafetyConfig(
                        enforce_hard_gate=safety_payload["enforce_hard_gate"],
                        enable_static_checks=safety_payload["enable_static_checks"],
                        max_exec_time=safety_payload["max_exec_time"],
                        allowed_imports=safety_payload["allowed_imports"],
                        blocked_calls=safety_payload["blocked_calls"],
                    )
                except Exception:
                    pass
            if safety_cfg is None and build_default_safety is not None:
                try:
                    safety_cfg = build_default_safety()
                except Exception:
                    safety_cfg = None

            interp = Interpreter(safety_config=safety_cfg) if safety_cfg is not None else Interpreter()
            t0 = time.time()
            r = interp.run(code, reset_session=True)
            dt = time.time() - t0

            term_out = ""
            if hasattr(r, "term_out"):
                term_out = "\n".join(getattr(r, "term_out", []) or [])
            elif hasattr(r, "output"):
                term_out = str(getattr(r, "output"))

            exc_info = getattr(r, "exc_info", None)
            passed = exc_info is None
            issues = []
            if exc_info:
                if isinstance(exc_info, dict):
                    if "Custom Safety Execution Info" in exc_info:
                        cse = exc_info["Custom Safety Execution Info"]
                        if isinstance(cse, dict) and "issues" in cse:
                            issues.extend([str(x) for x in cse["issues"]])
                else:
                    issues.append("Exception raised during execution.")

            result.update({
                "ok": passed,
                "term_out": term_out,
                "exec_time": dt,
                "raw_exc_info": exc_info,
            })
            if issues:
                result["safety_report"]["passed"] = False
                result["safety_report"]["issues"].extend(issues)
            if result["safety_report"]["issues"] and enforce_hard_gate:
                result["ok"] = False
            return result

        except Exception as e:
            result["ok"] = False
            result["exception"] = f"Interpreter error: {e}"
            result["raw_exc_info"] = traceback.format_exc()
            return result

    # Fallback demo mode
    local_globals = {"__builtins__": {"print": print, "len": len, "range": range}}
    t0 = time.time()
    try:
        exec(code, local_globals, local_globals)
        dt = time.time() - t0
        result["ok"] = True
        result["term_out"] = "(Executed in fallback demo mode — no repo interpreter found)"
        result["exec_time"] = dt
        return result
    except Exception as e:
        dt = time.time() - t0
        result["ok"] = False
        result["exception"] = str(e)
        result["exec_time"] = dt
        result["raw_exc_info"] = traceback.format_exc()
        return result

def list_experiments(base: Path) -> List[Path]:
    exps: List[Path] = []
    # Since we are INSIDE applications/ai_scientist_v2,
    # check local experiments/ first, but also allow repo-root search just in case.
    local = Path(__file__).resolve().parent / "experiments"
    if local.exists():
        for ext in ("*.txt", "*.md", "*.yaml", "*.yml", "*.json"):
            exps.extend(sorted(local.rglob(ext)))
    # Also check the canonical path from repo root
    canonical = repo_root / "applications" / "ai_scientist_v2" / "experiments"
    if canonical.exists():
        for ext in ("*.txt", "*.md", "*.yaml", "*.yml", "*.json"):
            exps.extend(sorted(canonical.rglob(ext)))
    return exps

# Layout
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🔒 Safety-on Interpreter Run")
    code_demo = st.text_area(
        "Paste code to execute via the safety-wrapped interpreter",
        height=220,
        value="x = 1 + 2\nprint('x is', x)\n",
    )
    run_code = st.button("▶️ Run Code")

with col2:
    st.subheader("🧪 Choose an Experiment (optional)")
    exp_paths = list_experiments(repo_root)
    if exp_paths:
        choices = {str(p): p.name for p in exp_paths}
        selected = st.selectbox("Experiment file", options=list(choices.keys()), format_func=lambda s: choices[s])
        if selected:
            sel_path = Path(selected)
            preview = sel_path.read_text(encoding="utf-8", errors="ignore")[:1200]
            st.code(preview, language="markdown")
    else:
        st.info("No experiments/ directory found next to this file.")

def render_result(label: str, res: Dict[str, Any]):
    st.markdown(f"### {label}")
    cols = st.columns(3)
    with cols[0]:
        st.metric("Passed", "✅ Yes" if res.get("ok") else "❌ No")
    with cols[1]:
        st.metric("Hard Gate", "On" if enforce_hard_gate else "Off")
    with cols[2]:
        st.metric("Exec Time (s)", f"{(res.get('exec_time') or 0):.3f}")

    with st.expander("Safety Report", expanded=True):
        st.json(res.get("safety_report", {}))

    if res.get("term_out"):
        st.subheader("Terminal Output")
        st.code(res["term_out"])

    if res.get("exception"):
        st.error(f"Exception: {res['exception']}")

    if res.get("raw_exc_info"):
        with st.expander("Raw Exception Info"):
            st.code(str(res["raw_exc_info"]))

def guess_test_file(name: str) -> Optional[Path]:
    candidates = [
        Path(__file__).resolve().parent / f"{name}.py",
        Path(__file__).resolve().parent / "ai_scientist_v2" / f"{name}.py",
        repo_root / "applications" / "ai_scientist_v2" / f"{name}.py",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None

def run_test_file(path: Path) -> Dict[str, Any]:
    code = path.read_text(encoding="utf-8", errors="ignore")
    return run_with_interpreter(code)

if run_code:
    res = run_with_interpreter(code_demo)
    render_result("Ad hoc Code Run", res)

if run_safe:
    p = guess_test_file("test_safe")
    if p:
        st.info(f"Running safe test: {p}")
        res = run_test_file(p)
    else:
        st.warning("Could not find test_safe.py; running a built-in safe snippet.")
        res = run_with_interpreter("print('hello from safe test'); y = sum([1,2,3]); print('y=', y)")
    render_result("Safe Test Result", res)

if run_unsafe:
    p = guess_test_file("test_unsafe")
    if p:
        st.info(f"Running unsafe test: {p}")
        res = run_test_file(p)
    else:
        st.warning("Could not find test_unsafe.py; running a built-in UNSAFE snippet.")
        res = run_with_interpreter("import subprocess\nsubprocess.call(['echo','hi'])")
    render_result("Unsafe Test Result", res)

st.markdown("---")
with st.expander("📐 Architecture (high-level)", expanded=False):
    st.markdown("""
**Researcher/Planner ➜ Code Generator ➜ _Safety Layer_ ➜ Interpreter Execution ➜ Results**

- **Safety Layer**: static checks, import allowlist, blocked-call list, runtime limits, hard/soft gating.
- **Interpreter**: runs code only if constraints pass (or soft-gate warns), returns `exc_info` / custom safety info.
- This UI shows pass/fail, issues, timing, and raw exception details.
""")
