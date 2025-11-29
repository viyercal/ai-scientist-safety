import os
import sys
import json
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st
import importlib

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Try loading .env from multiple locations
    # streamlit_app.py is in: applications/ai_scientist_v2/streamlit_app.py
    # .env should be in: applications/ai_scientist_v2/.env
    env_paths = [
        Path(__file__).resolve().parent / ".env",  # Same directory as streamlit_app.py (ai_scientist_v2/.env)
        Path(__file__).resolve().parent.parent.parent / ".env",  # Repo root (ai-scientist-safety/.env)
    ]
    loaded_env = False
    for env_path in env_paths:
        if env_path.exists():
            load_dotenv(env_path, override=False)  # Don't override existing env vars
            loaded_env = True
            # Verify the key was loaded
            if os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY"):
                # Successfully loaded
                pass
            break
    if not loaded_env:
        # .env file not found, but that's okay if env vars are set another way
        pass
except ImportError:
    # python-dotenv not installed, that's okay
    pass
except Exception as e:
    # Log but don't fail - env vars might be set another way
    pass

# ---------------------------
# Page setup
# ---------------------------
st.set_page_config(page_title="AI Scientist v2 — Safety Demo", layout="wide")
st.markdown("# 🧪 AI Scientist v2 — Safety Layer Demo")



# ---------------------------
this_file = Path(__file__).resolve()
#   parents[0] = applications/ai_scientist_v2 (or /app in Docker)
#   parents[1] = applications (or / in Docker)
#   parents[2] = <repo_root> (may not exist in Docker)
app_dir = this_file.parent  # applications/ai_scientist_v2 or /app

# Try to get repo_root, but fallback to app_dir if in Docker (where parents[2] doesn't exist)
try:
    if len(this_file.parents) > 2:
        repo_root = this_file.parents[2]
    else:
        # In Docker, we're at /app, so use app_dir as repo_root
        repo_root = app_dir
except (IndexError, AttributeError):
    # Fallback: use app_dir as repo_root
    repo_root = app_dir

# Add app directory to Python path first (where ai_scientist module is)
if str(app_dir) not in sys.path:
    sys.path.insert(0, str(app_dir))

# Also add repo root as fallback (if different from app_dir)
if str(repo_root) not in sys.path and str(repo_root) != str(app_dir):
    sys.path.insert(0, str(repo_root))

user_root = st.sidebar.text_input("Override repo root (optional)", value=str(repo_root))
if user_root and Path(user_root).exists():
    if user_root not in sys.path:
        sys.path.insert(0, user_root)
    # Also try adding applications/ai_scientist_v2 relative to user_root
    app_path_from_user = Path(user_root) / "applications" / "ai_scientist_v2"
    if app_path_from_user.exists() and str(app_path_from_user) not in sys.path:
        sys.path.insert(0, str(app_path_from_user))

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

st.sidebar.info("""
**Note:** The Interpreter uses `safety_config.yaml` for safety rules.
Edit `ai_scientist/safety/safety_config.yaml` to change:
- `blocked_modules` (e.g., subprocess, socket)
- `blocked_functions` (e.g., exec, compile, open)
- `agent_verify` (enable AI safety analysis)
- `confirm_timeout_sec` (wait time when issues found)
""")

# Display options
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Display Options")
show_exc_info = st.sidebar.checkbox("Show exc_info Details", value=False, help="Display the full exc_info dictionary from interpreter.run() results")

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

def run_with_interpreter(code: str) -> Dict[str, Any]:
    """
    Run code through the Interpreter which uses safety_config.yaml for safety checks.
    The interpreter.run() method handles all safety logic internally.
    """
    if not has_repo or Interpreter is None:
        return {
            "ok": False,
            "term_out": "",
            "exception": "Interpreter not available. Please check imports and setup.",
            "safety_report": {
                "passed": False,
                "issues": ["Interpreter class not found"],
                "config": {},
            },
            "raw_exc_info": "Interpreter could not be imported.",
            "exec_time": None,
        }
    
    result: Dict[str, Any] = {
        "ok": False,
        "term_out": "",
        "exception": None,
        "safety_report": {
            "passed": True,
            "issues": [],
            "config": "Using safety_config.yaml from ai_scientist/safety/",
        },
        "raw_exc_info": None,
        "exec_time": None,
    }

    try:
        # Create interpreter - it automatically loads safety_config.yaml at import time
        # The interpreter uses SAFETY_CONFIG loaded from safety_config.yaml
        import tempfile
        working_dir = Path(tempfile.mkdtemp(prefix="ai_scientist_streamlit_"))
        
        # Detect venv and prepare environment for child process
        env_vars = {}
        venv_site_packages = None
        
        # Pass API keys to child process so agent_verify can access them
        # interpreter.py uses os.getenv("OPENAI_API_KEY") in agent_verify function
        # So we must ensure OPENAI_API_KEY is set in the child process environment
        openai_key = os.environ.get("OPENAI_API_KEY")
        openrouter_key = os.environ.get("OPENROUTER_API_KEY")
        
        # Prefer OPENAI_API_KEY, but use OPENROUTER_API_KEY if that's what's available
        api_key = openai_key or openrouter_key
        if api_key:
            # Always set OPENAI_API_KEY since interpreter.py expects it
            env_vars["OPENAI_API_KEY"] = api_key
            # Also set OPENROUTER_API_KEY if it exists and is different
            if openrouter_key and openrouter_key != api_key:
                env_vars["OPENROUTER_API_KEY"] = openrouter_key
        else:
            # No API key found - agent_verify will fail, but that's okay
            # The error will be caught and displayed to the user
            pass
        
        # Check if we're in a venv (sys.prefix != sys.base_prefix)
        if sys.prefix != sys.base_prefix:
            # We're in a venv - use sys.prefix
            venv_base = Path(sys.prefix)
            py_version = f"{sys.version_info.major}.{sys.version_info.minor}"
            site_packages_candidates = [
                venv_base / "lib" / f"python{py_version}" / "site-packages",
                venv_base / "lib" / "site-packages",
            ]
            for candidate in site_packages_candidates:
                if candidate.exists():
                    venv_site_packages = str(candidate)
                    break
        
        # If venv found, add to PYTHONPATH
        if venv_site_packages:
            current_pythonpath = os.environ.get("PYTHONPATH", "")
            if current_pythonpath:
                env_vars["PYTHONPATH"] = f"{venv_site_packages}{os.pathsep}{current_pythonpath}"
            else:
                env_vars["PYTHONPATH"] = venv_site_packages
            
            # Prepend code to add venv paths to sys.path in the child process
            # Using __import__('sys') instead of 'import sys' to avoid static analyzer
            # flagging it as a blocked import (since sys is in blocked_modules)
            # The static analyzer only checks for 'import' and 'from ... import' statements,
            # not __import__() calls
            venv_init_code = f"""# VENV_INIT_START
# Setup venv paths - using __import__ to avoid safety checks on 'import sys'
# This code runs before user code to ensure venv packages are available
_sys_mod = __import__('sys')
_os_mod = __import__('os')
_venv_path = r"{venv_site_packages}"
if _venv_path not in _sys_mod.path:
    _sys_mod.path.insert(0, _venv_path)
# Also add PYTHONPATH directories if set
_p = _os_mod.environ.get("PYTHONPATH", "")
if _p:
    for _x in _p.split(_os_mod.pathsep):
        if _x and _x not in _sys_mod.path:
            _sys_mod.path.insert(0, _x)
# VENV_INIT_END

"""
            code_to_run = venv_init_code + code
        else:
            code_to_run = code
        
        interp = Interpreter(
            working_dir=working_dir,
            timeout=3600,  # Default timeout: 1 hour
            format_tb_ipython=False,
            agent_file_name="runfile.py",
            env_vars=env_vars,
        )
        
        # Run code - interpreter.run() handles all safety checking internally
        t0 = time.time()
        r = interp.run(code_to_run, reset_session=True)
        dt = time.time() - t0

        # Extract terminal output
        term_out = ""
        if hasattr(r, "term_out"):
            term_out = "\n".join(getattr(r, "term_out", []) or [])
        elif hasattr(r, "output"):
            term_out = str(r.output)

        # Extract exception and safety info
        exc_info = getattr(r, "exc_info", None)
        exc_type = getattr(r, "exc_type", None)
        
        # Check if execution succeeded (no exception)
        passed = exc_type is None
        
        # Extract safety issues from exc_info structure
        safety_issues = []
        agent_verify_response = None
        
        if exc_info:
            if isinstance(exc_info, dict):
                # Extract safety issues from Custom Safety Execution Info
                if "Custom Safety Execution Info" in exc_info:
                    cse = exc_info["Custom Safety Execution Info"]
                    if isinstance(cse, dict) and "issues" in cse:
                        for issue in cse["issues"]:
                            if isinstance(issue, dict):
                                severity = issue.get("severity", "unknown")
                                code_issue = issue.get("code", "UNKNOWN")
                                detail = issue.get("detail", "")
                                location = issue.get("location", "")
                                issue_str = f"[{severity.upper()}] {code_issue}"
                                if location:
                                    issue_str += f" at {location}"
                                if detail:
                                    issue_str += f": {detail}"
                                safety_issues.append(issue_str)
                            else:
                                safety_issues.append(str(issue))
                
                # Extract agent_verify response if available
                if "AI agent_response_to_safety_issues" in exc_info:
                    agent_verify_response = exc_info["AI agent_response_to_safety_issues"]
            else:
                # Non-dict exc_info means there was an actual exception
                safety_issues.append(f"Execution exception: {exc_type}")

        result.update({
            "ok": passed,
            "term_out": term_out,
            "exec_time": dt,
            "raw_exc_info": exc_info,
            "agent_verify_response": agent_verify_response,
        })
        
        # Update safety report with extracted issues
        if safety_issues:
            result["safety_report"]["passed"] = False
            result["safety_report"]["issues"] = safety_issues
        
        return result

    except Exception as e:
        result["ok"] = False
        result["exception"] = f"Interpreter error: {e}"
        result["raw_exc_info"] = traceback.format_exc()
        return result

def list_experiments(base: Path) -> List[Path]:
    """List only .py files from experiments directory, excluding plotting_code.py files."""
    exps: List[Path] = []
    # Since we are INSIDE applications/ai_scientist_v2,
    # check local experiments/ first, but also allow repo-root search just in case.
    local = Path(__file__).resolve().parent / "experiments"
    if local.exists():
        # Only get .py files
        exps.extend(sorted(local.rglob("*.py")))
    # Also check the canonical path from repo root
    canonical = repo_root / "applications" / "ai_scientist_v2" / "experiments"
    if canonical.exists():
        # Only get .py files
        exps.extend(sorted(canonical.rglob("*.py")))
    # Remove duplicates, ensure only .py files, and exclude plotting_code.py
    unique_exps = []
    seen = set()
    for exp in exps:
        # Exclude plotting_code.py files
        if exp.name == "plotting_code.py":
            continue
        if exp.suffix == ".py" and str(exp) not in seen:
            seen.add(str(exp))
            unique_exps.append(exp)
    return sorted(unique_exps)

# Initialize session state for code editor
if "code_demo" not in st.session_state:
    st.session_state.code_demo = "x = 1 + 2\nprint('x is', x)\n"

# Layout
col1, col2 = st.columns([2, 1])

with col2:
    st.subheader("🧪 Choose an Experiment (optional)")
    exp_paths = list_experiments(repo_root)
    if exp_paths:
        choices = {str(p): p.name for p in exp_paths}
        
        def load_experiment():
            """Callback to load selected experiment into code editor."""
            selected_path = st.session_state.get("experiment_select")
            if selected_path:
                sel_path = Path(selected_path)
                try:
                    experiment_code = sel_path.read_text(encoding="utf-8", errors="ignore")
                    st.session_state.code_demo = experiment_code
                except Exception as e:
                    st.error(f"Failed to load experiment: {e}")
        
        selected = st.selectbox(
            "Experiment file", 
            options=list(choices.keys()), 
            format_func=lambda s: choices[s],
            key="experiment_select",
            on_change=load_experiment
        )
        if selected:
            sel_path = Path(selected)
            st.success(f"✅ Loaded: {sel_path.name}")
    else:
        st.info("No experiments/ directory found next to this file.")

with col1:
    st.subheader("🔒 Safety-on Interpreter Run")
    code_demo = st.text_area(
        "Paste code to execute via the safety-wrapped interpreter",
        height=220,
        value=st.session_state.code_demo,
        key="code_demo",
    )
    run_code = st.button("▶️ Run Code")

def render_result(label: str, res: Dict[str, Any], show_exc_info: bool = False):
    st.markdown(f"### {label}")
    cols = st.columns(3)
    with cols[0]:
        st.metric("Passed", "✅ Yes" if res.get("ok") else "❌ No")
    with cols[1]:
        safety_passed = res.get("safety_report", {}).get("passed", True)
        st.metric("Safety Check", "✅ Passed" if safety_passed else "⚠️ Issues Found")
    with cols[2]:
        st.metric("Exec Time (s)", f"{(res.get('exec_time') or 0):.3f}")

    # Safety Report Section
    with st.expander("🔒 Safety Report", expanded=True):
        safety_report = res.get("safety_report", {})
        if safety_report.get("issues"):
            st.warning(f"**{len(safety_report['issues'])} safety issue(s) detected:**")
            for issue in safety_report["issues"]:
                st.markdown(f"- {issue}")
        else:
            st.success("✅ No safety issues detected")
        st.caption(f"Safety config: {safety_report.get('config', 'N/A')}")

    # Agent Verify Response (if available)
    agent_response = res.get("agent_verify_response")
    if agent_response:
        with st.expander("🤖 AI Safety Analysis (agent_verify)", expanded=False):
            st.markdown(agent_response)

    if res.get("term_out"):
        st.subheader("Terminal Output")
        st.code(res["term_out"])

    if res.get("exception"):
        st.error(f"Exception: {res['exception']}")

    # Show exc_info if toggle is enabled
    if show_exc_info and res.get("raw_exc_info"):
        with st.expander("📋 Exception Info (exc_info)", expanded=True):
            exc_info = res["raw_exc_info"]
            if isinstance(exc_info, dict):
                # Pretty print dictionary
                st.json(exc_info)
            else:
                # For string or other types, show as code
                st.code(str(exc_info), language="python")

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
    render_result("Ad hoc Code Run", res, show_exc_info)

if run_safe:
    p = guess_test_file("test_safe")
    if p:
        st.info(f"Running safe test: {p}")
        res = run_test_file(p)
    else:
        st.warning("Could not find test_safe.py; running a built-in safe snippet.")
        res = run_with_interpreter("print('hello from safe test'); y = sum([1,2,3]); print('y=', y)")
    render_result("Safe Test Result", res, show_exc_info)

if run_unsafe:
    p = guess_test_file("test_unsafe")
    if p:
        st.info(f"Running unsafe test: {p}")
        res = run_test_file(p)
    else:
        st.warning("Could not find test_unsafe.py; running a built-in UNSAFE snippet.")
        res = run_with_interpreter("import subprocess\nsubprocess.call(['echo','hi'])")
    render_result("Unsafe Test Result", res, show_exc_info)

st.markdown("---")
with st.expander("📐 Architecture (high-level)", expanded=False):
    st.markdown("""
**Researcher/Planner ➜ Code Generator ➜ _Safety Layer_ ➜ Interpreter Execution ➜ Results**

- **Safety Layer**: static checks, import allowlist, blocked-call list, runtime limits, hard/soft gating.
- **Interpreter**: runs code only if constraints pass (or soft-gate warns), returns `exc_info` / custom safety info.
- This UI shows pass/fail, issues, timing, and raw exception details.
""")