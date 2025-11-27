"""
Python interpreter for executing code snippets and capturing their output.
Supports:
- captures stdout and stderr
- captures exceptions and stack traces
- limits execution time
"""

import logging
import os
import queue
import signal
import sys
import time
import traceback
from dataclasses import dataclass
from multiprocessing import Process, Queue
from pathlib import Path

import humanize
from dataclasses_json import DataClassJsonMixin

from ai_scientist.safety.config import load_safety_config, SafetyIssue, SafetyReport
from ai_scientist.safety.static_analyzer import analyze_code_str

logger = logging.getLogger("ai-scientist")

# Load safety config once at import time
SAFETY_CONFIG = load_safety_config(
    Path(__file__).resolve().parent.parent / "safety" / "safety_config.yaml"
)

def confirm_before_run(timeout_sec: int = 15) -> bool:
    """
    Give user a short window to cancel execution.
    In Colab/notebooks, they can click 'Interrupt execution' or hit Ctrl+C.
    """
    print(
        "[SAFETY] About to execute model-generated code.\n"
        f"         If you want to STOP, press Ctrl+C / interrupt NOW.\n"
        f"         Otherwise execution will continue in {timeout_sec} seconds..."
    )
    try:
        time.sleep(timeout_sec)
    except KeyboardInterrupt:
        print("[SAFETY] Execution cancelled by user before starting.\n")
        return False
    return True

@dataclass
class ExecutionResult(DataClassJsonMixin):
    """
    Result of executing a code snippet in the interpreter.
    Contains the output, execution time, and exception information.
    """

    term_out: list[str]
    exec_time: float
    exc_type: str | None
    exc_info: dict | None = None
    exc_stack: list[tuple] | None = None


def exception_summary(e, working_dir, exec_file_name, format_tb_ipython):
    """Generates a string that summarizes an exception and its stack trace (either in standard python repl or in IPython format)."""
    if format_tb_ipython:
        import IPython.core.ultratb

        tb = IPython.core.ultratb.VerboseTB(tb_offset=1, color_scheme="NoColor")
        tb_str = str(tb.text(*sys.exc_info()))
    else:
        tb_lines = traceback.format_exception(e)
        # skip parts of stack trace in weflow code
        tb_str = "".join(
            [l for l in tb_lines if "treesearch/" not in l and "importlib" not in l]
        )

    # replace whole path to file with just filename (to remove agent workspace dir)
    tb_str = tb_str.replace(str(working_dir / exec_file_name), exec_file_name)

    exc_info = {}
    if hasattr(e, "args"):
        exc_info["args"] = [str(i) for i in e.args]
    for att in ["name", "msg", "obj"]:
        if hasattr(e, att):
            exc_info[att] = str(getattr(e, att))

    tb = traceback.extract_tb(e.__traceback__)
    exc_stack = [(t.filename, t.lineno, t.name, t.line) for t in tb]

    return tb_str, e.__class__.__name__, exc_info, exc_stack


class RedirectQueue:
    def __init__(self, queue):
        self.queue = queue

    def write(self, msg):
        self.queue.put(msg)

    def flush(self):
        pass


class Interpreter:
    def __init__(
        self,
        working_dir: Path | str,
        timeout: int = 3600,
        format_tb_ipython: bool = False,
        agent_file_name: str = "runfile.py",
        env_vars: dict[str, str] = {},
    ):
        """
        Simulates a standalone Python REPL with an execution time limit.

        Args:
            working_dir (Path | str): working directory of the agent
            timeout (int, optional): Timeout for each code execution step. Defaults to 3600.
            format_tb_ipython (bool, optional): Whether to use IPython or default python REPL formatting for exceptions. Defaults to False.
            agent_file_name (str, optional): The name for the agent's code file. Defaults to "runfile.py".
            env_vars (dict[str, str], optional): Environment variables to set in the child process. Defaults to {}.
        """
        # this really needs to be a path, otherwise causes issues that don't raise exc
        self.working_dir = Path(working_dir).resolve()
        assert (
            self.working_dir.exists()
        ), f"Working directory {self.working_dir} does not exist"
        self.timeout = timeout
        self.format_tb_ipython = format_tb_ipython
        self.agent_file_name = agent_file_name
        self.process: Process = None  # type: ignore
        self.env_vars = env_vars

    def child_proc_setup(self, result_outq: Queue) -> None:
        # disable all warnings (before importing anything)
        import shutup

        shutup.mute_warnings()

        for key, value in self.env_vars.items():
            os.environ[key] = value

        os.chdir(str(self.working_dir))

        # this seems to only  benecessary because we're exec'ing code from a string,
        # a .py file should be able to import modules from the cwd anyway
        sys.path.append(str(self.working_dir))

        # capture stdout and stderr
        # trunk-ignore(mypy/assignment)
        sys.stdout = sys.stderr = RedirectQueue(result_outq)

    def _run_session(
        self, code_inq: Queue, result_outq: Queue, event_outq: Queue
    ) -> None:
        self.child_proc_setup(result_outq)

        global_scope: dict = {}
        while True:
            code = code_inq.get()
            os.chdir(str(self.working_dir))
            with open(self.agent_file_name, "w") as f:
                f.write(code)

            event_outq.put(("state:ready",))
            try:
                exec(compile(code, self.agent_file_name, "exec"), global_scope)
            except BaseException as e:
                tb_str, e_cls_name, exc_info, exc_stack = exception_summary(
                    e,
                    self.working_dir,
                    self.agent_file_name,
                    self.format_tb_ipython,
                )
                result_outq.put(tb_str)
                if e_cls_name == "KeyboardInterrupt":
                    e_cls_name = "TimeoutError"

                event_outq.put(("state:finished", e_cls_name, exc_info, exc_stack))
            else:
                event_outq.put(("state:finished", None, None, None))

            # put EOF marker to indicate that we're done
            result_outq.put("<|EOF|>")

    def create_process(self) -> None:
        # we use three queues to communicate with the child process:
        # - code_inq: send code to child to execute
        # - result_outq: receive stdout/stderr from child
        # - event_outq: receive events from child (e.g. state:ready, state:finished)
        # trunk-ignore(mypy/var-annotated)
        self.code_inq, self.result_outq, self.event_outq = Queue(), Queue(), Queue()
        self.process = Process(
            target=self._run_session,
            args=(self.code_inq, self.result_outq, self.event_outq),
        )
        self.process.start()

    def _drain_queues(self):
        """Quickly drain all in-flight messages to prevent blocking."""
        while not self.result_outq.empty():
            try:
                self.result_outq.get_nowait()
            except Exception:
                break

        while not self.event_outq.empty():
            try:
                self.event_outq.get_nowait()
            except Exception:
                break

        while not self.code_inq.empty():
            try:
                self.code_inq.get_nowait()
            except Exception:
                break

    def cleanup_session(self):
        if self.process is None:
            return
        # give the child process a chance to terminate gracefully
        self.process.terminate()
        self._drain_queues()
        self.process.join(timeout=2)
        # kill the child process if it's still alive
        if self.process.exitcode is None:
            logger.warning("Child process failed to terminate gracefully, killing it..")
            self.process.kill()
            self._drain_queues()
            self.process.join(timeout=2)
        # don't wait for gc, clean up immediately
        self.process.close()
        self.process = None  # type: ignore

    def run(self, code: str, reset_session=True) -> ExecutionResult:
        """
        Execute the provided Python command in a separate process and return its output.

        Parameters:
            code (str): Python code to execute.
            reset_session (bool, optional): Whether to reset the interpreter session before executing the code. Defaults to True.

        Returns:
            ExecutionResult: Object containing the output and metadata of the code execution.

        """
        safety_meta: dict | None = None
        # --- SAFETY GATE: static analysis before any child process is created ---
        safety_report: SafetyReport = analyze_code_str(code, SAFETY_CONFIG)
        # If there are *any* issues but we still decided to allow execution,
        # warn the user and give a 15-second window to interrupt.
        if safety_report.issues:
            # Build a short summary of issues
            issues_lines = [
                f"- [{issue.severity.upper()}] {issue.code}"
                + (f" at {issue.location}" if issue.location else "")
                + f": {issue.detail}"
                for issue in safety_report.issues
            ]
            issues_text = "\n".join(issues_lines)
            safety_meta = {
                "issues": [
                    {
                        "severity": i.severity,
                        "code": i.code,
                        "detail": i.detail,
                        "location": i.location,
                    }
                    for i in safety_report.issues
                ]
            }
            # Show a truncated view of the code that will be executed
            code_lines = code.splitlines()
            max_lines = 20
            shown = code_lines[:max_lines]
            if len(code_lines) > max_lines:
                shown.append("... [truncated]")

            code_preview = "\n".join(shown)

            print("\n[SAFETY] About to execute model-generated code with flagged issues.\n")
            print("Detected safety findings:")
            print(issues_text or "(no details available)")
            print("\nCode preview:")
            print("-------------------------------------------------------------------------------")
            print(code_preview)
            print("-------------------------------------------------------------------------------")
            print(
                "\nIf you want to STOP, press Ctrl+C / interrupt the cell NOW.\n"
                f"Otherwise execution will continue in {SAFETY_CONFIG.confirm_timeout_sec} seconds..."
            )
            sys.stdout.flush()
            time.sleep(SAFETY_CONFIG.confirm_timeout_sec)
        logger.debug(f"REPL is executing code (reset_session={reset_session})")

        if reset_session:
            if self.process is not None:
                # terminate and clean up previous process
                self.cleanup_session()
            self.create_process()
        else:
            # reset_session needs to be True on first exec
            assert self.process is not None

        assert self.process.is_alive()

        self.code_inq.put(code)

        # wait for child to actually start execution (we don't want interrupt child setup)
        try:
            state = self.event_outq.get(timeout=10)
        except queue.Empty:
            msg = "REPL child process failed to start execution"
            logger.critical(msg)
            while not self.result_outq.empty():
                logger.error(f"REPL output queue dump: {self.result_outq.get()}")
            raise RuntimeError(msg) from None
        assert state[0] == "state:ready", state
        start_time = time.time()

        # this flag indicates that the child ahs exceeded the time limit and an interrupt was sent
        # if the child process dies without this flag being set, it's an unexpected termination
        child_in_overtime = False

        while True:
            try:
                # check if the child is done
                state = self.event_outq.get(timeout=1)  # wait for state:finished
                assert state[0] == "state:finished", state
                exec_time = time.time() - start_time
                break
            except queue.Empty:
                # we haven't heard back from the child -> check if it's still alive (assuming overtime interrupt wasn't sent yet)
                if not child_in_overtime and not self.process.is_alive():
                    msg = "REPL child process died unexpectedly"
                    logger.critical(msg)
                    while not self.result_outq.empty():
                        logger.error(
                            f"REPL output queue dump: {self.result_outq.get()}"
                        )
                    raise RuntimeError(msg) from None

                # child is alive and still executing -> check if we should sigint..
                if self.timeout is None:
                    continue
                running_time = time.time() - start_time
                if running_time > self.timeout:
                    # [TODO] handle this in a better way
                    assert reset_session, "Timeout ocurred in interactive session"

                    # send interrupt to child
                    os.kill(self.process.pid, signal.SIGINT)  # type: ignore
                    child_in_overtime = True
                    # terminate if we're overtime by more than a minute
                    if running_time > self.timeout + 60:
                        logger.warning("Child failed to terminate, killing it..")
                        self.cleanup_session()

                        state = (None, "TimeoutError", {}, [])
                        exec_time = self.timeout
                        break

        output: list[str] = []
        # read all stdout/stderr from child up to the EOF marker
        # waiting until the queue is empty is not enough since
        # the feeder thread in child might still be adding to the queue
        while not self.result_outq.empty() or not output or output[-1] != "<|EOF|>":
            output.append(self.result_outq.get())
        output.pop()  # remove the EOF marker

        e_cls_name, exc_info, exc_stack = state[1:]

        if e_cls_name == "TimeoutError":
            output.append(
                f"TimeoutError: Execution exceeded the time limit of {humanize.naturaldelta(self.timeout)}"
            )
        else:
            output.append(
                f"Execution time: {humanize.naturaldelta(exec_time)} seconds (time limit is {humanize.naturaldelta(self.timeout)})."
            )
        basic_exc_info = exc_info
        if safety_meta is not None:
            exc_info = {
            "AI Scientist Execution Info": basic_exc_info,
            "Custom Safety Execution Info": safety_meta
            }
            def agent_verify(code, safety_meta, model: str = "gpt-4o"):
                """
                Analyzes metadata to provide analysis and summary of the code execution's 
                potential safety issues. This can be used for future improvements to the safety 
                guidelines, and any required changes to the safety configuration/overall research pipeline.
                As of now, integration of this analysis is manual, however this could be an automated
                feedback loop for the agentic research system if desired.
                Args:
                    code: The surroudning code of the flagged issues
                    safety_meta: Custom Safety Execution Info defined in the interpreter.run() function
                    model: The model used to analyze these parameterws (defaults to gpt-4o)
                Returns:
                    A summary of the code execution's potential safety issues and any required changes to the safety configuration/overall research pipeline.
                """
                print("Agent verification started")
                if safety_meta is None:
                    return "No safety issues found"
                else:
                    from openai import OpenAI
                    prompt_sys = """You are a safety analysis expert for AI-generated code execution.
                            Your role is to analyze safety metadata from code execution and provide:
                            1. A clear summary of key safety flags and their severity
                            2. Specific recommendations for improving safety in future code generation if and only if the safety issues are severe.
                            Be concise, actionable, and focus on practical improvements"""
                    prompt_user = f"""Analyze the following safety information from an AI-generated code execution:
                    SAFETY ISSUES DETECTED:
                    {safety_meta}
                    CODE PREVIEW (surrounding code lines):
                    {code}Please provide:
                    1. **Key Safety Flags Summary**: A brief overview of the most critical safety concerns
                    2. **Severity Assessment**: Which issues pose the highest risk and why. If the severity is not severe, just say "No severe safety issues found".
                    3. **Improvement Recommendations**: Specific, actionable suggestions if and only if the safety issues are severe:
                    - How the code generation process could be improved to avoid these issues
                    - What safety checks or constraints should be added
                    - Best practices for future code generation to be more safety-conscious
                    Format your response clearly with sections for each of the above points."""
                        
                    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url="https://openrouter.ai/api/v1")
                    response = client.chat.completions.create(
                        model=model,
                        messages=[{"role": "system", "content": prompt_sys},
                            {"role": "user", "content": prompt_user}])
                    if response:
                      return response.choices[0].message.content
            agent_response = agent_verify(code, safety_meta, model = "gpt-4o")
            exc_info["AI agent_response_to_safety_issues"] = agent_response
        return ExecutionResult(output, exec_time, e_cls_name, exc_info, exc_stack)