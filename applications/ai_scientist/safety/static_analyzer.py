import ast
from pathlib import Path
from typing import List
from .config import SafetyConfig, SafetyReport, SafetyIssue


class _Visitor(ast.NodeVisitor):
    def __init__(self, cfg: SafetyConfig):
        self.cfg = cfg
        self.issues: List[SafetyIssue] = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            name = alias.name.split(".")[0]
            if name in self.cfg.blocked_modules:
                self.issues.append(
                    SafetyIssue(
                        severity="error",
                        code="BLOCKED_IMPORT",
                        detail=f"Importing blocked module '{name}'",
                        location=f"line {node.lineno}",
                    )
                )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            name = node.module.split(".")[0]
            if name in self.cfg.blocked_modules:
                self.issues.append(
                    SafetyIssue(
                        severity="error",
                        code="BLOCKED_IMPORT",
                        detail=f"Importing from blocked module '{name}'",
                        location=f"line {node.lineno}",
                    )
                )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        # Function name: might be Name or Attribute
        func_name = None
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            # something.func; take "module.func"
            base = []
            cur = node.func
            while isinstance(cur, ast.Attribute):
                base.append(cur.attr)
                cur = cur.value
            if isinstance(cur, ast.Name):
                base.append(cur.id)
            base_rev = list(reversed(base))
            func_name = ".".join(base_rev)

        if func_name:
            # Check blocked functions and common dangerous patterns
            for blocked in self.cfg.blocked_functions:
                if func_name == blocked or func_name.endswith("." + blocked):
                    self.issues.append(
                        SafetyIssue(
                            severity="error",
                            code="BLOCKED_CALL",
                            detail=f"Call to blocked function '{func_name}'",
                            location=f"line {node.lineno}",
                        )
                    )
            if func_name.startswith("subprocess."):
                self.issues.append(
                    SafetyIssue(
                        severity="error",
                        code="DANGEROUS_PROC",
                        detail=f"Spawning subprocess via '{func_name}'",
                        location=f"line {node.lineno}",
                    )
                )

        self.generic_visit(node)


def analyze_code_str(code: str, cfg: SafetyConfig) -> SafetyReport:
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        issue = SafetyIssue(
            severity="error",
            code="SYNTAX_ERROR",
            detail=str(e),
            location=f"line {e.lineno}",
        )
        return SafetyReport(passed=False, issues=[issue], risk_score=1.0)

    visitor = _Visitor(cfg)
    visitor.visit(tree)

    has_error = any(i.severity == "error" for i in visitor.issues)
    risk_score = 1.0 if has_error else (0.2 if visitor.issues else 0.0)
    passed = not has_error
    return SafetyReport(passed=passed, issues=visitor.issues, risk_score=risk_score)


def analyze_file(path: str | Path, cfg: SafetyConfig) -> SafetyReport:
    path = Path(path)
    code = path.read_text()
    return analyze_code_str(code, cfg)
