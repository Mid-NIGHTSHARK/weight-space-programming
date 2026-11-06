"""Safe AST-based weight parser for the Experiment 5 edits language.

Parses markdown edits blocks into zero-initialized neural network tensors without using eval() or exec().
Enforces strict security boundaries and verifies all tensor assignments against dimension constraints.
"""

import ast
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set
import numpy as np

from network.architecture import (
    TENSOR_SHAPES,
    TOTAL_PARAMETERS,
    create_zero_weights,
)

# Approved tensor identifiers in Experiment 5
ALLOWED_TENSORS: Set[str] = {"W1", "b1", "W2", "b2", "W3", "b3"}

# Maximum allowed absolute value as defined in the prompt: (|v| <= 1e6)
MAX_ABS_VALUE: float = 1e6


class WeightParserSecurityError(Exception):
    """Raised when an unauthorized syntax construct or security violation is detected."""
    pass


class WeightParserSyntaxError(Exception):
    """Raised when an invalid assignment syntax or dimension mismatch occurs."""
    pass


@dataclass
class ParseErrorRecord:
    line_number: int
    raw_statement: str
    error_type: str
    message: str


@dataclass
class ParseReport:
    block_found: bool = False
    statements_total: int = 0
    statements_applied: int = 0
    entries_written: int = 0
    tensors_touched: List[str] = field(default_factory=list)
    errors: List[ParseErrorRecord] = field(default_factory=list)
    nonzero: Dict[str, int] = field(default_factory=dict)
    nonzero_total: int = 0
    max_abs: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "block_found": self.block_found,
            "statements_total": self.statements_total,
            "statements_applied": self.statements_applied,
            "entries_written": self.entries_written,
            "tensors_touched": sorted(self.tensors_touched),
            "errors": [
                {
                    "line": e.line_number,
                    "statement": e.raw_statement,
                    "error_type": e.error_type,
                    "message": e.message,
                }
                for e in self.errors
            ],
            "nonzero": self.nonzero,
            "nonzero_total": self.nonzero_total,
            "max_abs": self.max_abs,
        }


def extract_last_edits_block(text: str) -> Optional[str]:
    """Extract the last ```edits ... ``` block from a markdown or HTML response.
    
    Per prompt instructions: 'Only the LAST edits block of your reply is read'.
    Supports both standard markdown code fences (```edits ... ```) and HTML <pre> blocks
    commonly produced when saving responses from official web interfaces.
    """
    pattern = r"```(?:edits|python)?\s*\n(.*?)\n```"
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[-1].strip()

    # Also check for HTML <pre> code tags (from web export / docx)
    pattern_pre = r"<pre(?:.*?)>(.*?)</pre>"
    matches_pre = re.findall(pattern_pre, text, re.DOTALL | re.IGNORECASE)
    if matches_pre:
        import html
        # Return the last block that sets weights
        for block in reversed(matches_pre):
            candidate = html.unescape(block).strip()
            if any(t in candidate for t in ("W1", "W2", "W3", "b1", "b2", "b3")):
                return candidate
        return html.unescape(matches_pre[-1]).strip()

    return None


class SafeASTWeightParser:
    """Secure AST interpreter for neural network weight edits."""

    def __init__(self, strict: bool = False):
        self.strict = strict

    def _eval_literal_node(self, node: ast.AST) -> Any:
        """Safely evaluate AST literal nodes without eval()."""
        if isinstance(node, ast.Constant):
            val = node.value
            if not isinstance(val, (int, float)):
                raise WeightParserSyntaxError(f"Expected numerical constant, got {type(val).__name__}")
            if abs(val) > MAX_ABS_VALUE:
                raise WeightParserSyntaxError(f"Value {val} exceeds magnitude limit {MAX_ABS_VALUE}")
            return val

        elif isinstance(node, ast.UnaryOp):
            operand_val = self._eval_literal_node(node.operand)
            if isinstance(node.op, ast.USub):
                return -operand_val
            elif isinstance(node.op, ast.UAdd):
                return operand_val
            raise WeightParserSecurityError(f"Unauthorized UnaryOp: {type(node.op).__name__}")

        elif isinstance(node, ast.List):
            return [self._eval_literal_node(elt) for elt in node.elts]

        elif isinstance(node, ast.Tuple):
            return tuple(self._eval_literal_node(elt) for elt in node.elts)

        elif isinstance(node, ast.Slice):
            lower = self._eval_literal_node(node.lower) if node.lower is not None else None
            upper = self._eval_literal_node(node.upper) if node.upper is not None else None
            step = self._eval_literal_node(node.step) if node.step is not None else None
            return slice(lower, upper, step)

        elif isinstance(node, ast.Name):
            # Prohibit reading identifiers on the RHS
            raise WeightParserSecurityError(f"Variable reference on RHS is not permitted: {node.id}")

        else:
            raise WeightParserSecurityError(
                f"Unauthorized syntax node: {type(node).__name__}"
            )

    def _parse_subscript_slice(self, slice_node: ast.AST) -> Any:
        """Parse the index or slice expression of a tensor subscript."""
        if isinstance(slice_node, ast.Tuple):
            return tuple(self._eval_literal_node(elt) for elt in slice_node.elts)
        else:
            return self._eval_literal_node(slice_node)

    def parse_text(self, text: str) -> Tuple[Dict[str, np.ndarray], ParseReport]:
        """Parse raw model text, extract the edits block, and apply statements.
        
        Returns:
            weights: Dictionary of 6 float32 tensors with edits applied.
            report: ParseReport detailing execution statistics and any skipped errors.
        """
        weights = create_zero_weights()
        report = ParseReport()

        block_str = extract_last_edits_block(text)
        if block_str is None:
            # Fallback check if the text itself is raw edits code without code block backticks
            if any(tensor in text for tensor in ALLOWED_TENSORS):
                block_str = text.strip()
            else:
                report.errors.append(
                    ParseErrorRecord(
                        line_number=0,
                        raw_statement="",
                        error_type="MissingEditsBlock",
                        message="No ```edits code block found in response text",
                    )
                )
                return weights, report

        report.block_found = True
        touched_tensors = set()

        # Try parsing the whole block first
        try:
            tree = ast.parse(block_str)
            statements = tree.body
        except SyntaxError as e:
            # If block has a syntax error in some lines, fallback to parsing line by line
            # so valid statements still apply per prompt specification
            return self._parse_line_by_line(block_str, weights, report)

        report.statements_total = len(statements)

        for stmt in statements:
            lineno = getattr(stmt, "lineno", 0)
            raw_line = ast.unparse(stmt) if hasattr(ast, "unparse") else str(stmt)
            self._apply_statement(stmt, weights, touched_tensors, report, lineno, raw_line)

        # Finalize report metrics
        report.tensors_touched = list(touched_tensors)
        report.nonzero_total = 0
        for name, arr in weights.items():
            nz = int(np.count_nonzero(arr))
            report.nonzero[name] = nz
            report.nonzero_total += nz
            report.max_abs[name] = float(np.max(np.abs(arr))) if arr.size > 0 else 0.0

        return weights, report

    def _apply_statement(
        self,
        stmt: ast.AST,
        weights: Dict[str, np.ndarray],
        touched_tensors: set,
        report: ParseReport,
        lineno: int,
        raw_line: str,
    ) -> None:
        """Validate and apply an individual AST assignment/augassign statement."""
        try:
            if isinstance(stmt, ast.Assign):
                if len(stmt.targets) != 1:
                    raise WeightParserSyntaxError("Chained or multiple assignments not allowed")
                target = stmt.targets[0]
                if not isinstance(target, ast.Subscript):
                    raise WeightParserSyntaxError("Target must be a tensor subscript (e.g., W1[...])")
                if not isinstance(target.value, ast.Name) or target.value.id not in ALLOWED_TENSORS:
                    raise WeightParserSecurityError(f"Target identifier must be one of {ALLOWED_TENSORS}")

                tensor_name = target.value.id
                slice_expr = self._parse_subscript_slice(target.slice)
                val = np.asarray(self._eval_literal_node(stmt.value), dtype=np.float32)

                # Apply assignment
                weights[tensor_name][slice_expr] = val
                touched_tensors.add(tensor_name)
                report.statements_applied += 1
                report.entries_written += int(val.size)

            elif isinstance(stmt, ast.AugAssign):
                target = stmt.target
                if not isinstance(target, ast.Subscript):
                    raise WeightParserSyntaxError("Target must be a tensor subscript")
                if not isinstance(target.value, ast.Name) or target.value.id not in ALLOWED_TENSORS:
                    raise WeightParserSecurityError(f"Target identifier must be one of {ALLOWED_TENSORS}")

                tensor_name = target.value.id
                slice_expr = self._parse_subscript_slice(target.slice)
                val = np.asarray(self._eval_literal_node(stmt.value), dtype=np.float32)

                if isinstance(stmt.op, ast.Add):
                    weights[tensor_name][slice_expr] += val
                elif isinstance(stmt.op, ast.Sub):
                    weights[tensor_name][slice_expr] -= val
                elif isinstance(stmt.op, ast.Mult):
                    weights[tensor_name][slice_expr] *= val
                elif isinstance(stmt.op, ast.Div):
                    weights[tensor_name][slice_expr] /= val
                else:
                    raise WeightParserSecurityError(f"Unsupported operator: {type(stmt.op).__name__}")

                touched_tensors.add(tensor_name)
                report.statements_applied += 1
                report.entries_written += int(val.size)

            elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
                # Docstring / standalone comment literal, ignore safely
                pass

            else:
                raise WeightParserSecurityError(
                    f"Unauthorized statement construct: {type(stmt).__name__}"
                )

        except Exception as exc:
            err = ParseErrorRecord(
                line_number=lineno,
                raw_statement=raw_line,
                error_type=type(exc).__name__,
                message=str(exc),
            )
            report.errors.append(err)
            if self.strict:
                raise

    def _parse_line_by_line(
        self,
        block_str: str,
        weights: Dict[str, np.ndarray],
        report: ParseReport,
    ) -> Tuple[Dict[str, np.ndarray], ParseReport]:
        """Fallback line-by-line parser when the full block contains a syntax error."""
        lines = block_str.splitlines()
        report.statements_total = len([l for l in lines if l.strip() and not l.strip().startswith("#")])
        buffer = ""
        touched = set()

        for idx, line in enumerate(lines, 1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            if not buffer:
                # Must start with an allowed tensor target to begin a valid assignment
                if not any(stripped.startswith(t) for t in ALLOWED_TENSORS):
                    report.errors.append(
                        ParseErrorRecord(
                            line_number=idx,
                            raw_statement=stripped,
                            error_type="RejectedNonAssignment",
                            message=f"Line does not start with an allowed tensor name ({ALLOWED_TENSORS})",
                        )
                    )
                    continue

            buffer = f"{buffer} {stripped}" if buffer else stripped

            # Try parsing current buffer
            try:
                tree = ast.parse(buffer)
                if not tree.body:
                    continue
                stmt = tree.body[0]
                self._apply_statement(stmt, weights, touched, report, idx, buffer)
                buffer = ""
            except SyntaxError:
                # Incomplete statement (e.g. multiline list continuation)
                continue
            except Exception as e:
                report.errors.append(
                    ParseErrorRecord(
                        line_number=idx,
                        raw_statement=buffer,
                        error_type=type(e).__name__,
                        message=str(e),
                    )
                )
                buffer = ""

        if buffer:
            report.errors.append(
                ParseErrorRecord(
                    line_number=len(lines),
                    raw_statement=buffer,
                    error_type="UnterminatedStatement",
                    message="Statement buffer was not closed at end of input",
                )
            )

        report.tensors_touched = list(touched)
        report.nonzero_total = 0
        for name, arr in weights.items():
            nz = int(np.count_nonzero(arr))
            report.nonzero[name] = nz
            report.nonzero_total += nz
            report.max_abs[name] = float(np.max(np.abs(arr))) if arr.size > 0 else 0.0

        return weights, report

