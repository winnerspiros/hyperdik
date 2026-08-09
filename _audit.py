#!/usr/bin/env python3
"""Bulk audit of Python files for quick cleanup wins."""
import ast
import os
import sys
from collections import defaultdict

TARGET_DIR = "/home/ubuntu/hyperliquid-trader"
FILES = """action_executor.py
ai_decider.py
ai_validator.py
chart_patterns.py
context_enricher.py
continuous_predictor.py
correlation_tracker.py
cross_exchange.py
cryptogat_predictor.py
cvd_divergence.py
cvd_engine.py
economic_calendar.py
ev_gate.py
funding_acceleration.py
hrp_sizing.py
hurst_detector.py
hyperliquid_client.py
hyperliquid_delta_neutral.py
hyperliquid_evolution.py
hyperliquid_execution.py
hyperliquid_funding_sniper.py
hyperliquid_learner.py
hyperliquid_risk.py
hyperliquid_strategy.py
hyperliquid_whale.py
hyperliquid_ws.py
imbalance_trend.py
kronos_predictor.py
liquidation_monitor.py
liquidation_zones.py
market_analyzer.py
market_intel.py
market_regime.py
market_structure.py
ml_predictor.py
oi_delta.py
peak_exhaustion_detector.py
perfect_predictor.py
price_extremes.py
pump_detector.py
sector_rotation.py
taker_ratio.py
unified_direction.py
unified_predictor.py
volume_delta.py
vwap_signal.py
hyperliquid_daemon.py""".strip().split('\n')

# Note: the list had 47 but actually I see 47+hyperliquid_daemon = let me recount
# The original list has 46 unique entries (I count action_executor through vwap_signal = 46)
# But the user said 47. Let me just check all of them plus hyperliquid_daemon.

class FileAuditor(ast.NodeVisitor):
    def __init__(self, source_lines):
        self.source_lines = source_lines
        self.imports = {}      # name -> line
        self.functions = []    # (name, line, is_public)
        self.classes = []      # (name, line)
        self.used_names = set()
        self.dead_code = []    # (line, description)
        self.verbose_comments = []  # (line, description)
        self.unreachable = []  # (line, description)
        self.docstring_lines = 0
        self.total_lines = len(source_lines)

    def visit_Import(self, node):
        for alias in node.names:
            name = alias.asname or alias.name
            self.imports[name] = node.lineno
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        for alias in node.names:
            name = alias.asname or alias.name
            if name == '*':
                continue
            self.imports[name] = node.lineno
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        is_public = not node.name.startswith('_')
        self.functions.append((node.name, node.lineno, is_public))
        # Short docstrings are fine
        if (isinstance(node.body[0], ast.Expr) and 
            isinstance(node.body[0].value, (ast.Constant, ast.Str))):
            doc = ast.get_docstring(node)
            if doc and len(doc) > 200:
                self.verbose_comments.append((node.lineno, f"Verbose docstring ({len(doc)} chars) on {node.name}"))
        self.generic_visit(node)

    def visit_ClassDef(self, node):
        self.classes.append((node.name, node.lineno))
        self.generic_visit(node)

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load):
            self.used_names.add(node.id)
        self.generic_visit(node)

    def visit_Attribute(self, node):
        if isinstance(node.value, ast.Name):
            self.used_names.add(node.value.id)
        self.generic_visit(node)

    def visit_If(self, node):
        # Check for "if False:" or "if 0:" etc.
        if isinstance(node.test, ast.Constant):
            val = node.test.value
            if val is False or val == 0 or val is None:
                self.dead_code.append((node.lineno, f"Dead branch: if {repr(val)}:"))
        self.generic_visit(node)

def check_unused_imports(auditor):
    unused = []
    # Common stdlib names that might appear in strings or f-strings
    noise = {'self', 'True', 'False', 'None', 'int', 'float', 'str', 'list', 'dict', 
             'set', 'tuple', 'bool', 'type', 'len', 'range', 'print', 'zip', 'map',
             'filter', 'object', 'super', 'Exception', 'ValueError', 'TypeError',
             'KeyError', 'RuntimeError', 'IOError', 'OSError', 'NotImplementedError',
             'min', 'max', 'abs', 'round', 'sum', 'any', 'all', 'enumerate', 'isinstance',
             'issubclass', 'hasattr', 'getattr', 'setattr', 'delattr', 'repr', 'format',
             'open', 'id', 'hash', 'bytes', 'bytearray', 'iter', 'next', 'slice',
             'staticmethod', 'classmethod', 'property', 'reversed', 'sorted'}
    noise.update({'json', 'os', 'sys', 'time', 'datetime', 'logging', 'math', 're',
                  'collections', 'itertools', 'functools', 'typing', 'copy', 'random',
                  'enum', 'dataclasses', 'pathlib', 'traceback', 'unittest', 'io',
                  'asyncio', 'threading', 'multiprocessing', 'subprocess', 'shutil',
                  'tempfile', 'uuid', 'hashlib', 'base64', 'struct', 'inspect',
                  'warnings', 'pdb', 'abc', 'atexit', 'contextlib', 'textwrap',
                  'pprint', 'statistics', 'csv', 'pickle', 'json', 'yaml'})

    for name, line in auditor.imports.items():
        # Skip common noise
        if name in noise:
            continue
        # Check if used (with .attribute access also captured)
        if name not in auditor.used_names:
            unused.append((line, name))
    return unused

def check_unreachable_after_return(auditor):
    """Check for code after return/raise/continue/break"""
    unreachable = []
    lines = auditor.source_lines
    # Simple heuristic: scan for patterns
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped in ('return', 'continue', 'break', 'pass') and not stripped.startswith('#'):
            # Check if there are non-blank, non-comment lines after at same indent
            indent = len(line) - len(line.lstrip())
            for j in range(i+1, min(i+20, len(lines))):
                next_line = lines[j]
                next_stripped = next_line.strip()
                if not next_stripped or next_stripped.startswith('#'):
                    continue
                next_indent = len(next_line) - len(next_line.lstrip())
                if next_indent == indent:
                    unreachable.append((j+1, f"Likely unreachable after {stripped} at line {i+1}"))
                break
    return unreachable

def check_verbose_single_line_comments(lines):
    """Check for excessively long comments"""
    verbose = []
    in_block = False
    block_start = 0
    block_lines = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        # Multi-line comment blocks
        if stripped.startswith('#') and not stripped.startswith('##') and not stripped.startswith('#!'):
            if not in_block:
                in_block = True
                block_start = i
                block_lines = 1
            else:
                block_lines += 1
        else:
            if in_block:
                if block_lines > 8 and not any('License' in lines[j] or 'Copyright' in lines[j] for j in range(block_start, min(block_start+3, len(lines)))):
                    verbose.append((block_start+1, f"Long comment block ({block_lines} lines)"))
                in_block = False
                block_lines = 0
    # Also check for very long single comment lines
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('#') and len(stripped) > 100:
            if not any(x[0] == i+1 for x in verbose):
                verbose.append((i+1, f"Very long comment: {len(stripped)} chars"))
    return verbose

def count_comment_lines(lines):
    """Count total and block comment lines"""
    total = 0
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('#'):
            total += 1
    return total

def main():
    results = {}
    for fname in FILES:
        fpath = os.path.join(TARGET_DIR, fname)
        if not os.path.exists(fpath):
            results[fname] = {"error": "NOT FOUND", "category": "MISSING"}
            continue
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                source = f.read()
            lines = source.split('\n')
            tree = ast.parse(source, filename=fpath)
            auditor = FileAuditor(lines)
            auditor.visit(tree)

            unused_imports = check_unused_imports(auditor)
            unreachable = check_unreachable_after_return(auditor)
            verbose_c = check_verbose_single_line_comments(lines)
            comment_total = count_comment_lines(lines)

            # Function name -> line for cross-reference
            func_names = {f[0] for f in auditor.functions}

            issues = []

            for line, name in unused_imports:
                issues.append(f"  L{line}: Unused import '{name}'")

            for line, desc in auditor.dead_code:
                issues.append(f"  L{line}: {desc}")

            for line, desc in auditor.verbose_comments:
                issues.append(f"  L{line}: {desc}")

            for line, desc in unreachable:
                issues.append(f"  L{line}: {desc}")

            for line, desc in verbose_c:
                issues.append(f"  L{line}: {desc}")

            n_pub = sum(1 for _, _, pub in auditor.functions if pub)
            n_priv = sum(1 for _, _, pub in auditor.functions if not pub)
            n_class = len(auditor.classes)
            comment_pct = round(comment_total / max(auditor.total_lines, 1) * 100, 1)

            n_issues = len(issues)
            if n_issues == 0:
                cat = "CLEAN"
            elif n_issues <= 2:
                cat = "MINOR"
            else:
                cat = "NEEDS_WORK"

            results[fname] = {
                "lines": auditor.total_lines,
                "public_funcs": n_pub,
                "private_funcs": n_priv,
                "classes": n_class,
                "comment_pct": comment_pct,
                "issues": issues,
                "n_issues": n_issues,
                "category": cat
            }
        except SyntaxError as e:
            results[fname] = {"error": f"SYNTAX ERROR: {e}", "category": "ERROR"}
        except Exception as e:
            results[fname] = {"error": str(e), "category": "ERROR"}

    # Print results
    print(f"{'File':<38} {'Lines':>6} {'Pub':>4} {'Priv':>4} {'Cls':>3} {'Cmt%':>5} {'Issues':>6} {'Category'}")
    print("-" * 120)
    for fname in FILES:
        r = results.get(fname, {})
        if r.get("error"):
            print(f"{fname:<38} ERROR: {r['error']}")
            continue
        print(f"{fname:<38} {r['lines']:>6} {r['public_funcs']:>4} {r['private_funcs']:>4} {r['classes']:>3} {r['comment_pct']:>5.1f} {r['n_issues']:>6} {r['category']}")

    # Print issues detail
    print("\n\n=== DETAILED ISSUES ===")
    for fname in FILES:
        r = results.get(fname, {})
        issues = r.get("issues", [])
        if issues:
            print(f"\n--- {fname} ({r['category']}) ---")
            for issue in issues:
                print(issue)

if __name__ == '__main__':
    main()