# evocoder/core/evaluator.py

import ast
import importlib.util
import os
import pytest
import shutil
import tempfile
import sys # Added for sys.path manipulation in _run_pytest_on_evolved_code
from pathlib import Path
from typing import Dict, Any, Tuple

class Evaluator:
    """
    Evaluates a given string of Python code based on a problem's specific
    configuration, including running tests and calculating metrics.
    """

    def _count_function_lines(self, code_string: str, target_function_name: str) -> int:
        """
        Counts the number of lines in a specific function within a code string.
        Uses AST parsing to find the function's start and end lines.
        Returns -1 if the function is not found, -2 for SyntaxError, -3 for other AST error.
        """
        try:
            tree = ast.parse(code_string)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == target_function_name:
                    # end_lineno is the line number of the last line of the function's block.
                    # lineno is the line number of the "def" statement.
                    # For functions spanning multiple lines, this should give the correct count.
                    if node.end_lineno is not None and node.lineno is not None:
                        # Ensure end_lineno is not before lineno (can happen with decorators sometimes if not careful)
                        if node.end_lineno >= node.lineno:
                            return node.end_lineno - node.lineno + 1
                        else: # Should not happen for valid function ASTs
                            return -4 # Indicate inconsistent line numbers
                    else: # Fallback if end_lineno is None (e.g. for a one-liner lambda in AST, though unlikely for FunctionDef)
                        # This fallback is less precise, counts non-empty lines in the function's source segment
                        # This part is complex to get right robustly for all edge cases without more context.
                        # For now, rely on end_lineno and lineno. If they are None, it's an issue.
                        return -5 # Indicate missing line number info
            return -1 # Function not found
        except SyntaxError:
            return -2 # Code could not be parsed
        except Exception:
            return -3 # Other AST processing error

    def _run_pytest_on_evolved_code(
        self,
        evolved_code_str: str,
        problem_config: Dict[str, Any]
    ) -> Tuple[float, int, int]:
        """
        Runs pytest on the evolved code in a temporary, isolated environment.
        """
        test_suite_file_path = Path(problem_config["TEST_SUITE_FILE"])
        target_function_name = problem_config["TARGET_FUNCTION_NAME"]

        temp_dir = None
        original_cwd = os.getcwd()
        original_sys_path = list(sys.path) # Store original sys.path

        try:
            temp_dir = tempfile.mkdtemp()
            temp_dir_path = Path(temp_dir)

            evolved_module_path = temp_dir_path / "evolved_module.py"
            with open(evolved_module_path, "w") as f:
                f.write(evolved_code_str)

            original_test_suite_content = test_suite_file_path.read_text()
            modified_test_content_lines = []
            found_and_replaced = False
            
            # Module path to replace, specific to the simple_line_reducer problem's test_suite.py
            module_path_to_replace = "evocoder.problems.simple_line_reducer.initial_code"
            new_module_path = ".evolved_module" # Relative import from within the temp package

            for line in original_test_suite_content.splitlines():
                # Check if the line contains the import we want to modify
                # Example line: "    from evocoder.problems.simple_line_reducer.initial_code import target_function_to_optimize"
                if module_path_to_replace in line and f"import {target_function_name}" in line and "from " in line:
                    modified_line = line.replace(module_path_to_replace, new_module_path)
                    modified_test_content_lines.append(modified_line)
                    found_and_replaced = True
                else:
                    modified_test_content_lines.append(line)
            
            if not found_and_replaced:
                print(f"Warning: Could not find and replace the import for '{target_function_name}' "
                      f"from module '{module_path_to_replace}' in test suite. "
                      "Tests might not pick up the evolved code correctly.")
                modified_test_content = original_test_suite_content
            else:
                modified_test_content = "\n".join(modified_test_content_lines)

            temp_test_suite_path = temp_dir_path / test_suite_file_path.name
            with open(temp_test_suite_path, "w") as f:
                f.write(modified_test_content)
            
            (temp_dir_path / "__init__.py").touch()

            # Add parent of temp_dir to sys.path so that 'import temp_dir_name.evolved_module' could work
            # and pytest can find the temporary test suite as a module if needed.
            # More importantly, chdir into temp_dir_path makes relative imports like '.evolved_module' work.
            sys.path.insert(0, str(temp_dir_path.parent))
            os.chdir(temp_dir_path) # Change CWD for pytest

            class TestCollectorPlugin:
                def __init__(self):
                    self.passed_count = 0
                    self.total_count = 0

                def pytest_collection_finish(self, session):
                    self.total_count = len(session.items)

                def pytest_runtest_logreport(self, report):
                    if report.passed and report.when == 'call':
                        self.passed_count += 1
            
            collector_plugin = TestCollectorPlugin()
            
            # Run pytest on the specific test file name (it's now in the CWD)
            # Using "-q" for quieter output, can be removed for debugging.
            exit_code = pytest.main([str(temp_test_suite_path.name), "-q"], plugins=[collector_plugin])
            
            total_tests = collector_plugin.total_count
            passed_tests = collector_plugin.passed_count
            
            if total_tests > 0:
                pass_ratio = float(passed_tests) / total_tests
            else:
                if exit_code == 0 or exit_code == 5: # 5 = NO_TESTS_COLLECTED
                    # If no tests were collected but pytest exited OK, it implies an empty but valid suite.
                    # Or, if the suite was meant to be empty, this is 100% pass.
                    # For our case, if target_function_to_optimize was None due to import error in test_suite, tests are skipped.
                    pass_ratio = 1.0 # Or handle as an error if tests were expected.
                                     # Let's assume 1.0 if no tests means no failures.
                else: # Some error during collection or execution not caught by plugin
                    pass_ratio = 0.0
            
            return pass_ratio, passed_tests, total_tests

        except Exception as e:
            print(f"Error during pytest execution of evolved code: {e}")
            return 0.0, 0, 0
        finally:
            os.chdir(original_cwd) # Restore CWD
            sys.path = original_sys_path # Restore sys.path
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)


    def evaluate(self, code_string_to_test: str, problem_config: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluates the given code string based on the problem configuration.
        (Args and Returns documentation remains the same)
        """
        scores: Dict[str, float] = {}
        target_function_name = problem_config["TARGET_FUNCTION_NAME"]
        
        try:
            pass_ratio, passed, total = self._run_pytest_on_evolved_code(code_string_to_test, problem_config)
            scores["correctness_score"] = pass_ratio
        except Exception as e:
            print(f"Failed to run tests on evolved code: {e}")
            scores["correctness_score"] = 0.0

        lines_in_function = -1000 
        try:
            # Check syntax before trying to count lines for a specific function
            ast.parse(code_string_to_test) 
            lines_in_function = self._count_function_lines(code_string_to_test, target_function_name)
            if lines_in_function < 0: 
                print(f"Warning: Could not determine line count for '{target_function_name}'. Error code: {lines_in_function}")
        except SyntaxError:
            print(f"SyntaxError in evolved code. Cannot calculate line count for '{target_function_name}'.")
        
        # Score calculation based on correctness and line count
        correctness_threshold = problem_config.get("CORRECTNESS_THRESHOLD", 1.0)
        if scores.get("correctness_score", 0.0) >= correctness_threshold:
            if lines_in_function >= 0: 
                scores["line_count_score"] = float(-lines_in_function)
            else: 
                scores["line_count_score"] = -9999.0 # Penalize if line count failed for correct code
        else:
            scores["line_count_score"] = -99999.0 # Penalize even more if not correct

        for metric_name in problem_config.get("EVALUATION_METRICS", {}).keys():
            if metric_name not in scores:
                scores[metric_name] = 0.0 

        return scores

if __name__ == '__main__':
    import asyncio 
    # sys and Path are already imported at the top
    
    current_dir = Path(__file__).resolve().parent 
    package_root = current_dir.parent 
    project_root_dir = package_root.parent 
    if str(project_root_dir) not in sys.path: # Add only if not already present
        sys.path.insert(0, str(project_root_dir))

    from evocoder.problems.simple_line_reducer import problem_config as current_problem_config
    # target_function_to_optimize is not directly used in this test block anymore,
    # as we test the evaluator with code strings.

    evaluator = Evaluator()

    print("--- Testing Evaluator with initial_code.py ---")
    initial_code_content = Path(current_problem_config.INITIAL_CODE_FILE).read_text()
    
    problem_config_dict = {
        "PROBLEM_NAME": current_problem_config.PROBLEM_NAME,
        "INITIAL_CODE_FILE": str(current_problem_config.INITIAL_CODE_FILE),
        "TEST_SUITE_FILE": str(current_problem_config.TEST_SUITE_FILE),
        "TARGET_FUNCTION_NAME": current_problem_config.TARGET_FUNCTION_NAME,
        "EVALUATION_METRICS": current_problem_config.EVALUATION_METRICS,
        "PRIMARY_METRIC": current_problem_config.PRIMARY_METRIC,
        "CORRECTNESS_THRESHOLD": current_problem_config.CORRECTNESS_THRESHOLD,
        "PROBLEM_LLM_INSTRUCTIONS": current_problem_config.PROBLEM_LLM_INSTRUCTIONS,
    }
    
    initial_scores = evaluator.evaluate(initial_code_content, problem_config_dict)
    print(f"Scores for initial code: {initial_scores}")

    print("\n--- Testing Evaluator with a slightly modified (correct) code ---")
    modified_correct_code = """
def target_function_to_optimize(a: int, b: int, c: int) -> int:
    # Optimized version
    val = a + b if a > 0 else a - b
    return val * c
"""
    modified_scores = evaluator.evaluate(modified_correct_code, problem_config_dict)
    print(f"Scores for modified correct code: {modified_scores}")

    print("\n--- Testing Evaluator with incorrect code (e.g., always adds) ---")
    incorrect_code = """
def target_function_to_optimize(a: int, b: int, c: int) -> int:
    # Incorrect version - always adds
    intermediate_sum = a + b # Bug: Does not consider if a > 0
    result = intermediate_sum * c
    return result
"""
    incorrect_scores = evaluator.evaluate(incorrect_code, problem_config_dict)
    print(f"Scores for incorrect code: {incorrect_scores}")

    print("\n--- Testing Evaluator with code that doesn't parse (syntax error) ---")
    syntax_error_code = """
def target_function_to_optimize(a: int, b: int, c: int) -> int:
    return a + b * c this is a syntax error
"""
    syntax_error_scores = evaluator.evaluate(syntax_error_code, problem_config_dict)
    print(f"Scores for syntax error code: {syntax_error_scores}")

    print("\n--- Testing Evaluator with code where target function is missing ---")
    missing_function_code = """
def another_function(a: int, b: int, c: int) -> int:
    return a + b + c
"""
    missing_function_scores = evaluator.evaluate(missing_function_code, problem_config_dict)
    print(f"Scores for missing function code: {missing_function_scores}")
