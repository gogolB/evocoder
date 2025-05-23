# evocoder/core/evaluator.py

import ast
import importlib.util
import os
import pytest
import shutil
import tempfile
import sys 
import asyncio 
from pathlib import Path
from typing import Dict, Any, Tuple, Coroutine, Optional, List # Added List

class Evaluator:
    """
    Evaluates a given string of Python code based on a problem's specific
    configuration, including running tests and calculating metrics.
    Implements an evaluation cascade.
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
                    if node.end_lineno is not None and node.lineno is not None:
                        if node.end_lineno >= node.lineno:
                            return node.end_lineno - node.lineno + 1
                        else: 
                            return -4 # Inconsistent line numbers
                    else: 
                        return -5 # Missing line number info
            return -1 # Function not found
        except SyntaxError:
            return -2 # Code could not be parsed
        except Exception:
            return -3 # Other AST processing error

    async def _run_pytest_on_evolved_code(
        self,
        evolved_code_str: str,
        problem_config: Dict[str, Any],
        pytest_marker: Optional[str] = None # NEW: For running specific marked tests
    ) -> Tuple[float, int, int]:
        """
        Runs pytest on the evolved code in a temporary, isolated environment,
        now using asyncio.to_thread and supporting pytest markers.
        """
        test_suite_file_path = Path(problem_config["TEST_SUITE_FILE"])
        target_function_name = problem_config["TARGET_FUNCTION_NAME"]

        temp_dir = None
        original_cwd = os.getcwd()
        
        try:
            temp_dir = tempfile.mkdtemp()
            temp_dir_path = Path(temp_dir)

            evolved_module_path = temp_dir_path / "evolved_module.py"
            with open(evolved_module_path, "w", encoding="utf-8") as f:
                f.write(evolved_code_str)
            
            original_test_suite_content = test_suite_file_path.read_text(encoding="utf-8")
            modified_test_content_lines = []
            found_and_replaced = False
            
            module_path_to_replace = f"evocoder.problems.{problem_config['PROBLEM_NAME']}.initial_code"
            new_module_path = ".evolved_module" 

            for line in original_test_suite_content.splitlines():
                if module_path_to_replace in line and f"import {target_function_name}" in line and "from " in line:
                    modified_line = line.replace(module_path_to_replace, new_module_path)
                    modified_test_content_lines.append(modified_line)
                    found_and_replaced = True
                else:
                    modified_test_content_lines.append(line)
            
            if not found_and_replaced:
                print(f"Warning: Could not find and replace the import for '{target_function_name}' "
                      f"from module '{module_path_to_replace}' in test suite {test_suite_file_path.name}. "
                      "Tests might not pick up the evolved code correctly.")
                modified_test_content = original_test_suite_content
            else:
                modified_test_content = "\n".join(modified_test_content_lines)

            temp_test_suite_path = temp_dir_path / test_suite_file_path.name
            with open(temp_test_suite_path, "w", encoding="utf-8") as f:
                f.write(modified_test_content)
            
            (temp_dir_path / "__init__.py").touch()

            class TestCollectorPlugin:
                def __init__(self):
                    self.passed_count = 0
                    self.total_count = 0
                    self.reports = []

                def pytest_collection_finish(self, session):
                    self.total_count = len(session.items)

                def pytest_runtest_logreport(self, report):
                    self.reports.append(report)
                    if report.passed and report.when == 'call':
                        self.passed_count += 1
            
            collector_plugin = TestCollectorPlugin()
            
            pytest_args = [str(temp_test_suite_path.name), "-q", "--disable-pytest-warnings"]
            if pytest_marker:
                pytest_args.extend(["-m", pytest_marker])
            
            def run_pytest_blocking():
                return pytest.main(pytest_args, plugins=[collector_plugin])

            os.chdir(temp_dir_path)
            exit_code = await asyncio.to_thread(run_pytest_blocking)
            
            total_tests_run_in_stage = collector_plugin.total_count
            passed_tests_in_stage = collector_plugin.passed_count
            
            if total_tests_run_in_stage > 0:
                pass_ratio = float(passed_tests_in_stage) / total_tests_run_in_stage
            else: 
                if exit_code == 0 or exit_code == 5: 
                    pass_ratio = 1.0 
                else: 
                    pass_ratio = 0.0
            
            return pass_ratio, passed_tests_in_stage, total_tests_run_in_stage

        except Exception as e:
            print(f"Error during pytest execution of evolved code (marker: {pytest_marker}): {e}")
            return 0.0, 0, 0
        finally:
            os.chdir(original_cwd) 
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    async def evaluate(self, code_string_to_test: str, problem_config: Dict[str, Any]) -> Dict[str, float]:
        scores: Dict[str, float] = {metric: 0.0 for metric in problem_config.get("EVALUATION_METRICS", {}).keys()}
        target_function_name = problem_config["TARGET_FUNCTION_NAME"]
        evaluation_cascade: List[Dict[str, Any]] = problem_config.get("EVALUATION_CASCADE", [])

        if not evaluation_cascade: 
            print("Warning: No EVALUATION_CASCADE defined in problem_config. Running all tests for correctness.")
            try:
                pass_ratio, _, _ = await self._run_pytest_on_evolved_code(code_string_to_test, problem_config)
                scores["correctness_score"] = pass_ratio
            except Exception as e:
                print(f"Failed to run tests on evolved code (no cascade): {e}")
                scores["correctness_score"] = 0.0
        else:
            overall_evaluation_passed_all_stages = True # Tracks if all stages passed successfully
            for stage_idx, stage_config in enumerate(evaluation_cascade):
                stage_name = stage_config.get("stage_name", f"stage_{stage_idx}")
                stage_type = stage_config.get("type")
                metric_to_update = stage_config.get("metric_to_update")
                fail_fast = stage_config.get("fail_fast_if_not_all_passed", False)
                
                print(f"  Running Evaluation Stage: {stage_name} (Type: {stage_type})")
                stage_pass_ratio = 0.0 

                if stage_type == "pytest":
                    pytest_marker = stage_config.get("pytest_marker", None)
                    try:
                        pass_ratio, passed, total = await self._run_pytest_on_evolved_code(
                            code_string_to_test, problem_config, pytest_marker=pytest_marker
                        )
                        stage_pass_ratio = pass_ratio
                        print(f"    Stage '{stage_name}' (pytest, marker='{pytest_marker}'): "
                              f"{passed}/{total} tests passed (Ratio: {stage_pass_ratio:.2f})")
                        if metric_to_update:
                            scores[metric_to_update] = stage_pass_ratio 
                            if metric_to_update == "precision_score" and stage_pass_ratio < 1.0:
                                scores[metric_to_update] = -float('inf') 
                            elif metric_to_update == "convergence_steps_score" and stage_pass_ratio < 1.0:
                                scores[metric_to_update] = -float('inf') 
                    except Exception as e:
                        print(f"    Error in pytest stage '{stage_name}': {e}")
                        stage_pass_ratio = 0.0
                        if metric_to_update:
                            scores[metric_to_update] = 0.0 
                else:
                    print(f"    Warning: Unknown stage type '{stage_type}' for stage '{stage_name}'. Skipping.")
                    continue

                if stage_pass_ratio < 1.0:
                    overall_evaluation_passed_all_stages = False # At least one stage didn't fully pass
                    if fail_fast:
                        print(f"    Stage '{stage_name}' failed and fail_fast is True. Stopping cascade.")
                        # Ensure correctness_score reflects this critical failure if it's a correctness stage
                        if metric_to_update == "correctness_score": # This is the primary correctness metric
                             scores["correctness_score"] = stage_pass_ratio 
                        break 
            
            # If cascade completed but not all stages passed (and fail_fast wasn't triggered for correctness)
            # ensure correctness_score reflects the outcome of its designated stage.
            # This logic is a bit redundant if correctness_score stage is fail_fast.
            if "correctness_score" not in scores or scores["correctness_score"] is None:
                 # Find the correctness stage and use its pass_ratio if it exists
                 correctness_stage_config = next((s for s in evaluation_cascade if s.get("metric_to_update") == "correctness_score"), None)
                 if correctness_stage_config:
                     # This assumes the score was set during the loop; if fail_fast stopped before it, this won't help.
                     # The logic inside the loop for fail_fast + metric_to_update = correctness_score is more direct.
                     pass # Score should have been set
                 else: # No designated correctness stage, or it wasn't run
                     scores["correctness_score"] = 0.0 if not overall_evaluation_passed_all_stages else 1.0


        final_correctness_score = scores.get("correctness_score", 0.0)
        correctness_threshold = problem_config.get("CORRECTNESS_THRESHOLD", 1.0)

        lines_in_function = -1000 
        if final_correctness_score >= correctness_threshold: 
            try:
                ast.parse(code_string_to_test) 
                lines_in_function = self._count_function_lines(code_string_to_test, target_function_name)
                if lines_in_function < 0: 
                    print(f"Warning: Could not determine line count for '{target_function_name}'. Error code: {lines_in_function}")
            except SyntaxError:
                print(f"SyntaxError in evolved code. Cannot calculate line count for '{target_function_name}'.")
                scores["correctness_score"] = 0.0 # Treat as incorrect if syntax error
        
        if scores.get("correctness_score", 0.0) >= correctness_threshold: # Re-check after potential syntax error update
            if lines_in_function >= 0: 
                scores["line_count_score"] = float(-lines_in_function)
            else: 
                scores["line_count_score"] = -9999.0 
        else:
            scores["line_count_score"] = -99999.0 
            if "precision_score" in scores and scores["precision_score"] > -float('inf'): # Don't overwrite if already penalized
                 scores["precision_score"] = -float('inf')
            if "convergence_steps_score" in scores and scores["convergence_steps_score"] > -float('inf'): 
                 scores["convergence_steps_score"] = -float('inf')

        for metric_name in problem_config.get("EVALUATION_METRICS", {}).keys():
            if metric_name not in scores:
                scores[metric_name] = -float('inf') if problem_config["EVALUATION_METRICS"][metric_name]["goal"] == "maximize" else float('inf')
        return scores

if __name__ == '__main__':
    current_dir_test = Path(__file__).resolve().parent 
    package_root_test = current_dir_test.parent 
    project_root_dir_test = package_root_test.parent 
    if str(project_root_dir_test) not in sys.path: 
        sys.path.insert(0, str(project_root_dir_test))

    from evocoder.problems.numerical_optimizer import problem_config as current_problem_config
    
    evaluator = Evaluator()

    async def run_evaluator_tests():
        print("--- Testing Evaluator (Phase 3 - Evaluation Cascade) ---")
        
        problem_config_dict = {
            attr: getattr(current_problem_config, attr)
            for attr in dir(current_problem_config)
            if not callable(getattr(current_problem_config, attr)) and not attr.startswith("__")
        }
        problem_config_dict["INITIAL_CODE_FILE"] = str(current_problem_config.INITIAL_CODE_FILE)
        problem_config_dict["TEST_SUITE_FILE"] = str(current_problem_config.TEST_SUITE_FILE)

        print(f"\n--- Testing Evaluator with initial_code.py for '{current_problem_config.PROBLEM_NAME}' ---")
        initial_code_content = Path(current_problem_config.INITIAL_CODE_FILE).read_text()
        initial_scores = await evaluator.evaluate(initial_code_content, problem_config_dict)
        print(f"Scores for initial code: {initial_scores}")

        print("\n--- Testing with code that fails basic correctness ---")
        fails_correctness_code = """
def find_root_bisection(func, a, b, tolerance=1e-7, max_iterations=100):
    return None # Always fails to find a root
"""
        fails_correctness_scores = await evaluator.evaluate(fails_correctness_code, problem_config_dict)
        print(f"Scores for code failing correctness: {fails_correctness_scores}")
        assert fails_correctness_scores.get("correctness_score", 1.0) < 1.0 # Expect less than perfect correctness

        print("\n--- Testing with code that passes correctness but might be imprecise ---")
        passes_correctness_imprecise_code = f"""
import math
def f_poly1(x: float) -> float: return x**2 - 3 * x + 2
def f_transcendental1(x: float) -> float: return math.cos(x) - x

def find_root_bisection(func, a, b, tolerance=1e-7, max_iterations=100):
    # This version might pass some 'correctness' tests if they are broad,
    # but will fail 'precision' tests.
    if hasattr(func, '__name__') and func.__name__ == 'f_poly1': return 1.9 # Imprecise for f_poly1 root 2.0
    if hasattr(func, '__name__') and func.__name__ == 'f_transcendental1': return 0.7 # Imprecise for cos(x)-x root ~0.739
    # Fallback for lambda in precision tests (e.g. x**2-2)
    # This is a hacky way to pass the "correctness" stage for the lambda in precision tests.
    # A better test suite would separate the lambda into its own named function for correctness tests.
    if "test_root_precision.<locals>.<lambda>" in str(func): return 1.4 
    return (a + b) / 2.0 
"""
        imprecise_scores = await evaluator.evaluate(passes_correctness_imprecise_code, problem_config_dict)
        print(f"Scores for imprecise code: {imprecise_scores}")
        # For 'passes_correctness_imprecise_code', the 'basic_correctness_tests' stage might not be 1.0.
        # The 'fail_fast' on that stage would then determine if 'precision_tests' even run.
        # Let's check that correctness is not perfect, and precision is heavily penalized.
        assert imprecise_scores.get("correctness_score", 1.0) < 1.0 # It should fail some correctness tests
        assert imprecise_scores.get("precision_score", 0.0) == -float('inf') # Should be penalized

    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(run_evaluator_tests())
