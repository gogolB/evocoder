# evocoder/core/evaluator_cascade.py

import ast
import importlib.util
import os
import pytest
import shutil
import tempfile
import sys 
import asyncio 
from pathlib import Path
from typing import Dict, Any, Tuple, Coroutine, Optional, List 

# Import the logger setup function
try:
    from ..utils.logger import setup_logger
except ImportError:
    # Fallback for direct script execution
    if __name__ == '__main__':
        file_path = Path(__file__).resolve()
        project_root = file_path.parent.parent.parent
        sys.path.insert(0, str(project_root))
        from evocoder.utils.logger import setup_logger
    else:
        raise


class Evaluator:
    """
    Evaluates a given string of Python code based on a problem's specific
    configuration, including running tests and calculating metrics.
    Implements an evaluation cascade.
    """

    def __init__(self):
        self.logger = setup_logger(f"evocoder.evaluator.{self.__class__.__name__}")
        self.logger.info("Evaluator initialized.")

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
                            self.logger.warning(f"Inconsistent line numbers for {target_function_name}: start={node.lineno}, end={node.end_lineno}")
                            return -4 
                    else: 
                        self.logger.warning(f"Missing line number info for {target_function_name} in AST.")
                        return -5 
            self.logger.debug(f"Function '{target_function_name}' not found in AST.")
            return -1 
        except SyntaxError as e:
            self.logger.warning(f"SyntaxError parsing code for line count: {e}")
            return -2 
        except Exception as e:
            self.logger.exception(f"Unexpected error during AST processing for line count: {e}")
            return -3 

    async def _run_pytest_on_evolved_code(
        self,
        evolved_code_str: str,
        problem_config: Dict[str, Any],
        pytest_marker: Optional[str] = None 
    ) -> Tuple[float, int, int]:
        """
        Runs pytest on the evolved code in a temporary, isolated environment,
        now using asyncio.to_thread and supporting pytest markers.
        """
        test_suite_file_path = Path(problem_config["TEST_SUITE_FILE"])
        target_function_name = problem_config["TARGET_FUNCTION_NAME"]
        problem_name_for_log = problem_config.get("PROBLEM_NAME", "unknown_problem")

        self.logger.debug(f"Running pytest for problem '{problem_name_for_log}', marker: '{pytest_marker}'")

        temp_dir = None
        original_cwd = os.getcwd()
        
        try:
            temp_dir = tempfile.mkdtemp(prefix=f"evocoder_eval_{problem_name_for_log}_")
            temp_dir_path = Path(temp_dir)
            self.logger.debug(f"Created temp dir for pytest: {temp_dir_path}")

            evolved_module_path = temp_dir_path / "evolved_module.py"
            with open(evolved_module_path, "w", encoding="utf-8") as f:
                f.write(evolved_code_str)
            
            original_test_suite_content = test_suite_file_path.read_text(encoding="utf-8")
            modified_test_content_lines = []
            found_and_replaced = False
            
            # Construct the expected module path based on problem_name from problem_config
            module_path_to_replace = f"evocoder.problems.{problem_name_for_log}.initial_code"
            new_module_path = ".evolved_module" 

            for line in original_test_suite_content.splitlines():
                if module_path_to_replace in line and f"import {target_function_name}" in line and "from " in line:
                    modified_line = line.replace(module_path_to_replace, new_module_path)
                    modified_test_content_lines.append(modified_line)
                    found_and_replaced = True
                else:
                    modified_test_content_lines.append(line)
            
            if not found_and_replaced:
                self.logger.warning(f"Could not find/replace import for '{target_function_name}' "
                                    f"from module '{module_path_to_replace}' in test suite "
                                    f"'{test_suite_file_path.name}'. Tests might not use evolved code.")
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
            
            pytest_args = [
                str(temp_test_suite_path.name), 
                "-q", 
                "--disable-pytest-warnings",
                "-p", "no:anyio" 
            ]
            if pytest_marker:
                pytest_args.extend(["-m", pytest_marker])
            
            def run_pytest_blocking():
                exit_code_val = 0 
                try:
                    result = pytest.main(pytest_args, plugins=[collector_plugin])
                    if isinstance(result, pytest.ExitCode):
                        exit_code_val = result.value
                    elif isinstance(result, int):
                        exit_code_val = result
                except SystemExit as e:
                    self.logger.warning(f"Pytest attempted to SystemExit with code: {e.code} during evaluation.")
                    if isinstance(e.code, int):
                        exit_code_val = e.code
                    else: 
                        exit_code_val = 1 
                except Exception as e_pytest:
                    self.logger.error(f"Exception during pytest.main: {e_pytest}", exc_info=True)
                    exit_code_val = 3 
                return exit_code_val

            os.chdir(temp_dir_path)
            self.logger.debug(f"Running pytest with args: {pytest_args} in CWD: {temp_dir_path}")
            exit_code = await asyncio.to_thread(run_pytest_blocking)
            self.logger.debug(f"Pytest finished with exit code: {exit_code}")
            
            total_tests_run_in_stage = collector_plugin.total_count 
            passed_tests_in_stage = collector_plugin.passed_count
            
            if total_tests_run_in_stage > 0:
                pass_ratio = float(passed_tests_in_stage) / total_tests_run_in_stage
            else: 
                if exit_code == pytest.ExitCode.OK or exit_code == pytest.ExitCode.NO_TESTS_COLLECTED:
                    self.logger.debug(f"No tests collected or run for marker '{pytest_marker}', considering as 100% pass for stage.")
                    pass_ratio = 1.0 
                else: 
                    self.logger.warning(f"Pytest exited with code {exit_code} and no tests were parsed as run for marker '{pytest_marker}'.")
                    pass_ratio = 0.0
            
            if exit_code not in [pytest.ExitCode.OK, pytest.ExitCode.NO_TESTS_COLLECTED, pytest.ExitCode.TESTS_FAILED] \
               and pass_ratio == 1.0 and total_tests_run_in_stage > 0 : # Unexpected exit code despite tests passing
                 self.logger.warning(f"Pytest reported exit code {exit_code} but all collected tests passed. Overriding pass_ratio to 0 for safety.")
                 pass_ratio = 0.0

            return pass_ratio, passed_tests_in_stage, total_tests_run_in_stage

        except Exception as e:
            self.logger.exception(f"Error during pytest execution of evolved code (marker: {pytest_marker}): {e}")
            return 0.0, 0, 0
        finally:
            os.chdir(original_cwd) 
            if temp_dir and os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                    self.logger.debug(f"Successfully removed temp dir: {temp_dir}")
                except Exception as e_rm:
                    self.logger.error(f"Failed to remove temp dir {temp_dir}: {e_rm}")


    async def evaluate(self, code_string_to_test: str, problem_config: Dict[str, Any]) -> Dict[str, float]:
        self.logger.info(f"Evaluating code for problem: {problem_config.get('PROBLEM_NAME', 'Unknown')}")
        scores: Dict[str, float] = {metric: 0.0 for metric in problem_config.get("EVALUATION_METRICS", {}).keys()}
        target_function_name = problem_config["TARGET_FUNCTION_NAME"]
        evaluation_cascade: List[Dict[str, Any]] = problem_config.get("EVALUATION_CASCADE", [])

        if not evaluation_cascade: 
            self.logger.warning("No EVALUATION_CASCADE defined. Running all tests for correctness.")
            try:
                pass_ratio, _, _ = await self._run_pytest_on_evolved_code(code_string_to_test, problem_config)
                scores["correctness_score"] = pass_ratio
            except Exception as e:
                self.logger.exception(f"Failed to run tests on evolved code (no cascade): {e}")
                scores["correctness_score"] = 0.0
        else:
            overall_evaluation_passed_all_stages = True 
            for stage_idx, stage_config in enumerate(evaluation_cascade):
                stage_name = stage_config.get("stage_name", f"stage_{stage_idx}")
                stage_type = stage_config.get("type")
                metric_to_update = stage_config.get("metric_to_update")
                fail_fast = stage_config.get("fail_fast_if_not_all_passed", False)
                
                self.logger.info(f"  Running Evaluation Stage: {stage_name} (Type: {stage_type})")
                stage_pass_ratio = 0.0 

                if stage_type == "pytest":
                    pytest_marker = stage_config.get("pytest_marker", None)
                    try:
                        pass_ratio, passed, total = await self._run_pytest_on_evolved_code(
                            code_string_to_test, problem_config, pytest_marker=pytest_marker
                        )
                        stage_pass_ratio = pass_ratio
                        self.logger.info(f"    Stage '{stage_name}' (pytest, marker='{pytest_marker}'): "
                              f"{passed}/{total} tests passed (Ratio: {stage_pass_ratio:.2f})")
                        if metric_to_update:
                            scores[metric_to_update] = stage_pass_ratio 
                            if metric_to_update == "precision_score" and stage_pass_ratio < 1.0:
                                self.logger.debug(f"Precision tests failed for stage '{stage_name}', penalizing score.")
                                scores[metric_to_update] = -float('inf') 
                            elif metric_to_update == "convergence_steps_score" and stage_pass_ratio < 1.0:
                                self.logger.debug(f"Convergence tests failed for stage '{stage_name}', penalizing score.")
                                scores[metric_to_update] = -float('inf') 
                    except Exception as e:
                        self.logger.exception(f"    Error in pytest stage '{stage_name}': {e}")
                        stage_pass_ratio = 0.0
                        if metric_to_update:
                            scores[metric_to_update] = 0.0 
                else:
                    self.logger.warning(f"    Unknown stage type '{stage_type}' for stage '{stage_name}'. Skipping.")
                    continue

                if stage_pass_ratio < 1.0:
                    overall_evaluation_passed_all_stages = False 
                    if fail_fast:
                        self.logger.info(f"    Stage '{stage_name}' failed (pass_ratio={stage_pass_ratio:.2f}) and fail_fast is True. Stopping cascade.")
                        if metric_to_update == "correctness_score": 
                             scores["correctness_score"] = stage_pass_ratio 
                        break 
            
            # Ensure correctness_score is definitively set based on its stage outcome or overall cascade result
            correctness_metric_key = "correctness_score" # Assuming this is the standard key
            for stage_cfg in evaluation_cascade: # Find if correctness_score was set by a specific stage
                if stage_cfg.get("metric_to_update") == correctness_metric_key:
                    if correctness_metric_key not in scores: # If fail_fast stopped before its stage
                         scores[correctness_metric_key] = 0.0
                    break 
            else: # If no stage specifically updates "correctness_score"
                 scores[correctness_metric_key] = 1.0 if overall_evaluation_passed_all_stages else 0.0


        final_correctness_score = scores.get("correctness_score", 0.0)
        correctness_threshold = problem_config.get("CORRECTNESS_THRESHOLD", 1.0)
        self.logger.debug(f"Final correctness score before line count: {final_correctness_score}, Threshold: {correctness_threshold}")

        lines_in_function = -1000 
        if final_correctness_score >= correctness_threshold: 
            try:
                ast.parse(code_string_to_test) 
                lines_in_function = self._count_function_lines(code_string_to_test, target_function_name)
                if lines_in_function < 0: 
                    self.logger.warning(f"Could not determine line count for '{target_function_name}'. Error code: {lines_in_function}")
            except SyntaxError:
                self.logger.warning(f"SyntaxError in evolved code. Cannot calculate line count for '{target_function_name}'.")
                scores["correctness_score"] = 0.0 # Treat as incorrect if syntax error
        
        if scores.get("correctness_score", 0.0) >= correctness_threshold: 
            if lines_in_function >= 0: 
                scores["line_count_score"] = float(-lines_in_function)
            else: 
                scores["line_count_score"] = -9999.0 
        else:
            self.logger.info(f"Code did not meet correctness threshold ({scores.get('correctness_score', 0.0)} < {correctness_threshold}). Penalizing performance metrics.")
            scores["line_count_score"] = -99999.0 
            if "precision_score" in scores and scores["precision_score"] > -float('inf'): 
                 scores["precision_score"] = -float('inf')
            if "convergence_steps_score" in scores and scores["convergence_steps_score"] > -float('inf'): 
                 scores["convergence_steps_score"] = -float('inf')

        # Ensure all defined metrics have some value, defaulting to worst if not set
        for metric_name in problem_config.get("EVALUATION_METRICS", {}).keys():
            if metric_name not in scores:
                metric_goal = problem_config["EVALUATION_METRICS"][metric_name].get("goal", "maximize")
                scores[metric_name] = -float('inf') if metric_goal == "maximize" else float('inf')
                self.logger.debug(f"Metric '{metric_name}' was not set by cascade, defaulting to worst value.")
        
        self.logger.info(f"Final scores for evaluated code: {scores}")
        return scores

if __name__ == '__main__':
    # Setup logger for the __main__ block itself
    main_test_logger = setup_logger("evocoder.test_evaluator_cascade", level_str="DEBUG")

    current_dir_test = Path(__file__).resolve().parent 
    package_root_test = current_dir_test.parent 
    project_root_dir_test = package_root_test.parent 
    if str(project_root_dir_test) not in sys.path: 
        sys.path.insert(0, str(project_root_dir_test))

    from evocoder.problems.numerical_optimizer import problem_config as current_problem_config
    
    evaluator = Evaluator() # Evaluator will now use its own logger

    async def run_evaluator_tests():
        main_test_logger.info("--- Testing Evaluator (Phase 3 - Evaluation Cascade with Logging) ---")
        
        problem_config_dict = {
            attr: getattr(current_problem_config, attr)
            for attr in dir(current_problem_config)
            if not callable(getattr(current_problem_config, attr)) and not attr.startswith("__")
        }
        problem_config_dict["INITIAL_CODE_FILE"] = str(current_problem_config.INITIAL_CODE_FILE)
        problem_config_dict["TEST_SUITE_FILE"] = str(current_problem_config.TEST_SUITE_FILE)

        main_test_logger.info(f"--- Testing Evaluator with initial_code.py for '{current_problem_config.PROBLEM_NAME}' ---")
        initial_code_content = Path(current_problem_config.INITIAL_CODE_FILE).read_text()
        initial_scores = await evaluator.evaluate(initial_code_content, problem_config_dict)
        main_test_logger.info(f"Scores for initial code: {initial_scores}")

        main_test_logger.info("--- Testing with code that fails basic correctness ---")
        fails_correctness_code = """
def find_root_bisection(func, a, b, tolerance=1e-7, max_iterations=100):
    return None # Always fails to find a root
"""
        fails_correctness_scores = await evaluator.evaluate(fails_correctness_code, problem_config_dict)
        main_test_logger.info(f"Scores for code failing correctness: {fails_correctness_scores}")
        assert fails_correctness_scores.get("correctness_score", 1.0) < 1.0 

        main_test_logger.info("--- Testing with code that passes correctness but might be imprecise ---")
        passes_correctness_imprecise_code = f"""
import math
def f_poly1(x: float) -> float: return x**2 - 3 * x + 2
def f_transcendental1(x: float) -> float: return math.cos(x) - x

def find_root_bisection(func, a, b, tolerance=1e-7, max_iterations=100):
    if hasattr(func, '__name__') and func.__name__ == 'f_poly1': return 1.9 
    if hasattr(func, '__name__') and func.__name__ == 'f_transcendental1': return 0.7 
    if "test_root_precision.<locals>.<lambda>" in str(func): return 1.4 
    return (a + b) / 2.0 
"""
        imprecise_scores = await evaluator.evaluate(passes_correctness_imprecise_code, problem_config_dict)
        main_test_logger.info(f"Scores for imprecise code: {imprecise_scores}")
        assert imprecise_scores.get("correctness_score", 1.0) < 1.0 
        assert imprecise_scores.get("precision_score", 0.0) == -float('inf') 
        main_test_logger.info("--- Evaluator Tests Finished ---")

    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(run_evaluator_tests())
