# evocoder/core/evaluator.py

import ast
import importlib.util
import os
import pytest
import shutil
import tempfile
import sys 
import asyncio # Added for asyncio.to_thread
from pathlib import Path
from typing import Dict, Any, Tuple, Coroutine

class Evaluator:
    """
    Evaluates a given string of Python code based on a problem's specific
    configuration, including running tests and calculating metrics.
    Evaluation of tests is now done asynchronously.
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
                            return -4 
                    else: 
                        return -5 
            return -1 
        except SyntaxError:
            return -2 
        except Exception:
            return -3 

    async def _run_pytest_on_evolved_code(
        self,
        evolved_code_str: str,
        problem_config: Dict[str, Any]
    ) -> Tuple[float, int, int]: # Return type remains the same
        """
        Runs pytest on the evolved code in a temporary, isolated environment,
        now using asyncio.to_thread for the blocking pytest.main() call.
        """
        test_suite_file_path = Path(problem_config["TEST_SUITE_FILE"])
        target_function_name = problem_config["TARGET_FUNCTION_NAME"]

        temp_dir = None
        original_cwd = os.getcwd()
        
        try:
            temp_dir = tempfile.mkdtemp()
            temp_dir_path = Path(temp_dir)

            evolved_module_path = temp_dir_path / "evolved_module.py"
            with open(evolved_module_path, "w", encoding="utf-8") as f: # Specify encoding
                f.write(evolved_code_str)

            # --- ADDED DEBUG PRINT: Read back the written file ---
            try:
                with open(evolved_module_path, "r", encoding="utf-8") as f_read:
                    written_content = f_read.read()
                print(f"\n--- Content WRITTEN to temporary evolved_module.py ---\n"
                      f"{written_content}"
                      f"\n--- End of content WRITTEN to temporary evolved_module.py ---\n")
            except Exception as e_read:
                print(f"Error reading back temporary evolved_module.py for debug: {e_read}")
            # --- END ADDED DEBUG PRINT ---


            original_test_suite_content = test_suite_file_path.read_text(encoding="utf-8")
            modified_test_content_lines = []
            found_and_replaced = False
            
            module_path_to_replace = "evocoder.problems.simple_line_reducer.initial_code"
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
                      f"from module '{module_path_to_replace}' in test suite. "
                      "Tests might not pick up the evolved code correctly.")
                modified_test_content = original_test_suite_content
            else:
                modified_test_content = "\n".join(modified_test_content_lines)

            temp_test_suite_path = temp_dir_path / test_suite_file_path.name
            with open(temp_test_suite_path, "w", encoding="utf-8") as f: # Specify encoding
                f.write(modified_test_content)
            
            (temp_dir_path / "__init__.py").touch()

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
            
            def run_pytest_blocking():
                current_thread_cwd = os.getcwd() 
                # print(f"Pytest running in thread from CWD: {current_thread_cwd}") # Should be temp_dir_path
                return pytest.main([str(temp_test_suite_path.name), "-q", "--disable-pytest-warnings"], plugins=[collector_plugin])


            os.chdir(temp_dir_path)
            exit_code = await asyncio.to_thread(run_pytest_blocking)
            
            total_tests = collector_plugin.total_count
            passed_tests = collector_plugin.passed_count
            
            if total_tests > 0:
                pass_ratio = float(passed_tests) / total_tests
            else:
                if exit_code == 0 or exit_code == 5: 
                    pass_ratio = 1.0 
                else: 
                    pass_ratio = 0.0
            
            return pass_ratio, passed_tests, total_tests

        except Exception as e:
            print(f"Error during pytest execution of evolved code: {e}")
            # import traceback # For debugging
            # traceback.print_exc() # For debugging
            return 0.0, 0, 0
        finally:
            os.chdir(original_cwd) 
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)


    async def evaluate(self, code_string_to_test: str, problem_config: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluates the given code string based on the problem configuration.
        Now an async method.
        """
        scores: Dict[str, float] = {}
        target_function_name = problem_config["TARGET_FUNCTION_NAME"]
        
        try:
            pass_ratio, passed, total = await self._run_pytest_on_evolved_code(code_string_to_test, problem_config)
            scores["correctness_score"] = pass_ratio
        except Exception as e:
            print(f"Failed to run tests on evolved code: {e}")
            scores["correctness_score"] = 0.0

        lines_in_function = -1000 
        try:
            ast.parse(code_string_to_test) 
            lines_in_function = self._count_function_lines(code_string_to_test, target_function_name)
            if lines_in_function < 0: 
                print(f"Warning: Could not determine line count for '{target_function_name}'. Error code: {lines_in_function}")
        except SyntaxError:
            print(f"SyntaxError in evolved code. Cannot calculate line count for '{target_function_name}'.")
        
        correctness_threshold = problem_config.get("CORRECTNESS_THRESHOLD", 1.0)
        if scores.get("correctness_score", 0.0) >= correctness_threshold:
            if lines_in_function >= 0: 
                scores["line_count_score"] = float(-lines_in_function)
            else: 
                scores["line_count_score"] = -9999.0 
        else:
            scores["line_count_score"] = -99999.0 

        for metric_name in problem_config.get("EVALUATION_METRICS", {}).keys():
            if metric_name not in scores:
                scores[metric_name] = 0.0 

        return scores

if __name__ == '__main__':
    current_dir_test = Path(__file__).resolve().parent 
    package_root_test = current_dir_test.parent 
    project_root_dir_test = package_root_test.parent 
    if str(project_root_dir_test) not in sys.path: 
        sys.path.insert(0, str(project_root_dir_test))

    from evocoder.problems.simple_line_reducer import problem_config as current_problem_config
    
    evaluator = Evaluator()

    async def run_evaluator_tests():
        print("--- Testing Evaluator (Async Evaluation with Read-Back Debug) ---") # Updated title
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
        
        print("\n--- Testing Evaluator with initial_code.py ---")
        initial_scores = await evaluator.evaluate(initial_code_content, problem_config_dict)
        print(f"Scores for initial code: {initial_scores}")

        print("\n--- Testing Evaluator with a slightly modified (correct) code ---")
        modified_correct_code = """
def target_function_to_optimize(a: int, b: int, c: int) -> int:
    # Optimized version
    val = a + b if a > 0 else a - b
    return val * c
"""
        modified_scores = await evaluator.evaluate(modified_correct_code, problem_config_dict)
        print(f"Scores for modified correct code: {modified_scores}")

        print("\n--- Testing Evaluator with incorrect code (e.g., always adds) ---")
        incorrect_code = """
def target_function_to_optimize(a: int, b: int, c: int) -> int:
    # Incorrect version - always adds
    intermediate_sum = a + b # Bug: Does not consider if a > 0
    result = intermediate_sum * c
    return result
"""
        incorrect_scores = await evaluator.evaluate(incorrect_code, problem_config_dict)
        print(f"Scores for incorrect code: {incorrect_scores}")

        print("\n--- Testing Evaluator with code that doesn't parse (syntax error) ---")
        syntax_error_code = """
def target_function_to_optimize(a: int, b: int, c: int) -> int:
    return a + b * c this is a syntax error
"""
        syntax_error_scores = await evaluator.evaluate(syntax_error_code, problem_config_dict)
        print(f"Scores for syntax error code: {syntax_error_scores}")

        print("\n--- Testing Evaluator with code where target function is missing ---")
        missing_function_code = """
def another_function(a: int, b: int, c: int) -> int:
    return a + b + c
"""
        missing_function_scores = await evaluator.evaluate(missing_function_code, problem_config_dict)
        print(f"Scores for missing function code: {missing_function_scores}")

    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(run_evaluator_tests())
