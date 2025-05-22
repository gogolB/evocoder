# evocoder/problems/simple_line_reducer/problem_config.py

from pathlib import Path

# Get the directory of the current file (problem_config.py)
PROBLEM_DIR = Path(__file__).resolve().parent

PROBLEM_NAME = "simple_line_reducer"

# --- Problem Specific Files ---
# Path to the initial Python code file that the LLM will try to optimize.
# This file should contain the function(s) or class(es) to be evolved.
INITIAL_CODE_FILE: Path = PROBLEM_DIR / "initial_code.py"

# Path to the test suite for this problem.
# This file should contain pytest-compatible tests that verify the correctness
# of the evolved code. The evaluator will use these tests.
TEST_SUITE_FILE: Path = PROBLEM_DIR / "test_suite.py"

# --- Target for Evolution ---
# Specify the name of the function or class within INITIAL_CODE_FILE that is the target of evolution.
# This helps the evaluator and potentially the LLM focus on the correct part.
# For this problem, let's assume we are evolving a function named 'target_function'.
TARGET_FUNCTION_NAME: str = "target_function_to_optimize"


# --- Evaluation Metrics Configuration ---
# Define the metrics that the evaluator will calculate and the system will try to optimize.
# 'metric_name': {
#   'goal': 'minimize' or 'maximize',
#   'aggregator': Optional, how to combine if multiple runs (e.g., 'avg', 'min', 'max')
# }
# The evaluator will return a dictionary of scores matching these metric names.
EVALUATION_METRICS = {
    "correctness_score": { # e.g., 1.0 if all tests pass, 0.0 otherwise, or ratio of passed tests
        "goal": "maximize",
        "description": "Fraction of unit tests passed. 1.0 means all pass."
    },
    "line_count_score": { # We want to minimize line count, so the score could be -lines or 1/lines.
                          # Let's use negative line count so maximizing the score minimizes lines.
        "goal": "maximize", # Maximizing a negative number means minimizing the positive.
        "description": "Negative of the number of lines in the target function. Higher is better (fewer lines)."
    }
    # Add other metrics like "execution_time_score", "cyclomatic_complexity_score" later if desired.
}

# --- Primary Optimization Objective ---
# The main metric the evolutionary algorithm will focus on for selection,
# especially if not doing full multi-objective selection initially.
# This must be one of the keys in EVALUATION_METRICS.
PRIMARY_METRIC: str = "line_count_score" # We'll try to maximize this (i.e., minimize lines)
                                         # but only if correctness_score is high.

# --- Correctness Threshold ---
# A threshold for the 'correctness_score' below which solutions are heavily penalized
# or considered invalid, regardless of other scores.
# For example, if correctness_score is 0.0 (no tests pass), the solution is useless.
CORRECTNESS_THRESHOLD: float = 1.0 # Require all tests to pass.

# --- LLM Prompting Instructions (Specific to this problem) ---
# These can be used by the LLMManager to guide the LLM.
PROBLEM_LLM_INSTRUCTIONS = f"""
Your task is to optimize the Python function named '{TARGET_FUNCTION_NAME}' found in the provided code.
The primary goal is to reduce the number of lines of code in this specific function
while ensuring that its external behavior remains EXACTLY the same.
It must pass all existing unit tests.

Please provide the complete, modified Python code for the entire file/module,
containing the optimized '{TARGET_FUNCTION_NAME}' function.
Do not explain your changes, just output the full modified code block.
Focus only on reducing line count through refactoring, removing redundancy, or using more concise Python idioms.
Do not change the function's signature (name, parameters, return type if hinted).
"""

# --- Optional: Evaluation Cascade (for later phases) ---
# EVALUATION_CASCADE = [
#     {"stage_name": "static_analysis", "script": "path/to/linter_check.py", "pass_if_zero_exit": True},
#     {"stage_name": "unit_tests_quick", "test_file": TEST_SUITE_FILE, "pytest_marker": "quick"},
#     {"stage_name": "unit_tests_full", "test_file": TEST_SUITE_FILE},
# ]


if __name__ == "__main__":
    print(f"Problem Configuration for: {PROBLEM_NAME}")
    print(f"Problem Directory: {PROBLEM_DIR}")
    print(f"Initial Code File: {INITIAL_CODE_FILE}")
    print(f"Test Suite File: {TEST_SUITE_FILE}")
    print(f"Target Function Name: {TARGET_FUNCTION_NAME}")
    print(f"Primary Metric: {PRIMARY_METRIC}")
    print(f"Correctness Threshold: {CORRECTNESS_THRESHOLD}")
    print("Evaluation Metrics:")
    for metric, details in EVALUATION_METRICS.items():
        print(f"  - {metric}: Goal={details['goal']}, Desc='{details['description']}'")
    print("\nProblem LLM Instructions:")
    print(PROBLEM_LLM_INSTRUCTIONS)

