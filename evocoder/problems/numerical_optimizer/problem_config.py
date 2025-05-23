# evocoder/problems/numerical_optimizer/problem_config.py

from pathlib import Path
import math # For a known root in example instructions

PROBLEM_DIR = Path(__file__).resolve().parent
PROBLEM_NAME = "numerical_optimizer"

# --- Problem Specific Files ---
INITIAL_CODE_FILE: Path = PROBLEM_DIR / "initial_code.py"
TEST_SUITE_FILE: Path = PROBLEM_DIR / "test_suite.py"

# --- Target for Evolution ---
# We'll evolve a root-finding function, e.g., bisection method
TARGET_FUNCTION_NAME: str = "find_root_bisection"

# --- Evaluation Metrics Configuration ---
EVALUATION_METRICS = {
    "correctness_score": { # 1.0 if basic functionality tests pass, 0.0 otherwise
        "goal": "maximize",
        "description": "Pass rate for basic functional correctness tests."
    },
    "precision_score": { # Higher is better (e.g., negative log of error, or 1 / error)
                         # Let's use negative absolute error for simplicity: -(abs(found_root - actual_root))
        "goal": "maximize",
        "description": "Negative absolute error from the known root. Closer to zero (less negative) is better."
    },
    "speed_score": { # Negative of execution time. Higher is better (faster).
        "goal": "maximize",
        "description": "Negative of the average execution time for finding a root."
    },
    "convergence_steps_score": { # Negative of steps to converge. Higher is better (fewer steps).
        "goal": "maximize",
        "description": "Negative of the number of iterations to converge (if applicable)."
    }
}
# For this problem, precision is key, but only if correct. Speed is secondary.
PRIMARY_METRIC: str = "precision_score" 
CORRECTNESS_THRESHOLD: float = 1.0 # Must pass all basic correctness tests

# --- LLM Prompting Instructions ---
_SEARCH_MARKER_START = "<<<<<<<< SEARCH"
_SEARCH_MARKER_END = "========" 
_REPLACE_MARKER_END = ">>>>>>>>> REPLACE"

# Example: Optimizing a bisection method for f(x) = x^2 - 2, root is sqrt(2)
# Known root for an example function like x^2 - 2 = 0 is math.sqrt(2)
_EXAMPLE_TARGET_EQUATION = "x**2 - 2" # LLM should define this or use a passed lambda
_EXAMPLE_KNOWN_ROOT = math.sqrt(2)

PROBLEM_LLM_INSTRUCTIONS = f"""
You are an expert in numerical methods and Python programming.
Your task is to optimize the Python function named '{TARGET_FUNCTION_NAME}' within the provided code.
This function is intended to find a root of a given mathematical function `func` within an interval `[a, b]` up to a given `tolerance`.

The goals for optimization are, in order of importance:
1.  **Correctness**: The function must correctly find roots for various test equations.
2.  **Precision**: The found root should be as close as possible to the true root.
3.  **Efficiency**: The function should converge to the root in fewer iterations (steps) and/or execute faster.

You can modify the internal logic of the '{TARGET_FUNCTION_NAME}' function. This might involve:
- Adjusting the convergence criteria.
- Modifying how the interval is updated.
- Introducing more sophisticated checks or step adjustments.
- Reducing computational overhead.

Provide your changes STRICTLY in the following diff format:
{_SEARCH_MARKER_START}
[EXACT ORIGINAL CODE SEGMENT TO BE REPLACED FROM THE '{TARGET_FUNCTION_NAME}' FUNCTION.]
{_SEARCH_MARKER_END}
[NEW CODE SEGMENT TO REPLACE THE ORIGINAL. PRESERVE RELEVANT INDENTATION.]
{_REPLACE_MARKER_END}

IMPORTANT RULES FOR THE DIFF:
1.  The SEARCH block MUST be an exact, contiguous segment from the original '{TARGET_FUNCTION_NAME}' function's body, including its original indentation.
2.  The REPLACE block MUST be correctly indented to be syntactically valid Python code.
3.  You MUST include the '{_SEARCH_MARKER_END}' separator.
4.  Only output the diff block(s) as raw text. No surrounding text or explanations.
5.  Do NOT change the function's signature: `def {TARGET_FUNCTION_NAME}(func, a, b, tolerance=1e-7, max_iterations=100):`.
6.  The function should return the found root as a float, or None/raise an error if a root cannot be found under the conditions.
7.  If no beneficial changes can be made, output the exact string: NO_CHANGES_NECESSARY

The code to modify will be provided after these instructions.
Consider the example of finding the root of `f(x) = {_EXAMPLE_TARGET_EQUATION}` which is approximately {_EXAMPLE_KNOWN_ROOT}.
"""

# --- Evaluation Cascade Definition ---
EVALUATION_CASCADE = [
    {
        "stage_name": "basic_correctness_tests",
        "type": "pytest",
        # Pytest will use TEST_SUITE_FILE by default.
        # We can use markers if we have different sets of tests in that file.
        "pytest_marker": "correctness", # Assumes tests for basic correctness are marked with @pytest.mark.correctness
        "fail_fast_if_not_all_passed": True, # Essential for correctness
        "metric_to_update": "correctness_score" # This stage's pass_ratio updates this metric
    },
    {
        "stage_name": "precision_tests",
        "type": "pytest",
        "pytest_marker": "precision", # Assumes tests for precision are marked with @pytest.mark.precision
        "fail_fast_if_not_all_passed": False, # Don't fail fast, just record precision
        "metric_to_update": "precision_score" # This stage might return a custom precision score
                                              # The Evaluator will need to know how to get this.
                                              # For now, assume test suite handles precision calculation.
    },
    {
        "stage_name": "convergence_speed_tests",
        "type": "pytest",
        "pytest_marker": "convergence", # Assumes tests that return iteration counts
        "fail_fast_if_not_all_passed": False,
        "metric_to_update": "convergence_steps_score"
    },
    # Example of a speed test stage (implementation in Evaluator would be more complex)
    # {
    #     "stage_name": "execution_speed_benchmark",
    #     "type": "benchmark", # Custom type the Evaluator would need to handle
    #     "function_to_benchmark": TARGET_FUNCTION_NAME,
    #     "benchmark_iterations": 1000,
    #     "benchmark_inputs": [ # List of (func_str, a, b, tol) for benchmarking
    #         ("lambda x: x**2 - 2", 1.0, 2.0, 1e-7),
    #         ("lambda x: math.cos(x) - x", 0.0, 1.0, 1e-7)
    #     ],
    #     "metric_to_update": "speed_score"
    # }
]

if __name__ == "__main__":
    print(f"Problem Configuration for: {PROBLEM_NAME}")
    print(f"Initial Code File: {INITIAL_CODE_FILE}")
    print(f"Test Suite File: {TEST_SUITE_FILE}")
    print(f"Target Function Name: {TARGET_FUNCTION_NAME}")
    print("\nEvaluation Metrics:")
    for metric, details in EVALUATION_METRICS.items():
        print(f"  - {metric}: Goal='{details['goal']}', Desc='{details['description']}'")
    
    print("\nEvaluation Cascade:")
    if "EVALUATION_CASCADE" in locals():
        for i, stage in enumerate(EVALUATION_CASCADE):
            print(f"  Stage {i+1}: {stage['stage_name']} (type: {stage['type']})")
            for key, value in stage.items():
                if key != 'stage_name' and key != 'type':
                    print(f"    - {key}: {value}")
    else:
        print("  EVALUATION_CASCADE is not defined.")

    print("\nProblem LLM Instructions (Snippet):")
    print(PROBLEM_LLM_INSTRUCTIONS[:500] + "\n...")
