# evocoder/problems/simple_line_reducer/problem_config.py

from pathlib import Path

# Get the directory of the current file (problem_config.py)
PROBLEM_DIR = Path(__file__).resolve().parent

PROBLEM_NAME = "simple_line_reducer"

# --- Problem Specific Files ---
INITIAL_CODE_FILE: Path = PROBLEM_DIR / "initial_code.py"
TEST_SUITE_FILE: Path = PROBLEM_DIR / "test_suite.py"

# --- Target for Evolution ---
TARGET_FUNCTION_NAME: str = "target_function_to_optimize"


# --- Evaluation Metrics Configuration ---
EVALUATION_METRICS = {
    "correctness_score": { 
        "goal": "maximize",
        "description": "Fraction of unit tests passed. 1.0 means all pass."
    },
    "line_count_score": { 
        "goal": "maximize", 
        "description": "Negative of the number of lines in the target function. Higher is better (fewer lines)."
    }
}
PRIMARY_METRIC: str = "line_count_score"
CORRECTNESS_THRESHOLD: float = 1.0 

# --- LLM Prompting Instructions (Specific to this problem - UPDATED FOR DIFFS) ---
# Define the markers for the diff format (can be imported from diff_utils if preferred globally)
_SEARCH_MARKER_START = "<<<<<<<< SEARCH"
_SEARCH_MARKER_END = "========"
_REPLACE_MARKER_END = ">>>>>>>>> REPLACE"

# Example of a very simple diff to include in the instructions

_EXAMPLE_FUNCTION_CONTEXT_START = f"""
def {TARGET_FUNCTION_NAME}(param1, param2):
# Some initial lines
line_to_keep_before = "this line stays"
"""

# Note: The example search/replace now includes the indentation

_EXAMPLE_ORIGINAL_CODE_SNIPPET = """    # This is a comment to be replaced
x = param1 + param2 # old calculation"""
_EXAMPLE_NEW_CODE_SNIPPET = """    x = param1 * param2 # new calculation with correct indentation"""
_EXAMPLE_FUNCTION_CONTEXT_END = """
line_to_keep_after = "this line also stays"
return x
"""

PROBLEM_LLM_INSTRUCTIONS = f"""
You are an expert Python code optimizer. Your task is to refactor ONLY the Python function named '{TARGET_FUNCTION_NAME}' within the provided code.
The primary goal is to REDUCE THE LINE COUNT of this specific function while ensuring its external behavior remains EXACTLY the same and it passes all unit tests.

You MUST provide your changes STRICTLY in the following diff format using one or more blocks.
It is CRITICAL that you use the exact markers including the '========' separator.

THE REQUIRED DIFF FORMAT IS:
{_SEARCH_MARKER_START}
[EXACT ORIGINAL CODE SEGMENT TO BE REPLACED FROM THE '{TARGET_FUNCTION_NAME}' FUNCTION. THIS MUST BE A CONTIGUOUS BLOCK OF LINES.]
{_SEARCH_MARKER_END}
[NEW CODE SEGMENT TO REPLACE THE ORIGINAL. THIS CAN BE EMPTY TO DELETE THE SEARCH BLOCK. PRESERVE RELEVANT INDENTATION.]
{_REPLACE_MARKER_END}

For example, if the original code for '{TARGET_FUNCTION_NAME}' looked like this:

```python
{_EXAMPLE_FUNCTION_CONTEXT_START.strip()}
{_EXAMPLE_ORIGINAL_CODE_SNIPPET.strip()}
{_EXAMPLE_FUNCTION_CONTEXT_END.strip()}
```

And you wanted to change only the part indicated, your complete and only output would be:
{_SEARCH_MARKER_START}
{_EXAMPLE_ORIGINAL_CODE_SNIPPET.strip()}
{_SEARCH_MARKER_END}
{_EXAMPLE_NEW_CODE_SNIPPET.strip()}
{_REPLACE_MARKER_END}

Notice how the example NEW CODE SEGMENT maintains the same base indentation as the original lines it replaced.

IMPORTANT RULES:

The SEARCH block MUST be an exact, contiguous segment from the original '{TARGET_FUNCTION_NAME}' function's body.
You MUST include the '{_SEARCH_MARKER_END}' separator between the SEARCH and REPLACE blocks.
Only output the diff block(s) as raw text. Do NOT include any other text, explanations, or code markdown fences (like ```python) before or after the diff blocks.
If multiple distinct, non-overlapping changes are needed, provide multiple complete diff blocks sequentially, each with its own START, SEPARATOR, and END markers.
Focus ONLY on reducing line count within '{TARGET_FUNCTION_NAME}'.
Do NOT change the function's signature (name, parameters, return type hints) unless that is the specific optimization required.
If no beneficial changes can be made to reduce line count while maintaining correctness, you MUST output the exact string: NO_CHANGES_NECESSARY
- The code to modify will be provided after these instructions.
"""

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
