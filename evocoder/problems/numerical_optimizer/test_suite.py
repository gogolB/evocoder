# evocoder/problems/numerical_optimizer/test_suite.py

import pytest
import math
import time # For potential speed tests, though not directly used for score here
from typing import Callable, Optional

# The Evaluator will handle making the evolved version of this function available.
# For standalone testing of this test suite (e.g., pytest test_suite.py),
# it will try to import the version from initial_code.py.
try:
    from evocoder.problems.numerical_optimizer.initial_code import find_root_bisection
except ImportError:
    # This allows the file to be parsed, but tests will be skipped if the function isn't found.
    # The Evaluator must ensure the function is in the execution scope for the tests.
    find_root_bisection = None 

# --- Helper Functions for Tests ---

def f_poly1(x: float) -> float:
    # Root at x = 2
    return x**2 - 3 * x + 2  # (x-1)(x-2), so roots at 1 and 2

def f_poly2(x: float) -> float:
    # Root at x = -3
    return x**3 + 2 * x**2 - 5 * x - 6 # (x+1)(x-2)(x+3), roots -3, -1, 2

def f_transcendental1(x: float) -> float:
    # Root near 0.739085 (cos(x) = x)
    return math.cos(x) - x

def f_with_flat_region(x: float) -> float:
    if x > 0.9 and x < 1.1:
        return 0.0000001 # A very flat region, bisection might struggle or take many steps
    return x**2 - 1 # Roots at -1, 1

# --- Test Cases ---

# correctness_score: Based on pass/fail of these tests.
# precision_score: Based on how close the result is to known_root.
# convergence_steps_score: The find_root_bisection could be modified to return steps,
#                          or we can infer it by varying tolerance/max_iterations.
#                          For simplicity, we'll focus on correctness and precision first.
#                          The initial_code.py's find_root_bisection does not return step count.
#                          If we want to score steps, the evolved function should return it.
#                          For now, convergence tests will just check if it finds a root.

# To capture iteration count, the evolved function would need to return it.
# Let's assume for now the evolved function might be modified to return: (root, iterations)
# If not, convergence_steps_score will be hard to measure directly from these tests alone
# without modifying the function signature or using a global counter (which is bad).
# For now, let's assume problem_config's "convergence_steps_score" will be based on
# whether a root is found within max_iterations for difficult cases.

# Helper to call and unpack, assuming find_root_bisection might be evolved
# to return (root, steps_taken) in the future for the "convergence_steps_score".
# For the current initial_code, it only returns Optional[float].
def call_find_root(func, a, b, tol, max_iter):
    if find_root_bisection is None:
        pytest.skip("find_root_bisection not available for testing.")
    
    result = find_root_bisection(func, a, b, tolerance=tol, max_iterations=max_iter)
    
    # For now, since initial_code doesn't return steps, we'll mock steps or ignore it.
    # If the evolved function returns a tuple (root, steps):
    # if isinstance(result, tuple) and len(result) == 2:
    #     return result[0], result[1] 
    return result, -1 # Return -1 for steps if not provided by the function

@pytest.mark.correctness
@pytest.mark.parametrize("func, a, b, expected_root_known, tol, max_iter", [
    (f_poly1, 0.5, 1.5, 1.0, 1e-7, 100),      # Root at 1
    (f_poly1, 1.5, 2.5, 2.0, 1e-7, 100),      # Root at 2
    (f_poly2, -3.5, -2.5, -3.0, 1e-7, 100),   # Root at -3
    (f_poly2, -1.5, -0.5, -1.0, 1e-7, 100),   # Root at -1
    (f_poly2, 1.5, 2.5, 2.0, 1e-7, 100),      # Root at 2
    (f_transcendental1, 0.0, 1.0, 0.73908513, 1e-7, 100), # cos(x) = x
])
def test_basic_correctness(func, a, b, expected_root_known, tol, max_iter):
    """Tests basic root finding for various functions and intervals."""
    found_root, _ = call_find_root(func, a, b, tol, max_iter)
    assert found_root is not None, f"Should find a root for {func.__name__} in [{a}, {b}]"
    assert math.isclose(found_root, expected_root_known, abs_tol=tol * 10), \
        f"Root for {func.__name__} incorrect. Expected ~{expected_root_known}, got {found_root}"

@pytest.mark.correctness
def test_no_root_bracketed(f1=f_poly1): # Use default f1 for this test
    """Tests behavior when the root is not bracketed."""
    if find_root_bisection is None: pytest.skip("Skipping.")
    # f1(x) = x^2 - 3x + 2, roots at 1, 2. Interval [3,4] does not bracket a root.
    found_root, _ = call_find_root(f1, 3, 4, 1e-7, 100)
    assert found_root is None, "Should return None if root is not bracketed"

@pytest.mark.correctness
def test_max_iterations_exceeded(f1=f_poly1):
    """Tests behavior when max_iterations is too low to find the root to tolerance."""
    if find_root_bisection is None: pytest.skip("Skipping.")
    # For f1 in [1.5, 2.5], root is 2.0. With very few iterations, it might not converge.
    # The current find_root_bisection returns the midpoint if max_iter is hit.
    # This test will check if it returns *something* (not None if bracketed).
    # A more specific test would require the function to signal max_iter_exceeded.
    found_root, _ = call_find_root(f1, 1.5, 2.5, 1e-7, 3) # Only 3 iterations
    assert found_root is not None, "Should still return an estimate if max_iterations hit"
    # We don't assert precision here, just that it didn't fail entirely.
    # The precision tests will handle accuracy.

@pytest.mark.precision
@pytest.mark.parametrize("func, a, b, known_root, precision_tolerance", [
    (f_poly1, 1.9, 2.1, 2.0, 1e-7),
    (f_transcendental1, 0.7, 0.8, 0.739085133215, 1e-7),
    (lambda x: x**2 - 2, 1.0, 2.0, math.sqrt(2), 1e-8), # Higher precision demand
])
def test_root_precision(func, a, b, known_root, precision_tolerance):
    """Tests the precision of the found root against a known value."""
    # For precision tests, we might use more max_iterations by default
    found_root, _ = call_find_root(func, a, b, precision_tolerance / 10, 200) # Smaller tol for finding
    assert found_root is not None, "Root not found for precision test"
    error = abs(found_root - known_root)
    print(f"Precision test for {func.__name__ if hasattr(func, '__name__') else 'lambda'}: "
          f"found={found_root}, known={known_root}, error={error}, target_tol={precision_tolerance}")
    assert error < precision_tolerance, f"Precision not met. Error: {error}, Target: {precision_tolerance}"

@pytest.mark.convergence # These tests check if a root is found within reasonable iterations
@pytest.mark.parametrize("func, a, b, max_iter_expected_to_pass", [
    (f_poly1, 0.5, 1.5, 50),      # Root at 1
    (f_transcendental1, 0.0, 1.0, 50), # cos(x) = x
    (f_with_flat_region, 0.5, 1.5, 100) # Root at 1, but flat region might slow it
])
def test_convergence_within_iterations(func, a, b, max_iter_expected_to_pass):
    """Tests if the method converges to a root within a certain number of iterations."""
    # We use a fairly loose tolerance here, focus is on convergence itself.
    found_root, steps = call_find_root(func, a, b, 1e-5, max_iter_expected_to_pass)
    assert found_root is not None, \
        f"Should converge for {func.__name__ if hasattr(func, '__name__') else 'lambda'} " \
        f"in [{a},{b}] within {max_iter_expected_to_pass} iterations."
    # If `steps` was actually returned by an evolved function, we could assert on `steps`.
    # print(f"Convergence test for {func.__name__}: found root in {steps} steps (max_iter={max_iter_expected_to_pass})")

# Example of how the evaluator might interpret results from these tests for scores:
# - correctness_score: ratio of @pytest.mark.correctness tests passed.
# - precision_score: Could be average of -log10(error) from @pytest.mark.precision tests,
#                    or if tests assert error < tolerance, then pass/fail of these tests.
#                    The current problem_config implies it's -(abs error).
#                    The Evaluator needs to be designed to extract this if tests don't directly yield it.
#                    For now, our precision tests assert error < tolerance.
#                    A more advanced test could return the actual error.
# - convergence_steps_score: If the evolved function returns step count, tests could assert
#                            steps < threshold, or return steps for scoring.
