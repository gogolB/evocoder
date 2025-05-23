# evocoder/problems/numerical_optimizer/initial_code.py

import math
from typing import Callable, Optional

# This is the target function for optimization.
# Its name should match TARGET_FUNCTION_NAME in problem_config.py
def find_root_bisection(
    func: Callable[[float], float], 
    a: float, 
    b: float, 
    tolerance: float = 1e-7, 
    max_iterations: int = 100
) -> Optional[float]:
    """
    Finds a root of the function 'func' within the interval [a, b]
    using the bisection method.

    Args:
        func: The function for which to find a root. It must take a float and return a float.
        a: The start of the interval.
        b: The end of the interval.
        tolerance: The desired precision for the root. The algorithm stops
                   when the interval width (b - a) is less than this tolerance.
        max_iterations: The maximum number of iterations to perform.

    Returns:
        The approximate root as a float if found within max_iterations and 
        if a root is bracketed. Returns None if a root is not bracketed
        or if max_iterations are exceeded before tolerance is met.
    """
    # Ensure 'a' and 'b' are correctly ordered
    if a > b:
        a, b = b, a # Swap them

    fa = func(a)
    fb = func(b)

    if fa * fb >= 0:
        # Root is not bracketed or one of the endpoints is a root.
        # (This basic bisection doesn't explicitly check if fa or fb is zero)
        # print("Warning: Root not bracketed or one endpoint is a root. Bisection may fail or be suboptimal.")
        if abs(fa) < tolerance: return a
        if abs(fb) < tolerance: return b
        return None # Or raise an error, depending on desired behavior for unbracketed roots

    iterations = 0
    while (b - a) / 2.0 > tolerance and iterations < max_iterations:
        midpoint = a + (b - a) / 2.0 # Avoid (a+b)/2 to prevent potential overflow with large a, b
        fm = func(midpoint)

        if abs(fm) < tolerance: # Found a point very close to a root
            return midpoint

        if fa * fm < 0:
            b = midpoint
            fb = fm # Update fb, not strictly necessary for basic bisection but good practice
        else:
            a = midpoint
            fa = fm # Update fa
        
        iterations += 1
        # print(f"Iter {iterations}: a={a}, b={b}, midpoint={midpoint}, fm={fm}") # For debugging

    if iterations >= max_iterations:
        # print(f"Warning: Bisection reached max_iterations ({max_iterations}) without converging to tolerance.")
        # Return the current best estimate
        return a + (b - a) / 2.0 

    # Return the midpoint of the final interval
    return a + (b - a) / 2.0

if __name__ == '__main__':
    # Example functions for testing
    def f1(x: float) -> float:
        return x**2 - 2  # Roots at +/- sqrt(2)

    def f2(x: float) -> float:
        return math.cos(x) - x # Root near 0.739

    def f3(x: float) -> float:
        return x**3 - x - 2 # Root near 1.521
    
    print("--- Testing find_root_bisection directly ---")

    # Test f1(x) = x^2 - 2
    sqrt2 = math.sqrt(2)
    print(f"Actual root of x^2 - 2: {sqrt2}")
    root1 = find_root_bisection(f1, 1, 2, tolerance=1e-7, max_iterations=100)
    print(f"Root of x^2 - 2 in [1, 2]: {root1}, Error: {abs(root1 - sqrt2) if root1 else 'N/A'}")
    
    root1_neg = find_root_bisection(f1, -2, -1, tolerance=1e-7, max_iterations=100)
    print(f"Root of x^2 - 2 in [-2, -1]: {root1_neg}, Error: {abs(root1_neg - (-sqrt2)) if root1_neg else 'N/A'}")

    # Test f2(x) = cos(x) - x
    # Approximate root for f2 is 0.739085
    actual_root_f2 = 0.7390851332151607 
    print(f"\nActual root of cos(x) - x: {actual_root_f2}")
    root2 = find_root_bisection(f2, 0, 1, tolerance=1e-7, max_iterations=100)
    print(f"Root of cos(x) - x in [0, 1]: {root2}, Error: {abs(root2 - actual_root_f2) if root2 else 'N/A'}")

    # Test f3(x) = x^3 - x - 2
    # Approximate root for f3 is 1.5213797
    actual_root_f3 = 1.521379706804568
    print(f"\nActual root of x^3 - x - 2: {actual_root_f3}")
    root3 = find_root_bisection(f3, 1, 2, tolerance=1e-7, max_iterations=100)
    print(f"Root of x^3 - x - 2 in [1, 2]: {root3}, Error: {abs(root3 - actual_root_f3) if root3 else 'N/A'}")

    # Test case where root is not bracketed
    print("\nTesting unbracketed root for x^2 - 2 in [3, 4]:")
    root_unbracketed = find_root_bisection(f1, 3, 4)
    print(f"Result for unbracketed: {root_unbracketed}") # Expected: None

    # Test case where one endpoint is a root
    print("\nTesting endpoint as root for x^2 - 2 with a=sqrt(2):")
    root_at_endpoint = find_root_bisection(f1, math.sqrt(2), 3)
    print(f"Result for endpoint root: {root_at_endpoint}")
    if root_at_endpoint is not None:
        assert abs(root_at_endpoint - math.sqrt(2)) < 1e-7
