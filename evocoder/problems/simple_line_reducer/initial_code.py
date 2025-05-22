# evocoder/problems/simple_line_reducer/initial_code.py

# This is the target function for optimization.
# The goal is to reduce its line count while maintaining its exact behavior.
# The name of this function should match TARGET_FUNCTION_NAME in problem_config.py
def target_function_to_optimize(a: int, b: int, c: int) -> int:
    """
    A simple function with some deliberate verbosity for the LLM to optimize.
    It calculates: (a + b) * c if a is positive, otherwise (a - b) * c.
    """
    intermediate_sum: int
    
    # Condition based on 'a'
    if a > 0:
        # Calculate sum if a is positive
        intermediate_sum = a + b
        # This is an extra comment line
        # Another one
    else:
        # Calculate difference if a is not positive
        intermediate_sum = a - b
        # Yet another comment
    
    # Perform multiplication
    result: int
    result = intermediate_sum * c
    
    # Add some more lines for the sake of having more lines
    temp_variable_for_no_reason = result + 0 
    another_temp = temp_variable_for_no_reason - 0

    # Final print for no reason other than adding lines (will be removed by optimizer)
    # print(f"Intermediate sum was: {intermediate_sum}, result is: {another_temp}")

    # Return the final result
    return another_temp

if __name__ == '__main__':
    # Example usage (not part of the optimization target, just for testing the file)
    print(f"Test 1 (a > 0): target_function_to_optimize(5, 3, 2) = {target_function_to_optimize(5, 3, 2)}") # Expected: (5+3)*2 = 16
    print(f"Test 2 (a <= 0): target_function_to_optimize(-5, 3, 2) = {target_function_to_optimize(-5, 3, 2)}") # Expected: (-5-3)*2 = -16
    print(f"Test 3 (a = 0): target_function_to_optimize(0, 10, 5) = {target_function_to_optimize(0, 10, 5)}") # Expected: (0-10)*5 = -50

