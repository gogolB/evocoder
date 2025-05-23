# evocoder/utils/diff_utils.py

import re
from typing import List, Tuple, Optional

# Define the markers for the diff format
SEARCH_MARKER_START = "<<<<<<<< SEARCH"
SEARCH_MARKER_END = "========"
REPLACE_MARKER_END = ">>>>>>>>> REPLACE"

class DiffBlock:
    """Represents a single SEARCH/REPLACE block."""
    def __init__(self, search_lines: List[str], replace_lines: List[str]):
        # Normalize by removing trailing newlines from each line for consistent matching,
        # but preserve empty lines within the blocks.
        self.search_block: str = "".join(search_lines) # Keep as multiline string
        self.replace_block: str = "".join(replace_lines) # Keep as multiline string

    def __repr__(self) -> str:
        return (f"DiffBlock(search_block=\"{self.search_block[:30]}...\", "
                f"replace_block=\"{self.replace_block[:30]}...\")")

def parse_diff_string(diff_string: str) -> List[DiffBlock]:
    """
    Parses a string containing one or more SEARCH/REPLACE diff blocks.

    Args:
        diff_string (str): The string containing the diff blocks.

    Returns:
        List[DiffBlock]: A list of DiffBlock objects.
                         Returns an empty list if no valid blocks are found or parsing fails.
    """
    diff_blocks: List[DiffBlock] = []
    lines = diff_string.splitlines(keepends=True) # Keep newlines for reconstruction
    
    i = 0
    while i < len(lines):
        line = lines[i].rstrip('\n') # rstrip for marker matching

        if line == SEARCH_MARKER_START:
            search_lines: List[str] = []
            replace_lines: List[str] = []
            
            i += 1 # Move to lines after SEARCH_MARKER_START
            # Collect search lines
            while i < len(lines) and lines[i].rstrip('\n') != SEARCH_MARKER_END:
                search_lines.append(lines[i])
                i += 1
            
            if i < len(lines) and lines[i].rstrip('\n') == SEARCH_MARKER_END:
                i += 1 # Move to lines after SEARCH_MARKER_END
                # Collect replace lines
                while i < len(lines) and lines[i].rstrip('\n') != REPLACE_MARKER_END:
                    replace_lines.append(lines[i])
                    i += 1
                
                if i < len(lines) and lines[i].rstrip('\n') == REPLACE_MARKER_END:
                    # Successfully parsed a complete block
                    if search_lines: # Only add if search block is not empty
                        diff_blocks.append(DiffBlock(search_lines, replace_lines))
                    else:
                        # Handle case where search block is empty (might be an LLM error)
                        print("Warning: Parsed a diff block with an empty SEARCH section. Ignoring.")
                else:
                    # Malformed block: REPLACE_MARKER_END not found
                    # print(f"Warning: Malformed diff block. Missing '{REPLACE_MARKER_END}'. Started at line with '{SEARCH_MARKER_START}'.")
                    pass 
            else:
                # Malformed block: SEARCH_MARKER_END not found
                # print(f"Warning: Malformed diff block. Missing '{SEARCH_MARKER_END}'. Started at line with '{SEARCH_MARKER_START}'.")
                pass 
        i += 1
        
    return diff_blocks

def apply_diffs(original_code: str, diff_blocks: List[DiffBlock]) -> str:
    """
    Applies a list of DiffBlock objects to the original code.
    Diffs are applied sequentially. If a search block is not found,
    that specific diff is skipped.

    Args:
        original_code (str): The original code string.
        diff_blocks (List[DiffBlock]): A list of DiffBlock objects to apply.

    Returns:
        str: The modified code string after applying all found diffs.
    """
    modified_code = original_code
    for i, block in enumerate(diff_blocks):
        if not block.search_block.strip() and block.search_block: 
             print(f"Warning: Diff block {i+1} has a search pattern that is only whitespace. Skipping.")
             continue
        if not block.search_block: 
            print(f"Warning: Diff block {i+1} has an empty search pattern. Skipping.")
            continue

        # str.replace() returns the new string. It does not return a count of replacements.
        # To check if a replacement happened when count=1, we can see if the string changed.
        # However, it's possible the search and replace blocks are identical, leading to no change.
        # A more direct way to check if the search_block was found is to see if it's in the string.
        
        if block.search_block in modified_code:
            # Apply the replacement, count=1 replaces only the first occurrence.
            new_code = modified_code.replace(block.search_block, block.replace_block, 1)
            if new_code != modified_code:
                # print(f"Applied diff block {i+1}.")
                pass
            else:
                # This case means search_block was found, but search_block == replace_block
                # print(f"Diff block {i+1}: search and replace blocks are identical. No change made to code string.")
                pass
            modified_code = new_code
        else:
            print(f"Warning: Search block for diff {i+1} not found in the current code state. Skipping this diff.")
            # print(f"Search block was:\n---\n{block.search_block}\n---")
            # print(f"Current code state (first 200 chars):\n---\n{modified_code[:200]}...\n---")

    return modified_code

if __name__ == '__main__':
    print("--- Testing diff_utils.py ---")

    sample_original_code = """
def old_function_name(param1, param2):
    # This is some old logic
    x = param1 + param2
    print("Calculating sum...") # Old print
    y = x * 2
    # Another comment
    return y

def another_function():
    return "unchanged"
"""

    # Test case 1: Simple replacement
    diff_string_1 = f"""
{SEARCH_MARKER_START}
    # This is some old logic
    x = param1 + param2
    print("Calculating sum...") # Old print
{SEARCH_MARKER_END}
    # This is new, improved logic
    x = param1 + param2  # Summation
    print("Sum calculated.") # New print
{REPLACE_MARKER_END}
"""
    print("\n--- Test Case 1: Simple Replacement ---")
    blocks1 = parse_diff_string(diff_string_1)
    print(f"Parsed blocks: {blocks1}")
    if blocks1:
        modified_code_1 = apply_diffs(sample_original_code, blocks1)
        print("Original Code:\n", sample_original_code)
        print("Modified Code 1:\n", modified_code_1)
        assert "This is new, improved logic" in modified_code_1
        assert "Old print" not in modified_code_1

    # Test case 2: Replacing a function name and its body
    diff_string_2 = f"""
{SEARCH_MARKER_START}
def old_function_name(param1, param2):
    # This is some old logic
    x = param1 + param2
    print("Calculating sum...") # Old print
    y = x * 2
    # Another comment
    return y
{SEARCH_MARKER_END}
def new_function_name(p1, p2):
    # Entirely new body
    return p1 * p2
{REPLACE_MARKER_END}
"""
    print("\n--- Test Case 2: Replace entire function ---")
    blocks2 = parse_diff_string(diff_string_2)
    print(f"Parsed blocks: {blocks2}")
    if blocks2:
        modified_code_2 = apply_diffs(sample_original_code, blocks2)
        print("Original Code:\n", sample_original_code)
        print("Modified Code 2:\n", modified_code_2)
        assert "new_function_name" in modified_code_2
        assert "old_function_name" not in modified_code_2
        assert "Entirely new body" in modified_code_2

    # Test case 3: Deletion (replace with empty)
    diff_string_3 = f"""
{SEARCH_MARKER_START}
    print("Calculating sum...") # Old print
{SEARCH_MARKER_END}
{REPLACE_MARKER_END}
"""
    print("\n--- Test Case 3: Deletion ---")
    blocks3 = parse_diff_string(diff_string_3)
    print(f"Parsed blocks: {blocks3}")
    if blocks3:
        temp_original_for_delete = """
def old_function_name(param1, param2):
    # This is some old logic
    x = param1 + param2
    print("Calculating sum...") # Old print
    y = x * 2
    # Another comment
    return y

def another_function():
    return "unchanged"
"""
        modified_code_3 = apply_diffs(temp_original_for_delete, blocks3)
        print("Original for delete:\n", temp_original_for_delete)
        print("Modified Code 3 (after deletion):\n", modified_code_3)
        assert 'print("Calculating sum...") # Old print' not in modified_code_3

    # Test case 4: Multiple diff blocks
    diff_string_4 = f"""
{SEARCH_MARKER_START}
    # This is some old logic
{SEARCH_MARKER_END}
    # This is the first new logic part
{REPLACE_MARKER_END}

Some other text between diffs that should be ignored by parser.

{SEARCH_MARKER_START}
    # Another comment
{SEARCH_MARKER_END}
    # Second new logic part, replacing the comment
{REPLACE_MARKER_END}
"""
    print("\n--- Test Case 4: Multiple Diff Blocks ---")
    blocks4 = parse_diff_string(diff_string_4)
    print(f"Parsed blocks: {blocks4}")
    assert len(blocks4) == 2
    if len(blocks4) == 2:
        modified_code_4 = apply_diffs(sample_original_code, blocks4)
        print("Original Code:\n", sample_original_code)
        print("Modified Code 4:\n", modified_code_4)
        assert "This is the first new logic part" in modified_code_4
        assert "Second new logic part" in modified_code_4
        assert "# This is some old logic" not in modified_code_4
        assert "# Another comment" not in modified_code_4

    # Test case 5: Search block not found
    diff_string_5 = f"""
{SEARCH_MARKER_START}
THIS_TEXT_DOES_NOT_EXIST_IN_ORIGINAL
{SEARCH_MARKER_END}
SHOULD_NOT_BE_APPLIED
{REPLACE_MARKER_END}
"""
    print("\n--- Test Case 5: Search Block Not Found ---")
    blocks5 = parse_diff_string(diff_string_5)
    print(f"Parsed blocks: {blocks5}")
    if blocks5:
        modified_code_5 = apply_diffs(sample_original_code, blocks5)
        print("Original Code:\n", sample_original_code)
        print("Modified Code 5:\n", modified_code_5)
        assert modified_code_5 == sample_original_code # Code should be unchanged

    # Test case 6: Malformed diff (missing end replace marker)
    diff_string_6_malformed = f"""
{SEARCH_MARKER_START}
def old_function_name(param1, param2):
{SEARCH_MARKER_END}
def new_function_name(p1, p2):
# Missing REPLACE_MARKER_END
"""
    print("\n--- Test Case 6: Malformed Diff (Missing REPLACE_MARKER_END) ---")
    blocks6 = parse_diff_string(diff_string_6_malformed)
    print(f"Parsed blocks (should be empty or fewer): {blocks6}")
    assert len(blocks6) == 0 # Expecting no valid block

    # Test case 7: Empty search block (should be ignored by parser or apply_diffs)
    diff_string_7_empty_search = f"""
{SEARCH_MARKER_START}
{SEARCH_MARKER_END}
this should not be applied
{REPLACE_MARKER_END}
"""
    print("\n--- Test Case 7: Empty Search Block ---")
    blocks7 = parse_diff_string(diff_string_7_empty_search)
    print(f"Parsed blocks (should be empty): {blocks7}")
    assert len(blocks7) == 0
    if blocks7: # Should not happen
        modified_code_7 = apply_diffs(sample_original_code, blocks7)
        assert modified_code_7 == sample_original_code

    print("\nDiff utils tests finished.")

