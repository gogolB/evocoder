# evocoder/core/program_database.py

import sqlite3
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# Database file will be created in the project root or a specified data directory.
# For simplicity, let's plan for it to be in a 'data' subdirectory in the project root.
DEFAULT_DB_FILE_NAME = "evocoder_programs.db"
DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" # evocoder/data/

class ProgramDatabase:
    """
    Manages the storage and retrieval of evolved programs and their metadata
    using an SQLite database.
    """

    def __init__(self, db_path: Optional[Path] = None):
        """
        Initializes the ProgramDatabase.

        Args:
            db_path (Optional[Path]): Path to the SQLite database file.
                                      If None, uses a default path in 'project_root/data/'.
        """
        if db_path is None:
            self.db_path = DEFAULT_DATA_DIR / DEFAULT_DB_FILE_NAME
            # Ensure the data directory exists
            DEFAULT_DATA_DIR.mkdir(parents=True, exist_ok=True)
        else:
            self.db_path = db_path
            # If a specific db_path is given, ensure its parent directory exists
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._initialize_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Establishes and returns a database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row # Access columns by name
        return conn

    def _initialize_db(self):
        """
        Creates the necessary tables in the database if they don't already exist.
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS programs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        problem_name TEXT NOT NULL,
                        code_content TEXT NOT NULL,
                        generation INTEGER NOT NULL,
                        parent_id INTEGER,
                        scores TEXT, -- JSON string to store multiple scores
                        created_at REAL DEFAULT (STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')),
                        llm_prompt TEXT, -- Optional: store the prompt that generated this
                        llm_diff TEXT,   -- Optional: store the diff applied if applicable
                        FOREIGN KEY (parent_id) REFERENCES programs (id)
                    )
                """)
                # Add indexes for frequently queried columns
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_programs_generation ON programs (generation)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_programs_problem_name ON programs (problem_name)")
                conn.commit()
        except sqlite3.Error as e:
            print(f"Database error during initialization: {e}")
            raise

    def add_program(
        self,
        problem_name: str,
        code_content: str,
        generation: int,
        scores: Dict[str, float],
        parent_id: Optional[int] = None,
        llm_prompt: Optional[str] = None,
        llm_diff: Optional[str] = None
    ) -> int:
        """
        Adds a new program to the database.

        Args:
            problem_name (str): Identifier for the problem this program belongs to.
            code_content (str): The Python code string of the program.
            generation (int): The generation number in which this program was created.
            scores (Dict[str, float]): A dictionary of scores for the program.
            parent_id (Optional[int]): The ID of the parent program, if any.
            llm_prompt (Optional[str]): The prompt used to generate this program.
            llm_diff (Optional[str]): The diff applied to the parent to get this program.

        Returns:
            int: The ID of the newly inserted program.
        
        Raises:
            sqlite3.Error: If a database error occurs.
        """
        scores_json = json.dumps(scores)
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO programs (problem_name, code_content, generation, parent_id, scores, llm_prompt, llm_diff)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (problem_name, code_content, generation, parent_id, scores_json, llm_prompt, llm_diff))
                conn.commit()
                return cursor.lastrowid if cursor.lastrowid is not None else -1 # Should always return an ID
        except sqlite3.Error as e:
            print(f"Database error in add_program: {e}")
            raise

    def get_program_by_id(self, program_id: int) -> Optional[Dict[str, Any]]:
        """
        Retrieves a program by its ID.

        Args:
            program_id (int): The ID of the program to retrieve.

        Returns:
            Optional[Dict[str, Any]]: A dictionary representing the program row,
                                      or None if not found. Scores are JSON decoded.
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM programs WHERE id = ?", (program_id,))
                row = cursor.fetchone()
                if row:
                    program_dict = dict(row)
                    if program_dict.get("scores"):
                        program_dict["scores"] = json.loads(program_dict["scores"])
                    return program_dict
                return None
        except sqlite3.Error as e:
            print(f"Database error in get_program_by_id: {e}")
            return None
        except json.JSONDecodeError as e:
            print(f"JSON decode error for scores in get_program_by_id (id={program_id}): {e}")
            # Potentially return the row with raw scores or handle differently
            return None


    def get_best_programs(
        self,
        problem_name: str,
        primary_metric: str,
        metric_goal: str = "maximize", # "maximize" or "minimize"
        n: int = 1,
        generation_limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieves the top N programs for a given problem, ordered by a primary metric.

        Args:
            problem_name (str): The problem identifier.
            primary_metric (str): The key for the score in the 'scores' JSON to sort by.
            metric_goal (str): Whether to "maximize" or "minimize" the primary_metric.
            n (int): The number of top programs to retrieve.
            generation_limit (Optional[int]): If set, only consider programs up to this generation.

        Returns:
            List[Dict[str, Any]]: A list of program dictionaries.
        """
        # SQLite's JSON functions can be used for ordering if available and simple.
        # json_extract(scores, '$.metric_key')
        # However, for broader compatibility and potentially complex metrics,
        # it might be simpler to fetch more and sort in Python, or ensure scores are simple.
        # For now, let's use json_extract. This requires SQLite 3.38.0+.
        
        order_direction = "DESC" if metric_goal == "maximize" else "ASC"
        # Ensure the primary_metric key is safe for SQL injection if it were user-provided.
        # Here, it's assumed to be a fixed string from problem_config.
        # For robustness, one might validate primary_metric against a known set of keys.
        json_path = f"'$.{primary_metric}'" # Construct the JSON path

        query = f"""
            SELECT * FROM programs
            WHERE problem_name = ? 
            AND json_valid(scores) AND json_extract(scores, {json_path}) IS NOT NULL
        """
        params: List[Any] = [problem_name]

        if generation_limit is not None:
            query += " AND generation <= ?"
            params.append(generation_limit)
        
        query += f" ORDER BY CAST(json_extract(scores, {json_path}) AS REAL) {order_direction} LIMIT ?"
        params.append(n)

        programs = []
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, tuple(params))
                for row in cursor.fetchall():
                    program_dict = dict(row)
                    if program_dict.get("scores"):
                        program_dict["scores"] = json.loads(program_dict["scores"])
                    programs.append(program_dict)
            return programs
        except sqlite3.Error as e:
            # This might fail if json_extract is not available or metric key is bad
            print(f"Database error in get_best_programs (possibly json_extract issue or invalid metric '{primary_metric}'): {e}")
            print("Falling back to fetching all and sorting in Python for this query if it was a json_extract issue.")
            # Fallback: Fetch all for the problem and sort in Python (less efficient)
            return self._get_best_programs_fallback(problem_name, primary_metric, metric_goal, n, generation_limit)


    def _get_best_programs_fallback(
        self,
        problem_name: str,
        primary_metric: str,
        metric_goal: str = "maximize",
        n: int = 1,
        generation_limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Fallback for get_best_programs if json_extract fails or is not available."""
        all_programs_for_problem = []
        query = "SELECT * FROM programs WHERE problem_name = ?"
        params: List[Any] = [problem_name]
        if generation_limit is not None:
            query += " AND generation <= ?"
            params.append(generation_limit)

        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, tuple(params))
                for row in cursor.fetchall():
                    program_dict = dict(row)
                    try:
                        if program_dict.get("scores"):
                            program_dict["scores"] = json.loads(program_dict["scores"])
                        # Only consider programs that have the primary_metric in their scores
                        if isinstance(program_dict["scores"], dict) and primary_metric in program_dict["scores"]:
                             all_programs_for_problem.append(program_dict)
                    except json.JSONDecodeError:
                        continue # Skip if scores are malformed
            
            # Sort in Python
            reverse_sort = True if metric_goal == "maximize" else False
            all_programs_for_problem.sort(
                key=lambda p: p["scores"].get(primary_metric, float('-inf') if metric_goal == "maximize" else float('inf')),
                reverse=reverse_sort
            )
            return all_programs_for_problem[:n]
        except sqlite3.Error as e:
            print(f"Database error in _get_best_programs_fallback: {e}")
            return []


    def get_programs_by_generation(self, problem_name: str, generation: int) -> List[Dict[str, Any]]:
        """Retrieves all programs from a specific generation for a given problem."""
        programs = []
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT * FROM programs WHERE problem_name = ? AND generation = ?",
                    (problem_name, generation)
                )
                for row in cursor.fetchall():
                    program_dict = dict(row)
                    if program_dict.get("scores"):
                        program_dict["scores"] = json.loads(program_dict["scores"])
                    programs.append(program_dict)
            return programs
        except sqlite3.Error as e:
            print(f"Database error in get_programs_by_generation: {e}")
            return []
        except json.JSONDecodeError as e:
            print(f"JSON decode error for scores in get_programs_by_generation: {e}")
            return []

    def get_all_programs(self, problem_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieves all programs, optionally filtered by problem_name."""
        programs = []
        query = "SELECT * FROM programs"
        params = []
        if problem_name:
            query += " WHERE problem_name = ?"
            params.append(problem_name)
        
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, tuple(params))
                for row in cursor.fetchall():
                    program_dict = dict(row)
                    if program_dict.get("scores"):
                        program_dict["scores"] = json.loads(program_dict["scores"])
                    programs.append(program_dict)
            return programs
        except sqlite3.Error as e:
            print(f"Database error in get_all_programs: {e}")
            return []
        except json.JSONDecodeError as e:
            print(f"JSON decode error for scores in get_all_programs: {e}")
            return []


if __name__ == '__main__':
    # Standalone test for ProgramDatabase
    # This will create/use 'evocoder_programs.db' in a 'data' subdirectory
    # relative to where this script is located (evocoder/core/data/ if run from core)
    # or (evocoder/data/ if run from project root after path adjustment)
    
    import sys
    # Adjust path to run from project root for imports if necessary
    current_dir_test = Path(__file__).resolve().parent # evocoder/core
    package_root_test = current_dir_test.parent # evocoder/evocoder
    project_root_dir_test = package_root_test.parent # evocoder/
    if str(project_root_dir_test) not in sys.path:
        sys.path.insert(0, str(project_root_dir_test))

    # Ensure the data directory exists before creating the DB object for the test
    DEFAULT_DATA_DIR.mkdir(parents=True, exist_ok=True) # Ensures evocoder/data/ exists

    # Create a temporary DB for testing or use the default
    test_db_path = DEFAULT_DATA_DIR / "test_program_db.sqlite3"
    if test_db_path.exists():
        test_db_path.unlink() # Clear previous test DB

    print(f"Using test database: {test_db_path}")
    db = ProgramDatabase(db_path=test_db_path) # This will now ensure test_db_path.parent exists

    print("\n--- Testing ProgramDatabase ---")

    # Test add_program
    print("\nAdding programs...")
    prog1_id = db.add_program("line_reducer", "def f1():\n  return 1", 0, {"correctness": 1.0, "lines": -2.0})
    print(f"Added program 1 with ID: {prog1_id}")
    
    prog2_id = db.add_program("line_reducer", "def f2():\n  x=1\n  return x", 1, {"correctness": 1.0, "lines": -3.0}, parent_id=prog1_id)
    print(f"Added program 2 with ID: {prog2_id}")

    prog3_id = db.add_program("line_reducer", "def f3():\n  # best\n  return 0", 1, {"correctness": 1.0, "lines": -1.0}, parent_id=prog1_id)
    print(f"Added program 3 with ID: {prog3_id}")
    
    prog4_id = db.add_program("line_reducer", "def f4():\n  pass # incorrect", 1, {"correctness": 0.0, "lines": -1.0}, parent_id=prog2_id)
    print(f"Added program 4 with ID: {prog4_id}")

    prog5_id = db.add_program("another_problem", "content for another problem", 0, {"metric_a": 100.0})
    print(f"Added program 5 with ID: {prog5_id}")


    # Test get_program_by_id
    print("\nGetting program by ID...")
    retrieved_prog2 = db.get_program_by_id(prog2_id)
    if retrieved_prog2:
        print(f"Retrieved program ID {prog2_id}: Gen={retrieved_prog2['generation']}, Scores={retrieved_prog2['scores']}")
    else:
        print(f"Failed to retrieve program ID {prog2_id}")

    # Test get_best_programs
    print("\nGetting best programs for 'line_reducer' (by 'lines', maximize score i.e. minimize actual lines):")
    best_line_reducer_progs = db.get_best_programs(
        problem_name="line_reducer",
        primary_metric="lines", # This is the key in the scores dict
        metric_goal="maximize", # We stored negative line count, so maximize this score
        n=2
    )
    for i, prog in enumerate(best_line_reducer_progs):
        print(f"  Best {i+1}: ID={prog['id']}, Gen={prog['generation']}, Lines Score={prog['scores']['lines']}, Code:\n{prog['code_content'][:30]}...")

    print("\nGetting best programs for 'line_reducer' (by 'correctness', maximize):")
    best_correctness_progs = db.get_best_programs(
        problem_name="line_reducer",
        primary_metric="correctness",
        metric_goal="maximize",
        n=2
    )
    for i, prog in enumerate(best_correctness_progs):
        print(f"  Best {i+1}: ID={prog['id']}, Gen={prog['generation']}, Correctness={prog['scores']['correctness']}")


    # Test get_programs_by_generation
    print("\nGetting programs from generation 1 for 'line_reducer':")
    gen1_programs = db.get_programs_by_generation("line_reducer", 1)
    for prog in gen1_programs:
        print(f"  ID={prog['id']}, Scores={prog['scores']}")
    
    # Test get_all_programs
    print("\nGetting all programs for 'line_reducer':")
    all_lr_programs = db.get_all_programs("line_reducer")
    print(f"Found {len(all_lr_programs)} programs for line_reducer.")

    print("\nGetting all programs (all problems):")
    all_programs = db.get_all_programs()
    print(f"Found {len(all_programs)} total programs in DB.")

    print("\n--- Test Finished ---")
    # Clean up the test database file
    if test_db_path.exists():
        test_db_path.unlink()
        print(f"Cleaned up test database: {test_db_path}")
