# evocoder/core/program_database.py

import sqlite3
import json
import random # For sampling random programs
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

try:
    from ..utils.logger import setup_logger
except ImportError:
    if __name__ == '__main__':
        import sys
        file_path = Path(__file__).resolve()
        project_root = file_path.parent.parent.parent 
        sys.path.insert(0, str(project_root))
        from evocoder.utils.logger import setup_logger
    else:
        raise

DEFAULT_DB_FILE_NAME = "evocoder_programs.db"
DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" 

class ProgramDatabase:
    def __init__(self, db_path: Optional[Path] = None):
        self.logger = setup_logger(f"evocoder.database.{self.__class__.__name__}")
        self.logger.info("Initializing ProgramDatabase...")

        if db_path is None:
            self.db_path = DEFAULT_DATA_DIR / DEFAULT_DB_FILE_NAME
            self.logger.debug(f"No db_path provided, using default: {self.db_path}")
            DEFAULT_DATA_DIR.mkdir(parents=True, exist_ok=True)
        else:
            self.db_path = db_path
            self.logger.debug(f"Using provided db_path: {self.db_path}")
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Database file path set to: {self.db_path}")
        self._initialize_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10) 
        conn.row_factory = sqlite3.Row 
        return conn

    def _initialize_db(self):
        self.logger.info("Initializing database tables if they don't exist...")
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
                        scores TEXT, 
                        created_at REAL DEFAULT (STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')),
                        llm_prompt TEXT, 
                        llm_diff TEXT,   
                        FOREIGN KEY (parent_id) REFERENCES programs (id)
                    )
                """)
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_programs_generation ON programs (generation)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_programs_problem_name ON programs (problem_name)")
                conn.commit()
                self.logger.info("Database tables initialized successfully.")
        except sqlite3.Error as e:
            self.logger.exception(f"Database error during initialization: {e}")
            raise

    def _deserialize_row(self, row: sqlite3.Row) -> Optional[Dict[str, Any]]:
        """Helper to deserialize a database row into a dictionary with JSON scores."""
        if not row:
            return None
        program_dict = dict(row)
        if program_dict.get("scores"):
            try:
                program_dict["scores"] = json.loads(program_dict["scores"])
            except json.JSONDecodeError as je:
                self.logger.error(f"JSON decode error for scores in program ID {program_dict.get('id')}: {je}. Scores: '{program_dict['scores']}'")
                program_dict["scores"] = {} 
        return program_dict

    def add_program(
        self, problem_name: str, code_content: str, generation: int,
        scores: Dict[str, float], parent_id: Optional[int] = None,
        llm_prompt: Optional[str] = None, llm_diff: Optional[str] = None
    ) -> int:
        self.logger.debug(f"Adding program for problem '{problem_name}', gen {generation}, parent_id {parent_id}.")
        scores_json = json.dumps(scores)
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO programs (problem_name, code_content, generation, parent_id, scores, llm_prompt, llm_diff)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (problem_name, code_content, generation, parent_id, scores_json, llm_prompt, llm_diff))
                conn.commit()
                new_id = cursor.lastrowid if cursor.lastrowid is not None else -1
                self.logger.info(f"Added program ID: {new_id} for '{problem_name}'. Scores: {scores}")
                return new_id
        except sqlite3.Error as e:
            self.logger.exception(f"Database error in add_program for '{problem_name}': {e}")
            raise

    def get_program_by_id(self, program_id: int) -> Optional[Dict[str, Any]]:
        self.logger.debug(f"Retrieving program by ID: {program_id}")
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM programs WHERE id = ?", (program_id,))
                return self._deserialize_row(cursor.fetchone())
        except sqlite3.Error as e:
            self.logger.exception(f"Database error in get_program_by_id for ID {program_id}: {e}")
            return None

    def get_best_programs(
        self, problem_name: str, primary_metric: str, metric_goal: str = "maximize", 
        n: int = 1, generation_limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        self.logger.debug(f"Getting best {n} programs for '{problem_name}' by metric '{primary_metric}' ({metric_goal}), gen limit: {generation_limit}")
        order_direction = "DESC" if metric_goal == "maximize" else "ASC"
        json_path = f"'$.{primary_metric}'" 
        query = f"""
            SELECT * FROM programs
            WHERE problem_name = ? AND json_valid(scores) AND json_extract(scores, {json_path}) IS NOT NULL
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
                for row in cursor.execute(query, tuple(params)):
                    deserialized = self._deserialize_row(row)
                    if deserialized: programs.append(deserialized)
            self.logger.debug(f"Found {len(programs)} best programs using json_extract.")
            return programs
        except sqlite3.Error as e:
            self.logger.warning(f"DB error in get_best_programs (json_extract issue or metric '{primary_metric}'): {e}. Falling back.")
            return self._get_best_programs_fallback(problem_name, primary_metric, metric_goal, n, generation_limit)

    def _get_best_programs_fallback(
        self, problem_name: str, primary_metric: str, metric_goal: str = "maximize",
        n: int = 1, generation_limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        self.logger.debug(f"Fallback: Getting best {n} for '{problem_name}' by '{primary_metric}' ({metric_goal}) via Python sort.")
        all_progs = self.get_all_programs_with_filters(problem_name=problem_name, generation_limit=generation_limit)
        valid_progs = [p for p in all_progs if isinstance(p.get("scores"), dict) and primary_metric in p["scores"]]
        
        reverse_sort = True if metric_goal == "maximize" else False
        default_sort_val = float('-inf') if metric_goal == "maximize" else float('inf')
        valid_progs.sort(key=lambda p: p["scores"].get(primary_metric, default_sort_val), reverse=reverse_sort)
        self.logger.debug(f"Fallback: Found {len(valid_progs)} valid programs, returning top {n}.")
        return valid_progs[:n]

    def get_random_correct_programs(
        self, problem_name: str, correctness_metric: str, 
        correctness_threshold: float, n: int, 
        exclude_ids: Optional[List[int]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieves N random programs for a given problem that meet the correctness threshold.
        """
        self.logger.debug(f"Getting {n} random correct programs for '{problem_name}', excluding IDs: {exclude_ids}")
        
        json_path_correctness = f"'$.{correctness_metric}'"
        
        query = f"""
            SELECT * FROM programs
            WHERE problem_name = ? 
            AND json_valid(scores) 
            AND CAST(json_extract(scores, {json_path_correctness}) AS REAL) >= ?
        """
        params: List[Any] = [problem_name, correctness_threshold]

        if exclude_ids:
            placeholders = ', '.join('?' for _ in exclude_ids)
            query += f" AND id NOT IN ({placeholders})"
            params.extend(exclude_ids)
        
        query += " ORDER BY RANDOM() LIMIT ?" 
        params.append(n)

        programs = []
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                for row in cursor.execute(query, tuple(params)):
                    deserialized = self._deserialize_row(row)
                    if deserialized: programs.append(deserialized)
            self.logger.debug(f"Found {len(programs)} random correct programs using json_extract and RANDOM().")
            return programs
        except sqlite3.Error as e:
            self.logger.warning(f"DB error in get_random_correct_programs (json_extract or RANDOM() issue): {e}. Falling back.")
            return self._get_random_correct_programs_fallback(problem_name, correctness_metric, correctness_threshold, n, exclude_ids)

    def _get_random_correct_programs_fallback(
        self, problem_name: str, correctness_metric: str, 
        correctness_threshold: float, n: int, 
        exclude_ids: Optional[List[int]] = None
    ) -> List[Dict[str, Any]]:
        self.logger.debug(f"Fallback: Getting {n} random correct programs for '{problem_name}' via Python filter.")
        all_progs = self.get_all_programs_with_filters(problem_name=problem_name)
        
        correct_programs = []
        for p in all_progs:
            if exclude_ids and p.get("id") in exclude_ids:
                continue
            scores = p.get("scores", {})
            if isinstance(scores, dict) and scores.get(correctness_metric, 0.0) >= correctness_threshold:
                correct_programs.append(p)
        
        if len(correct_programs) <= n:
            self.logger.debug(f"Fallback: Found {len(correct_programs)} correct programs, returning all.")
            return correct_programs
        else:
            selected = random.sample(correct_programs, n)
            self.logger.debug(f"Fallback: Sampled {len(selected)} random correct programs.")
            return selected

    def get_programs_by_generation(self, problem_name: str, generation: int) -> List[Dict[str, Any]]:
        self.logger.debug(f"Getting programs for '{problem_name}', generation {generation}.")
        programs = []
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                for row in cursor.execute(
                    "SELECT * FROM programs WHERE problem_name = ? AND generation = ?",
                    (problem_name, generation)
                ):
                    deserialized = self._deserialize_row(row)
                    if deserialized: programs.append(deserialized)
            self.logger.debug(f"Found {len(programs)} programs for gen {generation}.")
            return programs
        except sqlite3.Error as e:
            self.logger.exception(f"Database error in get_programs_by_generation: {e}")
            return []

    def get_all_programs_with_filters(
        self, 
        problem_name: Optional[str] = None,
        generation_limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Helper to get all programs with optional filters, used by fallbacks."""
        log_msg = "Getting all programs with filters:"
        if problem_name: log_msg += f" problem='{problem_name}'"
        if generation_limit is not None: log_msg += f" gen_limit<={generation_limit}"
        self.logger.debug(log_msg)

        programs = []
        query = "SELECT * FROM programs"
        conditions: List[str] = []
        params: List[Any] = []

        if problem_name:
            conditions.append("problem_name = ?")
            params.append(problem_name)
        if generation_limit is not None:
            conditions.append("generation <= ?")
            params.append(generation_limit)
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                for row in cursor.execute(query, tuple(params)):
                    deserialized = self._deserialize_row(row)
                    if deserialized: programs.append(deserialized)
            self.logger.debug(f"Found {len(programs)} programs for query: {query} with params {params}")
            return programs
        except sqlite3.Error as e:
            self.logger.exception(f"Database error in get_all_programs_with_filters: {e}")
            return []


if __name__ == '__main__':
    test_logger = setup_logger("evocoder.test_program_database", level_str="DEBUG") 
    
    current_dir_test = Path(__file__).resolve().parent 
    package_root_test = current_dir_test.parent 
    project_root_dir_test = package_root_test.parent 
    if str(project_root_dir_test) not in sys.path:
        sys.path.insert(0, str(project_root_dir_test))

    (project_root_dir_test / "data").mkdir(parents=True, exist_ok=True)
    test_db_path = DEFAULT_DATA_DIR / "test_program_db_logging_v2.sqlite3" 
    if test_db_path.exists():
        test_db_path.unlink() 

    test_logger.info(f"--- Testing ProgramDatabase (with Logging & Diverse Selection) ---")
    test_logger.info(f"Using test database: {test_db_path}")
    db = ProgramDatabase(db_path=test_db_path) 

    test_logger.info("Adding programs...")
    p = {} 
    p[1] = db.add_program("sampler_test", "code1_best_correct", 0, {"correctness": 1.0, "primary": 100.0, "lines": -2.0})
    p[2] = db.add_program("sampler_test", "code2_good_correct", 1, {"correctness": 1.0, "primary": 90.0, "lines": -3.0}, parent_id=p[1])
    p[3] = db.add_program("sampler_test", "code3_ok_correct", 1, {"correctness": 1.0, "primary": 80.0, "lines": -1.0}, parent_id=p[1])
    p[4] = db.add_program("sampler_test", "code4_bad_incorrect", 1, {"correctness": 0.0, "primary": 120.0, "lines": -1.0}, parent_id=p[2]) # Highest primary, but incorrect
    p[5] = db.add_program("sampler_test", "code5_best_correct_gen2", 2, {"correctness": 1.0, "primary": 110.0, "lines": -5.0}, parent_id=p[2])
    p[6] = db.add_program("sampler_test", "code6_good_correct_gen2", 2, {"correctness": 1.0, "primary": 85.0, "lines": -4.0}, parent_id=p[3])
    p[7] = db.add_program("sampler_test", "code7_bad_incorrect_gen2", 2, {"correctness": 0.5, "primary": 70.0, "lines": -2.0}, parent_id=p[3])

    test_logger.info("\nTesting get_random_correct_programs...")
    random_correct = db.get_random_correct_programs(
        problem_name="sampler_test", 
        correctness_metric="correctness", 
        correctness_threshold=1.0, 
        n=2
    )
    test_logger.info(f"Found {len(random_correct)} random correct programs:")
    for prog in random_correct:
        test_logger.info(f"  ID={prog['id']}, Scores={prog['scores']}")
    assert len(random_correct) <= 2
    for prog in random_correct:
        assert prog['scores']['correctness'] >= 1.0

    test_logger.info("\nTesting get_random_correct_programs excluding some IDs...")
    ids_to_exclude = [p[1], p[5]] 
    random_correct_excluded = db.get_random_correct_programs(
        problem_name="sampler_test", 
        correctness_metric="correctness", 
        correctness_threshold=1.0, 
        n=3, 
        exclude_ids=ids_to_exclude
    )
    test_logger.info(f"Found {len(random_correct_excluded)} random correct programs (excluding {ids_to_exclude}):")
    for prog in random_correct_excluded:
        test_logger.info(f"  ID={prog['id']}, Scores={prog['scores']}")
        assert prog['id'] not in ids_to_exclude
    assert len(random_correct_excluded) <= 3 


    test_logger.info("\n--- Original Tests ---")
    retrieved_prog2 = db.get_program_by_id(p[2]) 
    if retrieved_prog2:
        test_logger.info(f"Retrieved program ID {p[2]}: Gen={retrieved_prog2['generation']}, Scores={retrieved_prog2['scores']}")
    
    best_progs = db.get_best_programs(
        problem_name="sampler_test", primary_metric="primary", metric_goal="maximize", n=2
    )
    test_logger.info("Best programs by 'primary' metric:")
    for i, prog in enumerate(best_progs):
        test_logger.info(f"  Best {i+1}: ID={prog['id']}, Scores={prog['scores']}")
    
    # --- CORRECTED ASSERTIONS ---
    if best_progs:
        assert best_progs[0]['id'] == p[4] # ID 4 has primary = 120.0
    if len(best_progs) > 1:
         assert best_progs[1]['id'] == p[5] # ID 5 has primary = 110.0
    # --- END CORRECTED ASSERTIONS ---


    test_logger.info("\n--- Test Finished ---")
    if test_db_path.exists():
        test_db_path.unlink()
        test_logger.info(f"Cleaned up test database: {test_db_path}")
