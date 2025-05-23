# evocoder/core/evolutionary_controller.py

import asyncio
import random
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import importlib.util

try:
    from ..config import settings 
    from ..llm_interface.llm_manager import LLMManager
    # Assuming program_database.py is the correct file for ProgramDatabase
    from .program_database import ProgramDatabase 
    from .evaluator_cascade import Evaluator # Assuming using evaluator_cascade.py
    from ..utils import diff_utils 
    from ..utils.logger import setup_logger 
except ImportError:
    if __name__ == '__main__': 
        import sys
        file_path = Path(__file__).resolve()
        project_root = file_path.parent.parent.parent
        sys.path.insert(0, str(project_root))

        from evocoder.config import settings
        from evocoder.llm_interface.llm_manager import LLMManager
        from evocoder.core.program_database import ProgramDatabase
        from evocoder.core.evaluator_cascade import Evaluator 
        from evocoder.utils import diff_utils 
        from evocoder.utils.logger import setup_logger
        import importlib.util
    else:
        raise

class EvolutionaryController:
    def __init__(self, problem_config_path: str, experiment_config: Optional[Dict[str, Any]] = None):
        self.logger = setup_logger(f"evocoder.controller.{self.__class__.__name__}")
        self.logger.info(f"Initializing EvolutionaryController for problem config: {problem_config_path}")

        self.problem_config_path_str = problem_config_path
        self.experiment_config = experiment_config if experiment_config is not None else {}
        
        self.problem_config: Dict[str, Any] = self._load_problem_config(problem_config_path)
        self.problem_name: str = self.problem_config["PROBLEM_NAME"]
        self.correctness_metric_key: str = "correctness_score" 
        for metric, details in self.problem_config.get("EVALUATION_METRICS", {}).items():
            if "correctness" in metric.lower() and "score" in metric.lower():
                self.correctness_metric_key = metric
                break

        self.db = ProgramDatabase() 
        self.evaluator = Evaluator() 
        
        llm_provider_name = self.experiment_config.get("llm_provider", settings.DEFAULT_LLM_PROVIDER)
        
        llm_specific_config_for_manager: Dict[str, Any] = {}
        if llm_provider_name == "open_webui":
            llm_specific_config_for_manager['api_key'] = self.experiment_config.get(
                "open_webui_api_key", settings.OPEN_WEBUI_API_KEY
            )
            llm_specific_config_for_manager['base_url'] = self.experiment_config.get(
                "open_webui_base_url", settings.OPEN_WEBUI_BASE_URL
            )
        elif llm_provider_name == "gemini": 
             llm_specific_config_for_manager['api_key'] = self.experiment_config.get(
                "gemini_api_key", settings.GEMINI_API_KEY
            )

        self.llm_manager = LLMManager(
            provider_name=llm_provider_name,
            llm_config=llm_specific_config_for_manager
        )
        
        self.llm_model_name = self._get_llm_model_name()

        self.logger.info(f"EvolutionaryController initialized for problem: {self.problem_name}")
        self.logger.info(f"Using LLM Provider: {self.llm_manager.provider_name}, Model: {self.llm_model_name}")
        self.logger.info(f"Correctness metric key identified as: '{self.correctness_metric_key}'")

    def _get_llm_model_name(self) -> str:
        provider_name = self.llm_manager.provider_name
        model_name_override_key = f"{provider_name}_model_name" 
        default_model_from_settings = None

        if provider_name == "open_webui":
            default_model_from_settings = settings.OPEN_WEBUI_MODEL_NAME
        elif provider_name == "gemini":
            default_model_from_settings = settings.GEMINI_MODEL_NAME
        
        model_name = self.experiment_config.get(model_name_override_key, default_model_from_settings)
        
        if not model_name: 
            model_name = self.experiment_config.get("default_model_name")

        if not model_name:
             self.logger.error(f"LLM model name could not be determined for provider: {provider_name}")
             raise ValueError(f"LLM model name could not be determined for provider: {provider_name}")
        return model_name

    def _load_problem_config(self, module_path_str: str) -> Dict[str, Any]:
        self.logger.debug(f"Loading problem configuration from: {module_path_str}")
        try:
            spec = importlib.util.find_spec(module_path_str)
            if spec is None or spec.loader is None:
                self.logger.error(f"Problem configuration module not found: {module_path_str}")
                raise ImportError(f"Problem configuration module not found: {module_path_str}")
            
            problem_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(problem_module) 
            
            config_dict = {
                attr: getattr(problem_module, attr)
                for attr in dir(problem_module)
                if not callable(getattr(problem_module, attr)) and not attr.startswith("__")
            }
            required_keys = ["PROBLEM_NAME", "INITIAL_CODE_FILE", "TEST_SUITE_FILE", "TARGET_FUNCTION_NAME", 
                             "EVALUATION_METRICS", "PRIMARY_METRIC", "PROBLEM_LLM_INSTRUCTIONS",
                             "CORRECTNESS_THRESHOLD"]
            for key in required_keys:
                if key not in config_dict:
                    self.logger.error(f"Missing required key '{key}' in problem config: {module_path_str}")
                    raise ValueError(f"Missing required key '{key}' in problem configuration module: {module_path_str}")
            self.logger.info(f"Successfully loaded problem configuration for '{config_dict['PROBLEM_NAME']}'.")
            return config_dict
        except Exception as e:
            self.logger.exception(f"Error loading problem configuration from {module_path_str}: {e}")
            raise

    async def seed_initial_program(self): 
        self.logger.info("Seeding initial program...")
        initial_code_path = Path(self.problem_config["INITIAL_CODE_FILE"])
        if not initial_code_path.exists():
            self.logger.error(f"Initial code file not found: {initial_code_path}")
            raise FileNotFoundError(f"Initial code file not found: {initial_code_path}")

        initial_code_content = initial_code_path.read_text()
        
        self.logger.info("Evaluating initial program...")
        initial_scores = await self.evaluator.evaluate(initial_code_content, self.problem_config)
        self.logger.info(f"Initial program scores: {initial_scores}")

        program_id = self.db.add_program(
            problem_name=self.problem_name,
            code_content=initial_code_content,
            generation=0,
            scores=initial_scores,
            parent_id=None
        )
        self.logger.info(f"Initial program seeded with ID: {program_id}, Generation: 0")

    def _tournament_selection(
        self, population: List[Dict[str, Any]], tournament_size: int,
        primary_metric: str, metric_goal: str,
        correctness_metric: str, correctness_threshold: float
    ) -> Optional[Dict[str, Any]]:
        if not population: return None
        actual_tournament_size = min(tournament_size, len(population))
        if actual_tournament_size == 0: return None
        tournament_competitors = random.sample(population, actual_tournament_size)
        best_competitor_in_tournament = None
        for competitor in tournament_competitors:
            comp_scores = competitor.get("scores", {})
            comp_correctness = comp_scores.get(correctness_metric, 0.0)
            comp_primary_score = comp_scores.get(primary_metric)
            if best_competitor_in_tournament is None:
                best_competitor_in_tournament = competitor; continue
            best_scores = best_competitor_in_tournament.get("scores", {})
            best_correctness = best_scores.get(correctness_metric, 0.0)
            best_primary_score = best_scores.get(primary_metric)
            is_comp_correct = comp_correctness >= correctness_threshold
            is_best_correct = best_correctness >= correctness_threshold
            if is_comp_correct and not is_best_correct:
                best_competitor_in_tournament = competitor; continue
            if not is_comp_correct and is_best_correct: continue
            if comp_primary_score is None and best_primary_score is not None: continue
            if best_primary_score is None and comp_primary_score is not None:
                best_competitor_in_tournament = competitor; continue
            if comp_primary_score is None and best_primary_score is None: continue
            if metric_goal == "maximize":
                if comp_primary_score > best_primary_score: best_competitor_in_tournament = competitor
            else: 
                if comp_primary_score < best_primary_score: best_competitor_in_tournament = competitor
        return best_competitor_in_tournament

    async def _get_diverse_selection_pool(
        self,
        current_generation: int,
        num_best_to_fetch: int,
        num_random_correct_to_fetch: int,
        num_prev_gen_to_fetch: int
    ) -> List[Dict[str, Any]]:
        """
        Creates a diverse pool of candidates for parent and inspiration selection.
        Combines best overall, random correct, and recent programs.
        """
        self.logger.debug(f"Building diverse selection pool: "
                          f"{num_best_to_fetch} best, "
                          f"{num_random_correct_to_fetch} random correct, "
                          f"{num_prev_gen_to_fetch} from previous gen.")
        
        selection_pool: List[Dict[str, Any]] = []
        seen_ids = set()

        # 1. Get best overall programs
        best_programs = self.db.get_best_programs(
            problem_name=self.problem_name,
            primary_metric=self.problem_config["PRIMARY_METRIC"],
            metric_goal=self.problem_config["EVALUATION_METRICS"][self.problem_config["PRIMARY_METRIC"]]["goal"],
            n=num_best_to_fetch
        )
        for prog in best_programs:
            if prog["id"] not in seen_ids:
                selection_pool.append(prog)
                seen_ids.add(prog["id"])
        
        # 2. Get random correct programs (excluding already selected best ones)
        if num_random_correct_to_fetch > 0:
            random_correct_programs = self.db.get_random_correct_programs(
                problem_name=self.problem_name,
                correctness_metric=self.correctness_metric_key,
                correctness_threshold=self.problem_config["CORRECTNESS_THRESHOLD"],
                n=num_random_correct_to_fetch + len(seen_ids), # Fetch more to account for overlaps
                exclude_ids=list(seen_ids) 
            )
            for prog in random_correct_programs:
                if prog["id"] not in seen_ids:
                    selection_pool.append(prog)
                    seen_ids.add(prog["id"])
                    if len(selection_pool) >= num_best_to_fetch + num_random_correct_to_fetch: # Optimization
                        break
        
        # 3. Get programs from previous generation (excluding already selected)
        if current_generation > 0 and num_prev_gen_to_fetch > 0:
            prev_gen_programs = self.db.get_programs_by_generation(self.problem_name, current_generation - 1)
            random.shuffle(prev_gen_programs) # Shuffle to get a random sample if too many
            for prog in prev_gen_programs:
                if prog["id"] not in seen_ids:
                    selection_pool.append(prog)
                    seen_ids.add(prog["id"])
                    if len(selection_pool) >= num_best_to_fetch + num_random_correct_to_fetch + num_prev_gen_to_fetch: # Optimization
                        break
        
        # 4. If pool is still empty (e.g., only gen 0 exists), add gen 0 programs
        if not selection_pool:
            self.logger.warning("Selection pool is empty after fetching best, random, and previous. Adding gen 0 programs.")
            gen0_programs = self.db.get_programs_by_generation(self.problem_name, 0)
            for prog in gen0_programs:
                 if prog["id"] not in seen_ids:
                    selection_pool.append(prog)
                    seen_ids.add(prog["id"])

        self.logger.info(f"Diverse selection pool size: {len(selection_pool)}")
        return selection_pool


    async def run_evolution(
        self, 
        num_generations: int, 
        population_size_per_gen: int = 10, 
        num_inspirations: int = 2,
        tournament_size_parent: int = 3,
        max_concurrent_tasks: int = 5 
    ):
        self.logger.info(f"--- Starting Evolution for {self.problem_name} ---")
        self.logger.info(f"Config: Generations={num_generations}, PopSizePerGen={population_size_per_gen}, "
                         f"Inspirations={num_inspirations}, TournamentSize={tournament_size_parent}, "
                         f"MaxConcurrent={max_concurrent_tasks}")

        semaphore = asyncio.Semaphore(max_concurrent_tasks)
        
        # Seed initial program if no generation 0 exists
        gen0_programs = self.db.get_programs_by_generation(self.problem_name, 0)
        if not gen0_programs:
            await self.seed_initial_program() 
            gen0_programs = self.db.get_programs_by_generation(self.problem_name, 0) # Re-fetch
            if not gen0_programs:
                 self.logger.critical("Failed to seed or retrieve initial program for generation 0.")
                 raise RuntimeError("Failed to seed or retrieve initial program for generation 0.")

        for gen in range(1, num_generations + 1):
            start_time_gen = time.time()
            self.logger.info(f"--- Generation {gen} ---")
            
            newly_generated_this_gen = 0
            attempts_this_gen = 0
            
            # --- MODIFIED: Use _get_diverse_selection_pool ---
            # Determine how many of each type to fetch for the pool
            num_best_for_pool = max(population_size_per_gen // 2, tournament_size_parent, num_inspirations + 1, 5)
            num_random_correct_for_pool = population_size_per_gen // 2
            num_prev_gen_for_pool = population_size_per_gen // 3
            
            potential_candidates_for_selection = await self._get_diverse_selection_pool(
                current_generation=gen,
                num_best_to_fetch=num_best_for_pool,
                num_random_correct_to_fetch=num_random_correct_for_pool,
                num_prev_gen_to_fetch=num_prev_gen_for_pool
            )
            # --- END MODIFIED SECTION ---

            if not potential_candidates_for_selection:
                self.logger.error(f"Error: No candidate programs found for selection in Gen {gen}. Cannot proceed.")
                continue # Or break, depending on desired behavior

            generation_tasks = []
            for i in range(population_size_per_gen):
                parent_program_data = self._tournament_selection(
                    population=potential_candidates_for_selection,
                    tournament_size=tournament_size_parent,
                    primary_metric=self.problem_config["PRIMARY_METRIC"],
                    metric_goal=self.problem_config["EVALUATION_METRICS"][self.problem_config["PRIMARY_METRIC"]]["goal"],
                    correctness_metric=self.correctness_metric_key, 
                    correctness_threshold=self.problem_config["CORRECTNESS_THRESHOLD"]
                )
                if not parent_program_data:
                    self.logger.warning("Tournament selection failed to pick a parent. Using random choice from pool as fallback.")
                    parent_program_data = random.choice(potential_candidates_for_selection) # Fallback
                
                inspiration_programs_data: List[Dict[str, Any]] = []
                if num_inspirations > 0 and len(potential_candidates_for_selection) > 1:
                    possible_inspirations = [p for p in potential_candidates_for_selection if p["id"] != parent_program_data["id"]]
                    if len(possible_inspirations) >= num_inspirations: 
                        inspiration_programs_data = random.sample(possible_inspirations, num_inspirations)
                    else:
                        inspiration_programs_data = possible_inspirations # Take all available if fewer
                
                task = self._generate_and_evaluate_individual_with_semaphore(
                    semaphore, gen, parent_program_data, inspiration_programs_data
                )
                generation_tasks.append(task)
            
            results = await asyncio.gather(*generation_tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, Exception):
                    self.logger.error(f"Error generating/evaluating individual (gathered): {result}", exc_info=True)
                elif result: 
                    program_id, scores = result
                    # self.logger.info(f"Successfully generated and evaluated individual. ID: {program_id}, Scores: {scores}") # Logged in _generate_and_evaluate_individual
                    newly_generated_this_gen +=1
                attempts_this_gen +=1

            gen_duration = time.time() - start_time_gen
            self.logger.info(f"--- Generation {gen} Summary ---")
            self.logger.info(f"  Individuals attempted: {attempts_this_gen}")
            self.logger.info(f"  New individuals added to DB: {newly_generated_this_gen}")
            self.logger.info(f"  Generation duration: {gen_duration:.2f} seconds")

            current_best_list = self.db.get_best_programs( 
                problem_name=self.problem_name,
                primary_metric=self.problem_config["PRIMARY_METRIC"],
                metric_goal=self.problem_config["EVALUATION_METRICS"][self.problem_config["PRIMARY_METRIC"]]["goal"],
                n=1
            )
            if current_best_list:
                best_prog_for_log = current_best_list[0]
                # Log the best *correct* program if possible
                all_db_programs = self.db.get_all_programs_with_filters(problem_name=self.problem_name)
                correct_candidates = [p for p in all_db_programs if p.get("scores", {}).get(self.correctness_metric_key, 0.0) >= self.problem_config["CORRECTNESS_THRESHOLD"]]
                if correct_candidates:
                     best_correct_prog = self._tournament_selection(correct_candidates, len(correct_candidates), self.problem_config["PRIMARY_METRIC"], self.problem_config["EVALUATION_METRICS"][self.problem_config["PRIMARY_METRIC"]]["goal"], self.correctness_metric_key, self.problem_config["CORRECTNESS_THRESHOLD"])
                     if best_correct_prog:
                         best_prog_for_log = best_correct_prog
                
                self.logger.info(f"  Current best for '{self.problem_name}' (ID {best_prog_for_log['id']}): "
                      f"Scores: {best_prog_for_log['scores']}")
            else:
                self.logger.warning("  No best program found yet (all might have failed correctness).")

        self.logger.info("--- Evolution Finished ---")
        await self.llm_manager.close_provider()

    async def _generate_and_evaluate_individual_with_semaphore(
        self, semaphore: asyncio.Semaphore, current_generation: int,
        parent_program_data: Dict[str, Any], inspiration_programs_data: List[Dict[str, Any]] 
    ) -> Optional[Tuple[int, Dict[str, float]]]:
        async with semaphore:
            # self.logger.debug(f"Acquired semaphore for parent ID: {parent_program_data['id']}")
            try:
                result = await self._generate_and_evaluate_individual(
                    current_generation, parent_program_data, inspiration_programs_data
                )
                return result
            except Exception as e:
                self.logger.exception(f"Critical error in semaphore-wrapped task for parent {parent_program_data['id']}: {e}")
                return None 

    async def _generate_and_evaluate_individual(
        self, current_generation: int, parent_program_data: Dict[str, Any],
        inspiration_programs_data: List[Dict[str, Any]] 
    ) -> Optional[Tuple[int, Dict[str, float]]]:
        parent_id = parent_program_data["id"]
        parent_code = parent_program_data["code_content"]
        
        self.logger.info(f"Generating from parent ID: {parent_id} (Gen {parent_program_data['generation']}, Scores: {parent_program_data.get('scores')}) "
              f"with {len(inspiration_programs_data)} inspirations.")

        context_examples_for_llm: List[Dict[str, str]] = []
        for insp_prog in inspiration_programs_data:
            context_examples_for_llm.append({
                "role": "user", 
                "content": f"Here's an example of a previously successful related code (scores: {insp_prog.get('scores')}):\n```python\n{insp_prog['code_content']}\n```"
            })

        llm_diff_string = "" 
        try:
            llm_diff_string = await self.llm_manager.generate_code_modification(
                current_code=parent_code, model_name=self.llm_model_name,
                prompt_instructions=self.problem_config["PROBLEM_LLM_INSTRUCTIONS"],
                context_examples=context_examples_for_llm if context_examples_for_llm else None,
                temperature=self.experiment_config.get("llm_temperature", 0.1), 
                max_tokens=self.experiment_config.get("llm_max_tokens_diff", 1024) 
            )
            
            self.logger.debug(f"Parent {parent_id}: LLM returned diff string (first 300 chars):\n{llm_diff_string[:300]}...")
            
            if not llm_diff_string or not llm_diff_string.strip() or llm_diff_string.strip() == "NO_CHANGES_NECESSARY":
                log_msg = f"  Parent {parent_id}: LLM indicated no changes necessary." if llm_diff_string.strip() == "NO_CHANGES_NECESSARY" else f"  Parent {parent_id}: LLM returned empty/whitespace diff."
                self.logger.info(log_msg)
                return None 
            
            parsed_diff_blocks = diff_utils.parse_diff_string(llm_diff_string)
            self.logger.debug(f"  Parent {parent_id}: Parsed Diff Blocks count: {len(parsed_diff_blocks)}") 

            if not parsed_diff_blocks:
                self.logger.warning(f"  Parent {parent_id}: Could not parse valid diff blocks from LLM response: '{llm_diff_string[:100]}...'")
                return None 

            evolved_code_str_final = diff_utils.apply_diffs(parent_code, parsed_diff_blocks)
            # self.logger.debug(f"  Parent {parent_id}: Code after applying diff:\n{evolved_code_str_final}")
            
            if evolved_code_str_final == parent_code: 
                self.logger.info(f"  Parent {parent_id}: LLM diff resulted in no effective code change after application.")
                return None

            evolved_scores = await self.evaluator.evaluate(evolved_code_str_final, self.problem_config)
            
            new_program_id = self.db.add_program(
                problem_name=self.problem_name, code_content=evolved_code_str_final, 
                generation=current_generation, scores=evolved_scores,
                parent_id=parent_id, llm_diff=llm_diff_string 
            )
            self.logger.info(f"  Parent {parent_id}: Stored new program ID: {new_program_id}, Scores: {evolved_scores}")
            return new_program_id, evolved_scores

        except Exception as e:
            self.logger.exception(f"  Exception during individual processing for parent {parent_id}: {e}")
            return None


if __name__ == '__main__':
    import os
    
    async def test_controller():
        test_logger = setup_logger("evocoder.test_controller", level_str="INFO")
        test_logger.info("--- Testing EvolutionaryController (Phase 3 - Advanced Prompt Sampling) ---") 

        if not hasattr(settings, 'DEFAULT_LLM_PROVIDER'):
            test_logger.error("Settings not loaded. Ensure .env is configured and script is run from project root.")
            return
        
        problem_module_path = "evocoder.problems.numerical_optimizer.problem_config" 
        
        experiment_settings = {
            "llm_provider": "open_webui", 
            "llm_temperature": 0.2, 
            "llm_max_tokens_diff": 1024 
        }
        
        db_data_dir = Path(__file__).resolve().parent.parent.parent / "data"
        db_data_dir.mkdir(parents=True, exist_ok=True)
        test_db_for_controller = db_data_dir / "evocoder_programs.db" 
        if test_db_for_controller.exists():
            test_logger.info(f"Deleting existing test DB: {test_db_for_controller}")
            test_db_for_controller.unlink()

        controller = None
        try:
            controller = EvolutionaryController(
                problem_config_path=problem_module_path,
                experiment_config=experiment_settings
            )
            
            await controller.run_evolution(
                num_generations=2, 
                population_size_per_gen=3, # Keep pop size manageable for testing
                num_inspirations=1, # Test with 1 inspiration
                tournament_size_parent=2,
                max_concurrent_tasks=2 
            )

        except FileNotFoundError as fnf:
            test_logger.error(f"Test Error: File not found - {fnf}. Ensure problem files exist.")
        except (ValueError, ImportError) as e: 
            test_logger.error(f"Test Error: Configuration or Import Error - {e}. Check configurations.")
        except Exception as e:
            test_logger.exception(f"An unexpected error occurred in controller test: {e}")
        finally:
            if controller and hasattr(controller, 'llm_manager') and controller.llm_manager.provider:
                await controller.llm_manager.close_provider()
            test_logger.info("--- Controller Test Finished ---")

    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(test_controller())
