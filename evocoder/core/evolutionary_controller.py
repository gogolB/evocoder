# evocoder/core/evolutionary_controller.py

import asyncio
import random
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

# Assuming these modules are in the same package or sys.path is configured
try:
    from ..config import settings 
    from ..llm_interface.llm_manager import LLMManager
    from .program_database import ProgramDatabase
    from .evaluator import Evaluator # Evaluator.evaluate is now async
    from ..utils import diff_utils 
    import importlib.util
except ImportError:
    if __name__ == '__main__':
        import sys
        file_path = Path(__file__).resolve()
        project_root = file_path.parent.parent.parent
        sys.path.insert(0, str(project_root))

        from evocoder.config import settings
        from evocoder.llm_interface.llm_manager import LLMManager
        from evocoder.core.program_database import ProgramDatabase
        from evocoder.core.evaluator import Evaluator # Evaluator.evaluate is now async
        from evocoder.utils import diff_utils 
        import importlib.util
    else:
        raise

class EvolutionaryController:
    """
    Orchestrates the evolutionary algorithm to evolve code solutions.
    """

    def __init__(self, problem_config_path: str, experiment_config: Optional[Dict[str, Any]] = None):
        self.problem_config_path_str = problem_config_path
        self.experiment_config = experiment_config if experiment_config is not None else {}
        
        self.problem_config: Dict[str, Any] = self._load_problem_config(problem_config_path)
        self.problem_name: str = self.problem_config["PROBLEM_NAME"]

        self.db = ProgramDatabase() 
        self.evaluator = Evaluator() # Evaluator.evaluate is now async
        
        llm_provider_name = self.experiment_config.get("llm_provider", settings.DEFAULT_LLM_PROVIDER)
        
        llm_specific_config_for_manager: Dict[str, Any] = {}
        if llm_provider_name == "open_webui":
            llm_specific_config_for_manager['api_key'] = self.experiment_config.get(
                "open_webui_api_key", settings.OPEN_WEBUI_API_KEY
            )
            llm_specific_config_for_manager['base_url'] = self.experiment_config.get(
                "open_webui_base_url", settings.OPEN_WEBUI_BASE_URL
            )
        
        self.llm_manager = LLMManager(
            provider_name=llm_provider_name,
            llm_config=llm_specific_config_for_manager
        )
        
        self.llm_model_name = self._get_llm_model_name()

        print(f"EvolutionaryController initialized for problem: {self.problem_name}")
        print(f"Using LLM Provider: {self.llm_manager.provider_name}, Model: {self.llm_model_name}")

    def _get_llm_model_name(self) -> str:
        if self.llm_manager.provider_name == "open_webui":
            model_name = self.experiment_config.get("open_webui_model_name", settings.OPEN_WEBUI_MODEL_NAME)
        else:
            model_name = self.experiment_config.get("default_model_name")
            if not model_name:
                raise ValueError(f"LLM model name not configured for provider: {self.llm_manager.provider_name}")
        
        if not model_name:
             raise ValueError(f"LLM model name could not be determined for provider: {self.llm_manager.provider_name}")
        return model_name

    def _load_problem_config(self, module_path_str: str) -> Dict[str, Any]:
        try:
            spec = importlib.util.find_spec(module_path_str)
            if spec is None or spec.loader is None:
                raise ImportError(f"Problem configuration module not found: {module_path_str}")
            
            problem_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(problem_module) 
            
            config_dict = {
                attr: getattr(problem_module, attr)
                for attr in dir(problem_module)
                if not callable(getattr(problem_module, attr)) and not attr.startswith("__")
            }
            required_keys = ["PROBLEM_NAME", "INITIAL_CODE_FILE", "TEST_SUITE_FILE", "TARGET_FUNCTION_NAME", "EVALUATION_METRICS", "PRIMARY_METRIC", "PROBLEM_LLM_INSTRUCTIONS"]
            for key in required_keys:
                if key not in config_dict:
                    raise ValueError(f"Missing required key '{key}' in problem configuration module: {module_path_str}")
            return config_dict
        except Exception as e:
            print(f"Error loading problem configuration from {module_path_str}: {e}")
            raise

    async def seed_initial_program(self): # Made async
        """
        Loads the initial program code from the problem configuration,
        evaluates it (asynchronously), and adds it to the database as generation 0.
        """
        print("Seeding initial program...")
        initial_code_path = Path(self.problem_config["INITIAL_CODE_FILE"])
        if not initial_code_path.exists():
            raise FileNotFoundError(f"Initial code file not found: {initial_code_path}")

        initial_code_content = initial_code_path.read_text()
        
        print("Evaluating initial program...")
        # --- MODIFIED: Added await ---
        initial_scores = await self.evaluator.evaluate(initial_code_content, self.problem_config)
        print(f"Initial program scores: {initial_scores}")

        program_id = self.db.add_program(
            problem_name=self.problem_name,
            code_content=initial_code_content,
            generation=0,
            scores=initial_scores,
            parent_id=None
        )
        print(f"Initial program seeded with ID: {program_id}, Generation: 0")

    def _tournament_selection(
        self,
        population: List[Dict[str, Any]],
        tournament_size: int,
        primary_metric: str,
        metric_goal: str
    ) -> Optional[Dict[str, Any]]:
        if not population:
            return None
        
        actual_tournament_size = min(tournament_size, len(population))
        if actual_tournament_size == 0:
            return None

        tournament_competitors = random.sample(population, actual_tournament_size)
        
        best_competitor = None
        best_score = float('-inf') if metric_goal == "maximize" else float('inf')

        for competitor in tournament_competitors:
            score_dict = competitor.get("scores", {})
            competitor_score = score_dict.get(primary_metric)

            if competitor_score is None: 
                continue 

            if metric_goal == "maximize":
                if competitor_score > best_score:
                    best_score = competitor_score
                    best_competitor = competitor
            else: 
                if competitor_score < best_score:
                    best_score = competitor_score
                    best_competitor = competitor
        
        return best_competitor


    async def run_evolution(
        self, 
        num_generations: int, 
        population_size_per_gen: int = 10, 
        num_inspirations: int = 2,
        tournament_size_parent: int = 3 
    ):
        print(f"\n--- Starting Evolution for {self.problem_name} ---")
        print(f"Total generations: {num_generations}, Population target per gen: {population_size_per_gen}")
        print(f"Inspirations: {num_inspirations}, Parent Tournament Size: {tournament_size_parent}")

        # --- MODIFIED: Await seed_initial_program ---
        gen0_programs = self.db.get_programs_by_generation(self.problem_name, 0)
        if not gen0_programs:
            await self.seed_initial_program() # Now awaited
            gen0_programs = self.db.get_programs_by_generation(self.problem_name, 0)
            if not gen0_programs:
                 raise RuntimeError("Failed to seed or retrieve initial program for generation 0.")

        for gen in range(1, num_generations + 1):
            start_time_gen = time.time()
            print(f"\n--- Generation {gen} ---")
            
            newly_generated_this_gen = 0
            attempts_this_gen = 0
            
            candidate_pool_size = max(population_size_per_gen * tournament_size_parent, num_inspirations + population_size_per_gen + 5)
            potential_candidates_for_selection = self.db.get_best_programs(
                problem_name=self.problem_name,
                primary_metric=self.problem_config["PRIMARY_METRIC"],
                metric_goal=self.problem_config["EVALUATION_METRICS"][self.problem_config["PRIMARY_METRIC"]]["goal"],
                n=candidate_pool_size 
            )

            if not potential_candidates_for_selection:
                print(f"Warning: No suitable candidate programs found from previous generations for Gen {gen}. "
                      "Using generation 0 seed(s) if available.")
                potential_candidates_for_selection = gen0_programs
                if not potential_candidates_for_selection:
                    print("Error: No generation 0 programs found either. Cannot proceed with this generation.")
                    continue 

            generation_tasks = []
            for i in range(population_size_per_gen):
                parent_program_data = self._tournament_selection(
                    population=potential_candidates_for_selection,
                    tournament_size=tournament_size_parent,
                    primary_metric=self.problem_config["PRIMARY_METRIC"],
                    metric_goal=self.problem_config["EVALUATION_METRICS"][self.problem_config["PRIMARY_METRIC"]]["goal"]
                )
                if not parent_program_data:
                    print("Warning: Tournament selection failed to pick a parent. Using random choice from pool as fallback.")
                    if potential_candidates_for_selection:
                        parent_program_data = random.choice(potential_candidates_for_selection)
                    elif gen0_programs:
                         parent_program_data = random.choice(gen0_programs)
                    else: 
                        print("Error: No candidates available for parent selection. Skipping individual.")
                        continue
                
                inspiration_programs_data: List[Dict[str, Any]] = []
                if num_inspirations > 0:
                    possible_inspirations = [p for p in potential_candidates_for_selection if p["id"] != parent_program_data["id"]]
                    if len(possible_inspirations) >= num_inspirations: 
                        inspiration_programs_data = random.sample(possible_inspirations, num_inspirations)
                    else:
                        inspiration_programs_data = possible_inspirations
                
                generation_tasks.append(
                    self._generate_and_evaluate_individual(gen, parent_program_data, inspiration_programs_data)
                )
            
            results = await asyncio.gather(*generation_tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, Exception):
                    print(f"  Error generating/evaluating individual: {result}")
                elif result: 
                    program_id, scores = result
                    print(f"  Successfully generated and evaluated individual. ID: {program_id}, Scores: {scores}")
                    newly_generated_this_gen +=1
                attempts_this_gen +=1

            gen_duration = time.time() - start_time_gen
            print(f"--- Generation {gen} Summary ---")
            print(f"  Individuals attempted: {attempts_this_gen}")
            print(f"  New individuals added to DB: {newly_generated_this_gen}")
            print(f"  Generation duration: {gen_duration:.2f} seconds")

            current_best = self.db.get_best_programs(
                problem_name=self.problem_name,
                primary_metric=self.problem_config["PRIMARY_METRIC"],
                metric_goal=self.problem_config["EVALUATION_METRICS"][self.problem_config["PRIMARY_METRIC"]]["goal"],
                n=1
            )
            if current_best:
                print(f"  Current best for '{self.problem_name}' (ID {current_best[0]['id']}): "
                      f"Scores: {current_best[0]['scores']}")
            else:
                print("  No best program found yet (all might have failed correctness).")

        print("\n--- Evolution Finished ---")
        await self.llm_manager.close_provider()

    async def _generate_and_evaluate_individual(
        self,
        current_generation: int,
        parent_program_data: Dict[str, Any],
        inspiration_programs_data: List[Dict[str, Any]] 
    ) -> Optional[Tuple[int, Dict[str, float]]]:
        parent_id = parent_program_data["id"]
        parent_code = parent_program_data["code_content"]
        
        print(f"  Generating from parent ID: {parent_id} (Gen {parent_program_data['generation']}) "
              f"with {len(inspiration_programs_data)} inspirations.")

        context_examples_for_llm: List[Dict[str, str]] = []
        for insp_prog in inspiration_programs_data:
            context_examples_for_llm.append({
                "role": "user", 
                "content": f"Here's an example of a previously successful related code (scores: {insp_prog.get('scores')}):\n```python\n{insp_prog['code_content']}\n```"
            })

        evolved_code_str_final = parent_code 
        llm_diff_string = "" 

        try:
            llm_diff_string = await self.llm_manager.generate_code_modification(
                current_code=parent_code,
                model_name=self.llm_model_name,
                prompt_instructions=self.problem_config["PROBLEM_LLM_INSTRUCTIONS"],
                context_examples=context_examples_for_llm if context_examples_for_llm else None,
                temperature=self.experiment_config.get("llm_temperature", 0.1), 
                max_tokens=self.experiment_config.get("llm_max_tokens_diff", 1024) 
            )
            
            print(f"  LLM returned FULL diff string:\n------BEGIN LLM DIFF------\n{llm_diff_string}\n------END LLM DIFF------")
            
            if not llm_diff_string or not llm_diff_string.strip() or llm_diff_string.strip() == "NO_CHANGES_NECESSARY":
                if llm_diff_string.strip() == "NO_CHANGES_NECESSARY":
                    print("  LLM indicated no changes necessary.")
                else:
                    print("  LLM returned empty or whitespace-only diff. Assuming no change from parent.")
                return None 
            
            parsed_diff_blocks = diff_utils.parse_diff_string(llm_diff_string)
            print(f"  Parsed Diff Blocks: {parsed_diff_blocks}") 

            if not parsed_diff_blocks:
                print("  Could not parse any valid diff blocks from LLM response.")
                return None 

            evolved_code_str_final = diff_utils.apply_diffs(parent_code, parsed_diff_blocks)
            print(f"  Code after applying diff (evolved_code_str_final):\n------BEGIN EVOLVED CODE------\n{evolved_code_str_final}\n------END EVOLVED CODE------")
            
            if evolved_code_str_final == parent_code: 
                print("  LLM diff resulted in no effective change to the code after application, or diff was not applicable.")
                return None

            # --- MODIFIED: Added await ---
            evolved_scores = await self.evaluator.evaluate(evolved_code_str_final, self.problem_config)
            
            new_program_id = self.db.add_program(
                problem_name=self.problem_name,
                code_content=evolved_code_str_final, 
                generation=current_generation,
                scores=evolved_scores,
                parent_id=parent_id,
                llm_diff=llm_diff_string 
            )
            return new_program_id, evolved_scores

        except Exception as e:
            print(f"  Exception during diff generation/application/evaluation for parent {parent_id}: {e}")
            return None


if __name__ == '__main__':
    import os
    
    async def test_controller():
        print("--- Testing EvolutionaryController (Phase 2 - Async Evaluation Fix) ---") # Updated print

        if not hasattr(settings, 'DEFAULT_LLM_PROVIDER'):
            print("Error: Settings not loaded. Ensure .env is configured and script is run from project root.")
            return
        
        problem_module_path = "evocoder.problems.simple_line_reducer.problem_config"
        
        experiment_settings = {
            "llm_provider": "open_webui", 
            "llm_temperature": 0.1, 
            "llm_max_tokens_diff": 1024 
        }
        
        db_data_dir = Path(__file__).resolve().parent.parent.parent / "data"
        db_data_dir.mkdir(parents=True, exist_ok=True)
        test_db_for_controller = db_data_dir / "evocoder_programs.db" 
        if test_db_for_controller.exists():
            print(f"Deleting existing test DB: {test_db_for_controller}")
            test_db_for_controller.unlink()

        controller = None
        try:
            controller = EvolutionaryController(
                problem_config_path=problem_module_path,
                experiment_config=experiment_settings
            )
            
            await controller.run_evolution(
                num_generations=2, 
                population_size_per_gen=3, 
                num_inspirations=1, 
                tournament_size_parent=2 
            )

        except FileNotFoundError as fnf:
            print(f"Test Error: File not found - {fnf}. Ensure problem files exist.")
        except (ValueError, ImportError) as e: 
            print(f"Test Error: Configuration or Import Error - {e}. Check configurations.")
        except Exception as e:
            print(f"An unexpected error occurred in controller test: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if controller and hasattr(controller, 'llm_manager') and controller.llm_manager.provider:
                await controller.llm_manager.close_provider()

    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(test_controller())
