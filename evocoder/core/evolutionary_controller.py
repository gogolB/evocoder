# evocoder/core/evolutionary_controller.py

import asyncio
import random
import time
from pathlib import Path
from typing import Dict, Any, Optional, List

# Assuming these modules are in the same package or sys.path is configured
try:
    from ..config import settings # For default LLM provider, potentially other global settings
    from ..llm_interface.llm_manager import LLMManager
    from .program_database import ProgramDatabase
    from .evaluator import Evaluator
    # For loading problem_config modules dynamically
    import importlib.util
except ImportError:
    # This block is for handling potential import issues if this file is
    # somehow run in a context where the package structure isn't recognized.
    # The __main__ block below will also adjust sys.path for direct execution testing.
    if __name__ == '__main__':
        import sys
        file_path = Path(__file__).resolve()
        # evolutionary_controller.py is in evocoder/evocoder/core/
        # project_root is 3 levels up
        project_root = file_path.parent.parent.parent
        sys.path.insert(0, str(project_root))

        from evocoder.config import settings
        from evocoder.llm_interface.llm_manager import LLMManager
        from evocoder.core.program_database import ProgramDatabase
        from evocoder.core.evaluator import Evaluator
        import importlib.util
    else:
        raise

class EvolutionaryController:
    """
    Orchestrates the evolutionary algorithm to evolve code solutions.
    """

    def __init__(self, problem_config_path: str, experiment_config: Optional[Dict[str, Any]] = None):
        """
        Initializes the EvolutionaryController.

        Args:
            problem_config_path (str): Path to the problem's configuration module
                                       (e.g., "evocoder.problems.simple_line_reducer.problem_config").
            experiment_config (Optional[Dict[str, Any]]): A dictionary containing experiment-specific
                                                          settings, which can override global settings.
                                                          E.g., LLM provider, model name, evolutionary params.
        """
        self.problem_config_path_str = problem_config_path
        self.experiment_config = experiment_config if experiment_config is not None else {}
        
        self.problem_config: Dict[str, Any] = self._load_problem_config(problem_config_path)
        self.problem_name: str = self.problem_config["PROBLEM_NAME"]

        # Initialize components
        self.db = ProgramDatabase() # Uses default DB path or could be configured
        self.evaluator = Evaluator()
        
        # Configure LLMManager based on experiment_config or global settings
        llm_provider_name = self.experiment_config.get("llm_provider", settings.DEFAULT_LLM_PROVIDER)
        
        # Prepare llm_specific_config for LLMManager constructor
        # This allows overriding API keys, base URLs, etc., on a per-experiment basis
        # if they are present in experiment_config. Otherwise, LLMManager will use global settings.
        llm_specific_config_for_manager: Dict[str, Any] = {}
        if llm_provider_name == "open_webui":
            llm_specific_config_for_manager['api_key'] = self.experiment_config.get(
                "open_webui_api_key", settings.OPEN_WEBUI_API_KEY
            )
            llm_specific_config_for_manager['base_url'] = self.experiment_config.get(
                "open_webui_base_url", settings.OPEN_WEBUI_BASE_URL
            )
        # Add other providers as needed
        
        self.llm_manager = LLMManager(
            provider_name=llm_provider_name,
            llm_config=llm_specific_config_for_manager
        )
        
        # Determine model name for LLM requests - can be from experiment or global settings
        self.llm_model_name = self._get_llm_model_name()

        print(f"EvolutionaryController initialized for problem: {self.problem_name}")
        print(f"Using LLM Provider: {self.llm_manager.provider_name}, Model: {self.llm_model_name}")

    def _get_llm_model_name(self) -> str:
        """Determines the LLM model name to use based on provider and configuration."""
        # Priority: experiment_config -> global settings for the provider
        if self.llm_manager.provider_name == "open_webui":
            model_name = self.experiment_config.get("open_webui_model_name", settings.OPEN_WEBUI_MODEL_NAME)
        # Add elif for other providers like "gemini", "ollama_direct"
        # elif self.llm_manager.provider_name == "gemini":
        #     model_name = self.experiment_config.get("gemini_model_name", settings.GEMINI_MODEL_NAME)
        else:
            # Fallback or raise error if model name can't be determined for the provider
            model_name = self.experiment_config.get("default_model_name") # Generic fallback
            if not model_name:
                raise ValueError(f"LLM model name not configured for provider: {self.llm_manager.provider_name}")
        
        if not model_name: # Final check
             raise ValueError(f"LLM model name could not be determined for provider: {self.llm_manager.provider_name}")
        return model_name


    def _load_problem_config(self, module_path_str: str) -> Dict[str, Any]:
        """
        Loads a problem configuration module dynamically and returns its attributes as a dict.
        """
        try:
            spec = importlib.util.find_spec(module_path_str)
            if spec is None or spec.loader is None:
                raise ImportError(f"Problem configuration module not found: {module_path_str}")
            
            problem_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(problem_module) # type: ignore
            
            # Convert module attributes to a dictionary
            config_dict = {
                attr: getattr(problem_module, attr)
                for attr in dir(problem_module)
                if not callable(getattr(problem_module, attr)) and not attr.startswith("__")
            }
            # Ensure essential keys are present
            required_keys = ["PROBLEM_NAME", "INITIAL_CODE_FILE", "TEST_SUITE_FILE", "TARGET_FUNCTION_NAME", "EVALUATION_METRICS", "PRIMARY_METRIC", "PROBLEM_LLM_INSTRUCTIONS"]
            for key in required_keys:
                if key not in config_dict:
                    raise ValueError(f"Missing required key '{key}' in problem configuration module: {module_path_str}")
            return config_dict
        except Exception as e:
            print(f"Error loading problem configuration from {module_path_str}: {e}")
            raise

    def seed_initial_program(self):
        """
        Loads the initial program code from the problem configuration,
        evaluates it, and adds it to the database as generation 0.
        """
        print("Seeding initial program...")
        initial_code_path = Path(self.problem_config["INITIAL_CODE_FILE"])
        if not initial_code_path.exists():
            raise FileNotFoundError(f"Initial code file not found: {initial_code_path}")

        initial_code_content = initial_code_path.read_text()
        
        print("Evaluating initial program...")
        initial_scores = self.evaluator.evaluate(initial_code_content, self.problem_config)
        print(f"Initial program scores: {initial_scores}")

        program_id = self.db.add_program(
            problem_name=self.problem_name,
            code_content=initial_code_content,
            generation=0,
            scores=initial_scores,
            parent_id=None # No parent for initial seed
        )
        print(f"Initial program seeded with ID: {program_id}, Generation: 0")

    async def run_evolution(self, num_generations: int, population_size_per_gen: int = 10):
        """
        Runs the main evolutionary loop.

        Args:
            num_generations (int): The number of generations to run.
            population_size_per_gen (int): The number of new individuals to attempt to
                                           generate and evaluate per generation.
        """
        print(f"\n--- Starting Evolution for {self.problem_name} ---")
        print(f"Total generations: {num_generations}, Population target per gen: {population_size_per_gen}")

        # Ensure initial program is seeded if DB is empty for this problem at gen 0
        gen0_programs = self.db.get_programs_by_generation(self.problem_name, 0)
        if not gen0_programs:
            self.seed_initial_program()
            gen0_programs = self.db.get_programs_by_generation(self.problem_name, 0)
            if not gen0_programs: # Still no gen 0 programs after seeding
                 raise RuntimeError("Failed to seed or retrieve initial program for generation 0.")


        for gen in range(1, num_generations + 1):
            start_time_gen = time.time()
            print(f"\n--- Generation {gen} ---")
            
            newly_generated_this_gen = 0
            attempts_this_gen = 0

            # For MVP, let's try to generate `population_size_per_gen` new individuals
            # based on the best from the previous generation or diverse selection.
            
            # Selection strategy: For MVP, pick the best overall from all previous generations
            # or one of the best from the immediately preceding generation.
            # More advanced: tournament, diversity-based selection, etc.
            
            # Get best program from all prior generations as a potential parent
            best_overall_parents = self.db.get_best_programs(
                problem_name=self.problem_name,
                primary_metric=self.problem_config["PRIMARY_METRIC"],
                metric_goal=self.problem_config["EVALUATION_METRICS"][self.problem_config["PRIMARY_METRIC"]]["goal"],
                n=max(1, population_size_per_gen // 2) # Get a few best ones
            )

            if not best_overall_parents:
                print(f"Warning: No suitable parent programs found from previous generations for Gen {gen}. "
                      "This might happen if all prior attempts failed correctness checks. "
                      "Attempting to use generation 0 seed if available.")
                best_overall_parents = gen0_programs # Fallback to gen 0 seeds
                if not best_overall_parents:
                    print("Error: No generation 0 programs found either. Cannot proceed.")
                    break # Stop evolution

            # Create tasks for generating new individuals
            generation_tasks = []
            for i in range(population_size_per_gen):
                # Simple parent selection: cycle through best parents or pick randomly from them
                parent_program_data = random.choice(best_overall_parents)
                generation_tasks.append(self._generate_and_evaluate_individual(gen, parent_program_data))
            
            results = await asyncio.gather(*generation_tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, Exception):
                    print(f"  Error generating/evaluating individual: {result}")
                elif result: # Result is (program_id, scores)
                    program_id, scores = result
                    print(f"  Successfully generated and evaluated individual. ID: {program_id}, Scores: {scores}")
                    newly_generated_this_gen +=1
                attempts_this_gen +=1


            gen_duration = time.time() - start_time_gen
            print(f"--- Generation {gen} Summary ---")
            print(f"  Individuals attempted: {attempts_this_gen}")
            print(f"  New individuals added to DB: {newly_generated_this_gen}")
            print(f"  Generation duration: {gen_duration:.2f} seconds")

            # Display best program so far for this problem
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
        await self.llm_manager.close_provider() # Important to close LLM client session

    async def _generate_and_evaluate_individual(
        self,
        current_generation: int,
        parent_program_data: Dict[str, Any]
    ) -> Optional[tuple[int, Dict[str, float]]]:
        """
        Generates one new individual program, evaluates it, and stores it.
        """
        parent_id = parent_program_data["id"]
        parent_code = parent_program_data["code_content"]
        
        print(f"  Generating from parent ID: {parent_id} (Gen {parent_program_data['generation']})")

        try:
            # For MVP, context_examples can be None or simple.
            # Later, this could include other high-scoring programs from the DB as "inspirations".
            evolved_code_str = await self.llm_manager.generate_code_modification(
                current_code=parent_code,
                model_name=self.llm_model_name,
                prompt_instructions=self.problem_config["PROBLEM_LLM_INSTRUCTIONS"],
                temperature=self.experiment_config.get("llm_temperature", 0.7), # Allow experiment override
                max_tokens=self.experiment_config.get("llm_max_tokens", 2048)
            )

            if not evolved_code_str or not evolved_code_str.strip():
                print("  LLM returned empty or whitespace-only code. Skipping evaluation.")
                return None

            # print(f"  Evolved code by LLM:\n```python\n{evolved_code_str[:200]}...\n```") # For debugging

            evolved_scores = self.evaluator.evaluate(evolved_code_str, self.problem_config)
            
            # Store the new program
            new_program_id = self.db.add_program(
                problem_name=self.problem_name,
                code_content=evolved_code_str,
                generation=current_generation,
                scores=evolved_scores,
                parent_id=parent_id
                # Could also store llm_prompt and llm_diff if we capture them
            )
            # print(f"    Stored new program ID: {new_program_id}, Scores: {evolved_scores}")
            return new_program_id, evolved_scores

        except Exception as e:
            print(f"  Exception during generation/evaluation for parent {parent_id}: {e}")
            # Optionally, log this error more formally
            return None


if __name__ == '__main__':
    import os
    # sys, Path, asyncio, importlib.util are already imported or handled by top try-except

    async def test_controller():
        print("--- Testing EvolutionaryController ---")

        # Ensure .env is loaded for settings
        if not hasattr(settings, 'DEFAULT_LLM_PROVIDER'):
            print("Error: Settings not loaded. Ensure .env is configured and script is run from project root.")
            return
        
        # For this test, we'll use the simple_line_reducer problem
        problem_module_path = "evocoder.problems.simple_line_reducer.problem_config"
        
        # Example experiment config (could be loaded from a YAML/JSON file in a real scenario)
        # This will use the default OpenWebUI settings from .env
        # To test with a specific model from your OpenWebUI for this run:
        experiment_settings = {
            "llm_provider": "open_webui", # Could be settings.DEFAULT_LLM_PROVIDER
            # "open_webui_model_name": "llama3.2", # Override if needed, else uses from .env via settings
            "llm_temperature": 0.5,
            "llm_max_tokens": 512
        }
        
        # Ensure the data directory for the default DB path exists
        db_data_dir = Path(__file__).resolve().parent.parent.parent / "data"
        db_data_dir.mkdir(parents=True, exist_ok=True)
        # Clean up any previous test DB for a fresh run
        test_db_for_controller = db_data_dir / "evocoder_programs.db" # Default DB name
        if test_db_for_controller.exists():
            print(f"Deleting existing test DB: {test_db_for_controller}")
            test_db_for_controller.unlink()


        try:
            controller = EvolutionaryController(
                problem_config_path=problem_module_path,
                experiment_config=experiment_settings
            )
            # Seeding is now called within run_evolution if needed
            # controller.seed_initial_program() 
            
            await controller.run_evolution(num_generations=2, population_size_per_gen=3)

        except FileNotFoundError as fnf:
            print(f"Test Error: File not found - {fnf}. Ensure problem files exist.")
            print("  INITIAL_CODE_FILE should be at: evocoder/problems/simple_line_reducer/initial_code.py")
            print("  TEST_SUITE_FILE should be at: evocoder/problems/simple_line_reducer/test_suite.py")
        except ValueError as ve:
            print(f"Test Error: ValueError - {ve}. Check configurations.")
        except ImportError as ie:
            print(f"Test Error: ImportError - {ie}. Check module paths or problem config path.")
        except Exception as e:
            print(f"An unexpected error occurred in controller test: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Clean up test DB after controller test
            if test_db_for_controller.exists():
                print(f"Test finished. Deleting test DB: {test_db_for_controller}")
                # controller.db._get_connection().close() # Ensure connection is closed before unlink
                if hasattr(controller, 'llm_manager'): # Ensure controller was initialized
                    await controller.llm_manager.close_provider() # Close LLM if manager was created
                # test_db_for_controller.unlink() # Might cause issues if connection not fully closed by GC

    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(test_controller())
