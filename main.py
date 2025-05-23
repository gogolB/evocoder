# main.py (EvoCoder CLI Entry Point - Updated for YAML Experiment Configs)

import asyncio
import typer
from typing import Optional, Dict, Any 
from typing_extensions import Annotated 
from pathlib import Path
import os
import sys
import yaml # For loading YAML files

# Adjust sys.path to ensure the evocoder package can be imported
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

try:
    from evocoder.core.evolutionary_controller import EvolutionaryController
    from evocoder.config import settings # For global defaults if not in YAML
    from evocoder.utils.logger import setup_logger # For main CLI logging
except ImportError as e:
    print(f"Error importing EvoCoder modules: {e}")
    print("Please ensure you are running this script from the project root 'evocoder/' directory,")
    print("and that the 'evocoder' package is correctly in your PYTHONPATH or installed.")
    print(f"Current sys.path: {sys.path}")
    sys.exit(1)

# Setup a logger for the main CLI application
cli_logger = setup_logger("evocoder.cli")

app = typer.Typer(
    name="evocoder",
    help="EvoCoder: An AI agent to iteratively evolve code solutions using LLMs.",
    add_completion=False
)

def load_experiment_config_from_yaml(config_path: Path) -> Dict[str, Any]:
    """Loads experiment configuration from a YAML file."""
    if not config_path.exists() or not config_path.is_file():
        cli_logger.error(f"Experiment configuration file not found: {config_path}")
        raise typer.Exit(code=1)
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            exp_config = yaml.safe_load(f)
        cli_logger.info(f"Successfully loaded experiment configuration from: {config_path}")
        return exp_config
    except yaml.YAMLError as ye:
        cli_logger.error(f"Error parsing YAML configuration file {config_path}: {ye}")
        raise typer.Exit(code=1)
    except Exception as e:
        cli_logger.error(f"Unexpected error loading configuration file {config_path}: {e}")
        raise typer.Exit(code=1)


@app.command()
def run(
    experiment_config_file: Annotated[Path, typer.Argument(
        help="Path to the YAML experiment configuration file.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True
    )],
    # Optional overrides for very common parameters, if desired.
    # For now, let's assume most things are in the YAML.
    # num_generations_override: Annotated[Optional[int], typer.Option("--generations", "-g")] = None
):
    """
    Run an evolutionary process based on an experiment configuration YAML file.
    """
    cli_logger.info(f"Starting EvoCoder run with experiment config: {experiment_config_file}")

    # Load the experiment configuration from YAML
    exp_config = load_experiment_config_from_yaml(experiment_config_file)

    # Extract configurations, providing defaults or falling back to global settings
    problem_module = exp_config.get("problem_module")
    if not problem_module:
        cli_logger.error("Experiment config YAML must specify 'problem_module'.")
        raise typer.Exit(code=1)

    evolution_params = exp_config.get("evolution_params", {})
    num_generations = evolution_params.get("num_generations", 5) # Default if not in YAML
    population_size_per_gen = evolution_params.get("population_size_per_gen", 5) # Default
    num_inspirations = evolution_params.get("num_inspirations", 1)
    tournament_size_parent = evolution_params.get("tournament_size_parent", 3)
    max_concurrent_tasks = evolution_params.get("max_concurrent_tasks", 3)

    # LLM settings from YAML, to be passed to EvolutionaryController's experiment_config
    # The controller will then merge these with global settings from .env
    controller_experiment_config: Dict[str, Any] = exp_config.get("llm_settings", {}).copy()
    
    # Add evolutionary params to the controller_experiment_config so controller can access them
    # if it needs to (e.g., for LLM temp/tokens which are currently under llm_settings)
    # Or, we can pass them separately to controller.run_evolution.
    # For now, let's keep llm_settings separate for clarity in YAML.
    # The controller's __init__ will primarily use llm_settings for provider setup.
    # Temperature & max_tokens for LLM calls are already handled by controller's experiment_config.

    # Database path override
    db_path_str = exp_config.get("database_path")
    if db_path_str:
        controller_experiment_config["database_path"] = Path(db_path_str) # Pass to controller if it handles it
        cli_logger.info(f"Using custom database path for this run: {db_path_str}")
    
    # Logging overrides (logger is already set up globally, but can reconfigure for this run)
    log_level_override = exp_config.get("log_level")
    log_file_override = exp_config.get("log_file")

    if log_level_override or log_file_override:
        cli_logger.info("Reconfiguring logger for this experiment run...")
        # Reconfigure the main 'evocoder' logger and potentially others
        # This is a simple way; a more complex app might have a dedicated config function.
        # For now, let's assume the logger setup in each module respects global settings first.
        # Overriding here would require passing these to each module's setup_logger call,
        # or having setup_logger re-read from a dynamic config source.
        # For simplicity, we'll note that these YAML settings are for *intent* and
        # the logger setup in logger.py currently reads from global settings.
        # A more advanced setup would pass these overrides down.
        if log_level_override:
            cli_logger.info(f"  Requested Log Level: {log_level_override} (Note: Apply via .env or global settings for now)")
        if log_file_override:
            cli_logger.info(f"  Requested Log File: {log_file_override} (Note: Apply via .env or global settings for now)")


    # Ensure the data directory for the default DB path exists if no override
    if not db_path_str:
        db_data_dir = Path(__file__).resolve().parent / "data"
        db_data_dir.mkdir(parents=True, exist_ok=True)
        default_db_name = "evocoder_programs.db"
        db_for_run = db_data_dir / default_db_name
        if db_for_run.exists():
            cli_logger.warning(f"Default database '{db_for_run}' exists. Appending to it unless specific DB is set in YAML.")

    try:
        # Pass the problem_module and the llm_settings part as experiment_config to controller
        controller = EvolutionaryController(
            problem_config_path=problem_module,
            experiment_config=controller_experiment_config # This contains LLM provider/model/param overrides
        )
        
        asyncio.run(controller.run_evolution(
            num_generations=num_generations,
            population_size_per_gen=population_size_per_gen,
            num_inspirations=num_inspirations,
            tournament_size_parent=tournament_size_parent,
            max_concurrent_tasks=max_concurrent_tasks
        ))
        
        cli_logger.info(f"\nEvoCoder run finished successfully for {problem_module}.")

    except FileNotFoundError as fnf:
        cli_logger.error(f"ERROR: File not found - {fnf}", exc_info=True)
        typer.secho("Please ensure problem files (initial_code.py, test_suite.py) exist at the configured paths.", fg=typer.colors.RED)
    except (ValueError, ImportError) as e: 
        cli_logger.error(f"ERROR: Configuration or Import Error - {e}", exc_info=True)
        typer.secho("Please check your problem configuration path and ensure all dependencies are installed.", fg=typer.colors.RED)
    except Exception as e:
        cli_logger.error(f"An unexpected error occurred during the EvoCoder run: {e}", exc_info=True)
        # import traceback
        # traceback.print_exc()

@app.command()
def list_problems():
    """
    Lists available problem configurations found in the 'evocoder/problems/' directory.
    """
    problems_dir = current_dir / "evocoder" / "problems"
    cli_logger.info("Available problem configurations:")
    if not problems_dir.exists() or not problems_dir.is_dir():
        cli_logger.warning(f"  Problems directory not found at: {problems_dir}")
        return

    found_problems = False
    for item in problems_dir.iterdir():
        if item.is_dir() and (item / "problem_config.py").exists():
            problem_module_path = f"evocoder.problems.{item.name}.problem_config"
            cli_logger.info(f"  - {item.name} (module: {problem_module_path})")
            found_problems = True
    if not found_problems:
        cli_logger.warning("  No valid problem configurations found.")


if __name__ == "__main__":
    if os.name == 'nt': 
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    app()
