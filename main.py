# main.py (EvoCoder CLI Entry Point)

import asyncio
import typer
from typing import Optional, Dict, Any # Added Optional, Dict, Any
from typing_extensions import Annotated # For Typer < 0.9 compatibility with Python < 3.9 for Optional type hints
from pathlib import Path
import os
import sys

# Adjust sys.path to ensure the evocoder package can be imported
# This is useful if running `python main.py` from the project root.
# If installed as a package, this might not be strictly necessary,
# but it's good for development.
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

try:
    from evocoder.core.evolutionary_controller import EvolutionaryController
    from evocoder.config import settings # To access default settings if needed
except ImportError as e:
    print(f"Error importing EvoCoder modules: {e}")
    print("Please ensure you are running this script from the project root 'evocoder/' directory,")
    print("and that the 'evocoder' package is correctly in your PYTHONPATH or installed.")
    print(f"Current sys.path: {sys.path}")
    sys.exit(1)

app = typer.Typer(
    name="evocoder",
    help="EvoCoder: An AI agent to iteratively evolve code solutions using LLMs.",
    add_completion=False
)

@app.command()
def run(
    problem_config_module: Annotated[str, typer.Option(
        help="Dotted path to the problem configuration module (e.g., 'evocoder.problems.simple_line_reducer.problem_config')."
    )] = "evocoder.problems.simple_line_reducer.problem_config", # Default to our example
    num_generations: Annotated[int, typer.Option(
        "--generations", "-g",
        help="Number of generations to run the evolution."
    )] = 5,
    population_size_per_gen: Annotated[int, typer.Option(
        "--pop-size", "-p",
        help="Number of new individuals to generate per generation."
    )] = 5,
    llm_provider: Annotated[Optional[str], typer.Option(
        "--llm-provider",
        help="LLM provider to use (e.g., 'open_webui'). Overrides default from settings."
    )] = None,
    llm_model_name: Annotated[Optional[str], typer.Option(
        "--llm-model",
        help="Specific LLM model name to use. Overrides defaults for the chosen provider."
    )] = None,
    llm_temperature: Annotated[Optional[float], typer.Option(
        "--llm-temp",
        help="LLM generation temperature. Overrides defaults."
    )] = None,
    llm_max_tokens: Annotated[Optional[int], typer.Option(
        "--llm-max-tokens",
        help="LLM max tokens for generation. Overrides defaults."
    )] = None,
    # Add more experiment-specific overrides here if needed, e.g., API keys, base URLs
    # For simplicity, API keys and base URLs are primarily expected from .env via settings.py
    # but could be made configurable here for advanced use cases or CI.
):
    """
    Run an evolutionary process for a given problem configuration.
    """
    typer.echo(f"Starting EvoCoder run...")
    typer.echo(f"Problem Configuration Module: {problem_config_module}")
    typer.echo(f"Number of Generations: {num_generations}")
    typer.echo(f"Population Size per Generation: {population_size_per_gen}")

    experiment_config: Dict[str, Any] = {}
    if llm_provider:
        experiment_config["llm_provider"] = llm_provider
        typer.echo(f"Using LLM Provider (from CLI): {llm_provider}")
    
    # Model name override depends on the provider.
    # The EvolutionaryController's _get_llm_model_name will handle provider-specific keys.
    if llm_model_name:
        # We need to know which provider this model name applies to.
        # If llm_provider is also set, we can be specific.
        # Otherwise, this model name might be for the default provider.
        provider_key_for_model = llm_provider if llm_provider else settings.DEFAULT_LLM_PROVIDER
        if provider_key_for_model == "open_webui":
            experiment_config["open_webui_model_name"] = llm_model_name
        # Add elif for other providers if you want to allow model override for them
        # elif provider_key_for_model == "gemini":
        #     experiment_config["gemini_model_name"] = llm_model_name
        else:
            experiment_config["default_model_name"] = llm_model_name # Generic override
        typer.echo(f"Using LLM Model (from CLI for provider '{provider_key_for_model}'): {llm_model_name}")

    if llm_temperature is not None:
        experiment_config["llm_temperature"] = llm_temperature
        typer.echo(f"Using LLM Temperature (from CLI): {llm_temperature}")
    if llm_max_tokens is not None:
        experiment_config["llm_max_tokens"] = llm_max_tokens
        typer.echo(f"Using LLM Max Tokens (from CLI): {llm_max_tokens}")

    # Ensure the data directory for the default DB path exists
    # This is also done in ProgramDatabase, but good to ensure here too.
    db_data_dir = Path(__file__).resolve().parent / "data"
    db_data_dir.mkdir(parents=True, exist_ok=True)
    
    # Clean up any previous test DB for a fresh run if it's the default name
    # This is mainly for consistent testing; in production, you might not want to delete.
    # For now, let's make it conditional or remove for general runs.
    # For the MVP, let's assume we always start fresh with the default DB for a run via CLI.
    default_db_name = "evocoder_programs.db"
    db_for_run = db_data_dir / default_db_name
    if db_for_run.exists():
        typer.echo(f"Note: Default database '{db_for_run}' exists. For a clean run, delete it manually or it will be appended to.")
        # For a truly fresh run every time:
        # typer.echo(f"Deleting existing default DB for a fresh run: {db_for_run}")
        # db_for_run.unlink()


    try:
        controller = EvolutionaryController(
            problem_config_path=problem_config_module,
            experiment_config=experiment_config
        )
        
        asyncio.run(controller.run_evolution(
            num_generations=num_generations,
            population_size_per_gen=population_size_per_gen
        ))
        
        typer.echo(f"\nEvoCoder run finished successfully for {problem_config_module}.")

    except FileNotFoundError as fnf:
        typer.secho(f"ERROR: File not found - {fnf}", fg=typer.colors.RED, bold=True)
        typer.secho("Please ensure problem files (initial_code.py, test_suite.py) exist at the configured paths.", fg=typer.colors.RED)
    except (ValueError, ImportError) as e: # Catch config or import errors
        typer.secho(f"ERROR: Configuration or Import Error - {e}", fg=typer.colors.RED, bold=True)
        typer.secho("Please check your problem configuration path and ensure all dependencies are installed.", fg=typer.colors.RED)
    except Exception as e:
        typer.secho(f"An unexpected error occurred during the EvoCoder run: {e}", fg=typer.colors.RED, bold=True)
        # For more detailed debugging, you might want to re-raise or print traceback
        # import traceback
        # traceback.print_exc()

@app.command()
def list_problems():
    """
    Lists available problem configurations found in the 'evocoder/problems/' directory.
    (This is a basic implementation; could be made more robust)
    """
    problems_dir = current_dir / "evocoder" / "problems"
    typer.echo("Available problem configurations:")
    if not problems_dir.exists() or not problems_dir.is_dir():
        typer.secho(f"  Problems directory not found at: {problems_dir}", fg=typer.colors.YELLOW)
        return

    found_problems = False
    for item in problems_dir.iterdir():
        if item.is_dir() and (item / "problem_config.py").exists():
            problem_module_path = f"evocoder.problems.{item.name}.problem_config"
            typer.echo(f"  - {item.name} (module: {problem_module_path})")
            found_problems = True
    if not found_problems:
        typer.secho("  No valid problem configurations found.", fg=typer.colors.YELLOW)


# Add more commands later, e.g., for inspecting the database, listing LLM providers.

if __name__ == "__main__":
    # This makes the script runnable as `python main.py`
    # and also allows Typer to work correctly when installed as a package.
    if os.name == 'nt': # Windows specific asyncio policy for CLI apps
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    app()
