# EvoCoder - Architecture Overview
This document provides a more detailed look into the architecture of the EvoCoder system, outlining its major components and how they interact to achieve evolutionary code generation.

## Core Philosophy
EvoCoder is designed as a modular and extensible framework. The core idea is to separate concerns:

- Problem Definition: What code to evolve and how to measure its success.

- LLM Interaction: How to communicate with different Large Language Models.

- Evolutionary Control: The overarching algorithm that drives the search for better code.

- Evaluation: How to execute and assess the generated code.

- Persistence: Storing and retrieving programs and their metadata.

- Configuration: Managing settings for experiments and global parameters.

# Major Components and Data Flow
The system revolves around several key Python modules and a defined workflow:

1. main.py (Command-Line Interface - CLI)

    - Role: Serves as the primary user entry point. It uses Typer for a structured CLI.

    - Functionality:

        - Parses command-line arguments.

        - Loads experiment configurations from specified YAML files (evocoder/experiments/).

        - Initializes global settings (e.g., from .env via evocoder/config/settings.py).

        - Instantiates and invokes the EvolutionaryController to start an evolutionary run.

        - Provides utility commands (e.g., list-problems).

    - Interaction: Takes user input and experiment config, then delegates to EvolutionaryController.

2. Experiment Configuration (YAML files in evocoder/experiments/)

    - Role: Defines all parameters for a specific evolutionary run.

    - Content:

        - problem_module: Dotted path to the problem definition package (e.g., evocoder.problems.numerical_optimizer.problem_config).

        - evolution_params: Settings for the evolutionary algorithm (e.g., num_generations, population_size_per_gen, num_inspirations, tournament_size_parent, max_concurrent_tasks).

        - llm_settings: Configuration for the LLM, including the provider (e.g., "open_webui"), provider-specific model names (e.g., open_webui_model_name), and generation parameters (e.g., temperature, max_tokens_diff). Can also include overrides for API keys or base URLs.

        - database_path (optional): Path to a specific SQLite database file for the experiment.

        - log_level, log_file (optional): Overrides for logging settings for the experiment.

    - Interaction: Read by `main.py` and the resulting configuration dictionary is passed to the EvolutionaryController.

3. `evocoder/config/settings.py`

    - Role: Loads global default configurations and sensitive information (like API keys) from a .env file.

    - Interaction: Provides fallback settings if not specified in an experiment's YAML file. Used by LLMManager and potentially other components for default behaviors.

4. evocoder/core/EvolutionaryController.py (EvolutionaryController class)

    - Role: The central orchestrator of the evolutionary process.

    - Functionality:

        - Initializes other core components (LLMManager, ProgramDatabase, Evaluator) based on the loaded problem and experiment configurations.

        - Manages the main evolutionary loop (generations).

        - Handles seeding of the initial program.

        - Selection: Implements parent selection strategies (e.g., tournament selection prioritizing correctness then a primary metric) by querying the ProgramDatabase. It also selects "inspiration programs" from a diverse pool.

        - Variation: Coordinates with LLMManager to generate code modifications (diffs) for selected parent programs, providing context (parent code, inspiration programs, problem-specific instructions).

        - Modification Application: Uses diff_utils.py to apply LLM-generated diffs to parent code.

        - Evaluation Trigger: Sends the newly generated/modified code to the Evaluator.

        - Population Update: Stores new individuals (code, scores, diffs, lineage) in the ProgramDatabase.

        - Manages asynchronous tasks for generating and evaluating individuals concurrently, respecting max_concurrent_tasks via an asyncio.Semaphore.

    - Interaction: Drives the entire process, coordinating between the database, LLM interface, and evaluator.

5. `evocoder/problems/<problem_name>/` (Problem Definition Packages)

    - Role: Defines a specific task for EvoCoder. Each problem is a self-contained package.

    - Key Files:

        - problem_config.py: Contains problem-specific metadata: PROBLEM_NAME, TARGET_FUNCTION_NAME, paths to INITIAL_CODE_FILE and TEST_SUITE_FILE, EVALUATION_METRICS (including goals like "maximize"/"minimize"), PRIMARY_METRIC, CORRECTNESS_THRESHOLD, detailed PROBLEM_LLM_INSTRUCTIONS (including diff format guidance), and the EVALUATION_CASCADE definition.

        - initial_code.py: The starting Python code that will be evolved.

        - test_suite.py: pytest-compatible tests used by the Evaluator. Tests can be marked with pytest markers (e.g., @pytest.mark.correctness, @pytest.mark.precision) to be run in specific stages of the evaluation cascade.

    - Interaction: The problem_config.py is loaded by EvolutionaryController to configure the run. initial_code.py provides the seed. test_suite.py is used by the Evaluator.

6. `evocoder/llm_interface/` (LLM Interaction Layer)

    - BaseLLMProvider.py (BaseLLMProvider class): An abstract base class defining the common interface for all LLM providers (e.g., an async def generate_response(...) method).

    - providers/<provider_name>_provider.py (e.g., OpenWebUIProvider.py): Concrete implementations of BaseLLMProvider for specific LLM services or APIs. They handle API-specific authentication, request formatting, and response parsing.

    - LLMManager.py (LLMManager class):

        - Acts as a factory and a unified interface to the selected LLM provider.

        - Dynamically loads and instantiates the configured provider based on PROVIDER_REGISTRY and settings (from experiment config or global settings).

        - The generate_code_modification method constructs the final prompt (including parent code, inspiration examples, and problem-specific instructions asking for diffs) and calls the active provider's generate_response method.

    - Interaction: EvolutionaryController uses LLMManager to abstract away LLM provider details. LLMManager uses a specific provider instance to make API calls.

7. evocoder/utils/diff_utils.py

    - Role: Provides utilities for handling code modifications in diff format.

    - Functionality:

        - parse_diff_string(): Parses a string containing one or more SEARCH/REPLACE diff blocks into DiffBlock objects.

        - apply_diffs(): Applies a list of DiffBlock objects to an original code string to produce the modified code.

    - Interaction: Used by EvolutionaryController after receiving a diff string from the LLMManager.

8. evocoder/core/evaluator_cascade.py (Evaluator class)

    - Role: Evaluates the functional correctness and other quality metrics of a given code string.

    - Functionality:

        - Reads the EVALUATION_CASCADE definition from the problem's configuration.

        - Iterates through the defined stages.

        - For "pytest" stages:

            - Creates a temporary, isolated environment.

            - Writes the evolved code to a temporary module (evolved_module.py).

            - Copies and modifies the problem's test_suite.py to import from this temporary module.

            - Runs pytest programmatically (as a subprocess for isolation, using asyncio.to_thread) on the modified test suite, potentially filtering tests by markers specified in the cascade stage.

            - Parses pytest output to determine pass/fail counts.

            - Handles fail_fast_if_not_all_passed logic for cascade stages.

        - Calculates and returns a dictionary of scores based on the outcomes of the cascade stages and other direct measurements (like line count via AST parsing using _count_function_lines).

    - Interaction: Called by EvolutionaryController with the evolved code string and problem configuration.

9. evocoder/core/program_database.py (ProgramDatabase class)

    - Role: Manages persistence of programs and their associated data.

    - Functionality:

        - Uses SQLite as the backend.

        - Initializes the database schema (programs table).

        - Methods to add_program (stores code, generation, parent ID, JSON scores, LLM diff), get_program_by_id, get_best_programs (sorted by a primary metric, with fallback for older SQLite), get_programs_by_generation, and get_random_correct_programs.

    - Interaction: Used by EvolutionaryController to store new individuals and to retrieve candidates for selection (parents, inspirations).

10. evocoder/utils/logger.py

    - Role: Provides a centralized and configurable logging setup using Python's standard logging module.

    - Functionality: setup_logger function to create and configure named logger instances with consistent formatting, log levels (from settings), and optional file output.

    - Interaction: Imported and used by all major components (EvolutionaryController, LLMManager, Evaluator, ProgramDatabase, main.py) for logging messages.

## Data Flow Summary (Evolutionary Loop)
1. EvolutionaryController fetches a diverse pool of candidate programs (best, random correct, recent) from ProgramDatabase.

2. From this pool, a parent program is selected using tournament selection (correctness first, then primary metric). Inspiration programs are also selected.

3. The parent code, inspiration programs, and problem-specific instructions (asking for a diff) are passed to LLMManager.

4. LLMManager uses the configured LLM provider (e.g., OpenWebUIProvider) to send a prompt to the LLM.

5. The LLM returns a diff string.

6. EvolutionaryController uses diff_utils.apply_diffs to apply the diff to the parent code, creating the evolved code string.

7. The evolved code string is passed to the Evaluator.

8. Evaluator runs the evaluation cascade (e.g., pytest stages with markers) and returns a dictionary of scores.

9. EvolutionaryController stores the evolved code, its scores, the original diff string, and lineage information into the ProgramDatabase.

10. The loop repeats for the configured number of generations. All significant events and errors are logged via the logging system.

This architecture aims for a balance of power, flexibility, and maintainability, allowing EvoCoder to be adapted to various code evolution tasks and LLM backends.
