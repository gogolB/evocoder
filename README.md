# EvoCoder

## Description

EvoCoder is a Python-based AI system inspired by the [AlphaEvolve paper by Google DeepMind](#inspiration). It uses an evolutionary algorithm to guide a Large Language Model (LLM) in generating and refining code solutions for well-defined problems. The core capability of EvoCoder is to iteratively improve code through LLM interaction and automated evaluation, leveraging techniques like diff-based modifications and evaluation cascades.

## Key Features

EvoCoder incorporates a range of features to facilitate automated code evolution:

* **Evolutionary Core:** Implements a generational evolutionary algorithm orchestrated by the `EvolutionaryController` to iteratively refine code solutions.
* **Generic LLM Interface:** A flexible interface (`BaseLLMProvider`, `LLMManager`) allows EvoCoder to connect to various LLM providers.
    * Currently includes `OpenWebUIProvider` for OpenAI-compatible APIs (successfully tested with "protected.Claude 3.7 Sonnet" via an Open WebUI instance).
* **Diff-Based Code Modifications:** Supports requesting code changes from LLMs in a SEARCH/REPLACE diff format. These diffs are then parsed (`diff_utils.py`) and applied to the existing code.
* **Automated Evaluation Engine:**
    * Utilizes `pytest` for robustly testing the correctness and other characteristics of evolved code.
    * Implements an **Evaluation Cascade** system, allowing multi-stage testing (e.g., basic correctness, precision, convergence) defined per problem. This enables efficient filtering of candidates.
* **Program Database:** Employs an SQLite database (`ProgramDatabase`) to store all evolved programs, their lineage (parent ID), evaluation scores (as JSON), and the raw LLM-generated diffs that led to them.
* **Configuration System:**
    * Global settings (like API keys, default provider, log levels) are managed via a `.env` file (loaded by `evocoder/config/settings.py`).
    * Experiment-specific configurations (problem choice, evolutionary parameters, LLM overrides) are managed through YAML files located in the `evocoder/experiments/` directory.
* **Logging System:** An integrated, structured logging system (`evocoder/utils/logger.py`) is used throughout the core components for improved debugging, monitoring, and traceability of evolutionary runs.
* **Command-Line Interface (CLI):** A `typer`-based CLI (`main.py`) provides user-friendly commands to:
    * Run evolutionary experiments using YAML configuration files.
    * List available problem definitions.
* **Problem Definition Framework:** A clear and extensible structure for defining new problems for EvoCoder to solve. Each problem is a Python package within `evocoder/problems/` and includes:
    * `problem_config.py`: Defines problem name, target function, evaluation metrics, LLM instructions, and the evaluation cascade.
    * `initial_code.py`: The starting code for the LLM to evolve.
    * `test_suite.py`: `pytest` tests used by the evaluator.
    * Implemented example problems: "simple\_line\_reducer" and "numerical\_optimizer".
* **Selection Strategies:** Includes tournament selection for choosing parent programs, prioritizing correctness and then a primary metric.
* **Asynchronous Operations:** Leverages `asyncio` for concurrent LLM API calls and evaluations (running `pytest` in separate threads and using an `asyncio.Semaphore` for concurrency limiting) to improve throughput.

## Current Status

EvoCoder has completed its initial development phases, resulting in a functional system capable of evolving code solutions for the defined problems ("simple\_line\_reducer", "numerical\_optimizer"). Key features like diff-based evolution, evaluation cascades, YAML-based configuration, and integrated logging are implemented and have been tested.

More comprehensive documentation, including detailed architecture diagrams and developer guides, is planned for the future.

## Project Setup

### Prerequisites

* Python 3.12+

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd evocoder
    ```
2.  **Install Poetry:**
    If you don't have Poetry installed, follow the instructions on the [official Poetry website](https://python-poetry.org/docs/#installation).
3.  **Install Dependencies:**
    Navigate to the project root (`evocoder/`) and run:
    ```bash
    poetry install
    ```
    This will create a virtual environment and install all necessary dependencies listed in `pyproject.toml`.

### Configuration (`.env` file)

EvoCoder requires environment variables for configuration, especially for LLM provider API keys and endpoints.

1.  Copy the example environment file:
    ```bash
    cp .env.example .env
    ```
2.  Edit the `.env` file with your specific configurations. Key variables include:
    * `OPEN_WEBUI_API_KEY`: Your API key if your Open WebUI instance requires one.
    * `OPEN_WEBUI_BASE_URL`: The base URL for your Open WebUI API (e.g., `http://chat-api.preview.tamu.ai`).
    * `OPEN_WEBUI_MODEL_NAME`: The model identifier to use via Open WebUI (e.g., "protected.Claude 3.7 Sonnet", "llama3.2").
    * `GEMINI_API_KEY`: (Placeholder) Your Google Gemini API key if you plan to use the direct `GeminiProvider`.
    * `GEMINI_MODEL_NAME`: (Placeholder) Default model for Gemini (e.g., "gemini-1.5-pro-latest").
    * `DEFAULT_LLM_PROVIDER`: (Optional, defaults to "open\_webui" in `settings.py`) Can be set to "open\_webui" or "gemini" (once provider is fully tested).
    * `LOG_LEVEL`: (Optional, defaults to "INFO") Set to "DEBUG" for more verbose logging.
    * `LOG_FILE_PATH`: (Optional, e.g., `data/evocoder_run.log`) If set, logs will also be written to this file.

### Data Directory

The application will automatically create a `data/` directory in the project root if it doesn't exist. This directory is used to store:
* The SQLite database file (e.g., `evocoder_programs.db` or experiment-specific databases).
* Log files (if `LOG_FILE_PATH` is configured).

## Usage / How to Run

All commands should be run from the root of the `evocoder` project directory, after activating the Poetry environment.

1.  **Activate the Virtual Environment:**
    ```bash
    poetry shell
    ```

2.  **Running an Experiment:**
    Experiments are primarily configured and run using YAML files.
    ```bash
    python main.py run path/to/your_experiment_config.yaml
    ```
    An example configuration for the "numerical\_optimizer" problem is provided in:
    `evocoder/experiments/numerical_optimizer_default.yaml`

    To run this example:
    ```bash
    python main.py run evocoder/experiments/numerical_optimizer_default.yaml
    ```

3.  **Listing Available Problems:**
    To see the problems EvoCoder is aware of (based on subdirectories in `evocoder/problems/`):
    ```bash
    python main.py list-problems
    ```

4.  **Running Tests:**
    To run the unit and integration tests for the project:
    ```bash
    pytest
    ```
    Or to run tests for a specific file:
    ```bash
    pytest tests/core/test_evaluator_cascade.py 
    ```

5.  **Defining a New Problem (Overview):**
    To add a new problem for EvoCoder to solve:
    * Create a new subdirectory within `evocoder/problems/`, e.g., `evocoder/problems/my_new_problem/`.
    * Add an empty `__init__.py` file to make it a package.
    * Inside this directory, create three key files:
        * `problem_config.py`: Defines `PROBLEM_NAME`, paths to `INITIAL_CODE_FILE` and `TEST_SUITE_FILE`, `TARGET_FUNCTION_NAME`, `EVALUATION_METRICS`, `PROBLEM_LLM_INSTRUCTIONS`, and an `EVALUATION_CASCADE`. Refer to existing problems for examples.
        * `initial_code.py`: Contains the initial Python code (e.g., a function or class) that EvoCoder will evolve.
        * `test_suite.py`: Contains `pytest`-compatible tests to evaluate the correctness and other desired properties of the evolved code. Use `@pytest.mark.<marker_name>` to tag tests for different stages of the evaluation cascade.

6.  **Experiment Configuration (Overview):**
    * Experiment files (e.g., in `evocoder/experiments/`) are YAML files that specify:
        * `problem_module`: The dotted path to the problem's `problem_config.py`.
        * `evolution_params`: Parameters like `num_generations`, `population_size_per_gen`, etc.
        * `llm_settings`: Overrides for LLM provider, model name, temperature, etc.
        * `database_path` (optional): For experiment-specific databases.
        * `log_level`, `log_file` (optional): For experiment-specific logging.

## Project Structure

A brief overview of the main directories:

* `evocoder/`: The root project directory.
    * `main.py`: The main CLI entry point.
    * `evocoder/`: The main Python source package.
        * `config/`: Global configuration loading (`settings.py`).
        * `core/`: Core evolutionary logic (`EvolutionaryController`, `Evaluator`, `ProgramDatabase`).
        * `llm_interface/`: Generic LLM interface (`BaseLLMProvider`, `LLMManager`) and specific provider implementations (e.g., `OpenWebUIProvider`).
        * `problems/`: Contains subdirectories for each problem definition (config, initial code, tests).
        * `utils/`: Shared utility modules (logging, diff tools).
        * `cli/`: Modules for CLI command logic.
    * `experiments/`: Contains YAML configuration files for different evolutionary runs.
    * `tests/`: Contains all automated tests (`pytest`).
    * `docs/`: (Planned) For more detailed documentation like architecture diagrams and developer guides.
    * `.env.example`: Template for environment variable configuration.
    * `pyproject.toml`: Project metadata and dependencies for Poetry.

## How it Works (High-Level Architecture Overview)

EvoCoder operates through an evolutionary loop orchestrated by the `EvolutionaryController`:

1.  **Configuration:** An experiment run is initiated via the CLI (`main.py`), loading an **Experiment YAML file**. This file defines the target problem, evolutionary parameters, and LLM settings. Global defaults are also loaded from `.env` via `settings.py`.
2.  **Initialization:** The `EvolutionaryController` initializes:
    * The `ProgramDatabase` (SQLite) to store and retrieve program versions, scores, and diffs. The initial code for the problem is seeded if the database is empty for that problem.
    * The `LLMManager`, which loads the configured LLM provider (e.g., `OpenWebUIProvider`).
    * The `Evaluator`, which is configured by the chosen problem's `problem_config.py`.
3.  **Evolutionary Loop (Generations):**
    * **Selection:** Parents for the new generation are selected from the `ProgramDatabase` using **tournament selection** (prioritizing correctness, then the primary metric defined in the problem config). A diverse pool of candidates (best, random correct, recent) is considered.
    * **Variation (LLM Interaction):**
        * For each selected parent, "inspiration programs" (other good/diverse solutions) are also selected.
        * The `LLMManager` constructs a prompt containing the parent code, inspiration examples, and specific instructions (from `problem_config.py`) asking the LLM to produce modifications in a **diff format** (SEARCH/REPLACE blocks).
        * The configured LLM provider sends this prompt to the LLM.
    * **Modification Application:** The diff string returned by the LLM is parsed and applied to the parent code using `diff_utils.py` to create a new candidate program.
    * **Evaluation:** The new candidate program is evaluated by the `Evaluator`:
        * The `Evaluator` runs an **Evaluation Cascade** defined in the problem's configuration.
        * Each stage in the cascade (e.g., "correctness tests", "precision tests") runs `pytest` on the problem's `test_suite.py`, potentially using specific markers.
        * Stages can be "fail-fast". Scores for defined metrics are updated based on stage outcomes.
    * **Database Update:** The new program, its evaluation scores, and the LLM diff are stored in the `ProgramDatabase`.
4.  **Iteration:** The loop repeats for a configured number of generations.
5.  **Logging:** A structured **logging system** records the process, including LLM interactions, evaluation results, and errors, to console and/or a file.

## Inspiration

This project is inspired by the concepts and approach presented in the paper:
**"AlphaEvolve: A coding agent for scientific and algorithmic discovery"** by Novikov et al., Google DeepMind.

## License

This project is licensed under the **MIT License**.
