# evocoder/config/settings.py

import os
from dotenv import load_dotenv
from pathlib import Path

# Determine the project root directory.
# This assumes settings.py is in evocoder/config/ and .env is in the project root (evocoder/).
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent # Moves up three levels: config -> evocoder -> project_root

# Construct the path to the .env file
DOTENV_PATH = PROJECT_ROOT / ".env"

# Load environment variables from .env file
# If .env is not found, it will not raise an error,
# so variables might be None if not set in the environment either.
load_dotenv(dotenv_path=DOTENV_PATH)

# --- LLM Provider Configuration ---

# Open WebUI (or any OpenAI-compatible API) Settings
OPEN_WEBUI_API_KEY: str | None = os.getenv("OPEN_WEBUI_API_KEY")
OPEN_WEBUI_BASE_URL: str | None = os.getenv("OPEN_WEBUI_BASE_URL")
OPEN_WEBUI_MODEL_NAME: str | None = os.getenv("OPEN_WEBUI_MODEL_NAME")

# --- Default LLM Configuration (can be overridden by experiment configs later) ---
# This allows us to specify which provider and model to use by default.
# For now, we'll assume Open WebUI is the default.
DEFAULT_LLM_PROVIDER: str = "open_webui" # A key to identify the provider, e.g., "open_webui", "gemini", "ollama"

# --- General Application Settings ---
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()


# --- Validation (Optional but Recommended) ---
# You can add checks here to ensure critical variables are set.
# For example, for the default provider:
if DEFAULT_LLM_PROVIDER == "open_webui":
    if not OPEN_WEBUI_API_KEY:
        print("Warning: OPEN_WEBUI_API_KEY is not set in .env or environment variables.")
    if not OPEN_WEBUI_BASE_URL:
        print("Warning: OPEN_WEBUI_BASE_URL is not set in .env or environment variables.")
    if not OPEN_WEBUI_MODEL_NAME:
        print("Warning: OPEN_WEBUI_MODEL_NAME is not set in .env or environment variables.")

# You could expand this with more providers or more sophisticated config loading (e.g., Pydantic) in later phases.

if __name__ == "__main__":
    # Quick test to see if variables are loaded
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Dotenv Path: {DOTENV_PATH}")
    print(f"Open WebUI API Key Loaded: {'Yes' if OPEN_WEBUI_API_KEY else 'No'}")
    print(f"Open WebUI Base URL: {OPEN_WEBUI_BASE_URL}")
    print(f"Open WebUI Model Name: {OPEN_WEBUI_MODEL_NAME}")
    print(f"Default LLM Provider: {DEFAULT_LLM_PROVIDER}")
    print(f"Log Level: {LOG_LEVEL}")


