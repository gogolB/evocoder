# evocoder/config/settings.py

import os
from dotenv import load_dotenv
from pathlib import Path
from typing import Optional, Any # Added Optional, Any

# Determine the project root directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent 

# Construct the path to the .env file
DOTENV_PATH = PROJECT_ROOT / ".env"

# Load environment variables from .env file
load_dotenv(dotenv_path=DOTENV_PATH)

# --- LLM Provider Configuration ---

# Open WebUI (or any OpenAI-compatible API) Settings
OPEN_WEBUI_API_KEY: Optional[str] = os.getenv("OPEN_WEBUI_API_KEY")
OPEN_WEBUI_BASE_URL: Optional[str] = os.getenv("OPEN_WEBUI_BASE_URL")
OPEN_WEBUI_MODEL_NAME: Optional[str] = os.getenv("OPEN_WEBUI_MODEL_NAME")

# Google Gemini API Settings
GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL_NAME: Optional[str] = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-pro-latest") 
GEMINI_SAFETY_SETTINGS: Optional[Any] = os.getenv("GEMINI_SAFETY_SETTINGS", None) 


# --- Default LLM Configuration ---
DEFAULT_LLM_PROVIDER: str = os.getenv("DEFAULT_LLM_PROVIDER", "open_webui")

# --- General Application Settings ---
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()
# NEW: Define a path for a global log file, can be overridden by logger setup
LOG_FILE_PATH: Optional[str] = os.getenv("LOG_FILE_PATH", None) # Example: "data/evocoder_app.log"


# --- Validation (Optional but Recommended) ---
if DEFAULT_LLM_PROVIDER == "open_webui":
    # API key might be optional for some OpenWebUI setups, so no strict check here unless base_url is not localhost
    if not OPEN_WEBUI_BASE_URL:
        print("Warning: OPEN_WEBUI_BASE_URL is not set in .env or environment variables.")
    if not OPEN_WEBUI_MODEL_NAME:
        print("Warning: OPEN_WEBUI_MODEL_NAME is not set in .env or environment variables.")
elif DEFAULT_LLM_PROVIDER == "gemini":
    if not GEMINI_API_KEY:
        print("Warning: GEMINI_API_KEY is not set in .env or environment variables for Gemini provider.")
    # GEMINI_MODEL_NAME has a default, so no warning if not set.


if __name__ == "__main__":
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Dotenv Path: {DOTENV_PATH}")
    print(f"--- OpenWebUI Config ---")
    print(f"  API Key Loaded: {'Yes' if OPEN_WEBUI_API_KEY else 'No / Optional'}")
    print(f"  Base URL: {OPEN_WEBUI_BASE_URL}")
    print(f"  Model Name: {OPEN_WEBUI_MODEL_NAME}")
    print(f"--- Gemini Config ---")
    print(f"  API Key Loaded: {'Yes' if GEMINI_API_KEY else 'No'}")
    print(f"  Model Name: {GEMINI_MODEL_NAME}")
    print(f"  Safety Settings: {GEMINI_SAFETY_SETTINGS}")
    print(f"--- General Config ---")
    print(f"  Default LLM Provider: {DEFAULT_LLM_PROVIDER}")
    print(f"  Log Level: {LOG_LEVEL}")
    print(f"  Log File Path: {LOG_FILE_PATH}") # NEW print for testing

