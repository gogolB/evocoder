# evocoder/config/settings.py

import os
from typing import Any, Optional
from dotenv import load_dotenv
from pathlib import Path

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

# Google Gemini API Settings (NEW)
GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL_NAME: Optional[str] = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-pro-latest") # Default if not in .env
# Optional: Gemini Safety Settings (can be a JSON string or loaded differently)
# Example: "BLOCK_MEDIUM_AND_ABOVE" for HARM_CATEGORY_HARASSMENT, etc.
# For simplicity, we'll assume it's None or handled by the provider if not set.
GEMINI_SAFETY_SETTINGS: Optional[Any] = os.getenv("GEMINI_SAFETY_SETTINGS", None) 


# --- Default LLM Configuration (can be overridden by experiment configs later) ---
DEFAULT_LLM_PROVIDER: str = os.getenv("DEFAULT_LLM_PROVIDER", "open_webui")

# --- General Application Settings ---
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()


# --- Validation (Optional but Recommended) ---
# This section can be expanded as more providers are added.
# For now, it just prints warnings if default provider settings are missing.
if DEFAULT_LLM_PROVIDER == "open_webui":
    if not OPEN_WEBUI_API_KEY and OPEN_WEBUI_BASE_URL != "http://localhost:8080": # Example: local might not need key
        # Adjust condition if your specific OpenWebUI needs a key
        # print("Warning: OPEN_WEBUI_API_KEY is not set in .env or environment variables for non-default local URL.")
        pass # API key might be optional for some OpenWebUI setups
    if not OPEN_WEBUI_BASE_URL:
        print("Warning: OPEN_WEBUI_BASE_URL is not set in .env or environment variables.")
    if not OPEN_WEBUI_MODEL_NAME:
        print("Warning: OPEN_WEBUI_MODEL_NAME is not set in .env or environment variables.")
elif DEFAULT_LLM_PROVIDER == "gemini":
    if not GEMINI_API_KEY:
        print("Warning: GEMINI_API_KEY is not set in .env or environment variables for Gemini provider.")
    if not GEMINI_MODEL_NAME:
        print("Warning: GEMINI_MODEL_NAME is not set for Gemini provider (will use default).")


if __name__ == "__main__":
    # Quick test to see if variables are loaded
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

