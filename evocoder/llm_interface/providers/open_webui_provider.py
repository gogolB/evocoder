# evocoder/llm_interface/providers/open_webui_provider.py

import httpx
import json
from typing import List, Dict, Any, Optional
import sys # For path manipulation if run as script
from pathlib import Path # For path manipulation

# Attempt relative import for package use, fallback for direct script execution
try:
    from ..base_llm_provider import BaseLLMProvider
except ImportError:
    # This block executes if the relative import fails,
    # which happens when the script is run directly.
    # We adjust sys.path to allow absolute import of BaseLLMProvider.
    if __name__ == '__main__': # Only adjust path if this script is the entry point
        # Determine the project root. This script is in:
        # evocoder/evocoder/llm_interface/providers/open_webui_provider.py
        # The 'evocoder' package root is 3 levels up from this file's directory.
        # The project root (containing the top-level 'evocoder' dir) is 4 levels up.
        file_path = Path(__file__).resolve()
        # Path to the 'evocoder' package directory (evocoder/evocoder/)
        package_dir_parent = file_path.parent.parent.parent 
        # Path to the project root (evocoder/)
        project_root = package_dir_parent.parent 

        # Add the project root to sys.path so 'from evocoder. ...' works
        sys.path.insert(0, str(project_root))
        
        # Now attempt absolute import
        from evocoder.llm_interface.base_llm_provider import BaseLLMProvider
    else:
        # If not __main__ and relative import failed, it's a genuine issue
        raise

class OpenWebUIProvider(BaseLLMProvider):
    """
    Concrete LLM provider implementation for an Open WebUI instance
    (or any OpenAI-compatible API endpoint).
    """

    def __init__(
        self,
        api_key: Optional[str],
        base_url: Optional[str], # e.g., "http://chat-api.preview.tamu.ai"
        **kwargs: Any
    ):
        """
        Initialize the OpenWebUIProvider.

        Args:
            api_key (Optional[str]): The API key for Open WebUI.
            base_url (Optional[str]): The base URL of the Open WebUI API endpoint.
                                      Should not include '/api/chat/completions'.
            **kwargs (Any): Additional provider-specific configuration arguments.
        """
        super().__init__(api_key=api_key, base_url=base_url, **kwargs)
        if not self.base_url:
            raise ValueError("OpenWebUIProvider requires a 'base_url'.")

        self.api_endpoint = f"{self.base_url.rstrip('/')}/api/chat/completions"
        self._client = httpx.AsyncClient(timeout=kwargs.get("timeout", 60.0))

    async def generate_response(
        self,
        prompt: str,
        model_name: str,
        context: Optional[List[Dict[str, str]]] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None, 
        stream: bool = False, 
        **kwargs: Any
    ) -> str:
        """
        Generate a response from the Open WebUI compatible LLM.
        """
        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        messages = []
        if context:
            messages.extend(context)
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            "stream": stream,
        }

        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        
        payload.update(kwargs)

        try:
            response = await self._client.post(
                self.api_endpoint,
                headers=headers,
                json=payload,
            )
            response.raise_for_status() 
            response_data = response.json()

            if response_data.get("choices") and isinstance(response_data["choices"], list) and len(response_data["choices"]) > 0:
                first_choice = response_data["choices"][0]
                if first_choice.get("message") and isinstance(first_choice["message"], dict):
                    content = first_choice["message"].get("content")
                    if isinstance(content, str): # Check if content is a string
                        return content
                    elif content is None:
                        # Specific error if content is None
                        error_message = f"LLM returned None content for model {model_name}. Response: {response_data}"
                        print(f"Error: {error_message}")
                        raise ValueError(error_message)
            
            # Fallback for other unexpected structures
            error_message = f"Unexpected response structure from OpenWebUI or missing content: {response_data}"
            print(f"Error: {error_message}") 
            raise ValueError(error_message)

        except httpx.HTTPStatusError as e:
            error_body = ""
            try:
                error_body = e.response.text
            except Exception:
                pass 
            print(f"HTTP error occurred: {e.response.status_code} - {e.request.url} - Body: {error_body}")
            raise
        except httpx.RequestError as e:
            print(f"Request error occurred while connecting to OpenWebUI: {e}")
            raise
        except json.JSONDecodeError as e:
            print(f"Failed to decode JSON response from OpenWebUI: {e.doc}")
            raise ValueError(f"Invalid JSON response from OpenWebUI: {e.msg}") from e
        except Exception as e:
            print(f"An unexpected error occurred with OpenWebUIProvider: {e}")
            raise

    async def close(self) -> None:
        """
        Close the underlying HTTPX client session.
        """
        if hasattr(self, '_client') and self._client is not None:
            await self._client.aclose()

if __name__ == "__main__":
    import asyncio
    import os
    # sys and Path are already imported at the top if this block is reached due to ImportError

    # The sys.path modification is done in the `except ImportError` block at the top
    # if this script is run directly.

    # Now, attempt absolute imports for the testing block,
    # assuming sys.path has been correctly modified by the except block.
    try:
        from evocoder.config import settings
        # BaseLLMProvider should already be in scope due to the try-except at the top
        # If you need to reference it explicitly for type hints here, it's available.
    except ImportError as e:
        print(f"Error importing 'evocoder.config.settings' for standalone test: {e}")
        print("This usually means the script was not run from the project root directory "
              "OR the sys.path modification in the initial except block failed.")
        print(f"Current sys.path: {sys.path}")
        sys.exit(1)

    async def test_open_webui_provider():
        # --- START: Configurable parameters for easy testing ---
        # To use a different prompt for this test, change the string below:
        custom_test_prompt: str = "Say 'Hello, EvoCoder!'"
        # To use a different model for this test (overriding .env), set it here.
        # If set to None or empty string, it will use OPEN_WEBUI_MODEL_NAME from .env
        custom_model_to_test: Optional[str] = "protected.llama3.2" # Example: "ollama_chat/llama3" or your specific model ID
        # --- END: Configurable parameters ---

        if not settings.OPEN_WEBUI_BASE_URL:
            print("OPEN_WEBUI_BASE_URL not found in settings/environment.")
            print("Skipping OpenWebUIProvider live test.")
            return
        if not settings.OPEN_WEBUI_API_KEY:
             print("Warning: OPEN_WEBUI_API_KEY not found. Proceeding without API key.")

        effective_model_name = custom_model_to_test if custom_model_to_test else settings.OPEN_WEBUI_MODEL_NAME

        print(f"Testing OpenWebUIProvider with:")
        print(f"  Base URL: {settings.OPEN_WEBUI_BASE_URL}")
        print(f"  API Key: {'Provided' if settings.OPEN_WEBUI_API_KEY else 'Not Provided / Optional'}")
        print(f"  Model for this test: {effective_model_name}") # Uses custom if set, else from settings

        provider = OpenWebUIProvider(
            api_key=settings.OPEN_WEBUI_API_KEY,
            base_url=settings.OPEN_WEBUI_BASE_URL
        )

        try:
            if not effective_model_name:
                 print("No model name specified (neither custom_model_to_test nor OPEN_WEBUI_MODEL_NAME in .env). Cannot run test.")
                 await provider.close()
                 return

            print(f"\nSending prompt to {effective_model_name}: '{custom_test_prompt}'")
            
            response = await provider.generate_response(
                prompt=custom_test_prompt,
                model_name=effective_model_name, 
                temperature=0.5,
                max_tokens=150
            )
            print(f"\nReceived response from {effective_model_name}:\n---\n{response}\n---")

        except Exception as e:
            print(f"An error occurred during the OpenWebUIProvider test: {e}")
        finally:
            await provider.close()

    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(test_open_webui_provider())
