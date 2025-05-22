# evocoder/llm_interface/llm_manager.py

from typing import Dict, Any, Optional, List, Type
import importlib
import sys # For path manipulation if run as script
from pathlib import Path # For path manipulation

# Attempt relative import for package use, fallback for direct script execution
try:
    from ..config import settings
    from .base_llm_provider import BaseLLMProvider
except ImportError:
    # This block executes if the relative import fails (e.g., when run directly)
    if __name__ == '__main__':
        # Adjust sys.path to allow absolute imports for the __main__ block
        file_path = Path(__file__).resolve()
        # llm_manager.py is in evocoder/evocoder/llm_interface/
        # project_root is 3 levels up from this file's directory
        project_root = file_path.parent.parent.parent
        sys.path.insert(0, str(project_root))
        
        from evocoder.config import settings
        from evocoder.llm_interface.base_llm_provider import BaseLLMProvider
    else:
        # If not __main__ and relative import failed, it's a genuine issue elsewhere
        raise


# A registry to map provider names (from config) to their classes
# This allows for easy extension with new providers.
PROVIDER_REGISTRY: Dict[str, str] = {
    "open_webui": "evocoder.llm_interface.providers.open_webui_provider.OpenWebUIProvider",
    # Add other providers here as they are implemented, e.g.:
    # "gemini": "evocoder.llm_interface.providers.gemini_provider.GeminiProvider",
    # "ollama_direct": "evocoder.llm_interface.providers.ollama_provider.OllamaProvider",
}

class LLMManager:
    """
    Manages the instantiation and interaction with a configured LLM provider.
    This class acts as a factory and a unified interface for LLM operations.
    """

    def __init__(self, provider_name: Optional[str] = None, llm_config: Optional[Dict[str, Any]] = None):
        """
        Initializes the LLMManager.

        It will select and instantiate an LLM provider based on the provided
        `provider_name` or the `DEFAULT_LLM_PROVIDER` from settings.
        Provider-specific configurations are also loaded from settings.

        Args:
            provider_name (Optional[str]): The name of the LLM provider to use (e.g., "open_webui").
                                           If None, uses DEFAULT_LLM_PROVIDER from settings.
            llm_config (Optional[Dict[str, Any]]): A dictionary containing specific configurations
                                                   for the LLM provider (e.g., api_key, base_url, model_name).
                                                   If None, attempts to load from global settings.
        
        Raises:
            ValueError: If the specified provider is not supported or configuration is missing.
            ImportError: If the provider class cannot be imported.
        """
        self.provider_name = provider_name or settings.DEFAULT_LLM_PROVIDER
        self.llm_config = llm_config if llm_config is not None else {}
        self.provider: Optional[BaseLLMProvider] = None

        self._load_provider()

    def _get_provider_class(self, provider_key: str) -> Type[BaseLLMProvider]:
        """Dynamically imports and returns the provider class."""
        if provider_key not in PROVIDER_REGISTRY:
            raise ValueError(f"Unsupported LLM provider: {provider_key}. "
                             f"Available providers: {list(PROVIDER_REGISTRY.keys())}")
        
        module_path_str, class_name = PROVIDER_REGISTRY[provider_key].rsplit('.', 1)
        try:
            # importlib.import_module expects a string like 'evocoder.llm_interface.providers.open_webui_provider'
            # This should work if the project root (containing the top-level 'evocoder' dir) is in sys.path
            module = importlib.import_module(module_path_str)
            provider_class = getattr(module, class_name)
            if not issubclass(provider_class, BaseLLMProvider):
                raise TypeError(f"Provider class {provider_class.__name__} does not inherit from BaseLLMProvider.")
            return provider_class
        except ImportError as e:
            raise ImportError(f"Could not import LLM provider module {module_path_str}: {e}")
        except AttributeError:
            raise ImportError(f"Could not find class {class_name} in module {module_path_str}")


    def _load_provider(self):
        """
        Loads and instantiates the configured LLM provider using settings.
        """
        provider_class = self._get_provider_class(self.provider_name)
        
        current_provider_config = self.llm_config.copy()

        if self.provider_name == "open_webui":
            if 'api_key' not in current_provider_config:
                current_provider_config['api_key'] = settings.OPEN_WEBUI_API_KEY
            if 'base_url' not in current_provider_config:
                current_provider_config['base_url'] = settings.OPEN_WEBUI_BASE_URL
        else:
            print(f"Warning: No specific global settings load path for provider '{self.provider_name}'. "
                  "Relying on llm_config passed to LLMManager or provider defaults.")

        try:
            self.provider = provider_class(**current_provider_config)
            print(f"LLMManager initialized with provider: {self.provider_name}")
        except Exception as e:
            raise ValueError(f"Failed to instantiate LLM provider '{self.provider_name}': {e}")


    async def generate_code_modification(
        self,
        current_code: str, 
        model_name: str,   
        prompt_instructions: str, 
        context_examples: Optional[List[Dict[str, str]]] = None, 
        temperature: float = 0.7,
        max_tokens: Optional[int] = 2048, 
        **kwargs: Any 
    ) -> str:
        """
        Generates a code modification suggestion from the configured LLM.
        (Args, Returns, Raises documentation remains the same)
        """
        if not self.provider:
            raise RuntimeError("LLM provider has not been initialized.")
        
        messages_context: List[Dict[str, str]] = []
        if context_examples:
            messages_context.extend(context_examples)
        
        full_user_prompt = f"{prompt_instructions}\n\n```python\n{current_code}\n```"
        
        try:
            return await self.provider.generate_response(
                prompt=full_user_prompt,
                model_name=model_name, 
                context=messages_context if messages_context else None,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
        except Exception as e:
            print(f"Error during LLM generation via provider '{self.provider_name}': {e}")
            raise


    async def close_provider(self):
        """Closes any open connections or resources held by the LLM provider."""
        if self.provider:
            await self.provider.close()
            # print(f"LLM provider '{self.provider_name}' connection closed.") # Quieter


if __name__ == '__main__':
    import asyncio
    import os # For os.name check
    # sys and Path are already imported at the top if this block is reached due to ImportError

    # The sys.path modification for `settings` and `BaseLLMProvider` is handled
    # by the try-except block at the very top of the file when run as __main__.
    # So, `settings` and `BaseLLMProvider` should be available here.

    async def main_test():
        print("--- Testing LLMManager ---")
        
        # Ensure settings were loaded (they are used implicitly by LLMManager constructor)
        if not hasattr(settings, 'DEFAULT_LLM_PROVIDER'):
            print("Error: 'settings' module does not seem to be loaded correctly for the test.")
            return

        print(f"\nTesting with default provider: {settings.DEFAULT_LLM_PROVIDER}")
        if settings.DEFAULT_LLM_PROVIDER == "open_webui":
            if not settings.OPEN_WEBUI_API_KEY or \
               not settings.OPEN_WEBUI_BASE_URL or \
               not settings.OPEN_WEBUI_MODEL_NAME:
                print("OpenWebUI settings (API_KEY, BASE_URL, MODEL_NAME) are not fully configured in .env.")
                print("Skipping LLMManager live test with OpenWebUI.")
                return
        
        manager = None # Define manager outside try block for finally
        try:
            manager = LLMManager() # Uses DEFAULT_LLM_PROVIDER from settings
            
            test_code = "def hello():\n    print('Hello world!')\n    # A redundant line\n    return None"
            test_instructions = "Optimize the following Python function to be more concise. Remove redundant lines. Return the full modified function."
            
            # Determine the model name to use from settings, specific to the default provider
            model_for_test = ""
            if settings.DEFAULT_LLM_PROVIDER == "open_webui":
                model_for_test = settings.OPEN_WEBUI_MODEL_NAME
            # Add elif for other default providers if necessary
            
            if not model_for_test:
                print(f"Could not determine a model name for the default provider '{settings.DEFAULT_LLM_PROVIDER}' from settings.")
                print("Skipping generation test.")
                return

            print(f"\nAttempting to generate code modification for {settings.DEFAULT_LLM_PROVIDER} via LLMManager...")
            print(f"Using model: {model_for_test}")
            
            modified_code = await manager.generate_code_modification(
                current_code=test_code,
                model_name=model_for_test, 
                prompt_instructions=test_instructions,
                temperature=0.2,
                max_tokens=100
            )
            print("\nLLM Response (Modified Code Suggestion):")
            print("```python")
            print(modified_code)
            print("```")

        except ValueError as ve:
            print(f"ValueError during LLMManager test: {ve}")
        except ImportError as ie:
            print(f"ImportError during LLMManager test: {ie}")
        except RuntimeError as re:
            print(f"RuntimeError during LLMManager test: {re}")
        except Exception as e:
            print(f"An unexpected error occurred during LLMManager test: {e}")
        finally:
            if manager and hasattr(manager, 'provider') and manager.provider: # Check if provider was initialized
                await manager.close_provider()
            elif manager : # Manager was created but provider might not have been
                print("LLMManager was created but provider might not have been initialized or already closed.")

    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main_test())
