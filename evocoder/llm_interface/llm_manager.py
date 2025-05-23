# evocoder/llm_interface/llm_manager.py

from typing import Dict, Any, Optional, List, Type
import importlib
import sys 
from pathlib import Path 

try:
    from ..config import settings
    from .base_llm_provider import BaseLLMProvider
except ImportError:
    if __name__ == '__main__':
        file_path = Path(__file__).resolve()
        project_root = file_path.parent.parent.parent 
        sys.path.insert(0, str(project_root))
        
        from evocoder.config import settings
        from evocoder.llm_interface.base_llm_provider import BaseLLMProvider
    else:
        raise


# A registry to map provider names (from config) to their classes
# This allows for easy extension with new providers.
PROVIDER_REGISTRY: Dict[str, str] = {
    "open_webui": "evocoder.llm_interface.providers.open_webui_provider.OpenWebUIProvider",
    "gemini": "evocoder.llm_interface.providers.gemini_provider.GeminiProvider", # ADDED THIS LINE
    # Add other providers here as they are implemented, e.g.:
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
            provider_name (Optional[str]): The name of the LLM provider to use (e.g., "open_webui", "gemini").
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
            if 'api_key' not in current_provider_config: # Allow override from llm_config
                current_provider_config['api_key'] = settings.OPEN_WEBUI_API_KEY
            if 'base_url' not in current_provider_config:
                current_provider_config['base_url'] = settings.OPEN_WEBUI_BASE_URL
        elif self.provider_name == "gemini": # ADDED ELIF FOR GEMINI
            if 'api_key' not in current_provider_config:
                current_provider_config['api_key'] = settings.GEMINI_API_KEY
            # GeminiProvider doesn't require a base_url from us.
            # It might take other specific params like 'safety_settings' via kwargs
            if 'safety_settings' not in current_provider_config and hasattr(settings, 'GEMINI_SAFETY_SETTINGS'):
                current_provider_config['safety_settings'] = settings.GEMINI_SAFETY_SETTINGS
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
        Generates a code modification (expected as a diff string or full code) from the configured LLM.
        """
        if not self.provider:
            raise RuntimeError("LLM provider has not been initialized.")

        messages_for_provider: List[Dict[str, str]] = []
        if context_examples:
            messages_for_provider.extend(context_examples)
        
        full_user_prompt = f"{prompt_instructions}\n\nHere is the current Python code to modify:\n```python\n{current_code}\n```"
        
        final_prompt_for_provider = full_user_prompt
        final_context_for_provider = messages_for_provider if messages_for_provider else None
        
        try:
            response_string = await self.provider.generate_response(
                prompt=final_prompt_for_provider, 
                model_name=model_name, 
                context=final_context_for_provider, 
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            return response_string
        except Exception as e:
            print(f"Error during LLM generation via provider '{self.provider_name}': {e}")
            raise

    async def close_provider(self):
        if self.provider:
            await self.provider.close()

if __name__ == '__main__':
    import asyncio
    import os 
    
    try:
        from evocoder.config import settings 
    except ImportError as e:
        print(f"Failed to import modules for LLMManager test: {e}")
        sys.exit(1)

    async def main_test():
        print("--- Testing LLMManager (Phase 2 - With Gemini Provider Option) ---")
        
        if not hasattr(settings, 'DEFAULT_LLM_PROVIDER'):
            print("Error: 'settings' module does not seem to be loaded correctly for the test.")
            return

        # Test 1: Default provider (OpenWebUI)
        print(f"\nTesting with default provider from settings: {settings.DEFAULT_LLM_PROVIDER}")
        manager_default = None
        if settings.DEFAULT_LLM_PROVIDER == "open_webui":
            if not settings.OPEN_WEBUI_API_KEY or \
               not settings.OPEN_WEBUI_BASE_URL or \
               not settings.OPEN_WEBUI_MODEL_NAME:
                print("OpenWebUI settings not fully configured. Skipping default provider test.")
            else:
                try:
                    manager_default = LLMManager() # Uses settings.DEFAULT_LLM_PROVIDER
                    # ... (rest of the OpenWebUI test from previous version can be adapted here if desired) ...
                    print(f"Successfully initialized LLMManager with OpenWebUI.")
                except Exception as e:
                    print(f"Error initializing LLMManager with OpenWebUI: {e}")
                finally:
                    if manager_default: await manager_default.close_provider()
        
        # Test 2: Specifically test GeminiProvider (mocked, as no live calls)
        print(f"\nAttempting to initialize LLMManager with GeminiProvider (mocked test)...")
        manager_gemini = None
        if not settings.GEMINI_API_KEY:
            print("GEMINI_API_KEY not set in .env. Cannot initialize GeminiProvider for this test.")
        else:
            try:
                # We are testing if LLMManager can *instantiate* GeminiProvider.
                # The GeminiProvider itself will use the API key for genai.configure().
                # No actual API call will be made by this __main__ block for Gemini.
                gemini_llm_config = {'api_key': settings.GEMINI_API_KEY} # Pass only necessary for init
                manager_gemini = LLMManager(provider_name="gemini", llm_config=gemini_llm_config)
                print(f"Successfully initialized LLMManager with GeminiProvider.")
                
                # Example of how one might call it (won't run an actual LLM call here)
                # test_code = "def f(): return 1"
                # test_instr = "Make this function return 2. Provide diff."
                # gemini_model_for_test = getattr(settings, "GEMINI_MODEL_NAME", "gemini-1.5-pro-latest")
                # print(f"If we were to call Gemini model '{gemini_model_for_test}' (not actually calling)...")
                # This would require a running event loop and actual call, skip for now.
                # diff = await manager_gemini.generate_code_modification(test_code, gemini_model_for_test, test_instr)
                # print(f"Hypothetical diff: {diff}")

            except ValueError as ve: # Catches init errors from GeminiProvider or LLMManager
                print(f"ValueError during GeminiProvider initialization test: {ve}")
            except ImportError as ie:
                print(f"ImportError during GeminiProvider test: {ie}")
            except Exception as e:
                print(f"An unexpected error occurred during GeminiProvider test: {e}")
            finally:
                if manager_gemini: await manager_gemini.close_provider()

    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main_test())
