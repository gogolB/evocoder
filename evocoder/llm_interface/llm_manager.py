# evocoder/llm_interface/llm_manager.py

from typing import Dict, Any, Optional, List, Type
import importlib
import sys 
from pathlib import Path 

try:
    from ..config import settings
    from .base_llm_provider import BaseLLMProvider
    from ..utils.logger import setup_logger # Import the logger setup function
except ImportError:
    if __name__ == '__main__':
        file_path = Path(__file__).resolve()
        project_root = file_path.parent.parent.parent 
        sys.path.insert(0, str(project_root))
        
        from evocoder.config import settings
        from evocoder.llm_interface.base_llm_provider import BaseLLMProvider
        from evocoder.utils.logger import setup_logger # Import for __main__
    else:
        raise

PROVIDER_REGISTRY: Dict[str, str] = {
    "open_webui": "evocoder.llm_interface.providers.open_webui_provider.OpenWebUIProvider",
    "gemini": "evocoder.llm_interface.providers.gemini_provider.GeminiProvider",
}

class LLMManager:
    """
    Manages the instantiation and interaction with a configured LLM provider.
    """

    def __init__(self, provider_name: Optional[str] = None, llm_config: Optional[Dict[str, Any]] = None):
        self.logger = setup_logger(f"evocoder.llm_manager.{self.__class__.__name__}") # Initialize logger
        self.logger.info(f"Initializing LLMManager with provider_name='{provider_name}', llm_config='{llm_config is not None}'")

        self.provider_name = provider_name or settings.DEFAULT_LLM_PROVIDER
        self.llm_config = llm_config if llm_config is not None else {}
        self.provider: Optional[BaseLLMProvider] = None

        self._load_provider()

    def _get_provider_class(self, provider_key: str) -> Type[BaseLLMProvider]:
        """Dynamically imports and returns the provider class."""
        self.logger.debug(f"Attempting to get provider class for key: {provider_key}")
        if provider_key not in PROVIDER_REGISTRY:
            self.logger.error(f"Unsupported LLM provider: {provider_key}. Available: {list(PROVIDER_REGISTRY.keys())}")
            raise ValueError(f"Unsupported LLM provider: {provider_key}. "
                             f"Available providers: {list(PROVIDER_REGISTRY.keys())}")
        
        module_path_str, class_name = PROVIDER_REGISTRY[provider_key].rsplit('.', 1)
        try:
            self.logger.debug(f"Importing module: {module_path_str}, class: {class_name}")
            module = importlib.import_module(module_path_str)
            provider_class = getattr(module, class_name)
            if not issubclass(provider_class, BaseLLMProvider):
                self.logger.error(f"Provider class {provider_class.__name__} does not inherit from BaseLLMProvider.")
                raise TypeError(f"Provider class {provider_class.__name__} does not inherit from BaseLLMProvider.")
            return provider_class
        except ImportError as e:
            self.logger.exception(f"Could not import LLM provider module {module_path_str}: {e}")
            raise
        except AttributeError:
            self.logger.exception(f"Could not find class {class_name} in module {module_path_str}")
            raise

    def _load_provider(self):
        """
        Loads and instantiates the configured LLM provider using settings.
        """
        self.logger.info(f"Loading LLM provider: {self.provider_name}")
        provider_class = self._get_provider_class(self.provider_name)
        
        current_provider_config = self.llm_config.copy()

        if self.provider_name == "open_webui":
            if 'api_key' not in current_provider_config: 
                current_provider_config['api_key'] = settings.OPEN_WEBUI_API_KEY
            if 'base_url' not in current_provider_config:
                current_provider_config['base_url'] = settings.OPEN_WEBUI_BASE_URL
            self.logger.debug(f"OpenWebUI provider config: api_key_present={current_provider_config.get('api_key') is not None}, base_url='{current_provider_config.get('base_url')}'")
        elif self.provider_name == "gemini": 
            if 'api_key' not in current_provider_config:
                current_provider_config['api_key'] = settings.GEMINI_API_KEY
            if 'safety_settings' not in current_provider_config and hasattr(settings, 'GEMINI_SAFETY_SETTINGS'):
                current_provider_config['safety_settings'] = settings.GEMINI_SAFETY_SETTINGS
            self.logger.debug(f"Gemini provider config: api_key_present={current_provider_config.get('api_key') is not None}")
        else:
            self.logger.warning(f"No specific global settings load path for provider '{self.provider_name}'. "
                                "Relying on llm_config passed to LLMManager or provider defaults.")

        try:
            self.provider = provider_class(**current_provider_config)
            self.logger.info(f"LLMManager initialized successfully with provider: {self.provider_name}")
        except Exception as e:
            self.logger.exception(f"Failed to instantiate LLM provider '{self.provider_name}': {e}")
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
            self.logger.error("LLM provider has not been initialized before calling generate_code_modification.")
            raise RuntimeError("LLM provider has not been initialized.")

        self.logger.debug(f"Generating code modification. Model: {model_name}, Temp: {temperature}, MaxTokens: {max_tokens}")
        self.logger.debug(f"Prompt instructions (first 100 chars): {prompt_instructions[:100]}...")
        if context_examples:
            self.logger.debug(f"Number of context_examples: {len(context_examples)}")

        messages_for_provider: List[Dict[str, str]] = []
        if context_examples:
            messages_for_provider.extend(context_examples)
        
        full_user_prompt = f"{prompt_instructions}\n\nHere is the current Python code to modify:\n```python\n{current_code}\n```"
        
        final_prompt_for_provider = full_user_prompt
        final_context_for_provider = messages_for_provider if messages_for_provider else None
        
        try:
            self.logger.debug(f"Calling provider.generate_response for model '{model_name}'.")
            response_string = await self.provider.generate_response(
                prompt=final_prompt_for_provider, 
                model_name=model_name, 
                context=final_context_for_provider, 
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            self.logger.debug(f"Provider returned response (first 100 chars): {response_string[:100]}...")
            return response_string
        except Exception as e:
            self.logger.exception(f"Error during LLM generation via provider '{self.provider_name}' for model '{model_name}': {e}")
            raise

    async def close_provider(self):
        if self.provider:
            self.logger.info(f"Closing LLM provider: {self.provider_name}")
            await self.provider.close()
            self.logger.info(f"LLM provider '{self.provider_name}' connection closed.")


if __name__ == '__main__':
    import asyncio
    import os 
    
    # Logger for the __main__ block
    main_test_logger = setup_logger("evocoder.test_llm_manager", level_str="DEBUG")

    try:
        from evocoder.config import settings 
    except ImportError as e:
        main_test_logger.error(f"Failed to import modules for LLMManager test: {e}")
        sys.exit(1)

    async def main_test():
        main_test_logger.info("--- Testing LLMManager (Phase 3 - Integrated Logging) ---")
        
        if not hasattr(settings, 'DEFAULT_LLM_PROVIDER'):
            main_test_logger.error("'settings' module does not seem to be loaded correctly for the test.")
            return

        main_test_logger.info(f"Testing with default provider from settings: {settings.DEFAULT_LLM_PROVIDER}")
        manager_default = None
        if settings.DEFAULT_LLM_PROVIDER == "open_webui":
            if not settings.OPEN_WEBUI_API_KEY or \
               not settings.OPEN_WEBUI_BASE_URL or \
               not settings.OPEN_WEBUI_MODEL_NAME:
                main_test_logger.warning("OpenWebUI settings not fully configured. Skipping default provider live test.")
            else:
                try:
                    manager_default = LLMManager() 
                    main_test_logger.info(f"Successfully initialized LLMManager with OpenWebUI for live test.")
                    # Example live call (ensure your OpenWebUI and model are running)
                    test_code = "def f():\n  return 1+1 # simple sum"
                    test_instr = "Optimize this function for conciseness. Return the full modified function."
                    model_to_use = settings.OPEN_WEBUI_MODEL_NAME
                    main_test_logger.info(f"Making live call to OpenWebUI model: {model_to_use}")
                    response = await manager_default.generate_code_modification(
                        current_code=test_code,
                        model_name=model_to_use,
                        prompt_instructions=test_instr,
                        temperature=0.1, max_tokens=50
                    )
                    main_test_logger.info(f"Live OpenWebUI response: {response}")

                except Exception as e:
                    main_test_logger.exception(f"Error during live test with OpenWebUI: {e}")
                finally:
                    if manager_default: await manager_default.close_provider()
        
        main_test_logger.info(f"Attempting to initialize LLMManager with GeminiProvider (mocked test setup)...")
        manager_gemini = None
        if not settings.GEMINI_API_KEY: # Assuming GEMINI_API_KEY is needed for instantiation
            main_test_logger.warning("GEMINI_API_KEY not set in .env. Cannot initialize GeminiProvider for this test.")
        else:
            try:
                gemini_llm_config = {'api_key': settings.GEMINI_API_KEY}
                # This test is primarily for instantiation. Actual calls would be mocked in pytest.
                manager_gemini = LLMManager(provider_name="gemini", llm_config=gemini_llm_config)
                main_test_logger.info(f"Successfully initialized LLMManager with GeminiProvider.")
            except ValueError as ve: 
                main_test_logger.error(f"ValueError during GeminiProvider initialization test: {ve}")
            except ImportError as ie:
                main_test_logger.error(f"ImportError during GeminiProvider test: {ie}")
            except Exception as e:
                main_test_logger.exception(f"An unexpected error occurred during GeminiProvider test: {e}")
            finally:
                if manager_gemini: await manager_gemini.close_provider()
        
        main_test_logger.info("--- LLMManager Test Finished ---")

    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main_test())
