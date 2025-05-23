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

PROVIDER_REGISTRY: Dict[str, str] = {
    "open_webui": "evocoder.llm_interface.providers.open_webui_provider.OpenWebUIProvider",
}

class LLMManager:
    """
    Manages the instantiation and interaction with a configured LLM provider.
    """

    def __init__(self, provider_name: Optional[str] = None, llm_config: Optional[Dict[str, Any]] = None):
        self.provider_name = provider_name or settings.DEFAULT_LLM_PROVIDER
        self.llm_config = llm_config if llm_config is not None else {}
        self.provider: Optional[BaseLLMProvider] = None
        self._load_provider()

    def _get_provider_class(self, provider_key: str) -> Type[BaseLLMProvider]:
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
        prompt_instructions: str, # These instructions should now ask for a diff
        context_examples: Optional[List[Dict[str, str]]] = None, 
        temperature: float = 0.7,
        max_tokens: Optional[int] = 2048, 
        **kwargs: Any 
    ) -> str:
        """
        Generates a code modification (expected as a diff string) from the configured LLM.
        The prompt_instructions should guide the LLM to produce output in the SEARCH/REPLACE format.
        """
        if not self.provider:
            raise RuntimeError("LLM provider has not been initialized.")

        messages_for_provider: List[Dict[str, str]] = []
        if context_examples:
            messages_for_provider.extend(context_examples)
        
        # The prompt_instructions (from problem_config) should now include the request for diff format.
        # current_code is provided for context.
        full_user_prompt = f"{prompt_instructions}\n\nHere is the current Python code to modify:\n```python\n{current_code}\n```"
        
        final_prompt_for_provider = full_user_prompt
        final_context_for_provider = messages_for_provider if messages_for_provider else None
        
        try:
            # The response is expected to be a diff string
            diff_string_response = await self.provider.generate_response(
                prompt=final_prompt_for_provider, 
                model_name=model_name, 
                context=final_context_for_provider, 
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            return diff_string_response
        except Exception as e:
            print(f"Error during LLM diff generation via provider '{self.provider_name}': {e}")
            raise

    async def close_provider(self):
        if self.provider:
            await self.provider.close()

if __name__ == '__main__':
    import asyncio
    import os 
    
    try:
        from evocoder.config import settings 
        from evocoder.utils.diff_utils import SEARCH_MARKER_START, SEARCH_MARKER_END, REPLACE_MARKER_END # For test instructions
    except ImportError as e:
        print(f"Failed to import modules for LLMManager test: {e}")
        sys.exit(1)

    async def main_test():
        print("--- Testing LLMManager (Phase 2 - Requesting Diffs) ---")
        
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
        
        manager = None
        try:
            manager = LLMManager() 
            
            test_parent_code = (
                "def inefficient_sum(arr):\n"
                "    s = 0\n"
                "    # Start of loop\n"
                "    for x in arr:\n"
                "        s += x # Add item to sum\n"
                "    # Another loop for no reason\n"
                "    for y in arr:\n"
                "        pass # This loop is redundant\n"
                "    return s"
            )
            
            # NEW: Instructions asking for diff output
            test_instructions_for_diff = (
                "You are an expert Python code optimizer. "
                "Your task is to refactor the provided Python function `inefficient_sum` to be more efficient and concise. "
                "Specifically, remove the redundant loop. "
                f"Provide your changes as a diff in the following format ONLY:\n"
                f"{SEARCH_MARKER_START}\n"
                f"ORIGINAL_CODE_BLOCK_TO_REPLACE\n"
                f"{SEARCH_MARKER_END}\n"
                f"NEW_CODE_BLOCK\n"
                f"{REPLACE_MARKER_END}\n"
                "Do not add any explanations before or after the diff block. Ensure the search block matches exactly."
            )
            
            inspiration_example_code = (
                "def efficient_multiply(arr):\n"
                "    p = 1\n"
                "    for x in arr:\n"
                "        p *= x\n"
                "    return p"
            )
            inspiration_context = [{
                "role": "user",
                "content": f"Here's an example of a previously successful related code (scores: {{'efficiency': 10, 'lines': -3.0}}):\n```python\n{inspiration_example_code}\n```"
            }]
            
            model_for_test = ""
            if settings.DEFAULT_LLM_PROVIDER == "open_webui":
                model_for_test = settings.OPEN_WEBUI_MODEL_NAME
            
            if not model_for_test:
                print(f"Could not determine a model name for the default provider '{settings.DEFAULT_LLM_PROVIDER}' from settings.")
                print("Skipping generation test.")
                return

            print(f"\nAttempting to generate code modification (as diff) for {settings.DEFAULT_LLM_PROVIDER} via LLMManager...")
            print(f"Using model: {model_for_test}")
            print(f"Parent code:\n```python\n{test_parent_code}\n```")
            
            llm_response_diff_string = await manager.generate_code_modification(
                current_code=test_parent_code,
                model_name=model_for_test, 
                prompt_instructions=test_instructions_for_diff, # Use new instructions
                context_examples=inspiration_context,
                temperature=0.1, # Lower temperature for more deterministic diff generation
                max_tokens=300 # Diffs can be short or long
            )
            print("\nLLM Response (Expected Diff String):")
            print(llm_response_diff_string) # This should be the raw diff string

            # For this standalone test, we won't apply the diff here.
            # That will be the EvolutionaryController's job.
            # We just want to see if the LLM attempts to produce the diff format.

        except ValueError as ve:
            print(f"ValueError during LLMManager test: {ve}")
        except ImportError as ie:
            print(f"ImportError during LLMManager test: {ie}")
        except RuntimeError as re:
            print(f"RuntimeError during LLMManager test: {re}")
        except Exception as e:
            print(f"An unexpected error occurred during LLMManager test: {e}")
        finally:
            if manager and hasattr(manager, 'provider') and manager.provider:
                await manager.close_provider()
            elif manager:
                print("LLMManager was created but provider might not have been initialized or already closed.")

    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main_test())
