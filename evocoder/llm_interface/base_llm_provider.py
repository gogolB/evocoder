# evocoder/llm_interface/base_llm_provider.py

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import os

class BaseLLMProvider(ABC):
    """
    Abstract Base Class for all LLM providers.

    This class defines the common interface that all concrete LLM provider
    implementations must adhere to. This allows the LLMManager to interact
    with different LLM services (cloud-based or local) in a consistent way.
    """

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, **kwargs: Any):
        """
        Initialize the LLM provider.

        Args:
            api_key (Optional[str]): The API key for the LLM provider, if applicable.
            base_url (Optional[str]): The base URL for the LLM provider's API,
                                      especially relevant for self-hosted or local models.
            **kwargs (Any): Additional provider-specific configuration arguments.
        """
        self.api_key = api_key
        self.base_url = base_url
        # Store any other relevant kwargs if needed by subclasses
        self.provider_specific_config = kwargs

    @abstractmethod
    async def generate_response(
        self,
        prompt: str,
        model_name: str,
        context: Optional[List[Dict[str, str]]] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any
    ) -> str:
        """
        Generate a response from the LLM.

        This is an abstract method that must be implemented by all concrete
        LLM provider classes.

        Args:
            prompt (str): The main prompt or instruction for the LLM.
            model_name (str): The specific model identifier to use for this generation request
                              (e.g., "gemini-pro-preview-03-25" for OpenWebUI,
                              "gpt-4o-mini" for OpenAI, etc.).
            context (Optional[List[Dict[str, str]]]): A list of message dictionaries
                representing the conversation history or additional context.
                Each dictionary should typically have "role" (e.g., "user", "assistant", "system")
                and "content" keys. This is particularly useful for chat-based models.
                If None or empty, the prompt is treated as a single user message.
            temperature (float): Controls the randomness of the output.
                                 Typically between 0.0 (deterministic) and 1.0 (more random).
                                 Some models might support up to 2.0.
            max_tokens (Optional[int]): The maximum number of tokens to generate in the response.
                                        Provider-specific default if None.
            **kwargs (Any): Additional provider-specific parameters for the generation request
                            (e.g., top_p, top_k, stop_sequences).

        Returns:
            str: The LLM's generated text response.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
            Exception: Provider-specific exceptions related to API calls, authentication, etc.
        """
        raise NotImplementedError("Subclasses must implement the 'generate_response' method.")

    async def close(self) -> None:
        """
        Optional method to clean up resources, like closing HTTP client sessions.
        Subclasses should override this if they have resources to release.
        """
        pass # Default implementation does nothing

    # You could add other common methods here if needed, e.g.,
    # - count_tokens(text: str, model_name: str) -> int
    # - get_model_info(model_name: str) -> dict
    # However, for EvoCoder's core function, generate_response is the most critical.

if __name__ == "__main__":
    # This class is abstract and cannot be instantiated directly.
    # Example of how a concrete class might be structured (for illustration only):
    class DummyProvider(BaseLLMProvider):
        def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, **kwargs: Any):
            super().__init__(api_key, base_url, **kwargs)
            print(f"DummyProvider initialized with API key: {'Yes' if api_key else 'No'}, Base URL: {base_url}")

        async def generate_response(
            self,
            prompt: str,
            model_name: str,
            context: Optional[List[Dict[str, str]]] = None,
            temperature: float = 0.7,
            max_tokens: Optional[int] = None,
            **kwargs: Any
        ) -> str:
            print(f"\n--- DummyProvider.generate_response called ---")
            print(f"Model: {model_name}")
            print(f"Prompt: {prompt}")
            if context:
                print("Context:")
                for msg in context:
                    print(f"  - Role: {msg.get('role')}, Content: {msg.get('content')[:50]}...")
            print(f"Temperature: {temperature}")
            print(f"Max Tokens: {max_tokens}")
            if kwargs:
                print(f"Other kwargs: {kwargs}")
            return f"Dummy response for prompt: '{prompt}' using model '{model_name}'"

    async def main():
        # Cannot instantiate BaseLLMProvider directly
        # provider = BaseLLMProvider() # This would raise TypeError

        dummy_provider = DummyProvider(api_key="dummy_key", custom_param="test_value")
        response = await dummy_provider.generate_response(
            prompt="Explain abstract base classes in Python.",
            model_name="dummy-model-001",
            context=[
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": "What is an ABC?"}
            ],
            temperature=0.5,
            max_tokens=100,
            top_p=0.9
        )
        print(f"\nResponse from DummyProvider: {response}")
        await dummy_provider.close()

    # To run this example (Python 3.7+ for asyncio.run):
    import asyncio
    if os.name == 'nt': # Fix for ProactorLoopNotRunning on Windows for asyncio in scripts
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())


