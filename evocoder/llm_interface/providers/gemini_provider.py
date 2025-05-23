# evocoder/llm_interface/providers/gemini_provider.py

# Use the new import style as per google-genai
from google import generativeai as genai # Changed import
from typing import List, Dict, Any, Optional

try:
    from ..base_llm_provider import BaseLLMProvider
except ImportError:
    if __name__ == '__main__':
        import sys
        from pathlib import Path
        file_path = Path(__file__).resolve()
        project_root = file_path.parent.parent.parent.parent 
        sys.path.insert(0, str(project_root))
        from evocoder.llm_interface.base_llm_provider import BaseLLMProvider
    else:
        raise

class GeminiProvider(BaseLLMProvider):
    """
    Concrete LLM provider implementation for Google Gemini models
    using the 'google-genai' library structure (imported as google.generativeai).
    """

    def __init__(self, api_key: Optional[str], **kwargs: Any):
        super().__init__(api_key=api_key, **kwargs)
        if not self.api_key:
            raise ValueError("GeminiProvider requires an 'api_key'.")
        
        try:
            # configure is directly on the imported genai module
            genai.configure(api_key=self.api_key) 
        except Exception as e:
            raise ValueError(f"Failed to configure Gemini API: {e}")

        self.model_instances: Dict[str, genai.GenerativeModel] = {}
        self.default_generation_config_params = {
            # Default params for genai.types.GenerationConfig
            # "candidate_count": 1, # Usually default
            # "stop_sequences": [],
            # "max_output_tokens": 2048, # Can be overridden
            # "temperature": 0.7,       # Can be overridden
            # "top_p": None,
            # "top_k": None,
        }
        # Safety settings should be a list of SafetySetting objects if provided
        self.safety_settings = kwargs.get("safety_settings", None)


    def _get_model_instance(self, model_name: str) -> genai.GenerativeModel:
        if model_name not in self.model_instances:
            try:
                # GenerativeModel is directly on the imported genai module
                self.model_instances[model_name] = genai.GenerativeModel(
                    model_name=model_name,
                    safety_settings=self.safety_settings 
                )
            except Exception as e:
                raise ValueError(f"Failed to initialize Gemini model '{model_name}': {e}")
        return self.model_instances[model_name]

    async def generate_response(
        self,
        prompt: str, 
        model_name: str, 
        context: Optional[List[Dict[str, str]]] = None, 
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any
    ) -> str:
        generative_model = self._get_model_instance(model_name)
        gemini_history: List[Dict[str, Any]] = []
        if context:
            for msg in context:
                role = "model" if msg.get("role") == "assistant" else msg.get("role", "user")
                # Assuming genai.types.Part is the way to structure parts
                # If genai.types is not directly available, Part might be genai.Part or from a submodule
                # For now, assume parts are just strings for simplicity in this structure.
                # The actual Content object creation might be more complex.
                gemini_history.append({"role": role, "parts": [msg.get("content", "")]})
        
        contents_to_send = []
        if gemini_history:
            contents_to_send.extend(gemini_history)
        contents_to_send.append({"role": "user", "parts": [prompt]})

        # Prepare GenerationConfig
        # The genai.types.GenerationConfig class is the correct way.
        generation_config_kwargs = self.default_generation_config_params.copy()
        generation_config_kwargs["temperature"] = temperature
        if max_tokens is not None:
            generation_config_kwargs["max_output_tokens"] = max_tokens
        if 'top_p' in kwargs:
            generation_config_kwargs["top_p"] = kwargs.get('top_p')
        if 'top_k' in kwargs:
            generation_config_kwargs["top_k"] = kwargs.get('top_k')
        if 'stop_sequences' in kwargs:
             generation_config_kwargs["stop_sequences"] = kwargs.get('stop_sequences')
        
        current_generation_config = genai.types.GenerationConfig(**generation_config_kwargs)

        try:
            response = await generative_model.generate_content_async(
                contents=contents_to_send,
                generation_config=current_generation_config,
            )

            # Accessing response text according to common Gemini SDK patterns
            if response.text: # Shortcut for simple text from first candidate
                return response.text
            elif response.parts: # If response itself has parts (less common for main response)
                return "".join(part.text for part in response.parts if hasattr(part, 'text') and part.text)
            elif response.candidates and response.candidates[0].content.parts:
                 return "".join(part.text for part in response.candidates[0].content.parts if hasattr(part, 'text') and part.text)
            else:
                block_reason_obj = response.prompt_feedback.block_reason if response.prompt_feedback else None
                block_reason_str = genai.types.BlockReason(block_reason_obj).name if block_reason_obj else "Unknown"
                
                finish_reason_obj = response.candidates[0].finish_reason if response.candidates else None
                finish_reason_str = genai.types.FinishReason(finish_reason_obj).name if finish_reason_obj else "Unknown"
                
                safety_ratings_str = str(response.candidates[0].safety_ratings) if response.candidates and hasattr(response.candidates[0], 'safety_ratings') else "N/A"
                
                error_message = (f"Gemini response was empty or blocked. "
                                 f"Block Reason: {block_reason_str}. Finish Reason: {finish_reason_str}. "
                                 f"Safety Ratings: {safety_ratings_str}.")
                print(f"Warning: {error_message}\nFull Response: {response}")
                raise ValueError(error_message)

        except Exception as e:
            print(f"Error during Gemini API call for model '{model_name}': {e}")
            raise

    async def close(self) -> None:
        pass

if __name__ == '__main__':
    import asyncio
    import os
    
    try:
        from evocoder.config import settings 
    except ImportError as e:
        print(f"Failed to import settings for GeminiProvider test: {e}")
        print("Make sure your .env file is set up in the project root and includes GEMINI_API_KEY.")
        sys.exit(1)

    async def test_gemini_provider():
        print("--- Testing GeminiProvider (Standalone with new google.generativeai import) ---")
        if not settings.GEMINI_API_KEY: 
            print("GEMINI_API_KEY not found in settings. Skipping GeminiProvider live test.")
            return

        test_model = getattr(settings, "GEMINI_MODEL_NAME", "gemini-1.5-pro-latest") 
        print(f"Using Gemini Model: {test_model}")

        provider = None
        try:
            # Initialize with safety_settings from settings if available
            safety_config = None
            if hasattr(settings, 'GEMINI_SAFETY_SETTINGS') and settings.GEMINI_SAFETY_SETTINGS:
                # Assuming GEMINI_SAFETY_SETTINGS is a list of dicts like:
                # [{"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"}]
                # Need to convert string category/threshold to enum values
                safety_config = []
                for Ssafety_setting_dict in settings.GEMINI_SAFETY_SETTINGS:
                     category_str = Ssafety_setting_dict.get("category")
                     threshold_str = Ssafety_setting_dict.get("threshold")
                     if category_str and threshold_str:
                         try:
                             category = genai.types.HarmCategory[category_str]
                             threshold = genai.types.HarmBlockThreshold[threshold_str]
                             safety_config.append(genai.types.SafetySetting(category=category, threshold=threshold))
                         except KeyError as ke:
                             print(f"Warning: Invalid HarmCategory or HarmBlockThreshold string in GEMINI_SAFETY_SETTINGS: {ke}")
                if not safety_config: safety_config = None # Reset if parsing failed

            provider = GeminiProvider(api_key=settings.GEMINI_API_KEY, safety_settings=safety_config)
            
            test_prompt = "Write a short Python function that calculates the factorial of a number."
            print(f"\nSending prompt to {test_model}: '{test_prompt}'")
            
            response_text = await provider.generate_response(
                prompt=test_prompt,
                model_name=test_model,
                temperature=0.7,
                max_tokens=250
            )
            print(f"\nReceived response from {test_model}:\n---\n{response_text}\n---")

        except ValueError as ve:
            print(f"ValueError during GeminiProvider test: {ve}")
        except Exception as e:
            print(f"An error occurred during the GeminiProvider test: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if provider:
                await provider.close() 

    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(test_gemini_provider())
