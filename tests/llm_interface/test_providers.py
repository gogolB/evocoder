# tests/llm_interface/test_providers.py

import pytest
import httpx
import json
from unittest.mock import AsyncMock, patch, MagicMock, PropertyMock

# We are removing Gemini specific imports for now
# import google.generativeai as genai 
# from google.generativeai.types import (
#     Part, 
#     Content, 
#     GenerateContentResponse, 
#     Candidate, 
#     PromptFeedback,
#     SafetyRating,
#     HarmCategory,
#     FinishReason, 
#     BlockReason  
# )

from evocoder.llm_interface.providers.open_webui_provider import OpenWebUIProvider
# from evocoder.llm_interface.providers.gemini_provider import GeminiProvider # Removed
from evocoder.llm_interface.base_llm_provider import BaseLLMProvider 

# Test constants
TEST_BASE_URL_OPENWEBUI = "http://fake-open-webui.test"
TEST_API_KEY_OPENWEBUI = "test_openwebui_api_key_123"
# TEST_API_KEY_GEMINI = "test_gemini_api_key_xyz789" # Removed
TEST_MODEL_NAME_GENERIC = "test-model"
# TEST_MODEL_NAME_GEMINI = "gemini-1.5-pro-latest" # Removed
TEST_PROMPT = "Hello, world!"

# --- Fixtures for OpenWebUIProvider ---
@pytest.fixture
def mock_openwebui_settings(monkeypatch):
    monkeypatch.setenv("OPEN_WEBUI_API_KEY", TEST_API_KEY_OPENWEBUI)
    monkeypatch.setenv("OPEN_WEBUI_BASE_URL", TEST_BASE_URL_OPENWEBUI)
    monkeypatch.setenv("OPEN_WEBUI_MODEL_NAME", TEST_MODEL_NAME_GENERIC)

@pytest.fixture
async def open_webui_provider(mock_openwebui_settings):
    provider = OpenWebUIProvider(api_key=TEST_API_KEY_OPENWEBUI, base_url=TEST_BASE_URL_OPENWEBUI)
    yield provider
    await provider.close()

@pytest.fixture
def dummy_request_obj_openwebui(open_webui_provider: OpenWebUIProvider):
    return httpx.Request("POST", open_webui_provider.api_endpoint)

# --- Fixtures for GeminiProvider (REMOVED) ---
# @pytest.fixture
# def mock_gemini_settings(monkeypatch):
#     monkeypatch.setenv("GEMINI_API_KEY", TEST_API_KEY_GEMINI)
#     monkeypatch.setenv("GEMINI_MODEL_NAME", TEST_MODEL_NAME_GEMINI)

# @pytest.fixture
# async def gemini_provider_mocker(mock_gemini_settings):
#     # ... (Gemini mocker fixture removed) ...

# @pytest.fixture
# async def gemini_provider(gemini_provider_mocker): 
#     # ... (Gemini provider fixture removed) ...


# --- Tests for OpenWebUIProvider ---
@pytest.mark.asyncio
async def test_open_webui_provider_initialization(open_webui_provider: OpenWebUIProvider):
    assert open_webui_provider.api_key == TEST_API_KEY_OPENWEBUI
    assert open_webui_provider.base_url == TEST_BASE_URL_OPENWEBUI
    assert open_webui_provider.api_endpoint == f"{TEST_BASE_URL_OPENWEBUI}/api/chat/completions"
    assert isinstance(open_webui_provider._client, httpx.AsyncClient)

@pytest.mark.asyncio
async def test_open_webui_provider_initialization_no_api_key():
    provider = OpenWebUIProvider(api_key=None, base_url=TEST_BASE_URL_OPENWEBUI)
    assert provider.api_key is None
    await provider.close()

@pytest.mark.asyncio
async def test_open_webui_provider_initialization_requires_base_url():
    with pytest.raises(ValueError, match="OpenWebUIProvider requires a 'base_url'."):
        OpenWebUIProvider(api_key=TEST_API_KEY_OPENWEBUI, base_url=None)

@pytest.mark.asyncio
async def test_open_webui_generate_response_success(open_webui_provider: OpenWebUIProvider, dummy_request_obj_openwebui: httpx.Request):
    mock_response_content = "This is a test response."
    mock_api_response = {"choices": [{"message": {"role": "assistant", "content": mock_response_content}}]}
    with patch.object(open_webui_provider._client, 'post', new_callable=AsyncMock) as mock_post:
        mock_post.return_value = httpx.Response(200, json=mock_api_response, request=dummy_request_obj_openwebui)
        response = await open_webui_provider.generate_response(prompt=TEST_PROMPT, model_name=TEST_MODEL_NAME_GENERIC)
        assert response == mock_response_content
        mock_post.assert_called_once()
        _pos_args, kw_args = mock_post.call_args
        assert kw_args['json']['model'] == TEST_MODEL_NAME_GENERIC

@pytest.mark.asyncio
async def test_open_webui_generate_response_http_error(open_webui_provider: OpenWebUIProvider, dummy_request_obj_openwebui: httpx.Request):
    with patch.object(open_webui_provider._client, 'post', new_callable=AsyncMock) as mock_post:
        mock_http_response = httpx.Response(status_code=500, content=b"Internal Server Error", request=dummy_request_obj_openwebui)
        mock_post.return_value = mock_http_response
        with pytest.raises(httpx.HTTPStatusError):
            await open_webui_provider.generate_response(prompt=TEST_PROMPT, model_name=TEST_MODEL_NAME_GENERIC)

# (Ensure all other OpenWebUIProvider specific tests like _success_with_context, 
# _handles_none_content, _handles_unexpected_structure, _request_error, 
# _json_decode_error, _close_method_closes_client are present and correct if they were defined)
# For example, assuming they were in the previous version you had:

@pytest.mark.asyncio
async def test_open_webui_generate_response_success_with_context(open_webui_provider: OpenWebUIProvider, dummy_request_obj_openwebui: httpx.Request):
    mock_response_content = "Response considering context."
    context_messages = [
        {"role": "user", "content": "Previous question."},
        {"role": "assistant", "content": "Previous answer."}
    ]
    with patch.object(open_webui_provider._client, 'post', new_callable=AsyncMock) as mock_post:
        mock_post.return_value = httpx.Response(
            200,
            json={"choices": [{"message": {"content": mock_response_content}}]},
            request=dummy_request_obj_openwebui
        )
        response = await open_webui_provider.generate_response(
            prompt=TEST_PROMPT,
            model_name=TEST_MODEL_NAME_GENERIC,
            context=context_messages
        )
        assert response == mock_response_content
        _positional_args, keyword_args = mock_post.call_args
        sent_messages = keyword_args['json']['messages']
        assert len(sent_messages) == 3
        assert sent_messages[0] == context_messages[0]
        assert sent_messages[1] == context_messages[1]
        assert sent_messages[2] == {"role": "user", "content": TEST_PROMPT}

@pytest.mark.asyncio
async def test_open_webui_generate_response_handles_none_content(open_webui_provider: OpenWebUIProvider, dummy_request_obj_openwebui: httpx.Request):
    mock_api_response_none_content = {"choices": [{"message": {"role": "assistant", "content": None}}]}
    with patch.object(open_webui_provider._client, 'post', new_callable=AsyncMock) as mock_post:
        mock_post.return_value = httpx.Response(200, json=mock_api_response_none_content, request=dummy_request_obj_openwebui)
        with pytest.raises(ValueError, match=f"LLM returned None content for model {TEST_MODEL_NAME_GENERIC}"):
            await open_webui_provider.generate_response(prompt=TEST_PROMPT, model_name=TEST_MODEL_NAME_GENERIC)

@pytest.mark.asyncio
async def test_open_webui_generate_response_handles_unexpected_structure(open_webui_provider: OpenWebUIProvider, dummy_request_obj_openwebui: httpx.Request):
    mock_api_response_bad_structure = {"error": "bad_request"} 
    with patch.object(open_webui_provider._client, 'post', new_callable=AsyncMock) as mock_post:
        mock_post.return_value = httpx.Response(200, json=mock_api_response_bad_structure, request=dummy_request_obj_openwebui)
        with pytest.raises(ValueError, match="Unexpected response structure from OpenWebUI or missing content"):
            await open_webui_provider.generate_response(prompt=TEST_PROMPT, model_name=TEST_MODEL_NAME_GENERIC)

@pytest.mark.asyncio
async def test_open_webui_generate_response_request_error(open_webui_provider: OpenWebUIProvider, dummy_request_obj_openwebui: httpx.Request):
    with patch.object(open_webui_provider._client, 'post', new_callable=AsyncMock) as mock_post:
        mock_post.side_effect = httpx.RequestError("Connection failed", request=dummy_request_obj_openwebui)
        with pytest.raises(httpx.RequestError):
            await open_webui_provider.generate_response(prompt=TEST_PROMPT, model_name=TEST_MODEL_NAME_GENERIC)

@pytest.mark.asyncio
async def test_open_webui_generate_response_json_decode_error(open_webui_provider: OpenWebUIProvider, dummy_request_obj_openwebui: httpx.Request):
    with patch.object(open_webui_provider._client, 'post', new_callable=AsyncMock) as mock_post:
        mock_http_response = httpx.Response(200, text="This is not JSON", request=dummy_request_obj_openwebui)
        mock_post.return_value = mock_http_response
        with pytest.raises(ValueError, match="Invalid JSON response from OpenWebUI"):
            await open_webui_provider.generate_response(prompt=TEST_PROMPT, model_name=TEST_MODEL_NAME_GENERIC)

@pytest.mark.asyncio
async def test_open_webui_close_method_closes_client(open_webui_provider: OpenWebUIProvider):
    with patch.object(open_webui_provider._client, 'aclose', new_callable=AsyncMock) as mock_aclose:
        await open_webui_provider.close()
        mock_aclose.assert_called_once()

# --- Tests for GeminiProvider (REMOVED) ---
# All tests related to GeminiProvider have been removed.

