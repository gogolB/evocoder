# tests/llm_interface/test_providers.py

import pytest
import httpx
import json
from unittest.mock import AsyncMock, patch, MagicMock

# Adjust the import path based on how pytest will discover your modules.
# Assuming pytest runs from the project root 'evocoder/'.
from evocoder.llm_interface.providers.open_webui_provider import OpenWebUIProvider
from evocoder.llm_interface.base_llm_provider import BaseLLMProvider # For type checking if needed

# Test constants
TEST_BASE_URL = "http://fake-open-webui.test"
TEST_API_KEY = "test_api_key_123"
TEST_MODEL_NAME = "test-model"
TEST_PROMPT = "Hello, world!"

@pytest.fixture
def mock_settings(monkeypatch):
    """Fixture to mock settings if OpenWebUIProvider tries to import them directly."""
    monkeypatch.setenv("OPEN_WEBUI_API_KEY", TEST_API_KEY)
    monkeypatch.setenv("OPEN_WEBUI_BASE_URL", TEST_BASE_URL)
    monkeypatch.setenv("OPEN_WEBUI_MODEL_NAME", TEST_MODEL_NAME)

@pytest.fixture
async def open_webui_provider(mock_settings):
    """Fixture to create an OpenWebUIProvider instance for testing."""
    provider = OpenWebUIProvider(api_key=TEST_API_KEY, base_url=TEST_BASE_URL)
    yield provider
    await provider.close()

@pytest.fixture
def dummy_request_obj(open_webui_provider: OpenWebUIProvider):
    """Fixture to create a reusable dummy httpx.Request object."""
    return httpx.Request("POST", open_webui_provider.api_endpoint)


@pytest.mark.asyncio
async def test_open_webui_provider_initialization(open_webui_provider: OpenWebUIProvider):
    """Test that the provider initializes correctly."""
    assert open_webui_provider.api_key == TEST_API_KEY
    assert open_webui_provider.base_url == TEST_BASE_URL
    assert open_webui_provider.api_endpoint == f"{TEST_BASE_URL}/api/chat/completions"
    assert isinstance(open_webui_provider._client, httpx.AsyncClient)

@pytest.mark.asyncio
async def test_open_webui_provider_initialization_no_api_key():
    """Test initialization without an API key (should not raise error at init)."""
    provider = OpenWebUIProvider(api_key=None, base_url=TEST_BASE_URL)
    assert provider.api_key is None
    await provider.close()

@pytest.mark.asyncio
async def test_open_webui_provider_initialization_requires_base_url():
    """Test that ValueError is raised if base_url is not provided."""
    with pytest.raises(ValueError, match="OpenWebUIProvider requires a 'base_url'."):
        OpenWebUIProvider(api_key=TEST_API_KEY, base_url=None)

@pytest.mark.asyncio
async def test_generate_response_success(open_webui_provider: OpenWebUIProvider, dummy_request_obj: httpx.Request):
    """Test successful response generation."""
    mock_response_content = "This is a test response."
    mock_api_response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": mock_response_content
                }
            }
        ]
    }

    with patch.object(open_webui_provider._client, 'post', new_callable=AsyncMock) as mock_post:
        mock_http_response = httpx.Response(
            200,
            json=mock_api_response,
            request=dummy_request_obj # Add request object
        )
        mock_post.return_value = mock_http_response

        response = await open_webui_provider.generate_response(
            prompt=TEST_PROMPT,
            model_name=TEST_MODEL_NAME,
            temperature=0.5,
            max_tokens=50
        )

        assert response == mock_response_content
        mock_post.assert_called_once()
        
        # The URL is the first positional argument to client.post()
        # Keyword arguments are in called_kwargs
        positional_args, keyword_args = mock_post.call_args
        assert positional_args[0] == open_webui_provider.api_endpoint # Check URL as positional arg
        assert keyword_args['headers']['Authorization'] == f"Bearer {TEST_API_KEY}"
        assert keyword_args['headers']['Content-Type'] == "application/json"
        
        expected_payload = {
            "model": TEST_MODEL_NAME,
            "messages": [{"role": "user", "content": TEST_PROMPT}],
            "temperature": 0.5,
            "stream": False,
            "max_tokens": 50
        }
        assert keyword_args['json'] == expected_payload

@pytest.mark.asyncio
async def test_generate_response_success_with_context(open_webui_provider: OpenWebUIProvider, dummy_request_obj: httpx.Request):
    """Test successful response generation with context."""
    mock_response_content = "Response considering context."
    context_messages = [
        {"role": "user", "content": "Previous question."},
        {"role": "assistant", "content": "Previous answer."}
    ]
    
    with patch.object(open_webui_provider._client, 'post', new_callable=AsyncMock) as mock_post:
        mock_post.return_value = httpx.Response(
            200,
            json={"choices": [{"message": {"content": mock_response_content}}]},
            request=dummy_request_obj # Add request object
        )

        response = await open_webui_provider.generate_response(
            prompt=TEST_PROMPT,
            model_name=TEST_MODEL_NAME,
            context=context_messages
        )
        assert response == mock_response_content
        
        _positional_args, keyword_args = mock_post.call_args # Use keyword_args for json
        sent_messages = keyword_args['json']['messages']
        assert len(sent_messages) == 3
        assert sent_messages[0] == context_messages[0]
        assert sent_messages[1] == context_messages[1]
        assert sent_messages[2] == {"role": "user", "content": TEST_PROMPT}


@pytest.mark.asyncio
async def test_generate_response_handles_none_content(open_webui_provider: OpenWebUIProvider, dummy_request_obj: httpx.Request):
    """Test handling of API response where 'content' is None."""
    mock_api_response_none_content = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None 
                }
            }
        ]
    }
    with patch.object(open_webui_provider._client, 'post', new_callable=AsyncMock) as mock_post:
        mock_post.return_value = httpx.Response(
            200, 
            json=mock_api_response_none_content,
            request=dummy_request_obj # Add request object
        )

        with pytest.raises(ValueError, match=f"LLM returned None content for model {TEST_MODEL_NAME}"):
            await open_webui_provider.generate_response(
                prompt=TEST_PROMPT,
                model_name=TEST_MODEL_NAME
            )

@pytest.mark.asyncio
async def test_generate_response_handles_unexpected_structure(open_webui_provider: OpenWebUIProvider, dummy_request_obj: httpx.Request):
    """Test handling of API response with unexpected structure."""
    mock_api_response_bad_structure = {"error": "bad_request"} 
    
    with patch.object(open_webui_provider._client, 'post', new_callable=AsyncMock) as mock_post:
        mock_post.return_value = httpx.Response(
            200, 
            json=mock_api_response_bad_structure,
            request=dummy_request_obj # Add request object
        )

        with pytest.raises(ValueError, match="Unexpected response structure from OpenWebUI or missing content"):
            await open_webui_provider.generate_response(
                prompt=TEST_PROMPT,
                model_name=TEST_MODEL_NAME
            )

@pytest.mark.asyncio
async def test_generate_response_http_error(open_webui_provider: OpenWebUIProvider, dummy_request_obj: httpx.Request):
    """Test handling of HTTPStatusError from the API."""
    with patch.object(open_webui_provider._client, 'post', new_callable=AsyncMock) as mock_post:
        mock_http_response = httpx.Response(
            status_code=500,
            content=b"Internal Server Error",
            request=dummy_request_obj 
        )
        mock_post.return_value = mock_http_response
        
        with pytest.raises(httpx.HTTPStatusError) as excinfo:
            await open_webui_provider.generate_response(
                prompt=TEST_PROMPT,
                model_name=TEST_MODEL_NAME
            )
        
        assert excinfo.value.response.status_code == 500
        assert excinfo.value.request == dummy_request_obj

@pytest.mark.asyncio
async def test_generate_response_request_error(open_webui_provider: OpenWebUIProvider, dummy_request_obj: httpx.Request):
    """Test handling of httpx.RequestError (e.g., network issue)."""
    with patch.object(open_webui_provider._client, 'post', new_callable=AsyncMock) as mock_post:
        mock_post.side_effect = httpx.RequestError("Connection failed", request=dummy_request_obj)
        
        with pytest.raises(httpx.RequestError):
            await open_webui_provider.generate_response(
                prompt=TEST_PROMPT,
                model_name=TEST_MODEL_NAME
            )

@pytest.mark.asyncio
async def test_generate_response_json_decode_error(open_webui_provider: OpenWebUIProvider, dummy_request_obj: httpx.Request):
    """Test handling of invalid JSON response."""
    with patch.object(open_webui_provider._client, 'post', new_callable=AsyncMock) as mock_post:
        mock_http_response = httpx.Response(
            200, 
            text="This is not JSON", 
            request=dummy_request_obj
        )
        mock_post.return_value = mock_http_response

        with pytest.raises(ValueError, match="Invalid JSON response from OpenWebUI"):
            await open_webui_provider.generate_response(
                prompt=TEST_PROMPT,
                model_name=TEST_MODEL_NAME
            )

@pytest.mark.asyncio
async def test_close_method_closes_client(open_webui_provider: OpenWebUIProvider):
    """Test that the close method calls aclose on the httpx client."""
    with patch.object(open_webui_provider._client, 'aclose', new_callable=AsyncMock) as mock_aclose:
        await open_webui_provider.close()
        mock_aclose.assert_called_once()

    provider_no_client = OpenWebUIProvider(api_key=None, base_url=TEST_BASE_URL)
    if hasattr(provider_no_client, '_client'): 
        delattr(provider_no_client, '_client') 
    await provider_no_client.close()
