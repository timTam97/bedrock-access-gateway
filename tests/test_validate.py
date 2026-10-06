"""Tests for ``BedrockModel.validate()``.

``validate()`` is the only check the router runs *before* a streaming response is
started, so anything that must reach the caller as an HTTP 400 rather than as an
in-stream SSE error event has to be caught here.
"""

import pytest
from fastapi import HTTPException

from api.models import bedrock
from api.models.bedrock import BedrockModel
from api.schema import ChatRequest, UserMessage

MODEL_ID = "global.anthropic.claude-haiku-4-5-20251001-v1:0"
MESSAGES = [UserMessage(role="user", content="Hello")]


@pytest.fixture
def model(monkeypatch) -> BedrockModel:
    monkeypatch.setattr(bedrock, "bedrock_model_list", {MODEL_ID: {"modalities": ["TEXT", "IMAGE"]}})
    return BedrockModel()


def test_supported_model_passes_validation(model):
    model.validate(ChatRequest(model=MODEL_ID, messages=MESSAGES))


def test_unsupported_model_is_rejected(model):
    with pytest.raises(HTTPException) as exc_info:
        model.validate(ChatRequest(model="anthropic.claude-does-not-exist", messages=MESSAGES))

    assert exc_info.value.status_code == 400
    assert "Unsupported model" in exc_info.value.detail


@pytest.mark.parametrize("field", ["max_tokens", "max_completion_tokens"])
@pytest.mark.parametrize("max_tokens", [1, 1023, 1024])
def test_reasoning_with_too_small_output_window_is_rejected_before_streaming(model, field, max_tokens):
    # _parse_request() also rejects this, but it runs inside the response generator,
    # which on a streaming request is after the 200 has been committed.
    request = ChatRequest(model=MODEL_ID, messages=MESSAGES, reasoning_effort="low", **{field: max_tokens})

    with pytest.raises(HTTPException) as exc_info:
        model.validate(request)

    assert exc_info.value.status_code == 400
    assert "1024" in exc_info.value.detail


def test_reasoning_with_a_large_enough_output_window_passes(model):
    model.validate(ChatRequest(model=MODEL_ID, messages=MESSAGES, reasoning_effort="low", max_completion_tokens=4096))


def test_max_completion_tokens_takes_precedence_over_max_tokens(model):
    # The reasoning path uses max_completion_tokens when both are set, so validation
    # has to read the same field _parse_request() does.
    model.validate(
        ChatRequest(
            model=MODEL_ID,
            messages=MESSAGES,
            reasoning_effort="low",
            max_tokens=512,
            max_completion_tokens=4096,
        )
    )


def test_small_max_tokens_without_reasoning_is_fine(model):
    model.validate(ChatRequest(model=MODEL_ID, messages=MESSAGES, max_tokens=512))
