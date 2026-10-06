"""Tests for the Converse request body built by ``BedrockModel._parse_request()``.

Each assertion reads the dict ``_parse_request()`` actually returns rather than
re-deriving its logic. No Bedrock call is made: ``_parse_request()`` is pure.

The model IDs are real Converse-reachable Anthropic model IDs, shown with the
cross-region inference profile prefixes each one is published under, because the
topP handling has to work on prefixed profile IDs as well as bare IDs.
"""

import logging

import pytest
from fastapi import HTTPException

from api.models.bedrock import BedrockModel, requires_top_p_removal
from api.schema import ChatRequest, UserMessage

MESSAGES = [UserMessage(role="user", content="Hello")]

# Claude 4.5 / 4.6 generation: topP must be dropped.
CLAUDE_45_46_MODEL_IDS = [
    "global.anthropic.claude-opus-4-6-v1",
    "us.anthropic.claude-opus-4-5-20251101-v1:0",
    "jp.anthropic.claude-sonnet-4-6",
    "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
]

# An older model, outside the family: topP must be kept.
OUT_OF_FAMILY_MODEL_ID = "us.anthropic.claude-3-5-sonnet-20240620-v1:0"


@pytest.fixture
def model() -> BedrockModel:
    return BedrockModel()


def parse(bedrock_model: BedrockModel, **kwargs) -> dict:
    kwargs.setdefault("messages", MESSAGES)
    return bedrock_model._parse_request(ChatRequest(**kwargs))


@pytest.mark.parametrize("model_id", CLAUDE_45_46_MODEL_IDS)
def test_top_p_is_dropped_for_the_claude_45_46_generation(model, model_id):
    inference_config = parse(model, model=model_id)["inferenceConfig"]

    assert "topP" not in inference_config
    # temperature is the parameter OpenAI clients set, so it has to survive.
    assert inference_config["temperature"] == 1.0


def test_top_p_is_kept_for_a_model_outside_the_family(model):
    inference_config = parse(model, model=OUT_OF_FAMILY_MODEL_ID)["inferenceConfig"]

    assert inference_config["topP"] == 1.0
    assert inference_config["temperature"] == 1.0


def test_reasoning_effort_drops_temperature_and_top_p(model):
    args = parse(
        model,
        model=OUT_OF_FAMILY_MODEL_ID,
        max_tokens=10000,
        reasoning_effort="medium",
    )

    assert "temperature" not in args["inferenceConfig"]
    assert "topP" not in args["inferenceConfig"]
    assert args["inferenceConfig"]["maxTokens"] == 10000
    assert args["additionalModelRequestFields"] == {"reasoning_config": {"type": "enabled", "budget_tokens": 6000}}


def test_thinking_in_extra_body_drops_temperature_and_top_p(model):
    extra_body = {"thinking": {"type": "enabled", "budget_tokens": 2048}}

    args = parse(model, model=OUT_OF_FAMILY_MODEL_ID, extra_body=extra_body)

    assert "temperature" not in args["inferenceConfig"]
    assert "topP" not in args["inferenceConfig"]
    assert args["additionalModelRequestFields"] == extra_body


def test_extra_body_unrelated_to_thinking_leaves_sampling_parameters_alone(model):
    # A beta header passthrough says nothing about thinking, so nothing is dropped.
    extra_body = {"anthropic_beta": ["interleaved-thinking-2025-05-14"]}

    args = parse(model, model=OUT_OF_FAMILY_MODEL_ID, extra_body=extra_body)

    assert args["inferenceConfig"]["temperature"] == 1.0
    assert args["inferenceConfig"]["topP"] == 1.0
    assert args["additionalModelRequestFields"] == extra_body


@pytest.mark.parametrize(
    "extra_body",
    [
        {"thinking": {"type": "adaptive"}},
        {"output_config": {"effort": "medium"}},
        {"effort": "medium"},
        {"reasoning_config": {"type": "enabled", "budget_tokens": 2048}},
    ],
)
def test_adaptive_thinking_extra_body_drops_temperature_and_top_p(model, extra_body):
    # Adaptive thinking on Opus 4.6 / Sonnet 4.6 is steered by an effort level rather
    # than a thinking block, and thinking is on either way, so temperature and topP
    # have to go. See THINKING_EXTRA_BODY_KEYS.
    args = parse(model, model="global.anthropic.claude-opus-4-6-v1", extra_body=extra_body)

    assert "temperature" not in args["inferenceConfig"]
    assert "topP" not in args["inferenceConfig"]
    assert args["additionalModelRequestFields"] == extra_body


def test_application_inference_profile_arn_keeps_top_p(model):
    # Known gap, recorded rather than silently assumed away: an application inference
    # profile ARN ends in an opaque profile id and never names the model, so the
    # Claude 4.5/4.6 fragment check cannot match it.
    arn = "arn:aws:bedrock:us-west-2:123456789012:application-inference-profile/abcdef123456"

    assert requires_top_p_removal(arn) is False
    assert parse(model, model=arn)["inferenceConfig"]["topP"] == 1.0


def test_budget_tokens_is_clamped_to_the_bedrock_minimum(model):
    # The default max_tokens with effort "low" used to give int(2048 * 0.3) = 614,
    # which Bedrock rejects with a ValidationException.
    assert model._calc_budget_tokens(2048, "low") == 1024


def test_budget_tokens_for_high_effort_is_one_below_max_tokens(model):
    assert model._calc_budget_tokens(10000, "high") == 9999


def test_clamped_budget_warns_when_little_is_left_for_the_answer(model, caplog):
    # max_tokens=1500 at effort "low" computes 450, so the clamp to 1024 leaves only
    # 476 tokens for the answer. The request is valid but the answer may be cut off.
    with caplog.at_level(logging.WARNING, logger="api.models.bedrock"):
        assert model._calc_budget_tokens(1500, "low") == 1024

    assert "may be truncated" in caplog.text


def test_budget_tokens_well_above_the_floor_does_not_warn(model, caplog):
    with caplog.at_level(logging.WARNING, logger="api.models.bedrock"):
        assert model._calc_budget_tokens(10000, "low") == 3000

    assert caplog.text == ""


@pytest.mark.parametrize("max_tokens", [1, 1023, 1024])
def test_budget_tokens_rejects_max_tokens_at_or_below_the_minimum(model, max_tokens):
    with pytest.raises(HTTPException) as exc_info:
        model._calc_budget_tokens(max_tokens, "high")

    assert exc_info.value.status_code == 400
    assert "1024" in exc_info.value.detail


def test_reasoning_effort_with_too_small_max_tokens_is_rejected(model):
    with pytest.raises(HTTPException) as exc_info:
        parse(
            model,
            model=OUT_OF_FAMILY_MODEL_ID,
            max_tokens=512,
            reasoning_effort="low",
        )

    assert exc_info.value.status_code == 400
