import base64
import json
import logging
import re
import time
from abc import ABC
from collections import defaultdict
from typing import AsyncIterable, Iterable, Literal

import boto3
import numpy as np
import requests
import tiktoken
from botocore.config import Config
from fastapi import HTTPException
from starlette.concurrency import run_in_threadpool

from api.models.base import BaseChatModel, BaseEmbeddingsModel
from api.schema import (
    AssistantMessage,
    ChatRequest,
    ChatResponse,
    ChatResponseMessage,
    ChatStreamResponse,
    Choice,
    ChoiceDelta,
    Embedding,
    EmbeddingsRequest,
    EmbeddingsResponse,
    EmbeddingsUsage,
    Error,
    ErrorMessage,
    Function,
    ImageContent,
    ResponseFunction,
    TextContent,
    ToolCall,
    ToolContent,
    ToolMessage,
    Usage,
    UserMessage,
)
from api.setting import (
    AWS_REGION,
    DEBUG,
    DEFAULT_MODEL,
    ENABLE_CROSS_REGION_INFERENCE,
    ENABLE_APPLICATION_INFERENCE_PROFILES,
)

logger = logging.getLogger(__name__)

config = Config(
            connect_timeout=60,      # Connection timeout: 60 seconds
            read_timeout=900,        # Read timeout: 15 minutes (suitable for long streaming responses)
            retries={
                'max_attempts': 8,   # Maximum retry attempts
                'mode': 'adaptive'   # Adaptive retry mode
            },
            max_pool_connections=50  # Maximum connection pool size
        )

bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name=AWS_REGION,
    config=config,
)
bedrock_client = boto3.client(
    service_name="bedrock",
    region_name=AWS_REGION,
    config=config,
)


# Claude ids for which Bedrock Converse rejects temperature and topP together, so
# topP is dropped. The restriction was first observed on Claude Sonnet 4.5 and is
# applied here to the rest of the Converse-reachable Claude 4.5/4.6 generation.
#
# This is deliberately conservative. Anthropic documents the sampling-parameter
# restriction for these models only while thinking is enabled - "temperature and
# top_k are incompatible with thinking, and top_p is allowed at values between 0.95
# and 1" - and publishes no unconditional restriction for Opus 4.5/4.6, Sonnet 4.6
# or Haiku 4.5. See "Limits and feature compatibility" > "Sampling parameters" in
# https://platform.claude.com/docs/en/build-with-claude/thinking
# Dropping topP rather than temperature keeps the parameter OpenAI clients actually
# set, and it is documented in docs/Usage.md so callers know their top_p is ignored.
# These are id fragments, not full ids, so a bare model id and a cross-region
# inference profile id (us./eu./apac./jp./au./global. prefixed) both match. An
# application inference profile ARN does NOT match: it ends in an opaque profile id
# (".../application-inference-profile/abcdef123456") and never carries the model
# name, so requests routed through one keep their topP. Closing that gap would need
# an extra GetInferenceProfile call per request to resolve the target model, which
# is out of scope here.
CLAUDE_MODELS_WITHOUT_TOP_P = (
    "claude-sonnet-4-5",
    "claude-opus-4-5",
    "claude-haiku-4-5",
    "claude-sonnet-4-6",
    "claude-opus-4-6",
)

# extra_body keys that turn Claude's thinking on, and therefore mean temperature and
# topP have to be dropped from inferenceConfig. "thinking" covers both spellings of
# the thinking block - adaptive ({"type": "adaptive"}) and manual extended thinking
# ({"type": "enabled", "budget_tokens": N}) - and "reasoning_config" is the spelling
# the reasoning_effort path in this file sends.
#
# "output_config" and "effort" are deliberately NOT listed. Anthropic's per-model
# configuration table reads "Thinking off" for Claude Opus 4.6 and Claude Sonnet 4.6
# when no thinking field is sent ("thinking is off until you set
# thinking: {type: 'adaptive'}"), and the effort page states that because effort
# applies to every output token "it works whether or not thinking is enabled". An
# effort-only request therefore is not a thinking request and keeps its temperature.
# output_config is also the structured-outputs container (output_config.format), so
# matching it would scrub the sampling parameters of every JSON-schema request -
# requests that demonstrably have thinking off.
# https://platform.claude.com/docs/en/build-with-claude/thinking
# https://platform.claude.com/docs/en/build-with-claude/effort
# https://platform.claude.com/docs/en/build-with-claude/structured-outputs
# Unverified against a live Converse call (no credentials available here): these are
# the Messages API field names, as noted in docs/Usage.md.
THINKING_EXTRA_BODY_KEYS = (
    "thinking",
    "reasoning_config",
)

# Bedrock rejects a smaller reasoning budget with a ValidationException. It may
# change per model in the future.
MIN_REASONING_BUDGET_TOKENS = 1024


def requires_top_p_removal(model_id: str) -> bool:
    """Whether topP must be dropped for this model. See CLAUDE_MODELS_WITHOUT_TOP_P."""
    lowered = model_id.lower()
    return any(fragment in lowered for fragment in CLAUDE_MODELS_WITHOUT_TOP_P)


def enables_thinking(extra_body: dict) -> bool:
    """Whether an extra_body payload turns thinking on. See THINKING_EXTRA_BODY_KEYS."""
    return any(key in extra_body for key in THINKING_EXTRA_BODY_KEYS)


def reasoning_max_tokens_error(max_tokens: int) -> str:
    """Message for a reasoning request whose output window cannot fit the budget."""
    return f"max_tokens must be greater than {MIN_REASONING_BUDGET_TOKENS} when reasoning is enabled, got {max_tokens}"


SUPPORTED_BEDROCK_EMBEDDING_MODELS = {
    "cohere.embed-multilingual-v3": "Cohere Embed Multilingual",
    "cohere.embed-english-v3": "Cohere Embed English",
    "amazon.titan-embed-text-v1": "Titan Embeddings G1 - Text",
    "amazon.titan-embed-text-v2:0": "Titan Embeddings G2 - Text",
    # Disable Titan embedding.
    # "amazon.titan-embed-image-v1": "Titan Multimodal Embeddings G1"
}

ENCODER = tiktoken.get_encoding("cl100k_base")


def list_bedrock_models() -> dict:
    """Automatically getting a list of supported models.

    Returns a model list combines:
        - ON_DEMAND models.
        - Cross-Region Inference Profiles (if enabled via Env)
        - Application Inference Profiles (if enabled via Env)
    """
    model_list = {}
    try:
        # Map foundation model_id -> list of system defined inference profile IDs
        cr_profiles_by_model = defaultdict(list)
        # Map foundation model_id -> set of application inference profile ARNs
        app_profiles_by_model = defaultdict(set)

        if ENABLE_CROSS_REGION_INFERENCE:
            # List system defined inference profile IDs. A profile ID is a single
            # region prefix token followed by the base model ID, for example
            # "us.anthropic.claude-sonnet-4-5-20250929-v1:0". Index the IDs this
            # Region actually returned by their base model ID instead of guessing a
            # prefix, so every prefix on offer - us, eu, apac, jp, au, global and
            # anything AWS adds later - is picked up without a code change. AWS
            # documents that profile IDs must be discovered per source Region rather
            # than inferred:
            # https://docs.aws.amazon.com/bedrock/latest/userguide/inference-profiles-support.html
            paginator = bedrock_client.get_paginator("list_inference_profiles")
            for page in paginator.paginate(maxResults=1000, typeEquals="SYSTEM_DEFINED"):
                for profile in page["inferenceProfileSummaries"]:
                    profile_id = profile["inferenceProfileId"]
                    # Split once: the prefix must be a single token with no dot in it.
                    parts = profile_id.split(".", 1)
                    if len(parts) != 2 or not parts[0] or not parts[1]:
                        continue
                    cr_profiles_by_model[parts[1]].append(profile_id)

        if ENABLE_APPLICATION_INFERENCE_PROFILES:
            # List application defined inference profile IDs and create mapping
            paginator = bedrock_client.get_paginator('list_inference_profiles')
            for page in paginator.paginate(maxResults=1000, typeEquals="APPLICATION"):
                for profile in page["inferenceProfileSummaries"]:
                    try:
                        profile_arn = profile.get("inferenceProfileArn")
                        if not profile_arn:
                            continue
                        
                        # Process all models in the profile
                        models = profile.get("models", [])
                        for model in models:
                            model_arn = model.get("modelArn", "")
                            if model_arn:
                                model_id = model_arn.split('/')[-1] if '/' in model_arn else model_arn
                                if model_id:
                                    app_profiles_by_model[model_id].add(profile_arn)
                    except Exception as e:
                        logger.warning(f"Error processing application profile: {e}")
                        continue
                    
        # List foundation models, only cares about text outputs here.
        response = bedrock_client.list_foundation_models(byOutputModality="TEXT")

        for model in response["modelSummaries"]:
            model_id = model.get("modelId", "N/A")
            stream_supported = model.get("responseStreamingSupported", True)
            status = model["modelLifecycle"].get("status", "ACTIVE")

            # currently, use this to filter out rerank models and legacy models
            if not stream_supported or status not in ["ACTIVE", "LEGACY"]:
                continue

            inference_types = model.get("inferenceTypesSupported", [])
            input_modalities = model["inputModalities"]
            # Add on-demand model list
            if "ON_DEMAND" in inference_types:
                model_list[model_id] = {"modalities": input_modalities}

            # Add every cross-region inference profile discovered for this model.
            # Newer Claude models are profile-only (no ON_DEMAND entitlement), so
            # this is the only way they get listed at all.
            for profile_id in cr_profiles_by_model.get(model_id, []):
                model_list[profile_id] = {"modalities": input_modalities}

            # Add application inference profiles (emit all profiles for this model)
            if model_id in app_profiles_by_model:
                for profile_arn in app_profiles_by_model[model_id]:
                    model_list[profile_arn] = {"modalities": input_modalities}

    except Exception as e:
        logger.error(f"Unable to list models: {str(e)}")

    if not model_list:
        # In case stack not updated.
        model_list[DEFAULT_MODEL] = {"modalities": ["TEXT", "IMAGE"]}

    return model_list


# Initialize the model list.
bedrock_model_list = list_bedrock_models()


class BedrockModel(BaseChatModel):
    def list_models(self) -> list[str]:
        """Always refresh the latest model list"""
        global bedrock_model_list
        bedrock_model_list = list_bedrock_models()
        return list(bedrock_model_list.keys())

    def validate(self, chat_request: ChatRequest):
        """Perform basic validation on requests"""
        error = ""
        # check if model is supported
        if chat_request.model not in bedrock_model_list.keys():
            error = f"Unsupported model {chat_request.model}, please use models API to get a list of supported models"
            logger.error("Unsupported model: %s", chat_request.model)

        # A reasoning request needs an output window big enough for the minimum
        # thinking budget plus an answer. _calc_budget_tokens() rejects it too, but
        # that runs inside the response generator, which on a streaming request is
        # after the 200 and the SSE content type have been sent - so the error would
        # reach the client as an in-stream event instead of an HTTP 400. validate()
        # is called by the router before streaming starts, so check it here as well.
        if not error and chat_request.reasoning_effort:
            max_tokens = chat_request.max_completion_tokens or chat_request.max_tokens
            if max_tokens is not None and max_tokens <= MIN_REASONING_BUDGET_TOKENS:
                error = reasoning_max_tokens_error(max_tokens)

        if error:
            raise HTTPException(
                status_code=400,
                detail=error,
            )

    async def _invoke_bedrock(self, chat_request: ChatRequest, stream=False):
        """Common logic for invoke bedrock models"""
        if DEBUG:
            logger.info("Raw request: " + chat_request.model_dump_json())

        # convert OpenAI chat request to Bedrock SDK request
        args = self._parse_request(chat_request)
        if DEBUG:
            logger.info("Bedrock request: " + json.dumps(str(args)))

        try:
            if stream:
                # Run the blocking boto3 call in a thread pool
                response = await run_in_threadpool(
                    bedrock_runtime.converse_stream, **args
                )
            else:
                # Run the blocking boto3 call in a thread pool
                response = await run_in_threadpool(bedrock_runtime.converse, **args)
        except bedrock_runtime.exceptions.ValidationException as e:
            logger.error("Bedrock validation error for model %s: %s", chat_request.model, str(e))
            raise HTTPException(status_code=400, detail=str(e))
        except bedrock_runtime.exceptions.ThrottlingException as e:
            logger.warning("Bedrock throttling for model %s: %s", chat_request.model, str(e))
            raise HTTPException(status_code=429, detail=str(e))
        except Exception as e:
            logger.error("Bedrock invocation failed for model %s: %s", chat_request.model, str(e))
            raise HTTPException(status_code=500, detail=str(e))
        return response

    async def chat(self, chat_request: ChatRequest) -> ChatResponse:
        """Default implementation for Chat API."""

        message_id = self.generate_message_id()
        response = await self._invoke_bedrock(chat_request)

        output_message = response["output"]["message"]
        input_tokens = response["usage"]["inputTokens"]
        output_tokens = response["usage"]["outputTokens"]
        finish_reason = response["stopReason"]

        chat_response = self._create_response(
            model=chat_request.model,
            message_id=message_id,
            content=output_message["content"],
            finish_reason=finish_reason,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        if DEBUG:
            logger.info("Proxy response :" + chat_response.model_dump_json())
        return chat_response

    async def _async_iterate(self, stream):
        """Helper method to convert sync iterator to async iterator"""
        for chunk in stream:
            await run_in_threadpool(lambda: chunk)
            yield chunk

    async def chat_stream(self, chat_request: ChatRequest) -> AsyncIterable[bytes]:
        """Default implementation for Chat Stream API"""
        try:
            response = await self._invoke_bedrock(chat_request, stream=True)
            message_id = self.generate_message_id()
            stream = response.get("stream")
            self.think_emitted = False
            async for chunk in self._async_iterate(stream):
                args = {"model_id": chat_request.model, "message_id": message_id, "chunk": chunk}
                stream_response = self._create_response_stream(**args)
                if not stream_response:
                    continue
                if DEBUG:
                    logger.info("Proxy response :" + stream_response.model_dump_json())
                if stream_response.choices:
                    yield self.stream_response_to_bytes(stream_response)
                elif chat_request.stream_options and chat_request.stream_options.include_usage:
                    # An empty choices for Usage as per OpenAI doc below:
                    # if you set stream_options: {"include_usage": true}.
                    # an additional chunk will be streamed before the data: [DONE] message.
                    # The usage field on this chunk shows the token usage statistics for the entire request,
                    # and the choices field will always be an empty array.
                    # All other chunks will also include a usage field, but with a null value.
                    yield self.stream_response_to_bytes(stream_response)

            # return an [DONE] message at the end.
            yield self.stream_response_to_bytes()
            self.think_emitted = False  # Cleanup
        except Exception as e:
            logger.error("Stream error for model %s: %s", chat_request.model, str(e))
            error_event = Error(error=ErrorMessage(message=str(e)))
            yield self.stream_response_to_bytes(error_event)

    def _parse_system_prompts(self, chat_request: ChatRequest) -> list[dict[str, str]]:
        """Create system prompts.
        Note that not all models support system prompts.

        example output: [{"text" : system_prompt}]

        See example:
        https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference.html#message-inference-examples
        """

        system_prompts = []
        for message in chat_request.messages:
            if message.role != "system":
                # ignore system messages here
                continue
            assert isinstance(message.content, str)
            system_prompts.append({"text": message.content})

        return system_prompts

    def _parse_messages(self, chat_request: ChatRequest) -> list[dict]:
        """
        Converse API only support user and assistant messages.

        example output: [{
            "role": "user",
            "content": [{"text": input_text}]
        }]

        See example:
        https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference.html#message-inference-examples
        """
        messages = []
        for message in chat_request.messages:
            if isinstance(message, UserMessage):
                messages.append(
                    {
                        "role": message.role,
                        "content": self._parse_content_parts(
                            message, chat_request.model
                        ),
                    }
                )
            elif isinstance(message, AssistantMessage):
                # Check if message has content that's not empty
                has_content = False
                if isinstance(message.content, str):
                    has_content = message.content.strip() != ""
                elif isinstance(message.content, list):
                    has_content = len(message.content) > 0
                elif message.content is not None:
                    has_content = True
                
                if has_content:
                    # Text message
                    messages.append(
                        {
                            "role": message.role,
                            "content": self._parse_content_parts(
                                message, chat_request.model
                            ),
                        }
                    )
                if message.tool_calls:
                    # Tool use message
                    for tool_call in message.tool_calls:
                        tool_input = json.loads(tool_call.function.arguments)
                        messages.append(
                            {
                                "role": message.role,
                                "content": [
                                    {
                                        "toolUse": {
                                            "toolUseId": tool_call.id,
                                            "name": tool_call.function.name,
                                            "input": tool_input,
                                        }
                                    }
                                ],
                            }
                        )
            elif isinstance(message, ToolMessage):
                # Bedrock does not support tool role,
                # Add toolResult to content
                # https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ToolResultBlock.html
                
                # Handle different content formats from OpenAI SDK
                tool_content = self._extract_tool_content(message.content)
                
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "toolResult": {
                                    "toolUseId": message.tool_call_id,
                                    "content": [{"text": tool_content}],
                                }
                            }
                        ],
                    }
                )

            else:
                # ignore others, such as system messages
                continue
        return self._reframe_multi_payloard(messages)

    def _extract_tool_content(self, content) -> str:
        """Extract text content from various OpenAI SDK tool message formats.
        
        Handles:
        - String content (legacy format)
        - List of content objects (OpenAI SDK 1.91.0+)
        - Nested JSON structures within text content
        """
        try:
            if isinstance(content, str):
                return content
            
            if isinstance(content, list):
                text_parts = []
                for i, item in enumerate(content):
                    if isinstance(item, dict):
                        # Handle dict with 'text' field
                        if "text" in item:
                            item_text = item["text"]
                            if isinstance(item_text, str):
                                # Try to parse as JSON if it looks like JSON
                                if item_text.strip().startswith('{') and item_text.strip().endswith('}'):
                                    try:
                                        parsed_json = json.loads(item_text)
                                        # Convert JSON object to readable text
                                        text_parts.append(json.dumps(parsed_json, indent=2))
                                    except json.JSONDecodeError:
                                        # Silently fallback to original text
                                        text_parts.append(item_text)
                                else:
                                    text_parts.append(item_text)
                            else:
                                text_parts.append(str(item_text))
                        else:
                            # Handle other dict formats - convert to JSON string
                            text_parts.append(json.dumps(item, indent=2))
                    elif hasattr(item, 'text'):
                        # Handle ToolContent objects
                        text_parts.append(item.text)
                    else:
                        # Convert any other type to string
                        text_parts.append(str(item))
                return "\n".join(text_parts)
            
            # Fallback for any other type
            return str(content)
        except Exception as e:
            logger.warning("Tool content extraction failed: %s", str(e))
            # Return a safe fallback
            return str(content) if content is not None else ""

    def _reframe_multi_payloard(self, messages: list) -> list:
        """Receive messages and reformat them to comply with the Claude format

        With OpenAI format requests, it's not a problem to repeatedly receive messages from the same role, but
        with Claude format requests, you cannot repeatedly receive messages from the same role.

        This method searches through the OpenAI format messages in order and reformats them to the Claude format.

        ```
        openai_format_messages=[
            {"role": "user", "content": "Hello"},
            {"role": "user", "content": "Who are you?"},
        ]

        bedrock_format_messages=[
            {
                "role": "user",
                "content": [
                    {"text": "Hello"},
                    {"text": "Who are you?"}
                ]
            },
        ]
        """
        reformatted_messages = []
        current_role = None
        current_content = []

        # Search through the list of messages and combine messages from the same role into one list
        for message in messages:
            next_role = message["role"]
            next_content = message["content"]

            # If the next role is different from the previous message, add the previous role's messages to the list
            if next_role != current_role:
                if current_content:
                    reformatted_messages.append(
                        {"role": current_role, "content": current_content}
                    )
                # Switch to the new role
                current_role = next_role
                current_content = []

            # Add the message content to current_content
            if isinstance(next_content, str):
                current_content.append({"text": next_content})
            elif isinstance(next_content, list):
                current_content.extend(next_content)

        # Add the last role's messages to the list
        if current_content:
            reformatted_messages.append(
                {"role": current_role, "content": current_content}
            )

        return reformatted_messages

    def _parse_request(self, chat_request: ChatRequest) -> dict:
        """Create default converse request body.

        Also perform validations to tool call etc.

        Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html
        """
        messages = self._parse_messages(chat_request)
        system_prompts = self._parse_system_prompts(chat_request)

        # Base inference parameters.
        inference_config = {
            "temperature": chat_request.temperature,
            "maxTokens": chat_request.max_tokens,
            "topP": chat_request.top_p,
        }

        # The Claude 4.5/4.6 generation doesn't support both temperature and topP
        # Remove topP for these models
        if requires_top_p_removal(chat_request.model):
            inference_config.pop("topP", None)

        if chat_request.stop is not None:
            stop = chat_request.stop
            if isinstance(stop, str):
                stop = [stop]
            inference_config["stopSequences"] = stop

        args = {
            "modelId": chat_request.model,
            "messages": messages,
            "system": system_prompts,
            "inferenceConfig": inference_config,
        }
        if chat_request.reasoning_effort:
            # From OpenAI api, the max_token is not supported in reasoning mode
            # Use max_completion_tokens if provided.

            max_tokens = (
                chat_request.max_completion_tokens
                if chat_request.max_completion_tokens
                else chat_request.max_tokens
            )
            budget_tokens = self._calc_budget_tokens(
                max_tokens, chat_request.reasoning_effort
            )
            inference_config["maxTokens"] = max_tokens
            # unset temperature and topP - both are incompatible with thinking
            # https://platform.claude.com/docs/en/build-with-claude/thinking
            inference_config.pop("temperature", None)
            inference_config.pop("topP", None)

            args["additionalModelRequestFields"] = {
                "reasoning_config": {"type": "enabled", "budget_tokens": budget_tokens}
            }
        # add tool config
        if chat_request.tools:
            tool_config = {"tools": [self._convert_tool_spec(t.function) for t in chat_request.tools]}

            if chat_request.tool_choice and not chat_request.model.startswith(
                "meta.llama3-1-"
            ):
                if isinstance(chat_request.tool_choice, str):
                    # auto (default) is mapped to {"auto" : {}}
                    # required is mapped to {"any" : {}}
                    if chat_request.tool_choice == "required":
                        tool_config["toolChoice"] = {"any": {}}
                    else:
                        tool_config["toolChoice"] = {"auto": {}}
                else:
                    # Specific tool to use
                    assert "function" in chat_request.tool_choice
                    tool_config["toolChoice"] = {"tool": {"name": chat_request.tool_choice["function"].get("name", "")}}
            args["toolConfig"] = tool_config
        # add Additional fields to enable extend thinking
        if chat_request.extra_body:
            # reasoning_config will not be used
            args["additionalModelRequestFields"] = chat_request.extra_body
            # Thinking supports neither temperature nor topP, in either the adaptive
            # or the manual spelling of the thinking block. Remove both to avoid
            # validation errors. An extra_body that only carries an effort level or a
            # structured-output schema is not a thinking request and keeps them. See
            # THINKING_EXTRA_BODY_KEYS.
            # https://platform.claude.com/docs/en/build-with-claude/thinking
            if enables_thinking(chat_request.extra_body):
                inference_config.pop("temperature", None)
                inference_config.pop("topP", None)
        return args

    def _create_response(
        self,
        model: str,
        message_id: str,
        content: list[dict] | None = None,
        finish_reason: str | None = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> ChatResponse:
        message = ChatResponseMessage(
            role="assistant",
        )
        if finish_reason == "tool_use":
            # https://docs.aws.amazon.com/bedrock/latest/userguide/tool-use.html#tool-use-examples
            tool_calls = []
            for part in content:
                if "toolUse" in part:
                    tool = part["toolUse"]
                    tool_calls.append(
                        ToolCall(
                            id=tool["toolUseId"],
                            type="function",
                            function=ResponseFunction(
                                name=tool["name"],
                                arguments=json.dumps(tool["input"]),
                            ),
                        )
                    )
            message.tool_calls = tool_calls
            message.content = None
        else:
            message.content = ""
            for c in content:
                if "reasoningContent" in c:
                    message.reasoning_content = c["reasoningContent"][
                        "reasoningText"
                    ].get("text", "")
                elif "text" in c:
                    message.content = c["text"]
                else:
                    logger.warning(
                        "Unknown tag in message content " + ",".join(c.keys())
                    )
            if message.reasoning_content:
                message.content = f"<think>{message.reasoning_content}</think>{message.content}"
                message.reasoning_content = None

        response = ChatResponse(
            id=message_id,
            model=model,
            choices=[
                Choice(
                    index=0,
                    message=message,
                    finish_reason=self._convert_finish_reason(finish_reason),
                    logprobs=None,
                )
            ],
            usage=Usage(
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
            ),
        )
        response.system_fingerprint = "fp"
        response.object = "chat.completion"
        response.created = int(time.time())
        return response

    def _create_response_stream(
        self, model_id: str, message_id: str, chunk: dict
    ) -> ChatStreamResponse | None:
        """Parsing the Bedrock stream response chunk.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference.html#message-inference-examples
        """
        if DEBUG:
            logger.info("Bedrock response chunk: " + str(chunk))

        finish_reason = None
        message = None
        usage = None
        if "messageStart" in chunk:
            message = ChatResponseMessage(
                role=chunk["messageStart"]["role"],
                content="",
            )
        if "contentBlockStart" in chunk:
            # tool call start
            delta = chunk["contentBlockStart"]["start"]
            if "toolUse" in delta:
                # first index is content
                index = chunk["contentBlockStart"]["contentBlockIndex"] - 1
                message = ChatResponseMessage(
                    tool_calls=[
                        ToolCall(
                            index=index,
                            type="function",
                            id=delta["toolUse"]["toolUseId"],
                            function=ResponseFunction(
                                name=delta["toolUse"]["name"],
                                arguments="",
                            ),
                        )
                    ]
                )
        if "contentBlockDelta" in chunk:
            delta = chunk["contentBlockDelta"]["delta"]
            if "text" in delta:
                # stream content
                message = ChatResponseMessage(
                    content=delta["text"],
                )
            elif "reasoningContent" in delta:
                if "text" in delta["reasoningContent"]:
                    content = delta["reasoningContent"]["text"]
                    if not self.think_emitted:
                        # Port of "content_block_start" with "thinking"
                        content = "<think>" + content
                        self.think_emitted = True
                    message = ChatResponseMessage(content=content)
                elif "signature" in delta["reasoningContent"]:
                    # Port of "signature_delta"
                    if self.think_emitted:
                        message = ChatResponseMessage(content="\n </think> \n\n")
                    else:
                        return None  # Ignore signature if no <think> started
            else:
                # tool use
                index = chunk["contentBlockDelta"]["contentBlockIndex"] - 1
                message = ChatResponseMessage(
                    tool_calls=[
                        ToolCall(
                            index=index,
                            function=ResponseFunction(
                                arguments=delta["toolUse"]["input"],
                            ),
                        )
                    ]
                )
        if "messageStop" in chunk:
            message = ChatResponseMessage()
            finish_reason = chunk["messageStop"]["stopReason"]

        if "metadata" in chunk:
            # usage information in metadata.
            metadata = chunk["metadata"]
            if "usage" in metadata:
                # token usage
                return ChatStreamResponse(
                    id=message_id,
                    model=model_id,
                    choices=[],
                    usage=Usage(
                        prompt_tokens=metadata["usage"]["inputTokens"],
                        completion_tokens=metadata["usage"]["outputTokens"],
                        total_tokens=metadata["usage"]["totalTokens"],
                    ),
                )
        if message:
            return ChatStreamResponse(
                id=message_id,
                model=model_id,
                choices=[
                    ChoiceDelta(
                        index=0,
                        delta=message,
                        logprobs=None,
                        finish_reason=self._convert_finish_reason(finish_reason),
                    )
                ],
                usage=usage,
            )

        return None

    def _parse_image(self, image_url: str) -> tuple[bytes, str]:
        """Try to get the raw data from an image url.

        Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ImageSource.html
        returns a tuple of (Image Data, Content Type)
        """
        pattern = r"^data:(image/[a-z]*);base64,\s*"
        content_type = re.search(pattern, image_url)
        # if already base64 encoded.
        # Only supports 'image/jpeg', 'image/png', 'image/gif' or 'image/webp'
        if content_type:
            image_data = re.sub(pattern, "", image_url)
            return base64.b64decode(image_data), content_type.group(1)

        # Send a request to the image URL
        response = requests.get(image_url)
        # Check if the request was successful
        if response.status_code == 200:
            content_type = response.headers.get("Content-Type")
            if not content_type.startswith("image"):
                content_type = "image/jpeg"
            # Get the image content
            image_content = response.content
            return image_content, content_type
        else:
            raise HTTPException(
                status_code=500, detail="Unable to access the image url"
            )

    def _parse_content_parts(
        self,
        message: UserMessage | AssistantMessage,
        model_id: str,
    ) -> list[dict]:
        if isinstance(message.content, str):
            return [
                {
                    "text": message.content,
                }
            ]
        content_parts = []
        for part in message.content:
            if isinstance(part, TextContent):
                content_parts.append(
                    {
                        "text": part.text,
                    }
                )
            elif isinstance(part, ImageContent):
                if not self.is_supported_modality(model_id, modality="IMAGE"):
                    raise HTTPException(
                        status_code=400,
                        detail=f"Multimodal message is currently not supported by {model_id}",
                    )
                image_data, content_type = self._parse_image(part.image_url.url)
                content_parts.append(
                    {
                        "image": {
                            "format": content_type[6:],  # image/
                            "source": {"bytes": image_data},
                        },
                    }
                )
            else:
                # Ignore..
                continue
        return content_parts

    @staticmethod
    def is_supported_modality(model_id: str, modality: str = "IMAGE") -> bool:
        model = bedrock_model_list.get(model_id, {})
        modalities = model.get("modalities", [])
        if modality in modalities:
            return True
        return False

    def _convert_tool_spec(self, func: Function) -> dict:
        return {
            "toolSpec": {
                "name": func.name,
                "description": func.description,
                "inputSchema": {
                    "json": func.parameters,
                },
            }
        }

    def _calc_budget_tokens(
        self, max_tokens: int, reasoning_effort: Literal["low", "medium", "high"]
    ) -> int:
        # Helper function to calculate budget_tokens based on the max_tokens.
        # Ratio for efforts:  Low - 30%, medium - 60%, High: Max token - 1
        # The result is clamped to MIN_REASONING_BUDGET_TOKENS: the ratios alone can
        # fall below the minimum Bedrock accepts (the default max_tokens of 2048 with
        # effort "low" gives 614, which Bedrock rejects).
        # budget_tokens must also stay below maxTokens, so a max_tokens at or below
        # the minimum cannot be satisfied at all. validate() rejects this earlier for
        # requests that go through the router; this is the backstop.
        if max_tokens <= MIN_REASONING_BUDGET_TOKENS:
            raise HTTPException(
                status_code=400,
                detail=reasoning_max_tokens_error(max_tokens),
            )

        if reasoning_effort == "low":
            budget_tokens = int(max_tokens * 0.3)
        elif reasoning_effort == "medium":
            budget_tokens = int(max_tokens * 0.6)
        else:
            budget_tokens = max_tokens - 1

        if budget_tokens < MIN_REASONING_BUDGET_TOKENS:
            # The clamp takes a disproportionate share of a small output window: with
            # max_tokens=1500 and effort "low" the budget becomes 1024 and only 476
            # tokens are left for the answer, which may come back truncated. Warn
            # rather than fail - the request is still valid for Bedrock.
            logger.warning(
                "Raising reasoning budget_tokens from %d to the Bedrock minimum %d; "
                "only %d of max_tokens=%d is left for the answer, which may be truncated. "
                "Use a larger max_tokens (Anthropic recommends a budget of at least 4000).",
                budget_tokens,
                MIN_REASONING_BUDGET_TOKENS,
                max_tokens - MIN_REASONING_BUDGET_TOKENS,
                max_tokens,
            )
            budget_tokens = MIN_REASONING_BUDGET_TOKENS
        return budget_tokens

    def _convert_finish_reason(self, finish_reason: str | None) -> str | None:
        """
        Below is a list of finish reason according to OpenAI doc:

        - stop: if the model hit a natural stop point or a provided stop sequence,
        - length: if the maximum number of tokens specified in the request was reached,
        - content_filter: if content was omitted due to a flag from our content filters,
        - tool_calls: if the model called a tool
        """
        if finish_reason:
            finish_reason_mapping = {
                "tool_use": "tool_calls",
                "finished": "stop",
                "end_turn": "stop",
                "max_tokens": "length",
                "stop_sequence": "stop",
                "complete": "stop",
                "content_filtered": "content_filter",
            }
            return finish_reason_mapping.get(
                finish_reason.lower(), finish_reason.lower()
            )
        return None


class BedrockEmbeddingsModel(BaseEmbeddingsModel, ABC):
    accept = "application/json"
    content_type = "application/json"

    def _invoke_model(self, args: dict, model_id: str):
        body = json.dumps(args)
        if DEBUG:
            logger.info("Invoke Bedrock Model: " + model_id)
            logger.info("Bedrock request body: " + body)
        try:
            return bedrock_runtime.invoke_model(
                body=body,
                modelId=model_id,
                accept=self.accept,
                contentType=self.content_type,
            )
        except bedrock_runtime.exceptions.ValidationException as e:
            logger.error("Validation Error: " + str(e))
            raise HTTPException(status_code=400, detail=str(e))
        except bedrock_runtime.exceptions.ThrottlingException as e:
            logger.error("Throttling Error: " + str(e))
            raise HTTPException(status_code=429, detail=str(e))
        except Exception as e:
            logger.error(e)
            raise HTTPException(status_code=500, detail=str(e))

    def _create_response(
        self,
        embeddings: list[float],
        model: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        encoding_format: Literal["float", "base64"] = "float",
    ) -> EmbeddingsResponse:
        data = []
        for i, embedding in enumerate(embeddings):
            if encoding_format == "base64":
                arr = np.array(embedding, dtype=np.float32)
                arr_bytes = arr.tobytes()
                encoded_embedding = base64.b64encode(arr_bytes)
                data.append(Embedding(index=i, embedding=encoded_embedding))
            else:
                data.append(Embedding(index=i, embedding=embedding))
        response = EmbeddingsResponse(
            data=data,
            model=model,
            usage=EmbeddingsUsage(
                prompt_tokens=input_tokens,
                total_tokens=input_tokens + output_tokens,
            ),
        )
        if DEBUG:
            logger.info("Proxy response :" + response.model_dump_json())
        return response


class CohereEmbeddingsModel(BedrockEmbeddingsModel):
    def _parse_args(self, embeddings_request: EmbeddingsRequest) -> dict:
        texts = []
        if isinstance(embeddings_request.input, str):
            texts = [embeddings_request.input]
        elif isinstance(embeddings_request.input, list):
            texts = embeddings_request.input
        elif isinstance(embeddings_request.input, Iterable):
            # For encoded input
            # The workaround is to use tiktoken to decode to get the original text.
            encodings = []
            for inner in embeddings_request.input:
                if isinstance(inner, int):
                    # Iterable[int]
                    encodings.append(inner)
                else:
                    # Iterable[Iterable[int]]
                    text = ENCODER.decode(list(inner))
                    texts.append(text)
            if encodings:
                texts.append(ENCODER.decode(encodings))

        # Maximum of 2048 characters
        args = {
            "texts": texts,
            "input_type": "search_document",
            "truncate": "END",  # "NONE|START|END"
        }
        return args

    def embed(self, embeddings_request: EmbeddingsRequest) -> EmbeddingsResponse:
        response = self._invoke_model(
            args=self._parse_args(embeddings_request), model_id=embeddings_request.model
        )
        response_body = json.loads(response.get("body").read())
        if DEBUG:
            logger.info("Bedrock response body: " + str(response_body))

        return self._create_response(
            embeddings=response_body["embeddings"],
            model=embeddings_request.model,
            encoding_format=embeddings_request.encoding_format,
        )


class TitanEmbeddingsModel(BedrockEmbeddingsModel):
    def _parse_args(self, embeddings_request: EmbeddingsRequest) -> dict:
        if isinstance(embeddings_request.input, str):
            input_text = embeddings_request.input
        elif (
            isinstance(embeddings_request.input, list)
            and len(embeddings_request.input) == 1
        ):
            input_text = embeddings_request.input[0]
        else:
            raise ValueError(
                "Amazon Titan Embeddings models support only single strings as input."
            )
        args = {
            "inputText": input_text,
            # Note: inputImage is not supported!
        }
        if embeddings_request.model == "amazon.titan-embed-image-v1":
            args["embeddingConfig"] = (
                embeddings_request.embedding_config
                if embeddings_request.embedding_config
                else {"outputEmbeddingLength": 1024}
            )
        return args

    def embed(self, embeddings_request: EmbeddingsRequest) -> EmbeddingsResponse:
        response = self._invoke_model(
            args=self._parse_args(embeddings_request), model_id=embeddings_request.model
        )
        response_body = json.loads(response.get("body").read())
        if DEBUG:
            logger.info("Bedrock response body: " + str(response_body))

        return self._create_response(
            embeddings=[response_body["embedding"]],
            model=embeddings_request.model,
            input_tokens=response_body["inputTextTokenCount"],
        )


def get_embeddings_model(model_id: str) -> BedrockEmbeddingsModel:
    model_name = SUPPORTED_BEDROCK_EMBEDDING_MODELS.get(model_id, "")
    if DEBUG:
        logger.info("model name is " + model_name)
    match model_name:
        case "Cohere Embed Multilingual" | "Cohere Embed English":
            return CohereEmbeddingsModel()
        case "Titan Embeddings G2 - Text":
            return TitanEmbeddingsModel()
        case _:
            logger.error("Unsupported model id " + model_id)
            raise HTTPException(
                status_code=400,
                detail="Unsupported embedding model id " + model_id,
            )
