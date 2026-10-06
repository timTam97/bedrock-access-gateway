[中文](./Usage_CN.md)

# Usage Guide

Assuming you have set up below environment variables after deployed:

```bash
export OPENAI_API_KEY=<API key>
export OPENAI_BASE_URL=<API base url>
```

**API Example:**
- [Supported Anthropic models](#supported-anthropic-models)
- [Models API](#models-api)
- [Embedding API](#embedding-api)
- [Multimodal API](#multimodal-api)
- [Tool Call](#tool-call)
- [Reasoning](#reasoning)
- [Interleaved thinking (beta)](#Interleaved thinking (beta))

## Supported Anthropic models

AWS offers the current Claude models through **cross-region inference profiles** rather than on-demand throughput. Passing the bare base model ID to those models fails with an HTTP 400, so you must call them with a prefixed ID such as `us.<base-id>` or `global.<base-id>` (or the full inference profile ARN). The prefix determines routing and pricing: `global.` routes dynamically for maximum availability with no pricing premium, while the regional prefixes guarantee a geography and carry a 10% premium.

| Model | Base Bedrock model ID | Available prefixes |
|---|---|---|
| Claude Opus 4.6 | `anthropic.claude-opus-4-6-v1` | `global.` `us.` `eu.` `jp.` `apac.` |
| Claude Opus 4.5 | `anthropic.claude-opus-4-5-20251101-v1:0` | `global.` `us.` `eu.` |
| Claude Sonnet 4.6 | `anthropic.claude-sonnet-4-6` | `global.` `us.` `eu.` `jp.` |
| Claude Haiku 4.5 | `anthropic.claude-haiku-4-5-20251001-v1:0` | `global.` `us.` `eu.` |

Source: Anthropic's [Claude on Amazon Bedrock (legacy integration)](https://platform.claude.com/docs/en/build-with-claude/claude-on-amazon-bedrock-legacy) guide, section *API model IDs*. That page is the authoritative base-ID and prefix table for the `InvokeModel`/`Converse` APIs this proxy calls.

**Claude Sonnet 4.5** (`anthropic.claude-sonnet-4-5-20250929-v1:0`, prefixes `global.` `us.` `eu.` `jp.`) still works but is deprecated. Move to **Claude Sonnet 4.6** instead.

A prefix column above only tells you the profile *can* exist. The [Models API](#models-api) is the authoritative list for the region you actually deployed into - run it and use an ID it returns. See also AWS's [Supported Regions and models for inference profiles](https://docs.aws.amazon.com/bedrock/latest/userguide/inference-profiles-support.html).

### Cross-region inference profile discovery

The proxy builds its model list from the `ListInferenceProfiles` response of the region it runs in, so every system-defined prefix AWS offers there is picked up automatically - `us.`, `eu.`, `apac.`, `jp.`, `au.` and `global.`, plus any prefix AWS adds in future. Nothing is inferred from `AWS_REGION`, and no profile ID is ever synthesised: if the region does not return it, the proxy does not advertise it.

Set `ENABLE_CROSS_REGION_INFERENCE=false` to suppress all profile IDs and list only models with on-demand throughput.

> **Note:** Because the current Claude models have no on-demand entitlement, turning cross-region inference off will hide them from the model list entirely.

## Models API

You can use this API to get a list of supported model IDs.

Also, you can use this API to refresh the model list if new models are added to Amazon Bedrock.


**Example Request**

```bash
curl -s $OPENAI_BASE_URL/models -H "Authorization: Bearer $OPENAI_API_KEY" | jq .data
```

**Example Response**

```bash
[
  ...
  {
    "id": "global.anthropic.claude-haiku-4-5-20251001-v1:0",
    "created": 1734416893,
    "object": "model",
    "owned_by": "bedrock"
  },
  {
    "id": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "created": 1734416893,
    "object": "model",
    "owned_by": "bedrock"
  },
  ...
]
```

## Chat Completions API

### Basic Example with Claude Haiku 4.5

Claude Haiku 4.5 is the fastest and cheapest model of the current Claude generation, which makes it a good default for everyday requests. It is the default model this proxy falls back to when `DEFAULT_MODEL` is not set.

**Example Request**

```bash
curl $OPENAI_BASE_URL/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
    "model": "global.anthropic.claude-haiku-4-5-20251001-v1:0",
    "messages": [
      {
        "role": "user",
        "content": "Write a Python function to calculate the Fibonacci sequence using dynamic programming."
      }
    ]
  }'
```

**Example SDK Usage**

```python
from openai import OpenAI

client = OpenAI()
completion = client.chat.completions.create(
    model="global.anthropic.claude-haiku-4-5-20251001-v1:0",
    messages=[{"role": "user", "content": "Write a Python function to calculate the Fibonacci sequence using dynamic programming."}],
)

print(completion.choices[0].message.content)
```

### Frontier Example with Claude Opus 4.6

Claude Opus 4.6 is the most capable model reachable through this proxy. Swap the model ID in for the hardest coding, reasoning and agentic work.

```bash
curl $OPENAI_BASE_URL/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
    "model": "global.anthropic.claude-opus-4-6-v1",
    "messages": [
      {
        "role": "user",
        "content": "Write a Python function to calculate the Fibonacci sequence using dynamic programming."
      }
    ]
  }'
```

## Embedding API

**Important Notice**: Please carefully review the following points before using this proxy API for embedding.

1. If you have previously used OpenAI embedding models to create vectors, be aware that switching to a new model may not be straightforward. Different models have varying dimensions (e.g., embed-multilingual-v3.0 has 1024 dimensions), and even for the same text, they may produce different results.
2. If you are using OpenAI embedding models for encoded integers (such as with LangChain), this solution will attempt to decode the integers using `tiktoken` to retrieve the original text. However, there is no guarantee that the decoded text will be accurate.
3. If you are using OpenAI embedding models for long texts, you should verify the maximum number of tokens supported for Bedrock models, e.g. for optimal performance, Bedrock recommends limiting the text length to less than 512 tokens.


**Example Request**

```bash
curl $OPENAI_BASE_URL/embeddings \
-H "Authorization: Bearer $OPENAI_API_KEY" \
-H "Content-Type: application/json" \
-d '{
    "input": "The food was delicious and the waiter...",
    "model": "text-embedding-ada-002",
    "encoding_format": "float"
  }'
```

**Example Response**

```json
{
    "object": "list",
    "data": [
        {
            "object": "embedding",
            "embedding": [
                -0.02279663,
                -0.024612427,
                0.012863159,
                ...
                0.01612854,
                0.0038928986
            ],
            "index": 0
        }
    ],
    "model": "cohere.embed-multilingual-v3",
    "usage": {
        "prompt_tokens": 0,
        "total_tokens": 0
    }
}
```

Alternatively, you can use the OpenAI SDK

```python
from openai import OpenAI

client = OpenAI()

def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding

text = "hello"
# will output like [0.003578186, 0.028717041, 0.031021118, -0.0014066696,...]
print(get_embedding(text))
```

Or LangChain

```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
)
text = "This is a test document."
query_result = embeddings.embed_query(text)
print(query_result[:5])
doc_result = embeddings.embed_documents([text])
print(doc_result[0][:5])
```

## Multimodal API

**Example Request**

```bash
curl $OPENAI_BASE_URL/chat/completions \
curl $OPENAI_BASE_URL/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "please identify and count all the objects in these images, list all the names"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://github.com/aws-samples/bedrock-access-gateway/blob/main/assets/obj-detect.png?raw=true"
                    }
                }
            ]
        }
    ]
}'
```

If you need to use this API with non-public images, you can do base64 the image first and pass the encoded string. 
Replace `image/jpeg` with the actual content type. Currently, only 'image/jpeg', 'image/png', 'image/gif' or 'image/webp' is supported.

```bash
curl $OPENAI_BASE_URL/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "please identify and count all the objects in this images, list all the names"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/jpeg;base64,<your image data>"
                    }
                }
            ]
        }
    ]
}'
```

**Example Response**

```json
{
    "id": "msg_01BY3wcz41x7XrKhxY3VzWke",
    "created": 1712543069,
    "model": "global.anthropic.claude-haiku-4-5-20251001-v1:0",
    "system_fingerprint": "fp",
    "choices": [
        {
            "index": 0,
            "finish_reason": "stop",
            "message": {
                "role": "assistant",
                "content": "The image contains the following objects:\n\n1. A peach-colored short-sleeve button-up shirt\n2. An olive green plaid long coat/jacket\n3. A pair of white sneakers or canvas shoes\n4. A brown shoulder bag or purse\n5. A makeup brush or cosmetic applicator\n6. A tube or container (possibly lipstick or lip balm)\n7. A pair of sunglasses\n8. A thought bubble icon\n9. A footprint icon\n10. A leaf or plant icon\n11. A flower icon\n12. A cloud icon\n\nIn total, there are 12 distinct objects depicted in the illustrated scene."
            }
        }
    ],
    "object": "chat.completion",
    "usage": {
        "prompt_tokens": 197,
        "completion_tokens": 147,
        "total_tokens": 344
    }
}
```


## Tool Call

**Important Notice**: Please carefully review the following points before using this Tool Call for Chat completion API.

1. Function Call is now deprecated in favor of Tool Call by OpenAI, hence it's not supported here, you should use Tool Call instead.

**Example Request**

```bash
curl $OPENAI_BASE_URL/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [
        {
            "role": "user",
            "content": "What is the weather like in Shanghai today?"
        }
    ],
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city or state which is required."
                        },
                        "unit": {
                            "type": "string",
                            "enum": [
                                "celsius",
                                "fahrenheit"
                            ]
                        }
                    },
                    "required": [
                        "location"
                    ]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_current_location",
                "description": "Use this tool to get the current location if user does not provide a location",
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            }
        }
    ],
    "tool_choice": "auto"
}'
```

**Example Response**

```json
{
    "id": "msg_01PjrKDWhYGsrTNdeqzWd6D9",
    "created": 1712543689,
    "model": "global.anthropic.claude-haiku-4-5-20251001-v1:0",
    "system_fingerprint": "fp",
    "choices": [
        {
            "index": 0,
            "finish_reason": "stop",
            "message": {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "0",
                        "type": "function",
                        "function": {
                            "name": "get_current_weather",
                            "arguments": "{\"location\": \"Shanghai\", \"unit\": \"celsius\"}"
                        }
                    }
                ]
            }
        }
    ],
    "object": "chat.completion",
    "usage": {
        "prompt_tokens": 256,
        "completion_tokens": 64,
        "total_tokens": 320
    }
}
```

You can try it with different questions, such as:
1. Hello, who are you?  (No tools are needed)
2. What is the weather like today?  (Should use get_current_location tool first)


## Reasoning

**Important Notice**: Please carefully review the following points before using reasoning mode for Chat completion API.
- Reasoning is supported by the Claude models that offer extended thinking and by DeepSeek R1. Please make sure the model supports reasoning before use. Among the models reachable through this proxy, extended thinking is available on Claude Opus 4.6, Claude Opus 4.5, Claude Sonnet 4.6 and Claude Haiku 4.5. On Claude Opus 4.6 and Claude Sonnet 4.6 it is deprecated in favour of [adaptive thinking](#adaptive-thinking-and-the-effort-parameter).
- For Claude, the reasoning mode (or thinking mode) is not enabled by default, you must pass an additional `reasoning_effort` parameter in your request. Please also provide the right max_tokens (or max_completion_tokens) in your request. The budget_tokens is based on reasoning_effort (low: 30%, medium: 60%, high: 100% of max tokens), clamped to a minimum of 1,024 with Anthropic recommending at least 4,000 tokens for comprehensive reasoning. Because the budget must leave room for the answer, a reasoning request with `max_tokens` (or `max_completion_tokens`) of 1,024 or less is rejected with an HTTP 400 - ask for more tokens instead. Check [Bedrock Document](https://docs.aws.amazon.com/bedrock/latest/userguide/claude-messages-extended-thinking.html) for more details.
- `temperature` and `top_p` are **not** forwarded while thinking is on. Anthropic documents `temperature` and `top_k` as incompatible with thinking, and `top_p` as valid only between 0.95 and 1, so the proxy drops both rather than letting Bedrock reject the request. See [Extended thinking - sampling parameters](https://platform.claude.com/docs/en/build-with-claude/thinking). Any value you set for them is ignored on a thinking request.
- For DeepSeek R1, you don't need additional reasoning_effort parameter, otherwise, you may get an error.
- The reasoning response (CoT, thoughts) is added in an additional tag 'reasoning_content' which is not officially supported by OpenAI. This is to follow [Deepseek Reasoning Model](https://api-docs.deepseek.com/guides/reasoning_model#api-example). This may be changed in the future.

**Example Request**

- Claude Haiku 4.5

```bash
curl $OPENAI_BASE_URL/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
    "model": "global.anthropic.claude-haiku-4-5-20251001-v1:0",
    "messages": [
        {
            "role": "user",
            "content": "which one is bigger, 3.9 or 3.11?"
        }
    ],
    "max_completion_tokens": 4096,
    "reasoning_effort": "low",
    "stream": false
}'
```

- DeepSeek R1

```bash
curl $OPENAI_BASE_URL/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
    "model": "us.deepseek.r1-v1:0",
    "messages": [
        {
            "role": "user",
            "content": "which one is bigger, 3.9 or 3.11?"
        }
    ],
    "stream": false
}'
```

**Example Response**

```json
{
    "id": "chatcmpl-83fb7a88",
    "created": 1740545278,
    "model": "global.anthropic.claude-haiku-4-5-20251001-v1:0",
    "system_fingerprint": "fp",
    "choices": [
        {
            "index": 0,
            "finish_reason": "stop",
            "logprobs": null,
            "message": {
                "role": "assistant",
                "content": "3.9 is bigger than 3.11.\n\nWhen comparing decimal numbers, we need to understand what these numbers actually represent:...",
                "reasoning_content": "I need to compare the decimal numbers 3.9 and 3.11.\n\nFor decimal numbers, we first compare the whole number parts, and if they're equal, we compare the decimal parts. \n\nBoth numbers ..."
            }
        }
    ],
    "object": "chat.completion",
    "usage": {
        "prompt_tokens": 51,
        "completion_tokens": 565,
        "total_tokens": 616
    }
}
```

You can also use OpenAI SDK (run `pip3 install -U openai` first )

- Non-Streaming

```python
from openai import OpenAI
client = OpenAI()

messages = [{"role": "user", "content": "which one is bigger, 3.9 or 3.11?"}]
response = client.chat.completions.create(
    model="global.anthropic.claude-haiku-4-5-20251001-v1:0",
    messages=messages,
    reasoning_effort="low",
    max_completion_tokens=4096,
)

reasoning_content = response.choices[0].message.reasoning_content
content = response.choices[0].message.content
```

- Streaming

```python
from openai import OpenAI
client = OpenAI()

messages = [{"role": "user", "content": "9.11 and 9.8, which is greater?"}]
response = client.chat.completions.create(
    model="global.anthropic.claude-haiku-4-5-20251001-v1:0",
    messages=messages,
    reasoning_effort="low",
    max_completion_tokens=4096,
    stream=True,
)

reasoning_content = ""
content = ""

for chunk in response:
    if hasattr(chunk.choices[0].delta, 'reasoning_content') and chunk.choices[0].delta.reasoning_content:
        reasoning_content += chunk.choices[0].delta.reasoning_content
    elif chunk.choices[0].delta.content:
        content += chunk.choices[0].delta.content
```

### Adaptive thinking and the effort parameter

On **Claude Opus 4.6** and **Claude Sonnet 4.6**, manual extended thinking (`thinking.type: "enabled"` with an explicit `budget_tokens`) is deprecated. Those models use *adaptive* thinking instead: Claude decides when and how deeply to think, and you steer it with an `effort` level (`low`, `medium`, `high`, `max`) rather than a token budget. Anthropic documents adaptive thinking as the supported mode for both models and lists `effort` support on Claude Opus 4.5, Claude Opus 4.6 and Claude Sonnet 4.6. See [Extended thinking](https://platform.claude.com/docs/en/build-with-claude/thinking) and [Effort](https://platform.claude.com/docs/en/build-with-claude/effort).

This proxy needs no new parameter for either: anything you put in `extra_body` is forwarded verbatim as the Bedrock Converse `additionalModelRequestFields`, so you can pass the model's own thinking and effort fields straight through.

```bash
curl $OPENAI_BASE_URL/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
    "model": "global.anthropic.claude-opus-4-6-v1",
    "max_tokens": 8192,
    "messages": [
        {
            "role": "user",
            "content": "which one is bigger, 3.9 or 3.11?"
        }
    ],
    "extra_body": {
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": "medium"}
    }
}'
```

> **Unverified:** `thinking.type: "adaptive"` and `output_config.effort` are the field names Anthropic documents for its **Messages API**. We could not find an AWS-published statement of how those fields are named inside the Converse API's `additionalModelRequestFields`, and no live Bedrock call was made to confirm it, so treat the payload above as a template rather than a verified contract. If Bedrock rejects it with a `ValidationException` about an unexpected field, check the current [Bedrock inference request parameters](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-anthropic-claude-messages.html) for the accepted spelling and adjust `extra_body` - no change to this proxy is needed either way.

Note that `reasoning_effort` and `extra_body` are different paths: `reasoning_effort` makes the proxy compute a `budget_tokens` value for you (manual extended thinking), while `extra_body` hands your fields to Bedrock untouched. Use one or the other, not both.

## Interleaved thinking (beta)

**Important Notice**: Please carefully review the following points before using reasoning mode for Chat completion API.

Extended thinking with tool use in Claude 4 models supports [interleaved thinking](https://docs.aws.amazon.com/bedrock/latest/userguide/claude-messages-extended-thinking.html#claude-messages-extended-thinking-tool-use-interleaved) enables Claude 4 models to think between tool calls and run more sophisticated reasoning after receiving tool results. which is helpful for more complex agentic interactions.
With interleaved thinking, the `budget_tokens` can exceed the `max_tokens` parameter because it represents the total budget across all thinking blocks within one assistant turn.

**Supported Models**: the Claude models that offer extended thinking. Of the models reachable through this proxy that is Claude Opus 4.5 and Claude Haiku 4.5, plus Claude Opus 4.6 and Claude Sonnet 4.6 where extended thinking still works but is deprecated - prefer [adaptive thinking](#adaptive-thinking-and-the-effort-parameter) on those two.

> **Note:** `temperature` and `top_p` are dropped from the request whenever `extra_body` contains a `thinking` field, because Anthropic documents them as incompatible with thinking.

**Example Request**

- Non-Streaming (Claude Opus 4.5)

```bash
curl http://127.0.0.1:8000/api/v1/chat/completions \
-H "Content-Type: application/json" \
-H "Authorization: Bearer bedrock" \
-d '{
"model": "global.anthropic.claude-opus-4-5-20251101-v1:0",
"max_tokens": 2048,
"messages": [{
"role": "user",
"content": "Explain how to implement a binary search tree with self-balancing capabilities."
}],
"extra_body": {
"anthropic_beta": ["interleaved-thinking-2025-05-14"],
"thinking": {"type": "enabled", "budget_tokens": 4096}
}
}'
```

- Non-Streaming (Claude Haiku 4.5)

```bash
curl http://127.0.0.1:8000/api/v1/chat/completions \
-H "Content-Type: application/json" \
-H "Authorization: Bearer bedrock" \
-d '{
"model": "global.anthropic.claude-haiku-4-5-20251001-v1:0",
"max_tokens": 2048,
"messages": [{
"role": "user",
"content": "有一天，一个女孩参加数学考试只得了 38 分。她心里对父亲的惩罚充满恐惧，于是偷偷把分数改成了 88 分。她的父亲看到试卷后，怒发冲冠，狠狠地给了她一巴掌，怒吼道：“你这 8 怎么一半是绿的一半是红的，你以为我是傻子吗？”女孩被打后，委屈地哭了起来，什么也没说。过了一会儿，父亲突然崩溃了。请问这位父亲为什么过一会崩溃了？"
}],
"extra_body": {
"anthropic_beta": ["interleaved-thinking-2025-05-14"],
"thinking": {"type": "enabled", "budget_tokens": 4096}
}
}'
```

- Streaming (Claude Opus 4.5)

```bash
curl http://127.0.0.1:8000/api/v1/chat/completions \
-H "Content-Type: application/json" \
-H "Authorization: Bearer bedrock" \
-d '{
"model": "global.anthropic.claude-opus-4-5-20251101-v1:0",
"max_tokens": 2048,
"messages": [{
"role": "user",
"content": "Explain how to implement a binary search tree with self-balancing capabilities."
}],
"stream": true,
"extra_body": {
"anthropic_beta": ["interleaved-thinking-2025-05-14"],
"thinking": {"type": "enabled", "budget_tokens": 4096}
}
}'
```

- Streaming (Claude Haiku 4.5)

```bash
curl http://127.0.0.1:8000/api/v1/chat/completions \
-H "Content-Type: application/json" \
-H "Authorization: Bearer bedrock" \
-d '{
"model": "global.anthropic.claude-haiku-4-5-20251001-v1:0",
"max_tokens": 2048,
"messages": [{
"role": "user",
"content": "有一天，一个女孩参加数学考试只得了 38 分。她心里对父亲的惩罚充满恐惧，于是偷偷把分数改成了 88 分。她的父亲看到试卷后，怒发冲冠，狠狠地给了她一巴掌，怒吼道：“你这 8 怎么一半是绿的一半是红的，你以为我是傻子吗？”女孩被打后，委屈地哭了起来，什么也没说。过了一会儿，父亲突然崩溃了。请问这位父亲为什么过一会崩溃了？"
}],
"stream": true,
"extra_body": {
"anthropic_beta": ["interleaved-thinking-2025-05-14"],
"thinking": {"type": "enabled", "budget_tokens": 4096}
}
}'
```
