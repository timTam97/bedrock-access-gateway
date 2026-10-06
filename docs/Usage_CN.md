[English](./Usage.md)

# Usage Guide

假设您在部署后已设置以下环境变量:

```bash
export OPENAI_API_KEY=<API key>
export OPENAI_BASE_URL=<API base url>
```

**API 示例:**
- [支持的 Anthropic 模型](#支持的-anthropic-模型)
- [Models API](#models-api)
- [Embedding API](#embedding-api)
- [Multimodal API](#multimodal-api)
- [Tool Call](#tool-call)
- [Reasoning](#reasoning)
- [Interleaved thinking (beta)](#Interleaved thinking (beta))


## 支持的 Anthropic 模型

AWS 通过**跨区域推理配置文件（cross-region inference profile）**而非按需吞吐量（on-demand throughput）提供当前这一代 Claude 模型。对这些模型直接传入基础模型 ID 会返回 HTTP 400 错误，因此必须使用带前缀的 ID，例如 `us.<base-id>` 或 `global.<base-id>`（也可以传入完整的推理配置文件 ARN）。前缀决定了路由方式和价格：`global.` 允许 Bedrock 将请求路由到任意受支持的商业区域，AWS 对其定价比标准价格**低约 10%**；而地理前缀（`us.`、`eu.`、`apac.`、`jp.`、`au.`）保证处理过程留在该地理区域内，按**标准价格**计费。两者都不收取额外的路由费用，价格按你发起调用的区域计算 —— 详见 AWS 的 [cross-Region inference](https://docs.aws.amazon.com/bedrock/latest/userguide/cross-region-inference.html) 文档。有数据驻留要求时选择地理前缀，优先考虑成本时选择 `global.`。

| 模型 | Bedrock 基础模型 ID | 可用前缀 |
|---|---|---|
| Claude Opus 4.6 | `anthropic.claude-opus-4-6-v1` | `global.` `us.` `eu.` `jp.` `apac.` |
| Claude Opus 4.5 | `anthropic.claude-opus-4-5-20251101-v1:0` | `global.` `us.` `eu.` |
| Claude Sonnet 4.6 | `anthropic.claude-sonnet-4-6` | `global.` `us.` `eu.` `jp.` |
| Claude Haiku 4.5 | `anthropic.claude-haiku-4-5-20251001-v1:0` | `global.` `us.` `eu.` |

来源：Anthropic 的 [Claude on Amazon Bedrock (legacy integration)](https://platform.claude.com/docs/en/build-with-claude/claude-on-amazon-bedrock-legacy) 文档中的 *API model IDs* 一节。该页面是本代理所调用的 `InvokeModel` / `Converse` API 的权威基础 ID 与前缀对照表。

**Claude Sonnet 4.5**（`anthropic.claude-sonnet-4-5-20250929-v1:0`，前缀 `global.` `us.` `eu.` `jp.`）仍然可用，但已被标记为弃用（deprecated），建议迁移到 **Claude Sonnet 4.6**。

上表中的前缀列只说明该配置文件*可能*存在。[Models API](#models-api) 才是你实际部署区域的权威列表 —— 请先运行它，并使用它返回的 ID。另请参阅 AWS 的 [Supported Regions and models for inference profiles](https://docs.aws.amazon.com/bedrock/latest/userguide/inference-profiles-support.html)。

> **这一代模型不会转发 `top_p`。** Bedrock Converse 不接受同时传入 `temperature` 和 `topP` 这一现象，**只**在 Claude Sonnet 4.5 上被观察到；Anthropic 官方只针对[思考请求](https://platform.claude.com/docs/en/build-with-claude/thinking)说明了该限制，并未说明整代模型存在无条件的限制。本代理仍然保守地把这个规避措施推广到整代模型：只要模型 ID 中包含 `claude-sonnet-4-5`、`claude-opus-4-5`、`claude-haiku-4-5`、`claude-sonnet-4-6` 或 `claude-opus-4-6`，就会从**每一个**请求中丢弃 `top_p` —— 无论是否开启思考模式。丢弃 `top_p` 而保留 `temperature`，是因为后者才是 OpenAI 客户端真正会设置的参数。如果你确实需要 `top_p`，请改用该列表之外的模型。有一个例外：发往**应用推理配置文件 ARN**（application inference profile ARN）的请求仍会携带 `top_p`，因为该 ARN 以不透明的配置文件 ID 结尾、不含模型名称，代理无法判断它指向哪个模型。

### 跨区域推理配置文件的发现机制

本代理会根据所在区域的 `ListInferenceProfiles` 响应来构建模型列表，因此该区域提供的所有系统定义前缀都会被自动识别 —— `us.`、`eu.`、`apac.`、`jp.`、`au.` 和 `global.`，以及 AWS 未来新增的任何前缀。前缀不再根据 `AWS_REGION` 推断，也不会凭空拼接出任何配置文件 ID：如果该区域没有返回它，代理就不会对外提供它。

设置 `ENABLE_CROSS_REGION_INFERENCE=false` 可以屏蔽所有配置文件 ID，只列出具备按需吞吐量的模型。

> **注意**：由于当前这一代 Claude 模型没有按需吞吐量权限，关闭跨区域推理会导致它们完全不出现在模型列表中。这同时也会让内置的默认模型失效，因为它是一个 `global.` 配置文件 ID：在 `ENABLE_CROSS_REGION_INFERENCE=false` 时，你必须把 `DEFAULT_MODEL` 设置为模型列表中仍然存在的 ID，否则所有回退到默认模型的请求（包括 `gpt-*` 别名）都会被判定为不支持的模型而被拒绝。

## Models API

你可以通过这个API 获取支持的models 列表。 另外，如果Amazon Bedrock有新模型加入后，你也可以用它来更新刷新模型列表。

**Request 示例**

```bash
curl -s $OPENAI_BASE_URL/models -H "Authorization: Bearer $OPENAI_API_KEY" | jq .data
```

**Response 示例**

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

### Claude Haiku 4.5 基础示例

Claude Haiku 4.5 是当前这一代 Claude 模型中速度最快、成本最低的一款，适合作为日常请求的默认选择。当未设置 `DEFAULT_MODEL` 时，本代理也会回退到该模型。

**Request 示例**

```bash
curl $OPENAI_BASE_URL/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
    "model": "global.anthropic.claude-haiku-4-5-20251001-v1:0",
    "messages": [
      {
        "role": "user",
        "content": "编写一个使用动态规划计算斐波那契数列的Python函数。"
      }
    ]
  }'
```

**SDK 使用示例**

```python
from openai import OpenAI

client = OpenAI()
completion = client.chat.completions.create(
    model="global.anthropic.claude-haiku-4-5-20251001-v1:0",
    messages=[{"role": "user", "content": "编写一个使用动态规划计算斐波那契数列的Python函数。"}],
)

print(completion.choices[0].message.content)
```

### Claude Opus 4.6 旗舰示例

Claude Opus 4.6 是本代理可访问的能力最强的模型。处理最困难的编码、推理和 Agentic 任务时，只需替换模型 ID 即可。

```bash
curl $OPENAI_BASE_URL/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
    "model": "global.anthropic.claude-opus-4-6-v1",
    "messages": [
      {
        "role": "user",
        "content": "编写一个使用动态规划计算斐波那契数列的Python函数。"
      }
    ]
  }'
```

## Embedding API

**重要**: 在使用此代理 API 之前,请仔细阅读以下几点:

1. 如果您之前使用 OpenAI Embedding模型来创建向量,请注意切换到新模型可能没有那么直接。不同模型具有不同的维度(例如,embed-multilingual-v3.0 有 1024 个维度),即使对于相同的文本,它们也可能产生不同的结果。
2. 如果您使用 OpenAI Embedding模型传入的是整数编码(例如与 LangChain 一起使用),此方案将尝试使用 `tiktoken` 进行解码以检索原始文本。但是,无法保证解码后的文本准确无误。
3. 如果您对长文本使用 OpenAI Embedding,您应该验证 Bedrock 模型支持的最大Token数,例如为获得最佳性能,Bedrock 建议将文本长度限制在少于 512 个Token。

**Request 示例**

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

**Response 示例**

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

或者你可以使用OpenAI 的SDK

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

或者 LangChain

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

**Request 示例**

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
                        "url": "https://github.com/aws-samples/bedrock-access-gateway/blob/main/assets/obj-detect.png?raw=true"
                    }
                }
            ]
        }
    ]
}'
```

如果您需要使用此API处理非公开图像,您可以先对图像进行base64编码,然后传递编码后的字符串。
将"image/jpeg"替换为实际的内容类型(content type)。目前仅支持"image/jpeg"、"image/png"、"image/gif"或"image/webp"。

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

**Response 示例**

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

**重要**:在使用此代理API进行Tool Call之前,请仔细阅读以下几点:

1. OpenAI 已经废弃使用Function Call,而推荐使用Tool Call,因此Function Call在此处不受支持,您应该改为Tool Call。

**Request 示例**

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

**Response 示例**

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


**重要**: 使用此 reasoning 推理模式前，请仔细阅读以下要点。

- 支持扩展思考（extended thinking）的 Claude 模型以及 Deepseek R1 支持推理功能。使用前请确保所用模型支持推理。在本代理可访问的模型中，支持扩展思考的是 Claude Opus 4.6、Claude Opus 4.5、Claude Sonnet 4.6 和 Claude Haiku 4.5。在 Claude Opus 4.6 和 Claude Sonnet 4.6 上，扩展思考已被弃用，取而代之的是[自适应思考](#自适应思考与-effort-参数)。
- Claude 的推理模式（或思考模式）默认未启用，您必须在请求中传递额外的 reasoning_effort 参数，参数值可选:low，medium, high。另外，请在请求中提供正确的 max_tokens（或 max_completion_tokens）参数。budget_tokens 基于 reasoning_effort 设置（低：30%，中：60%，高：100% 的max tokens），并会被限制为最小 1,024，Anthropic 建议至少使用 4,000 个令牌以获得全面的推理。由于思考预算必须为答案留出空间，当 `max_tokens`（或 `max_completion_tokens`）小于或等于 1,024 时，推理请求会被拒绝并返回 HTTP 400 —— 请改为申请更多 token。该校验在响应开始之前执行，因此流式调用方同样会收到真正的 400，而不是流内部的错误事件。略高于该下限时，钳制会占去输出窗口的大部分：`max_tokens` 为 1,500 且 effort 为 `low` 时，留给答案的只有 476 个 token，回答可能被截断甚至为空（钳制生效时代理会记录一条 warning 日志）。请为推理请求预留数千个 token。详情请参阅 [Bedrock Document](https://docs.aws.amazon.com/bedrock/latest/userguide/claude-messages-extended-thinking.html)。
- 当思考模式开启时，`temperature` 和 `top_p` **不会**被转发。Anthropic 明确说明 `temperature` 和 `top_k` 与思考模式不兼容，而 `top_p` 仅在 0.95 到 1 之间有效，因此本代理会直接丢弃这两个参数，而不是让 Bedrock 拒绝该请求。详见 [Extended thinking - sampling parameters](https://platform.claude.com/docs/en/build-with-claude/thinking)。在思考请求中，你为它们设置的任何值都会被忽略。
- Deepseek R1 会自动使用推理模式，不需要在中传递额外的 reasoning_effort 参数（否则会报错）
- 推理结果（思维链结果、思考过程）被添加到名为 'reasoning_content' 的额外标签中，这不是 OpenAI 官方支持的格式。此设计遵循 [Deepseek Reasoning Model](https://api-docs.deepseek.com/guides/reasoning_model#api-example)  的规范。未来可能会有所变动。

**Request 示例**

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


**Response 示例**

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

或者使用 OpenAI SDK (请先运行`pip3 install -U openai` 升级到最新版本)

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

### 自适应思考与 effort 参数

在 **Claude Opus 4.6** 和 **Claude Sonnet 4.6** 上，手动扩展思考（`thinking.type: "enabled"` 配合显式的 `budget_tokens`）已被弃用。这两个模型改用*自适应*思考：由 Claude 自行决定何时思考以及思考多深，而你通过 `effort` 等级（`low`、`medium`、`high`、`max`）来引导它，而不是指定 token 预算。Anthropic 将自适应思考列为这两个模型的受支持模式，并说明 Claude Opus 4.5、Claude Opus 4.6 和 Claude Sonnet 4.6 支持 `effort`。详见 [Extended thinking](https://platform.claude.com/docs/en/build-with-claude/thinking) 和 [Effort](https://platform.claude.com/docs/en/build-with-claude/effort)。

在这两个模型上，只要你不传 thinking 块，思考模式就是**关闭**的：Anthropic 的逐模型配置表中，请求不含 thinking 字段时显示为 *Thinking off*，并说明思考会一直关闭，"直到你设置 `thinking: {type: "adaptive"}`"。单独传 `effort` **不会**开启思考模式 —— effort 作用于全部输出 token，"无论是否开启思考模式都有效"。因此：用 `thinking` 来开启思考，用 `effort` 来控制思考的多少。

这两种模式都不需要本代理新增任何参数：你放在 `extra_body` 中的内容会原样作为 Bedrock Converse 的 `additionalModelRequestFields` 转发，因此可以直接把模型自身的 thinking 和 effort 字段传下去。

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

> **未经验证**：`thinking.type: "adaptive"` 和 `output_config.effort` 是 Anthropic 为其 **Messages API** 所记录的字段名。我们未能找到 AWS 官方对这些字段在 Converse API 的 `additionalModelRequestFields` 中如何命名的说明，也没有发起真实的 Bedrock 调用来确认，因此请把上面的 payload 当作模板而非已验证的约定。如果 Bedrock 返回关于未知字段的 `ValidationException`，请查阅最新的 [Bedrock inference request parameters](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-anthropic-claude-messages.html) 以确认正确的字段名并调整 `extra_body` —— 无论如何都不需要修改本代理。

> **注意**：只要 `extra_body` 中包含 `thinking` 块（自适应或手动写法）或 `reasoning_config`，请求中的 `temperature` 和 `top_p` 就会被丢弃，因为 Anthropic 明确说明这两个参数与思考模式不兼容。但显式**关闭**思考的写法 `"thinking": {"type": "disabled"}` 例外：Anthropic 的分模型配置表把它在这一代模型上都标为 *Thinking off*，因此这类请求不算思考请求，会保留这两个参数。同样，如果 `extra_body` 中**只有** `output_config`，这两个参数也**不会**被丢弃：无论是 `output_config.effort` 等级，还是 `output_config.format` 的[结构化输出](https://platform.claude.com/docs/en/build-with-claude/structured-outputs) schema，都不会开启思考模式，因此这类请求会保留你设置的采样参数。需要注意的是，对 Claude 4.5/4.6 这一代模型，无论 `extra_body` 内容如何，`top_p` 仍会被[支持的 Anthropic 模型](#支持的-anthropic-模型)一节所述的模型系列规则丢弃。

请注意 `reasoning_effort` 和 `extra_body` 是两条不同的路径：`reasoning_effort` 会让代理替你计算 `budget_tokens`（手动扩展思考），而 `extra_body` 则把你的字段原样交给 Bedrock。请二者择一，不要同时使用。

## Interleaved thinking (beta)

**重要提示**：在使用 Chat Completion API 的推理模式（reasoning mode）前，请务必仔细阅读以下内容。

Claude 4 模型支持借助工具使用的扩展思维功能（Extended Thinking），其中包含交错思考（[interleaved thinking](https://docs.aws.amazon.com/bedrock/latest/userguide/claude-messages-extended-thinking.html#claude-messages-extended-thinking-tool-use-interleaved) ）。该功能使 Claude 4 可以在多次调用工具之间进行思考，并在收到工具结果后执行更复杂的推理，这对处理更复杂的 Agentic AI 交互非常有帮助。

在交错思考模式下，budget_tokens 可以超过 max_tokens 参数，因为它代表一次助手回合中所有思考块的总 Token 预算。

**支持的模型**: 支持扩展思考的 Claude 模型。在本代理可访问的模型中，即 Claude Opus 4.5 和 Claude Haiku 4.5，以及 Claude Opus 4.6 和 Claude Sonnet 4.6（在后两者上扩展思考仍然可用但已被弃用 —— 建议改用[自适应思考](#自适应思考与-effort-参数)）。

> **注意**：只要 `extra_body` 中包含 `thinking` 块或 `reasoning_config`，请求中的 `temperature` 和 `top_p` 就会被丢弃，因为 Anthropic 明确说明它们与思考模式不兼容。唯一的例外是 `"thinking": {"type": "disabled"}`：它会关闭思考模式，因此采样参数会被保留。

**Request 示例**

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
"content": "解释如何实现一个具有自平衡功能的二叉搜索树。"
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
"content": "解释如何实现一个具有自平衡功能的二叉搜索树。"
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
