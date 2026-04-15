# OpenAI 兼容 API

> TRIO 提供 OpenAI 兼容的 HTTP 接口，训完的模型可通过 `openai` SDK 直接对外服务——无需自建推理服务、无需下载权重。

## 配置

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://pytrio.cn/api/v1",
    api_key="YOUR_TRIO_API_KEY",  # 同 ~/.pytrio/config.toml 里的 api_key
)
```

`model` 字段传：
- **基础模型名**（如 `Qwen/Qwen3-4B-Instruct-2507`）：等价于独立 base model 推理
- **LoRA 权重路径**（WebUI「权重」页面查看，或 `train.save_weights_for_sampler().path`）：加载 LoRA 推理

## Chat Completions

```python
response = client.chat.completions.create(
    model="YOUR_MODEL_PATH",  # 或基模名
    messages=[{"role": "user", "content": "what's your name?"}],
    max_tokens=50,
    temperature=0.7,
    top_p=0.9,
)
print(response.choices[0].message.content)
```

## Text Completions（续写）

```python
response = client.completions.create(
    model="YOUR_MODEL_PATH",
    prompt="what's your name?",
    max_tokens=50,
    temperature=0.7,
    top_p=0.9,
)
print(response.choices[0].text)
```

## 列出模型

```python
models = client.models.list()
for m in models.data:
    print(m.id)
```

## 使用场景

- **产品集成**：训完 LoRA 权重直接用 OpenAI SDK 接入现有应用
- **A/B 对比**：用同一 `OpenAI` client 切换 `model` 字段对比基模和 LoRA 效果
- **无需部署**：省掉权重下载 + vLLM / transformers 本地部署流程

## 与 `SamplingClient` 的区别

| 特性 | `SamplingClient.sample()` | OpenAI 兼容 API |
|---|---|---|
| 协议 | WebSocket | HTTP |
| 返回 logprobs | 是（`sequence.logprobs`） | 需显式开启（OpenAI 语义） |
| 适用阶段 | 训练中推理 / RL rollout | 生产部署 / 应用集成 |
| 批量 `num_samples` | 原生支持 | 通过 `n` 参数 |

RL rollout 阶段应继续用 `SamplingClient`（能拿到逐 token logprobs），OpenAI 兼容 API 主要用于训完对外服务。
