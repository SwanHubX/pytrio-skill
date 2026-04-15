# SFT (Supervised Fine-Tuning) 最佳实践

## 官方标准数据构造模式

官方所有 SFT 示例（quick-start / train / async / datasets / chat_huanhuan）统一采用以下模式：

```python
def process_example(example: dict, tokenizer) -> trio.Datum:
    prompt = f"Question: {example['input']}\nAnswer:"

    # prompt 起始段: add_special_tokens=True 保留 BOS
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    prompt_weights = [0] * len(prompt_tokens)

    # completion 拼接段: add_special_tokens=False 避免重复 BOS
    completion_tokens = tokenizer.encode(f" {example['output']}\n\n", add_special_tokens=False)
    completion_weights = [1] * len(completion_tokens)

    tokens = prompt_tokens + completion_tokens
    weights = prompt_weights + completion_weights

    # 手动偏移对齐: input_tokens[i] 预测 target_tokens[i]
    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]
    weights = weights[1:]  # weights 必须和 target_tokens 同步偏移

    return trio.Datum(
        model_input=trio.ModelInput.from_ints(tokens=input_tokens),
        loss_fn_inputs=dict(weights=weights, target_tokens=target_tokens),
    )
```

关键点：
1. **构造 `full_weights` 再偏移**：先算 `prompt_weights + completion_weights`，再 `[1:]`，保证 prompt/completion 边界对齐
2. **tokenizer 按段区分**：起始段 True，拼接段 False
3. **completion 末尾拼 `\n\n`** 或 EOS，让模型学会停止

## Prompt Masking

`weights` 是由 0/1 组成的 list：
- `0` = 该 token 不参与 loss（prompt 段）
- `1` = 该 token 计入 loss（completion 段）

如果不做 prompt masking（weights 全 1），模型会把"复述指令"也当训练目标，浪费算力且降低生成质量。

## 使用 chat template

```python
messages = [
    {"role": "user", "content": user_text},
    {"role": "assistant", "content": assistant_text},
]
# chat template 已管理 special tokens，tokenize=False 后再 encode 时用 False
full_text = tokenizer.apply_chat_template(messages, tokenize=False)
tokens = tokenizer.encode(full_text, add_special_tokens=False)
```

对于 chat 风格，标记 assistant 部分为 `weights=1`、其余为 `0` 即可。

## EOS Token 追加

completion 末尾追加 `tokenizer.eos_token_id` 或 `\n\n` 等自然停止符，让模型学会在回答结束时停止生成。不加的话推理时只能靠 `max_tokens` 截断。

## per-token loss 计算

训练循环中可实时监控 per-token loss（两种等价写法）：

```python
import numpy as np

# Datum 构造时，loss_fn_inputs 里的 list 会被自动转为 TensorData，
# TensorData 提供 .tolist() 方法转回 list
weights = np.concatenate([ex.loss_fn_inputs["weights"].tolist() for ex in processed_examples])

# 方式 A: 从 metrics 算
loss_sum = fb_result.metrics["loss:sum"]
print(f"Loss per token: {loss_sum / weights.sum():.4f}")

# 方式 B: 从 loss_fn_outputs 算（和方式 A 完全等价）
logprobs = np.concatenate([o["logprobs"].tolist() for o in fb_result.loss_fn_outputs])
print(f"Loss per token: {-np.dot(logprobs, weights) / weights.sum():.4f}")
```

## HuggingFace datasets 集成

```python
from datasets import load_dataset

train_dataset = load_dataset("openai/gsm8k", "main")["train"]
processed = [process_example(ex, tokenizer) for ex in train_dataset]
```

直接用上面的 `process_example` 模板，把数据集里的字段（如 `question` / `answer`）替换 `input` / `output` 即可。
