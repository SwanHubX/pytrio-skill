# SFT (Supervised Fine-Tuning) 最佳实践

## Prompt Masking

SFT 场景下，prompt（指令）部分不应参与 loss 计算，只对 response（回答）部分计算损失。通过 `weights` 字段实现：

```python
prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
response_ids = tokenizer.encode(response_text, add_special_tokens=False)
eos_ids = [tokenizer.eos_token_id] if tokenizer.eos_token_id else []
full_ids = prompt_ids + response_ids + eos_ids

model_input = full_ids[:-1]
target_tokens = full_ids[1:]

# prompt 部分 weight=0（不计入 loss），response 部分 weight=1
prompt_len = max(len(prompt_ids) - 1, 0)
weights = [0.0] * prompt_len + [1.0] * (len(target_tokens) - prompt_len)

sample = Datum(
    model_input=ModelInput.from_ints(model_input),
    loss_fn_inputs={
        "target_tokens": target_tokens,
        "weights": weights,
    },
)
```

如果不做 prompt masking（weights 全 1），模型会把"学会复述指令"也当作训练目标，浪费算力且可能降低生成质量。

## EOS Token 追加

在训练序列末尾追加 `tokenizer.eos_token_id`，让模型学会在回答结束时自动停止生成。不加 EOS 会导致推理时生成不停止（只能靠 `max_tokens` 截断）。

## Tokenizer 使用

- 训练时：`tokenizer.encode(text, add_special_tokens=False)` — 避免 special tokens 干扰 loss
- 推理时：`tokenizer.encode(text, add_special_tokens=True)` — 模型需要 BOS token 作为起始
- 使用 `apply_chat_template()` 时：encode 也应 `add_special_tokens=False`，因为模板已自行管理
