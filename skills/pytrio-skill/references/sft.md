# SFT 能力说明

当用户要做监督微调、角色微调、多轮对话蒸馏、assistant-only loss mask、同步/异步 SFT 或 SwanLab 训练记录时，先读本文件。

## 推荐示例

| 场景 | 示例 |
|---|---|
| 最小 SFT 闭环 | `examples/quickstart_sft.py` |
| 角色 SFT / Chat-甄嬛 | `examples/chat-huanhuan.py` |
| 异步 SFT / 异步 SwanLab 记录 | `examples/chat-huanhuan-async.py` |
| 多轮对话蒸馏 | `examples/sft-distill-conversation.py` |
| 异步多轮对话蒸馏 | `examples/sft-distill-conversation-async.py` |

## 核心目标

SFT 的目标是让模型在给定 prompt/history 时预测目标 assistant token。prompt、system、user 和历史上下文通常只作为输入，不参与损失；当前需要学习的 assistant 回复参与损失。

## Datum 构造

SFT 通常手动右移，不使用 HuggingFace 的 `labels=-100`：

```python
input_tokens = tokens[:-1]
target_tokens = tokens[1:]
loss_weights = weights[1:]

datum = trio.Datum(
    model_input=trio.ModelInput.from_ints(tokens=input_tokens),
    loss_fn_inputs={
        "target_tokens": np.asarray(target_tokens, dtype=np.int32),
        "weights": np.asarray(loss_weights, dtype=np.float32),
    },
)
```

`weights` 的常见规则：

- prompt、system、user、历史上下文：`0`
- 当前 assistant completion：`1`
- EOS 或 `<|im_end|>`：如果希望模型学会停止，也设为 `1`

## Chat Template

使用 chat model 时，优先用 tokenizer 的 chat template：

- 构造 prompt：`apply_chat_template(..., tokenize=False, add_generation_prompt=True)`
- 构造完整多轮对话：按模板拼出每条 message，assistant 内容才设 loss weight
- 如果 chat template 已经处理特殊 token，`tokenizer.encode(..., add_special_tokens=False)`
- Qwen reasoning 蒸馏时，若只训练带思考过程的数据，应显式过滤包含完整 `<think>` 和 `</think>` 的 assistant 回复

## 训练循环

同步版：

```python
fwdbwd = training_client.forward_backward(batch, loss_fn="cross_entropy")
optim = training_client.optim_step(trio.AdamParams(learning_rate=learning_rate))
result = fwdbwd.result()
optim.result()
```

异步版：

```python
fwdbwd_future = await training_client.forward_backward_async(batch, "cross_entropy")
optim_future = await training_client.optim_step_async(trio.AdamParams(learning_rate=learning_rate))
result = await fwdbwd_future
await optim_future
```

异步日志不要只保存 coroutine；应使用 `asyncio.create_task(...)` 让 loss 计算和 `swanlab.log(...)` 尽早调度。

## SwanLab 指标

建议记录：

- `loss`
- `epoch`
- `batch`
- `global_step`
- `base_model`
- `dataset_path` 或 `dataset_id`
- `lora_rank`
- `learning_rate`
- `max_length`
- `weights_name`

## 常见错误

- 不要把 prompt token 也设为 `weights=1`，除非用户明确要训练完整语言建模。
- 不要把 HuggingFace 的 `-100` labels 放进 `loss_fn_inputs`。
- 同步 `sample(...)` 返回 future，需要 `.result()`；异步 `sample_async(...)` 在当前示例中 `await` 后直接得到 response。
- 读取 `.path` 前先解析 `save_weights_for_sampler(...)` 或 `save_weights_for_sampler_async(...)` 的 future。
