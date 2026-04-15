# TrainingClient

> LoRA 微调训练客户端。由 `ServiceClient.create_lora_training_client()` 创建。

## 导入

不直接实例化。通过 `ServiceClient` 工厂方法获取。

## 核心方法

### forward(data, loss_fn, loss_fn_config, auto_shift) -> APIFuture[ForwardBackwardOutput]

仅前向传播（不计算梯度）。

```python
future = train.forward(
    data: list[Datum],                           # 样本列表
    loss_fn: str = "cross_entropy",              # 损失函数
    loss_fn_config: dict[str, Any] | None = None,# 损失函数配置
    auto_shift: bool = False,                    # 自动对齐偏移
)
result = future.result()  # ForwardBackwardOutput
```

### forward_backward(data, loss_fn, loss_fn_config, auto_shift) -> APIFuture[ForwardBackwardOutput]

前向 + 反向传播（计算梯度，准备优化）。**训练主循环的核心方法。**

```python
future = train.forward_backward(
    data: list[Datum],
    loss_fn: str = "cross_entropy",
    loss_fn_config: dict[str, Any] | None = None,
    auto_shift: bool = False,
)
result = future.result()  # ForwardBackwardOutput
```

**loss_fn 可选值：**

| 值 | 必需的 loss_fn_inputs 字段 | 用途 |
|---|---|---|
| `"cross_entropy"` | `target_tokens`, 可选 `weights` | SFT 监督微调 |
| `"importance_sampling"` | `target_tokens`, `logprobs`, `advantages` | 无 clip 的 vanilla IS |
| `"ppo"` | `target_tokens`, `logprobs`, `advantages` | PPO / GRPO（带 clip） |

**loss_fn_config（仅 `"ppo"` 支持）：** 自定义 PPO 裁剪阈值，默认 ε=0.2（ratio 裁到 `[0.8, 1.2]`）
```python
train.forward_backward(
    data=data, loss_fn="ppo",
    loss_fn_config={"clip_low_threshold": 0.9, "clip_high_threshold": 1.1},
)
```

**auto_shift 参数：**
- `False`（默认）：用户自行对齐 input/target（input=[tok0, tok1, tok2]，target=[tok1, tok2, tok3]）
- `True`：远端自动将 labels 偏移一位，此时 model_input 和 target_tokens 传相同的完整序列即可

```python
# auto_shift=False（默认）：用户手动偏移
tokens = tokenizer.encode(text, add_special_tokens=False)
sample = trio.Datum(
    model_input=trio.ModelInput.from_ints(tokens[:-1]),     # [A, B, C, D]
    loss_fn_inputs={"target_tokens": tokens[1:]},      # [B, C, D, E]
)
train.forward_backward(data=[sample], auto_shift=False)

# auto_shift=True：传相同的完整序列，远端自动偏移
sample = trio.Datum(
    model_input=trio.ModelInput.from_ints(tokens),           # [A, B, C, D, E]
    loss_fn_inputs={"target_tokens": tokens},           # [A, B, C, D, E]
)
train.forward_backward(data=[sample], auto_shift=True)
```

**Tokenizer 使用说明（官方标准模式）：**
- **序列起始段**（prompt）：`add_special_tokens=True` — 保留 BOS 作为生成起始
- **拼接延续段**（completion）：`add_special_tokens=False` — 避免重复 BOS
- **推理时**：`add_special_tokens=True`（默认），模型生成时需要 BOS 作为起始
- **使用 chat template**：`tokenizer.apply_chat_template(..., tokenize=False)` 已自行管理 special tokens，后续 `encode` 应该用 `add_special_tokens=False` 避免重复

### forward_backward_custom(data, loss_fn) -> APIFuture[ForwardBackwardOutput]

自定义损失函数的前向+反向。**需要本地安装 PyTorch。**

```python
import torch

def my_loss_fn(
    samples: list[Datum], 
    logprobs: list[torch.Tensor]
) -> tuple[torch.Tensor, dict[str, float]]:
    # logprobs[i] 对应 samples[i] 的每个 token 的 logprob
    loss = -sum(lp.sum() for lp in logprobs)
    return loss, {"custom_loss": loss.item()}

future = train.forward_backward_custom(data=samples, loss_fn=my_loss_fn)
result = future.result()
```

工作原理：
1. 先调用远端 forward 获取 logprobs
2. 本地用自定义 loss_fn 计算 loss 并 backward 得到梯度
3. 将梯度作为 weights 传回远端再做一次 forward_backward（线性化近似）

### optim_step(adam_params) -> APIFuture[OptimStepResponse]

执行一步优化器更新。**必须在 `forward_backward` 之后调用。**

```python
import pytrio as trio

# 常见用法：只传 learning_rate，其余走 SDK 默认
future = train.optim_step(trio.AdamParams(learning_rate=1e-4))
result = future.result()  # OptimStepResponse

# SDK 默认值（和 PyTorch 不同）:
#   learning_rate=1e-4, beta1=0.9, beta2=0.95, eps=1e-12, weight_decay=0.0
# 若需与 PyTorch 行为一致（一般不需要），显式传 beta2=0.999, eps=1e-8
```

### save_state(name) -> APIFuture[SaveWeightsResponse]

保存完整 checkpoint（权重 + 优化器状态），用于断点续训。

```python
future = train.save_state("step_100")
result = future.result()
# result.path — 远端路径，用于 create_training_client_from_state_with_optimizer()
```

### save_weights_for_sampler(name) -> APIFuture[SaveWeightsForSamplerResponse]

仅保存模型权重（不含优化器状态），用于推理部署。

```python
future = train.save_weights_for_sampler("final_model")
result = future.result()
# result.path — 远端路径，用于 create_sampling_client()
```

### save_weights_and_get_sampling_client(name) -> SamplingClient

保存权重并直接创建推理客户端（便捷方法）。

```python
sampler = train.save_weights_and_get_sampling_client("my_model")
```

### create_sampling_client(model_path) -> SamplingClient

从已保存的权重路径创建推理客户端。

```python
sampler = train.create_sampling_client(model_path="path/from/save_weights")
```

### get_tokenizer()

获取与基础模型匹配的 tokenizer（来自 transformers/modelscope 的 AutoTokenizer）。

```python
tokenizer = train.get_tokenizer()
```

## 错误处理

- `APIFuture.result(timeout=None)` 支持 `timeout` 参数（秒），超时抛出 `concurrent.futures.TimeoutError`
- 服务端错误会抛出 `pydantic.ValidationError`（响应格式不匹配时）或 `ValueError`（SDK 层校验失败时）
- WebSocket 断连不会自动重连，需要重新创建 `ServiceClient`
- 建议在训练循环中用 try/except 包裹 `forward_backward` + `optim_step`，打印错误后继续或退出

## 异步版本

所有方法都有 `_async` 后缀版本。
