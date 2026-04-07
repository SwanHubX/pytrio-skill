# GRPO / PPO 强化学习最佳实践

## 数据构造

GRPO/PPO 场景需要四个等长字段：

```python
sample = Datum(
    model_input=ModelInput.from_ints(input_ids),
    loss_fn_inputs={
        "target_tokens": generated_ids,   # 模型生成的 token
        "logprobs": old_logprobs,          # 旧策略（参考策略）的 logprobs
        "advantages": advantages,          # 每个 token 的优势值（来自 reward model）
    },
)

result = train.forward_backward(
    data=[sample],
    loss_fn="importance_sampling",  # 或 "ppo"
).result()
```

## 典型流程

1. 用 `SamplingClient.sample()` 生成 response 并获取 logprobs
2. 用 reward model 计算 reward，再算出 advantages
3. 将生成的 tokens + 旧 logprobs + advantages 构造 Datum
4. 调用 `forward_backward(loss_fn="importance_sampling")`
5. 调用 `optim_step()`

## 自定义 Loss

如果内置的 importance_sampling / ppo 不满足需求，可用 `forward_backward_custom()`：

```python
def my_loss(samples: list[Datum], logprobs: list[torch.Tensor]):
    # logprobs[i] 是 samples[i] 每个 token 的 logprob（需要 grad）
    loss = ...  # 自定义计算
    return loss, {"custom_metric": loss.item()}

result = train.forward_backward_custom(data=samples, loss_fn=my_loss).result()
```

注意：`forward_backward_custom` 需要本地安装 PyTorch。
