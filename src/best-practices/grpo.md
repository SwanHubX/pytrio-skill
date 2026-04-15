# GRPO / PPO 强化学习最佳实践

## loss_fn 选择

| loss_fn | 是否 clip | 适用 |
|---|---|---|
| `"ppo"` | ✅ 有 clip（默认 ε=0.2） | **GRPO / PPO 都用这个** |
| `"importance_sampling"` | ❌ 无 clip，vanilla IS | 离线场景 / 教学演示，**不推荐做 GRPO** |

**GRPO 本质就是 clipped surrogate objective**（论文原文: clipped ratio + group-relative advantage），所以 `loss_fn="ppo"`。用 `importance_sampling` 没有裁剪保护，policy 容易发散。

GRPO 与 PPO 在 PyTRIO 里的唯一差异在**客户端的 advantage 计算方式**：
- PPO：advantage 来自 value model / GAE
- GRPO：advantage 来自同一 prompt 的 K 个 rollout 的 reward group-relative normalization（`(r - mean) / std`）

loss 函数本身调用完全一样。

## 数据构造

RL 场景的 Datum 需要四个字段，`weights` 必须传用于 prompt masking：

```python
sample = trio.Datum(
    model_input=trio.ModelInput.from_ints(input_tokens),
    loss_fn_inputs={
        "target_tokens": target_tokens,   # 和 model_input 偏移一位对齐
        "weights": weights,               # 0=prompt 部分 mask，1=completion 参与策略梯度
        "logprobs": old_logprobs,          # 采样时记录的旧策略 logprobs
        "advantages": advantages,          # 每个 token 的优势值（正=强化，负=抑制）
    },
)
```

**不要省略 `weights`** —— 否则 prompt 部分也会参与策略梯度，模型会学到"倾向输出 prompt token"的假信号，训练跑偏。

## 典型流程（官方 GSM8K 模式）

```python
def process_rollout(prompt_tokens, completion_tokens, completion_logprobs, reward):
    tokens = prompt_tokens + completion_tokens
    prompt_weights = [0] * len(prompt_tokens)
    completion_weights = [1] * len(completion_tokens)
    weights = prompt_weights + completion_weights

    # 采样记录的 completion_logprobs 对齐到完整序列
    old_logprobs = [0.0] * len(prompt_tokens) + list(completion_logprobs)
    old_logprobs = old_logprobs[:len(tokens)] + [0.0] * max(0, len(tokens) - len(old_logprobs))

    # 同步偏移
    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]
    weights = weights[1:]
    old_logprobs = old_logprobs[1:]
    advantages = [reward] * (len(tokens) - 1)  # token-level advantage 可按需改

    return trio.Datum(
        model_input=trio.ModelInput.from_ints(tokens=input_tokens),
        loss_fn_inputs=dict(
            weights=weights,
            target_tokens=target_tokens,
            logprobs=old_logprobs,
            advantages=advantages,
        ),
    )
```

训练主循环（GRPO 推荐流程）：

1. `sampler = training_client.save_weights_and_get_sampling_client(name=f"iter{i}")`
2. 对每条 prompt 调用 `sampler.sample(num_samples=K, ...)` 获取 K 个 completion 和 `sequence.logprobs`
3. reward function 打分 → **按 prompt 分组做 group-relative normalization** `advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)`（这是 GRPO 和 vanilla PPO 的关键区别）
4. `process_rollout()` 构造 Datum（注意 advantage 按 token 广播）
5. `training_client.forward_backward(data, loss_fn="ppo")` — **用 ppo 而不是 importance_sampling，保留裁剪**
6. `training_client.optim_step(AdamParams(learning_rate=1e-5))`

RL 的学习率通常比 SFT 小一个数量级（`1e-5` vs `1e-4`）。

## PPO / GRPO 裁剪阈值

默认裁剪阈值 `epsilon=0.2`（即 ratio 被裁到 `[0.8, 1.2]`）。可通过 `loss_fn_config` 自定义：

```python
fb = training_client.forward_backward(
    data=data,
    loss_fn="ppo",
    loss_fn_config={"clip_low_threshold": 0.9, "clip_high_threshold": 1.1},
)
```

`importance_sampling` 不支持 clip config（它就是未裁剪的 IS，公式里没有 clip 算子）。

## 监控 IS Loss

```python
import numpy as np

def to_np(x):
    return np.array(x.tolist() if hasattr(x, "tolist") else x, dtype=float)

logprobs = np.concatenate([to_np(o["logprobs"]) for o in fb_result.loss_fn_outputs])
weights = np.concatenate([to_np(ex.loss_fn_inputs["weights"]) for ex in data])
old_logprobs = np.concatenate([to_np(ex.loss_fn_inputs["logprobs"]) for ex in data])
advantages = np.concatenate([to_np(ex.loss_fn_inputs["advantages"]) for ex in data])

mask = weights > 0
loss = -np.sum(np.exp(logprobs[mask] - old_logprobs[mask]) * advantages[mask]) / mask.sum()
```

## 自定义 Loss

内置 IS/PPO 不够用时，用 `forward_backward_custom`：

```python
def my_loss(samples: list[trio.Datum], logprobs: list[torch.Tensor]):
    # logprobs[i] 是 samples[i] 每个 target_token 的 logprob（带 grad）
    loss = ...  # 任意可微
    return loss, {"custom_metric": loss.item()}

fb = train.forward_backward_custom(data=samples, loss_fn=my_loss).result()
```

注意：需要本地安装 PyTorch；FLOPs 约为 `forward_backward` 的 1.5×，实际耗时最多可达 3×（多一次 forward + 客户端-服务器往返）。

## 异步 RL（推荐）

RL 场景下 sample + forward_backward 并发对吞吐影响很大，推荐异步版，见 `best-practices/async.md`。
