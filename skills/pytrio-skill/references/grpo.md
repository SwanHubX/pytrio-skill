# GRPO 能力说明

当用户要做 GRPO、RLVR、GSM8K 数学题强化学习、reward-based post-training，或要求根据采样结果构造 group-relative advantage 时，先读本文件。

## 推荐示例

| 场景 | 示例 |
|---|---|
| 同步 GRPO / GSM8K | `examples/grpo-gsm8k.py` |
| 异步 GRPO / GSM8K | `examples/grpo-gsm8k-async.py` |

## 核心目标

GRPO 对同一个 prompt 采样多个 completion，使用 reward 函数打分，再把每条 completion 的 reward 减去同组均值，得到 group-relative advantage。训练时用 PyTRIO 的 `importance_sampling` loss 更新当前 student。

## 数据流

1. 从数据集中取 prompt，例如 GSM8K 的 `question`。
2. 用当前 student 权重创建 sampling client。
3. 对每个 prompt 采样 `group_size` 个 completion。
4. 保存每条 completion 的 tokens、old logprobs、文本和 reward。
5. 计算 `advantage = reward - mean(group_rewards)`。
6. 构造 `trio.Datum`，使用 `loss_fn="importance_sampling"`。
7. 记录 reward、退化 group 比例和 trainer metrics。

## Datum 构造

`importance_sampling` 需要字段长度严格对齐：

```python
observation_len = len(prompt_tokens) - 1
input_tokens = prompt_tokens + completion_tokens[:-1]
target_tokens = [0] * observation_len + completion_tokens
padded_logprobs = [0.0] * observation_len + old_logprobs
padded_advantages = [0.0] * observation_len + [advantage] * len(completion_tokens)

datum = trio.Datum(
    model_input=trio.ModelInput.from_ints(input_tokens),
    loss_fn_inputs={
        "target_tokens": np.asarray(target_tokens, dtype=np.int64),
        "logprobs": np.asarray(padded_logprobs, dtype=np.float32),
        "advantages": np.asarray(padded_advantages, dtype=np.float32),
    },
)
```

prompt/observation 区间不训练，所以用 `0` / `0.0` 占位。completion 区间使用采样时返回的旧策略 logprob 和 group-relative advantage。

## Reward

GSM8K 风格的最小 reward：

- 要求模型把最终答案写在 `\boxed{}`
- 从回答中取最后一个 `\boxed{...}`
- 和标准答案归一化后完全一致则 reward 为 `1.0`，否则为 `0.0`

用户换任务时，只替换 prompt 构造和 reward 函数，不要改 `importance_sampling` 对齐规则。

## 训练循环

同步版：

```python
sampling_client = training_client.save_weights_and_get_sampling_client()
result = sampling_client.sample(..., return_text=True).result()
fwdbwd = training_client.forward_backward(datums, loss_fn="importance_sampling")
optim = training_client.optim_step(adam_params)
fwdbwd_result = fwdbwd.result()
optim.result()
```

异步版使用 `save_weights_and_get_sampling_client_async()`、`sample_async()`、`forward_backward_async()` 和 `optim_step_async()`。

## 常见错误

- 不要用 `cross_entropy` 写 GRPO；GRPO 需要 old logprobs 和 advantages。
- 不要只采样 1 个 completion 后计算 group advantage；至少需要同 prompt 下多个 completion。
- 同组 reward 完全一样时 advantage 全为 0，通常跳过该组，避免无训练信号。
- 旧策略 logprobs 必须来自 rollout 当时的 student sampler，不能用更新后的模型重算替代。
- prompt token 区间只做上下文，不应该有非零 advantage。
