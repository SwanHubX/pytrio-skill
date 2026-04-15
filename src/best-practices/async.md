# 异步训练最佳实践

PyTRIO 所有核心方法都有 `_async` 后缀的异步版本。在 RL rollout、多步骤 pipeline、高并发场景下，异步版本可显著提升吞吐——本地与云端可并行工作。

## 两阶段模型

异步调用分两步：

1. **提交任务**：`await training_client.forward_backward_async(...)` 返回一个 `APIFutureResult` 对象（"令牌"），立即返回不阻塞
2. **获取结果**：`await fwdbwd_future` 显式等待云端计算完成

在两步之间，本地可继续执行其他操作（准备下一批数据、记录 metrics、发起并发任务）。

## 同步 vs 异步写法对比

```python
# 同步
for i in range(15):
    fb_future = training_client.forward_backward(data, "cross_entropy")
    fb_result = fb_future.result()

# 异步
async def main():
    for i in range(15):
        fb_future = await training_client.forward_backward_async(data, "cross_entropy")
        fb_result = await fb_future

asyncio.run(main())
```

## 官方 SFT 异步示例（完整）

```python
import asyncio
import numpy as np
import pytrio as trio


async def main():
    service_client = trio.ServiceClient()
    training_client = await service_client.create_lora_training_client_async(
        base_model="Qwen/Qwen3-4B-Instruct-2507",
        rank=32,
    )
    tokenizer = training_client.get_tokenizer()

    def process_example(example, tokenizer):
        prompt = f"Question: {example['input']}\nAnswer:"
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
        completion_tokens = tokenizer.encode(f" {example['output']}\n\n", add_special_tokens=False)
        tokens = prompt_tokens + completion_tokens
        weights = [0] * len(prompt_tokens) + [1] * len(completion_tokens)
        return trio.Datum(
            model_input=trio.ModelInput.from_ints(tokens=tokens[:-1]),
            loss_fn_inputs=dict(weights=weights[1:], target_tokens=tokens[1:]),
        )

    examples = [...]  # 你的数据集
    processed = [process_example(ex, tokenizer) for ex in examples]

    # 关键模式：提交 + 后台打印 loss，不阻塞下一次迭代
    print_tasks = []
    for i in range(15):
        fb_future = await training_client.forward_backward_async(processed, "cross_entropy")
        opt_future = await training_client.optim_step_async(trio.AdamParams(learning_rate=1e-4))

        async def log_loss(fb_future, opt_future, iteration):
            fb_result = await fb_future
            await opt_future
            logprobs = np.concatenate([o["logprobs"].tolist() for o in fb_result.loss_fn_outputs])
            weights = np.concatenate([ex.loss_fn_inputs["weights"].tolist() for ex in processed])
            print(f"Iter {iteration} | loss: {-np.dot(logprobs, weights) / weights.sum():.4f}")

        print_tasks.append(log_loss(fb_future, opt_future, i))

    await asyncio.gather(*print_tasks)


asyncio.run(main())
```

## RL 并发 rollout

RL 场景最能体现异步的价值——对一批 prompt 并发 sample，大大压缩 rollout 时间：

```python
async def rollout_batch(sampler, prompts):
    # 并发提交所有 sample 任务
    futures = [
        await sampler.sample_async(
            prompt=trio.ModelInput.from_ints(p),
            num_samples=4,
            sampling_params=trio.SamplingParams(max_tokens=128, temperature=0.7),
        )
        for p in prompts
    ]
    # 并发等待
    results = await asyncio.gather(*[f for f in futures])
    return results
```

## 已支持的异步方法

### ServiceClient

| 同步 | 异步 |
|---|---|
| `create_lora_training_client` | `create_lora_training_client_async` |
| `create_sampling_client` | `create_sampling_client_async` |
| `create_training_client_from_state` | `create_training_client_from_state_async` |
| `create_training_client_from_state_with_optimizer` | `create_training_client_from_state_with_optimizer_async` |

### TrainingClient

| 同步 | 异步 |
|---|---|
| `forward` | `forward_async` |
| `forward_backward` | `forward_backward_async` |
| `forward_backward_custom` | `forward_backward_custom_async` |
| `optim_step` | `optim_step_async` |
| `save_state` | `save_state_async` |
| `save_weights_for_sampler` | `save_weights_for_sampler_async` |
| `create_sampling_client` | `create_sampling_client_async` |
| `save_weights_and_get_sampling_client` | `save_weights_and_get_sampling_client_async` |

### SamplingClient

| 同步 | 异步 |
|---|---|
| `sample` | `sample_async` |
| `compute_logprobs` | `compute_logprobs_async` |

## 注意事项

- 异步方法返回的 future 既可以 `await future`，也可以 `future.result()`（同步阻塞）
- `asyncio.gather(*tasks)` 并发等待多个 future，整体耗时 ≈ 最慢那一个
- WebSocket 断连不自动重连，需要重建 `ServiceClient` — 异步场景里尤其要 try/except 兜底
- 异步 loop 里**不要**在一个 iteration 里 await 上一轮的 fb_result，会失去并行的意义；典型做法是把"等结果 + 打印 metrics"包成独立 coroutine 塞进 task queue
