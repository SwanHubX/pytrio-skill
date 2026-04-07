# SamplingClient

> 推理采样客户端。由 `ServiceClient.create_sampling_client()` 或 `TrainingClient.create_sampling_client()` 创建。

## 方法

### sample(prompt, num_samples, sampling_params, ...) -> APIFuture[SampleResponse]

根据 prompt 生成文本补全。

```python
import pytrio as trio

future = sampler.sample(
    prompt: ModelInput,                        # token id 列表
    num_samples: int,                          # 生成样本数
    sampling_params: SamplingParams,           # 采样参数
    include_prompt_logprobs: bool = False,     # 包含 prompt logprobs
    topk_prompt_logprobs: int = 0,             # prompt top-k logprobs
)
result = future.result()  # SampleResponse
```

**prompt 参数：** 必须使用 `ModelInput` 类型包装：
```python
import pytrio as trio
sampler.sample(prompt=trio.ModelInput.from_ints(token_ids), ...)
```

**SamplingParams：**
```python
trio.SamplingParams(
    max_tokens: int | None = None,        # 最大生成长度
    seed: int | None = None,              # 随机种子
    stop: str | list[str] | list[int] | None = None,  # 停止条件
    temperature: float = 1,               # 温度
    top_k: int = -1,                      # top-k 采样
    top_p: float = 1,                     # top-p (nucleus) 采样
    ignore_eos: bool = False,             # 忽略 EOS（必须设置 max_tokens）
)
```

**SampleResponse 结构：**
```python
result.sequences        # list[SampleSequence]
result.prompt_logprobs  # list[float | None]
result.output_tokens    # int
result.elapsed_time     # float (ms)

# SampleSequence:
seq = result.sequences[0]
seq.tokens     # list[int] — 生成的 token ids
seq.text       # str — 仅包含生成部分的文本（不含 prompt）
seq.logprobs   # list[float] — 每个 token 的 logprob
seq.stop_reason  # str — 停止原因
```

### compute_logprobs(prompt) -> APIFuture[dict]

计算 prompt tokens 的逐 token logprobs。

```python
future = sampler.compute_logprobs(prompt=trio.ModelInput.from_ints(token_ids))
result = future.result()
# {"prompt_logprobs": [None, -2.73, -5.76, ...], "elapsed_time": 91}
```

注意：`prompt` 参数与 `sample()` 相同，必须用 `trio.ModelInput.from_ints()` 包装。

### get_tokenizer()

获取匹配的 tokenizer。

```python
tokenizer = sampler.get_tokenizer()
```
