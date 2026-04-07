# 数据类型参考

## Datum (核心训练数据单元)

训练的基本数据单元。每个 `Datum` 代表一个训练样本。

```python
import pytrio as trio

sample = trio.Datum(
    model_input=trio.ModelInput.from_ints([tok0, tok1, tok2]),  # 必须用 ModelInput 包装
    loss_fn_inputs={                       # 损失函数参数 (值自动转 TensorData)
        "target_tokens": [tok1, tok2, tok3],
        "weights": [1.0, 1.0, 1.0],       # cross_entropy 可选
    }
)
```

**不同 loss_fn 的 loss_fn_inputs 要求：**

```python
# cross_entropy
loss_fn_inputs = {
    "target_tokens": [int, ...],      # 必填，长度 = len(model_input)
    "weights": [float, ...],          # 可选，默认全 1.0；具体用法见 best-practices/
}

# importance_sampling / ppo
loss_fn_inputs = {
    "target_tokens": [int, ...],      # 必填
    "logprobs": [float, ...],         # 必填，参考策略的 logprobs
    "advantages": [float, ...],       # 必填，优势值
}
```

**关于 model_input 和 target_tokens 的对齐（auto_shift=False 时）：**
```
model_input:    [A, B, C]    ← 模型看到的输入
target_tokens:  [B, C, D]    ← 模型应该预测的目标
```
即 `target_tokens[i]` 是 `model_input[i]` 位置应预测的下一个 token。

## ModelInput

token 序列的封装。

```python
import pytrio as trio

# 注意：Datum.model_input 不支持直接传 list[int]，必须用 ModelInput 包装
mi = trio.ModelInput.from_ints([1, 2, 3])
mi.to_ints()   # -> [1, 2, 3]
mi.tolist()    # -> [1, 2, 3]
len(mi)        # -> 3
```

## TensorData

张量数据的序列化格式，用于网络传输。

```python
from pytrio.types import TensorData

# 自动转换：在 Datum.loss_fn_inputs 中传 list/numpy/torch 时自动转换
# 手动构造：
td = TensorData(data=[1.0, 2.0, 3.0], dtype="float32", shape=[3])
td = TensorData.from_numpy(np_array)
td = TensorData.from_torch(torch_tensor)

# 转换输出：
td.to_numpy()   # -> numpy.ndarray
td.tolist()      # -> list
```

支持的 dtype: `"float32"`, `"int64"`

## AdamParams

Adam 优化器配置。

```python
import pytrio as trio

params = trio.AdamParams(
    learning_rate=0.0001,   # 学习率
    beta1=0.9,              # AdamW beta1
    beta2=0.95,             # AdamW beta2（注意：PyTorch 默认 0.999）
    eps=1e-12,              # epsilon（注意：PyTorch 默认 1e-8）
    weight_decay=0.0,       # 权重衰减
)
```

## SamplingParams

推理采样参数。

```python
import pytrio as trio

params = trio.SamplingParams(
    max_tokens=None,      # 最大生成长度
    seed=None,            # 随机种子
    stop=None,            # 停止条件 (str / list[str] / list[int])
    temperature=1.0,      # 温度
    top_k=-1,             # top-k (-1 表示不启用)
    top_p=1.0,            # top-p
    ignore_eos=False,     # 忽略 EOS（必须同时设 max_tokens）
)
```

## APIFuture[T]

异步结果封装。所有 API 调用返回此类型。

```python
future = train.forward_backward(data)

# 阻塞等待结果
result = future.result(timeout=None)

# 检查状态
future.done()       # bool
future.cancel()     # bool

# 链式变换
mapped = future.map(lambda r: r.metrics)

# 在 async 中使用
result = await future
```

## ForwardBackwardOutput

前向/反向传播结果。

```python
result.loss_fn_outputs  # list[dict[str, TensorData]] — 每个样本的 loss 输出（含 logprobs 等）
result.metrics          # dict[str, float] — 训练指标，key 为 "loss:sum"（注意是冒号不是下划线）
result.elapsed_time     # float — 耗时 (ms)

# 读取 loss 示例：
loss = result.metrics.get("loss:sum", 0.0)
```

## OptimStepResponse

优化步骤结果。

```python
result.metrics       # dict[str, float]
result.elapsed_time  # float (ms)
```

## SaveWeightsResponse / SaveWeightsForSamplerResponse

保存权重结果。

```python
result.path   # str — 远端存储路径
result.model  # str — 模型名
result.size   # int — 文件大小 (bytes)
```

## SampleResponse

推理结果。

```python
result.sequences            # list[SampleSequence]
result.prompt_logprobs      # list[float | None]
result.topk_prompt_logprobs # list[list[tuple[int, float]] | None]
result.output_tokens        # int
result.elapsed_time         # float (ms)
```

## SampleSequence

单个生成序列。

```python
seq.tokens       # list[int]
seq.text         # str
seq.logprobs     # list[float]
seq.stop_reason  # str
```
