# Custom Loss 能力说明

当用户要实现 PyTRIO 自定义损失函数、`forward_backward_custom()`、`forward_backward_custom_async()`、本地 PyTorch loss、序列级目标、pairwise loss，或需要把额外算法数据通过闭包传入 loss 时，先读本文件。

## 本页导航

- [官方文档](#官方文档)
- [适用场景](#适用场景)
- [执行模型](#执行模型)
- [Loss 函数契约](#loss-函数契约)
- [Datum 与本地元数据](#datum-与本地元数据)
- [同步与异步调用](#同步与异步调用)
- [梯度原理](#梯度原理)
- [项目内案例](#项目内案例)
- [性能与指标](#性能与指标)
- [常见错误](#常见错误)

## 官方文档

- 可视化页面：https://docs.pytrio.com/docs/guide/custom_loss
- Markdown：https://docs.pytrio.com/docs/content/guide/custom_loss/content.md
- 内置损失函数：https://docs.pytrio.com/docs/content/guide/loss_fn/content.md
- 异步调用：https://docs.pytrio.com/docs/content/guide/async/content.md

先检查内置 `cross_entropy`、`importance_sampling`、`ppo` 是否已经覆盖任务。只有算法需要新的可求导目标、归一化或跨样本组合时，再使用 custom loss。

## 适用场景

Custom loss 适合：

- DPO 等 chosen/rejected pairwise objective；
- GSPO 等 sequence-level ratio 与 clipping；
- DAPO 等自定义 PPO clip 和 token/sequence reduction；
- 需要 sampling/reference logprob、advantage、completion mask、分组关系或原始 batch 分母的研究算法。

reward、采样、reference model 打分和远程请求都应在 loss 回调前完成。loss 回调只组合当前模型 logprob 与已经准备好的本地数据。

## 执行模型

自定义函数始终在本地 Python 进程执行：

```text
服务端 forward
→ 返回当前模型逐 token logprob
→ 本地 PyTorch loss 与 autograd
→ 把 dL / dlogprob 交回服务端
→ 服务端 surrogate forward-backward
→ optimizer step
```

PyTRIO 不会 pickle 或上传用户定义的 Python 函数。运行 custom loss 的机器需要安装 `torch`，并承担本地 loss 计算与 autograd 开销。

## Loss 函数契约

固定签名为：

```python
def custom_loss_fn(
    data: list[trio.Datum],
    logprobs: list[torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    ...
```

- `data` 与调用 `forward_backward_custom(data=...)` 时的顺序一致。
- `logprobs[i]` 是当前模型对 `data[i]` 的逐 token 可求导 logprob。
- `logprobs[i].numel()` 必须等于 `data[i].model_input.length`。
- 返回值第一项必须是保留计算图的标量 `torch.Tensor`。
- metrics 只放可序列化的 Python 数值；调用 `.detach().item()`，不要把 tensor 直接放进字典。

不要对参与 loss 的 `logprobs` 调 `.detach()`、转换 NumPy 或转成 Python float，否则本地计算图会断开。

## Datum 与本地元数据

Custom loss 的 `Datum` 只提供服务端 forward 所需数据：

```python
datum = trio.Datum(
    model_input=trio.ModelInput.from_ints(input_tokens),
    loss_fn_inputs={
        "target_tokens": np.asarray(target_tokens, dtype=np.int64),
    },
)
```

`target_tokens` 与 `model_input` 必须等长。`loss_fn_inputs` 属于服务端损失 schema，不能放入自定义键、标量、dataclass 或其他本地对象。

额外算法数据使用独立结构保存，并通过闭包绑定：

```python
@dataclass(frozen=True)
class LossMeta:
    sampling_logprobs: list[float]
    advantage: float
    completion_tokens: int


def make_loss_fn(metas: list[LossMeta]):
    batch_metas = tuple(metas)

    def loss_fn(data, logprobs):
        if not (len(data) == len(logprobs) == len(batch_metas)):
            raise ValueError("data, logprobs and metas must align")
        loss = compute_loss(logprobs, batch_metas)
        return loss, {"custom/loss": float(loss.detach().item())}

    return loss_fn
```

每个 batch 创建自己的闭包快照。`data[i]`、`logprobs[i]` 和 `metas[i]` 必须描述同一条样本；过滤、排序或拆 micro-batch 时三者执行相同操作。

## 同步与异步调用

同步调用：

```python
result = training_client.forward_backward_custom(
    data=batch_data,
    loss_fn=make_loss_fn(batch_metas),
).result()
training_client.optim_step(adam).result()
```

Custom loss 包含额外的服务端与本地往返，官方建议优先使用异步方法：

```python
api_future = await training_client.forward_backward_custom_async(
    data=batch_data,
    loss_fn=make_loss_fn(batch_metas),
)
result = await api_future

optim_future = await training_client.optim_step_async(adam)
await optim_future
```

第一次 `await` 提交请求并取得 `APIFuture`，第二次 `await` 等待远程任务完成。流水化多个 batch 时仍要保证 optimizer update 不会越过所属 batch 的 backward 边界，并以当前安装 SDK 的签名为准。

## 梯度原理

本地 autograd 计算：

```text
g_i = ∂L / ∂log p_i
```

SDK 把 `-g_i` 转换为服务端代理权重，再构造线性 surrogate objective。服务端计算 `∂log p_i / ∂θ`，两段链式法则合成与原 custom loss 相同的参数梯度。

代理权重可以为任意实数，也可能为负数。它只承载 logprob 梯度，不能用于传递自定义元数据。

## 项目内案例

- DPO：先读 `references/dpo.md`，再参考 `examples/dpo-hh-rlhf.py`。
- DAPO：读 `references/dapo.md`，查看自定义 PPO clip 与 token-level reduction。
- GSPO：读 `references/gspo.md`，完整实现见 https://github.com/KMnO4-zx/agentic-rl-lab/tree/main/07-gspo 。

官网的 `logprob_squared_loss` 只用于解释 API 和梯度链路，没有实际训练价值，不要把它当成推荐目标函数。

## 性能与指标

- 记录 custom loss 返回的算法指标，例如 `dpo/loss`、`ppo/clip_fraction`、`gspo/seq_ratio_mean`。
- 同时记录样本数、有效 token、原始归一化分母、远端 forward/backward 时间和本地 loss 时间。
- 从调用返回值的 `metrics` 读取服务端与本地指标；对象访问形式以当前 SDK 为准。
- custom loss 通常比内置损失慢。先用小 batch 做 shape、dtype、device、有限值和梯度 smoke test。

## 常见错误

- 不要把 sampling/reference logprob、advantage、mask 或分组对象塞进 `loss_fn_inputs`；使用闭包元数据。
- 不要在 loss 回调中采样、访问网络、调用 reference client 或修改 batch 顺序。
- 不要让 `target_tokens`、`model_input` 和返回 logprob 长度不一致。
- 不要把 prompt 或 observation 占位区误计入 completion-only objective。
- 不要在返回 loss 前 `.item()`；返回值必须保留 autograd 计算图。
- 不要复用已经绑定旧 batch 元数据的闭包。
- 不要只看 loss 数值；使用独立评测验证算法目标是否转化为能力收益。
