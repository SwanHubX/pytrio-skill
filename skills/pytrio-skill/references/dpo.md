# DPO 能力说明

当用户要做 DPO、偏好优化或 chosen/rejected preference training 时，先读本文件。通用 `forward_backward_custom` 契约、闭包元数据和梯度原理见 `references/custom-loss.md`。

## 推荐示例

| 场景 | 示例 |
|---|---|
| 同步 DPO / HH-RLHF | `examples/dpo-hh-rlhf.py` |

## Custom Loss 前置说明

DPO 使用本地 pairwise custom loss。先按 `references/custom-loss.md` 构造只含 `target_tokens` 的 forward Datum，再把 response mask、reference logprob 和配对关系作为本地数据绑定到 batch 专属闭包。

## 核心目标

DPO 比较同一个 prompt 下 chosen 和 rejected 两个回复。reference model 给出参考 logprob，当前 student 给出可求导 logprob，本地 custom loss 计算 DPO loss，PyTRIO 负责远端 forward/backward 和 LoRA 更新。

```text
loss = -log sigmoid(beta * ((log pi_chosen - log ref_chosen)
                            - (log pi_rejected - log ref_rejected)))
```

## DPO 数据构造

每条 preference 样本要拆成：

- 共同 prompt messages；
- chosen assistant response；
- rejected assistant response。

每个 response 都构造成一条右移后的 `Datum`，response mask 作为独立的本地元数据返回：

```python
prompt_tokens = encode_messages(tokenizer, prompt_messages, add_generation_prompt=True)
full_tokens = encode_messages(
    tokenizer,
    [*prompt_messages, {"role": "assistant", "content": response}],
    add_generation_prompt=False,
)

completion_len = len(full_tokens) - len(prompt_tokens)
token_mask = [0.0] * len(prompt_tokens) + [1.0] * completion_len
response_mask = np.asarray(token_mask[1:], dtype=np.float32)

datum = trio.Datum(
    model_input=trio.ModelInput.from_ints(full_tokens[:-1]),
    loss_fn_inputs={
        "target_tokens": np.asarray(full_tokens[1:], dtype=np.int64),
    },
)
return datum, response_mask
```

batch 内保持 `[chosen_0, rejected_0, chosen_1, rejected_1, ...]` 的顺序。`response_masks` 和 `reference_logprobs` 使用相同顺序，并和 datums 一起完成过滤、排序与拆批。

## Reference Logprob

reference model 不参与优化，只负责计算每条 chosen/rejected 序列的参考 logprob：

```python
full_ids = datum.model_input.to_ints() + [int(datum.loss_fn_inputs["target_tokens"].data[-1])]
values = reference_client.compute_logprobs(trio.ModelInput.from_ints(full_ids)).result()
reference_logprobs = values[1:]
```

`reference_logprobs` 必须和 `datum.model_input` 右移后长度一致，且不能包含 `None`。

## DPO Loss

为当前 batch 创建 loss 闭包：

```python
loss_fn = make_dpo_loss_fn(
    reference_logprobs=reference_logprobs,
    response_masks=response_masks,
    dpo_beta=dpo_beta,
)
result = training_client.forward_backward_custom(batch, loss_fn).result()
```

loss 回调内一般这样做：

1. 用本地 `response_masks` 对 current student logprobs 加权求和，得到 `log pi_chosen` / `log pi_rejected`。
2. 用同一组 mask 对 reference logprobs 加权求和，得到 `log ref_chosen` / `log ref_rejected`。
3. 计算 DPO loss 并返回 metrics。

## 常见错误

- 不要用 `loss_fn="cross_entropy"` 或 `loss_fn="importance_sampling"` 直接替代 DPO；DPO 需要 pairwise custom loss。
- chosen/rejected 必须共享同一个 prompt，否则偏好比较无效。
- batch 长度必须为偶数，并保持 chosen/rejected 交错顺序。
- reference logprobs 要提前算好并闭包传入 custom loss；不要在 custom loss 内做网络请求。
- response mask 必须和 student/reference logprobs 对齐，并通过闭包传入；不要把它存进 `loss_fn_inputs`。
