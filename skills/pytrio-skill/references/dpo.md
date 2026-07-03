# DPO 与 Custom Loss 能力说明

当用户要做 DPO、偏好优化、chosen/rejected preference training，或需要解释 PyTRIO 的 `forward_backward_custom` / custom loss 怎么用时，先读本文件。

## 推荐示例

| 场景 | 示例 |
|---|---|
| 同步 DPO / HH-RLHF | `examples/dpo-hh-rlhf.py` |

## 核心目标

DPO 比较同一个 prompt 下 chosen 和 rejected 两个回复。reference model 给出参考 logprob，当前 student 给出可求导 logprob，本地 custom loss 计算 DPO loss，PyTRIO 负责远端 forward/backward 和 LoRA 更新。

```text
loss = -log sigmoid(beta * ((log pi_chosen - log ref_chosen)
                            - (log pi_rejected - log ref_rejected)))
```

## Custom Loss 的用法

`forward_backward_custom(data, loss_fn)` 会把当前 student 在 `data` 上的逐 token logprobs 传给本地 `loss_fn`。`loss_fn` 返回：

- 一个 torch loss tensor，用于反向传播；
- 一个 metrics dict，用于记录和调试。

最小形态：

```python
import torch

def custom_loss_fn(data, logprobs_list):
    # logprobs_list[i] 是第 i 条 Datum 的可求导逐 token logprob。
    weights = torch.as_tensor(data[0].loss_fn_inputs["weights"].data, device=logprobs_list[0].device)
    seq_logprob = torch.dot(logprobs_list[0].float(), weights.float())
    loss = -seq_logprob
    return loss, {"custom/loss": float(loss.detach().item())}

result = training_client.forward_backward_custom(batch, custom_loss_fn).result()
training_client.optim_step(trio.AdamParams(learning_rate=learning_rate)).result()
```

Custom loss 需要本地安装 `torch`，因为 loss 函数在本地 Python 进程里写，并用 torch tensor 保留梯度。

## DPO 数据构造

每条 preference 样本要拆成：

- 共同 prompt messages；
- chosen assistant response；
- rejected assistant response。

每个 response 都构造成一条 SFT 风格 `Datum`，只在 assistant response token 上设 `weights=1`：

```python
prompt_tokens = encode_messages(tokenizer, prompt_messages, add_generation_prompt=True)
full_tokens = encode_messages(
    tokenizer,
    [*prompt_messages, {"role": "assistant", "content": response}],
    add_generation_prompt=False,
)

completion_len = len(full_tokens) - len(prompt_tokens)
token_weights = [0.0] * len(prompt_tokens) + [1.0] * completion_len

datum = trio.Datum(
    model_input=trio.ModelInput.from_ints(full_tokens[:-1]),
    loss_fn_inputs={
        "target_tokens": np.asarray(full_tokens[1:], dtype=np.int64),
        "weights": np.asarray(token_weights[1:], dtype=np.float32),
    },
)
```

batch 内保持 `[chosen_0, rejected_0, chosen_1, rejected_1, ...]` 的顺序，custom loss 按偶数/奇数下标配对。

## Reference Logprob

reference model 不参与优化，只负责计算每条 chosen/rejected 序列的参考 logprob：

```python
full_ids = datum.model_input.to_ints() + [int(datum.loss_fn_inputs["target_tokens"].data[-1])]
values = reference_client.compute_logprobs(trio.ModelInput.from_ints(full_ids)).result()
reference_logprobs = values[1:]
```

`reference_logprobs` 必须和 `datum.model_input` 右移后长度一致，且不能包含 `None`。

## DPO Loss

custom loss 内一般这样做：

1. 用 response token 的 `weights` 对 current student logprobs 加权求和，得到 `log pi_chosen` / `log pi_rejected`。
2. 用同样 weights 对 reference logprobs 加权求和，得到 `log ref_chosen` / `log ref_rejected`。
3. 计算 DPO loss 并返回 metrics。

## 常见错误

- 不要用 `loss_fn="cross_entropy"` 或 `loss_fn="importance_sampling"` 直接替代 DPO；DPO 需要 pairwise custom loss。
- chosen/rejected 必须共享同一个 prompt，否则偏好比较无效。
- batch 长度必须为偶数，并保持 chosen/rejected 交错顺序。
- reference logprobs 要提前算好并闭包传入 custom loss；不要在 custom loss 内做网络请求。
- response token 的 `weights` 必须和 student/reference logprobs 对齐。
