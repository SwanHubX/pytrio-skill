# OPD 能力说明

当用户要做 OPD、on-policy distillation、teacher-KL 蒸馏、DeepMath prompt-only 蒸馏、Medical OPD、多 Teacher 能力调度，或让 student 模仿 teacher 的 token 偏好时，先读本文件。

## 本页导航

- [推荐示例](#推荐示例)
- [完整案例](#完整案例)
- [核心目标](#核心目标)
- [数据流](#数据流)
- [Teacher Logprob 对齐](#teacher-logprob-对齐)
- [Datum 构造](#datum-构造)
- [训练循环](#训练循环)
- [Medical OPD 与多 Teacher 扩展](#medical-opd-与多-teacher-扩展)
- [SwanLab 指标](#swanlab-指标)
- [常见错误](#常见错误)

## 推荐示例

| 场景 | 示例 |
|---|---|
| 同步 OPD / DeepMath | `examples/opd-deepmath.py` |
| 异步 OPD / DeepMath | `examples/opd-deepmath-async.py` |

## 完整案例

- Medical SFT、Medical OPD、SAR-OPD、IDT-OPD：https://github.com/KMnO4-zx/agentic-rl-lab/tree/main/02-opd
- 本 Skill 核对版本：https://github.com/KMnO4-zx/agentic-rl-lab/tree/906d54e/02-opd

DeepMath 最小闭环优先使用本地 example。涉及多阶段 checkpoint、多 Teacher 调度和独立评测时，沿用完整案例的多文件结构。

## 核心目标

OPD 让 student 先按当前策略采样 completion，再让 teacher 对同一条 student completion 计算 logprob。训练信号来自 reverse KL：

```text
reverse_kl = student_logprob - teacher_logprob
advantage = -kl_penalty_coef * reverse_kl
```

这会惩罚 student 相比 teacher 过度偏好的 token，并让 student 往 teacher 的分布靠近。

## 数据流

1. 从 prompt-only 数据集中取问题，例如 DeepMath 的 `question`。
2. 创建 student LoRA training client。
3. 创建 teacher sampling client，teacher 可以是更强 base model 或指定 `model_path`。
4. 用当前 student sampler 采样 completion，并保存 completion tokens 和 old logprobs。
5. teacher 对 `prompt_tokens + completion_tokens` 计算逐 token logprob。
6. 只取 completion 区间的 teacher logprob，和 student old logprob 对齐。
7. 计算 reverse KL 和 advantage。
8. 构造 `importance_sampling` Datum 并训练 student。

## Teacher Logprob 对齐

teacher 必须看见 student 实际生成的完整轨迹：

```python
all_ids = prompt_ids + completion_ids
all_logprobs = teacher_client.compute_logprobs(trio.ModelInput.from_ints(all_ids)).result()
teacher_completion_logprobs = all_logprobs[len(prompt_ids):]
```

`teacher_completion_logprobs` 的长度必须等于 `completion_ids`，且不能包含 `None`。不要让 teacher 自己生成新 completion 后再算 KL。

## Datum 构造

```python
prompt_loss_len = len(prompt_ids) - 1
input_ids = prompt_ids + completion_ids[:-1]
target_ids = [0] * prompt_loss_len + completion_ids
padded_logprobs = [0.0] * prompt_loss_len + student_old_logprobs
padded_advantages = [0.0] * prompt_loss_len + advantages

datum = trio.Datum(
    model_input=trio.ModelInput.from_ints(input_ids),
    loss_fn_inputs={
        "target_tokens": np.asarray(target_ids, dtype=np.int64),
        "logprobs": np.asarray(padded_logprobs, dtype=np.float32),
        "advantages": np.asarray(padded_advantages, dtype=np.float32),
    },
)
```

这里的 `student_old_logprobs` 必须来自 student rollout；`advantages` 是逐 completion token 的 `-kl_penalty_coef * reverse_kl`。

## 训练循环

同步版：

```python
student_sampler = training_client.save_weights_and_get_sampling_client()
sample_result = student_sampler.sample(..., return_text=False).result()
teacher_lps = teacher_client.compute_logprobs(...).result()
fwdbwd = training_client.forward_backward(datums, loss_fn="importance_sampling")
optim = training_client.optim_step(adam)
fwdbwd_result = fwdbwd.result()
optim.result()
```

异步版使用 `save_weights_and_get_sampling_client_async()`、`sample_async()`、`compute_logprobs_async()`、`forward_backward_async()` 和 `optim_step_async()`。

## Medical OPD 与多 Teacher 扩展

完整案例先用 Medical SFT 得到冻结 Teacher，再提供两种能力平衡实验：

```text
SAR-OPD:
Medical SFT Teacher
→ fresh Base Student 做 Medical OPD
→ 从 Student training state 恢复
→ 使用 Base Teacher 做通用能力锚定

IDT-OPD:
同一个 Student
→ Medical Teacher step
→ Base Teacher step
→ 按固定调度交替
```

保存产物要区分：

- `sampler_weights`：推理、评测或作为冻结 Teacher；
- `training_state`：恢复可训练 Student 并继续优化。

多 Teacher 场景仍要求每个 Teacher 只对 Student 当前 rollout 的同一组 completion token 打分。不同 Teacher 可以使用不同 prompt，但 tokenizer 和 completion token IDs 必须兼容。

训练数据与 held-out 评测严格分离。用选择题 prompt 做 OPD 时可以保留题目和选项，训练侧不读取 held-out answer。分别记录各 Teacher 的 step 数、数据比例和能力 benchmark；reverse KL 下降不能替代独立评测。

`SAR-OPD`、`IDT-OPD` 是完整案例用于描述两套实验方案的名称，并非论文中的标准算法名。医疗 benchmark 只衡量当前选择题设置，不能外推为临床安全或诊断能力。

## SwanLab 指标

建议记录：

- `opd/reverse_kl_mean`
- `opd/reverse_kl_std`
- `data/datums`
- `data/completion_tokens_mean`
- `data/completion_tokens_total`
- `data/completion_tokens_per_second`
- `trainer/*`

## 常见错误

- 不要用 teacher 自己生成的 completion 训练 OPD；OPD 是 student on-policy。
- teacher logprob 必须和 student completion token 一一对齐。
- 不要把 prompt 区间设为非零 advantage。
- `sampler_refresh_steps` 过大时，rollout 会偏离当前 student；默认每步刷新最稳。
- teacher 和 student tokenizer/chat template 不一致时，先确认 token 对齐再训练。
- 不要把 `sampler_weights` 当作可续训 state，也不要把 `training_state` 直接传给推理 client。
- 多 Teacher 交替训练时不要只看最后 checkpoint；分别评估各能力并根据目标做 early stopping。
