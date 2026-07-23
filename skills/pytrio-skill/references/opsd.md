# OPSD 能力说明

当用户要做 OPSD、On-Policy Self-Distillation、privileged teacher、自蒸馏，或让同一个初始模型用参考解答对 Student 自己的轨迹提供逐 token 反馈时，先读本文件。

## 本页导航

- [官方实现](#官方实现)
- [核心目标](#核心目标)
- [与普通 OPD 的区别](#与普通-opd-的区别)
- [本案例的训练目标](#本案例的训练目标)
- [数据流](#数据流)
- [Client 边界](#client-边界)
- [Teacher Logprob 对齐](#teacher-logprob-对齐)
- [Datum 构造](#datum-构造)
- [异步顺序](#异步顺序)
- [常见错误](#常见错误)

## 官方实现

OPSD 是包含数据准备、同步/异步训练、评测和分析的多文件案例：

- 官方案例：https://docs.pytrio.com/docs/example/opsd
- 官方 Markdown：https://docs.pytrio.cn/docs/content/example/opsd/content.md
- 完整代码：https://github.com/KMnO4-zx/llm-agent-rl-lab/tree/main/04-opsd
- 文档核对版本：https://github.com/KMnO4-zx/llm-agent-rl-lab/tree/52c2f1c/04-opsd

生成可运行项目时，优先沿用完整源码目录；本文件用于保证 Student / Teacher、token 对齐和 on-policy 边界正确。

## 核心目标

OPSD 让同一个初始模型承担两种角色：

- Student：只看问题，使用当前 LoRA 策略生成 completion。
- Teacher：固定在 step 0，额外看到参考解答，只对 Student 的同一条 completion 计算 logprob。

Teacher 不重新生成“标准答案”。参考解答只作为 privileged context 改变 Teacher 的条件分布，不是 Student 的 SFT label。

## 与普通 OPD 的区别

| 方法 | Student 轨迹 | Teacher |
|---|---|---|
| 普通 OPD | 当前 Student rollout | 通常是独立、更强的模型 |
| OPSD | 当前 Student rollout | 同一个初始模型，使用包含参考解答的 privileged prompt |

两者都要求 Teacher 对 Student 实际生成的 token 计算 logprob，并通过 `importance_sampling` 更新 Student。

## 本案例的训练目标

官方 PyTRIO 案例实现 sampled-token reverse KL。对于 Student 实际采样的 token：

```text
reverse_kl = student_logprob - teacher_logprob
advantage = -kl_penalty_coef * reverse_kl
```

Teacher 比 Student 更认可某个 token 时 advantage 为正，反之为负。

这是 sampled-token 训练实现，不等同于 OPSD 论文主实验的 full-vocabulary JSD；描述结果时必须保留这条边界。

## 数据流

1. 从数据集中读取 `problem + solution`。
2. 创建可训练的 Student LoRA `TrainingClient`。
3. 创建不带 `model_path` 的固定 base `SamplingClient` 作为 Teacher。
4. 每个 step 用最新 Student 权重刷新 sampler。
5. Student 只看 `problem` 并采样 completion。
6. Teacher 看 `problem + solution`，对同一组 completion tokens 调 `compute_logprobs`。
7. 计算逐 token reverse KL 和 advantage。
8. 构造 `importance_sampling` Datum，只训练 completion。
9. 更新 Student；Teacher 保持不变。

## Client 边界

异步创建方式：

```python
training_client = await service_client.create_lora_training_client_async(
    base_model=base_model,
    rank=lora_rank,
    train_attn=True,
    train_mlp=True,
)

teacher_client = await service_client.create_sampling_client_async(
    base_model=base_model,
)
```

每个 step 刷新 Student sampler：

```python
student_sampler = (
    await training_client.save_weights_and_get_sampling_client_async()
)
```

Teacher 没有 optimizer，也不随 Student 更新。

## Teacher Logprob 对齐

Teacher 必须对 `teacher_prompt_ids + student_completion_ids` 计算 logprob：

```python
all_ids = teacher_prompt_ids + completion_ids
all_logprobs = await teacher_client.compute_logprobs_async(
    trio.ModelInput.from_ints(all_ids)
)
teacher_logprobs = all_logprobs[len(teacher_prompt_ids):]
```

以下三者必须严格等长，且 Teacher 区间不能包含 `None`：

```text
Student completion tokens
Student rollout logprobs
Teacher completion logprobs
```

Student 和 Teacher 可以使用不同 prompt，但必须使用同一个 tokenizer，并对同一组 completion token IDs 打分。

## Datum 构造

OPSD 沿用 OPD 的 completion-only mask：

```python
prompt_loss_len = len(student_prompt_ids) - 1
input_ids = student_prompt_ids + completion_ids[:-1]
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

prompt 只提供上下文，真正进入 loss 的 target、old logprob 和 advantage 都来自 Student completion。

## 异步顺序

单题内部必须先 Student sample，再 Teacher score；不同题目之间可以并发：

```python
rollouts = await asyncio.gather(
    *(student_sample_then_teacher_score(row) for row in batch)
)
```

一个 batch 的 rollout 应来自同一个 Student checkpoint。全部完成后，训练 API 需要区分“提交请求”和“等待远程任务完成”两次 `await`：

```python
fwd_bwd_future = await training_client.forward_backward_async(
    datums,
    loss_fn="importance_sampling",
)
optim_future = await training_client.optim_step_async(adam)

fwd_bwd_result = await fwd_bwd_future
await optim_future
```

这里第一次 `await` 返回 `APIFuture`，第二次 `await` 才取得远程结果。`sample_async()`、`compute_logprobs_async()` 和 `save_weights_and_get_sampling_client_async()` 在当前案例中一次 `await` 直接返回结果；若 SDK 版本变化，应重新检查实际签名。

## 建议记录

- `trainer/loss_mean`
- `opd/reverse_kl_mean`
- `opd/reverse_kl_std`
- `opd/advantage_mean`
- `data/completion_tokens_total`
- `time/step_elapsed_time`

Loss 和 reverse KL 下降只说明 Student 正在靠近 privileged Teacher；能力是否提高必须使用独立 benchmark 评测。

## 常见错误

- 不要让 Teacher 自己采样另一条 completion。
- 不要把参考 `solution` 当成 Student 的 SFT target。
- 不要给 Teacher 创建 LoRA optimizer，或随 Student 一起更新。
- 不要把 OPSD 描述成更大 Teacher 模型蒸馏；它使用同一个初始模型和不同 prompt。
- 不要混淆 sampled-token reverse KL 与 full-vocabulary JSD。
- 不要把 prompt 区间设为非零 advantage。
- `sampler_refresh_steps` 过大时会削弱 on-policy 边界；默认每步刷新最清晰。
