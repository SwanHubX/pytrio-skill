# GSPO 能力说明

当用户要做 GSPO、sequence-level policy optimization、序列级重要性比率与裁剪，或需要用 PyTRIO `forward_backward_custom()` 实现序列级 RL loss 时，先读本文件。

## 本页导航

- [完整案例](#完整案例)
- [核心目标](#核心目标)
- [Rollout 与 Advantage](#rollout-与-advantage)
- [Datum 与序列级 Loss](#datum-与序列级-loss)
- [退化组与归一化](#退化组与归一化)
- [建议记录](#建议记录)
- [复现边界](#复现边界)
- [常见错误](#常见错误)

## 完整案例

GSPO 包含数据准备、group rollout、自定义 loss、checkpoint、评测和分析，不适合压缩成单个 example：

- 完整代码：https://github.com/KMnO4-zx/agentic-rl-lab/tree/main/07-gspo
- 本 Skill 核对版本：https://github.com/KMnO4-zx/agentic-rl-lab/tree/906d54e/07-gspo

## 核心目标

GSPO 保留 GRPO 的同题 group rollout、结果 reward 和 relative advantage，把重要性比率与裁剪单位提升到完整 completion sequence。

对一条长度为 `T` 的 completion：

```text
log_ratio = mean_t(current_logprob_t - sampling_logprob_t)
seq_ratio = exp(log_ratio)
```

这个几何平均比率、sequence advantage 和裁剪结果由整条回答共享。完整案例使用论文给出的窄裁剪范围：

```text
[1 - 3e-4, 1 + 4e-4]
```

## Rollout 与 Advantage

1. 用当前 Student sampler 对同一道题采样 `group_size` 条 completion。
2. 保存每条 completion 的 token、sampling logprob、文本和 reward。
3. 在完整同题 group 内用样本标准差标准化 reward：

```python
advantage = (reward - mean(group_rewards)) / (std(group_rewards) + epsilon)
```

4. 每条 completion 获得一个 sequence advantage 标量。

完整案例使用数学结果 reward，不启用 Dynamic Sampling，也不加入 Soft Overlong。全对或全错组 advantage 全为 0。

## Datum 与序列级 Loss

PyTRIO forward Datum 负责得到当前策略的可求导 token logprob：

```python
input_tokens = prompt_tokens + completion_tokens[:-1]
target_tokens = [0] * (len(prompt_tokens) - 1) + completion_tokens
```

sampling logprobs、sequence advantage 和 completion 长度保存在本地 metadata 中，通过闭包传给 custom loss。loss 只读取 `current_logprobs` 的末尾 completion 区间；不要让 prompt 占位 token 进入序列比率。

每条序列计算：

```text
unclipped = seq_ratio * advantage
clipped = clip(seq_ratio, 1-eps_low, 1+eps_high) * advantage
objective = min(unclipped, clipped)
```

所有 sequence objective 按原始 sequence 数归一化，loss 回调返回 torch loss 和 `gspo/*` metrics：

```python
training_client.forward_backward_custom(
    datums,
    make_gspo_loss_fn(metas, config, ...),
).result()
training_client.optim_step(adam).result()
```

## 退化组与归一化

退化组没有梯度，可以不送到远端 forward/backward；它们的零目标仍属于原始 rollout batch。构造 custom loss 前先统计：

- `normalization_sequences`：全部 rollout completion 数；
- `normalization_tokens`：全部 rollout completion token 数；
- `train_sequences` / `train_tokens`：过滤退化组后实际进入远端计算的数量。

loss 分母使用原始 batch 数量，避免过滤零目标后放大剩余有效样本的梯度。全部 group 都退化时跳过 backward 和 optimizer step。

## 建议记录

- `reward/base_mean`、`reward/accuracy`、`reward/format_rate`
- `rollout/groups`、`rollout/train_groups`、`rollout/degenerate_groups`
- `rollout/completion_tokens`、`rollout/mean_completion_tokens`
- `gspo/loss`、`gspo/seq_ratio_mean`
- `gspo/sequence_clip_fraction`、`gspo/token_clip_fraction`
- `gspo/train_sequences`、`gspo/normalization_sequences`
- `gspo/active_sequence_fraction`、`gspo/active_token_fraction`
- `time/step_seconds`

## 复现边界

完整案例验证了 GSPO custom loss、100-step 训练链路和 checkpoint 评测。当前结果最稳定地说明格式遵循率提高、回答长度缩短；数学准确率没有得到稳定提升，且缺少同模型、同数据、同预算的 matched GRPO 对照。

案例使用 Qwen3.5-4B LoRA、一次 rollout batch 对应一次更新和 AIME25，覆盖核心 loss 与训练链路。它没有复刻论文的大模型、MoE、mini-batch 和完整 benchmark 配置，也不能用于证明 GSPO 优于 GRPO。

## 常见错误

- 不要把 token-level ratio 分别裁剪后再称为 GSPO；整条 sequence 共享一个几何平均比率。
- 不要把 prompt token 计入 `log_ratio`。
- 不要过滤退化组后把 loss 分母改成有效序列数；这会放大梯度。
- 不要给 GSPO 自动叠加 DAPO Dynamic Sampling 或 Soft Overlong，除非用户明确要研究组合算法。
- 不要把单次 Pass@N 波动解释成稳定准确率收益。
- 不要在 custom loss 内发起采样、reward 或网络请求。
