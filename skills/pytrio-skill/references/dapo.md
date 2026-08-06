# DAPO 能力说明

当用户要做 DAPO、Dynamic Sampling、Clip-Higher、Token-level Policy Gradient Loss、Soft Overlong Punishment，或需要用 PyTRIO 对照 GRPO 与 DAPO 时，先读本文件。

## 本页导航

- [完整案例](#完整案例)
- [核心差异](#核心差异)
- [Rollout 与 Dynamic Sampling](#rollout-与-dynamic-sampling)
- [Reward 与 Advantage](#reward-与-advantage)
- [Datum 与 Custom PPO Loss](#datum-与-custom-ppo-loss)
- [预算与监控](#预算与监控)
- [复现边界](#复现边界)
- [常见错误](#常见错误)

## 完整案例

DAPO 需要数据准备、候选补采、自定义 PPO reduction、评测和分析，不适合压缩成单个 example：

- 完整代码：https://github.com/KMnO4-zx/agentic-rl-lab/tree/main/06-dapo
- 本 Skill 核对版本：https://github.com/KMnO4-zx/agentic-rl-lab/tree/906d54e/06-dapo

完整案例在同一个 `train.py` 中保留 GRPO / DAPO preset，适合做同数据、同模型、同预算下的机制对照。

## 核心差异

| 开关 | GRPO preset | DAPO preset |
|---|---|---|
| PPO clip | `0.8 / 1.2` | `0.8 / 1.28`（Clip-Higher） |
| Loss reduction | 每条 sequence 先取 token mean，再对 sequence mean | 合并 completion token 后取 token mean |
| Dynamic Sampling | 固定采样一批 | 丢弃全对/全错组并补采 |
| Soft Overlong | 关闭 | 在长度上限前线性增加惩罚 |

DAPO 的 token-level reduction 和算法对照指标通过本地 torch custom loss 实现，调用 `forward_backward_custom()`。不要仅替换内置 `ppo` 的 clip 参数后就宣称完成了 DAPO。

## Rollout 与 Dynamic Sampling

每道题先用当前 Student sampler 采样完整 group，并保存 completion token、old logprob、文本和 correctness。

有效组按原始正确性判断：

```text
0 < correct_count < group_size
```

Soft Overlong 产生的 shaped reward 用于组内 advantage；Dynamic Sampling 是否保留一组仍按原始 correctness 判断，避免长度惩罚把全错组伪装成有效组。

DAPO 在候选上限内持续补采，直到收集目标数量的有效组。候选预算耗尽时允许使用部分 batch；一个有效组都没有时跳过本次更新。

严格补齐会产生 refill barrier：接近填满时只差一组，后续补采宽度会缩小，learner 等待最后一个有效组。实现和评估时同时限制：

- 最大候选 group 数；
- rollout completion token 总量；
- 每个 step 的 wall-clock；
- 真正进入训练的 token 总量。

## Reward 与 Advantage

完整案例先计算数学正确性 reward：正确 `+1`，错误或格式非法 `-1`。DAPO 再加入 Soft Overlong：

```text
completion <= max_tokens - cache   → penalty = 0
进入最后 cache 区间                → penalty 从 0 线性降到 -1
超过 max_tokens                    → penalty = -1
shaped_reward = base_reward + penalty
```

同题完整 group 内使用样本标准差（`N-1` 分母）标准化：

```python
advantage = (shaped_reward - mean) / (std + epsilon)
```

组内 shaped reward 完全相同时，advantage 全为 0。

## Datum 与 Custom PPO Loss

每条 completion 构造右移对齐的 prompt-masked Datum：

```text
prompt token     → target/logprob/advantage 占位，completion_mask=False
completion token → 真实 target、rollout old logprob、sequence advantage
```

传给 `forward_backward_custom()` 的 forward Datum 只保留 `target_tokens`；old logprobs、advantages 和 completion mask 通过闭包元数据传给本地 loss。

逐 token PPO 目标：

```text
ratio_t = exp(current_logprob_t - old_logprob_t)
objective_t = min(ratio_t * A_t, clip(ratio_t) * A_t)
```

- GRPO preset：每条回答先对 token 求均值，再对回答求均值。
- DAPO preset：把全部 completion token 合并后求均值，长回答按 token 数获得更大权重。

custom loss 返回原始 torch loss 和 `ppo/*` metrics；网络请求、采样和 reward 计算都放在 loss 回调之外。

## 预算与监控

至少记录以下三套口径：

- 全部候选：`candidate_groups`、`completions`、`completion_tokens`；
- 最终训练：`effective_groups`、`train_completions`、`train_completion_tokens`；
- 时间与利用率：`oversample_ratio`、`effective_fill_ratio`、`time/step_seconds`。

建议同时记录：

- `reward/base_mean`、`reward/length_penalty_mean`、`reward/shaped_mean`
- `reward/accuracy`、`reward/format_rate`
- `rollout/effective_group_ratio`
- `rollout/mean_completion_tokens`、`rollout/max_completion_tokens`
- `ppo/clip_fraction`、`ppo/lower_clip_fraction`、`ppo/upper_clip_fraction`
- `ppo/train_tokens`、`ppo/gradient_active_tokens`

## 复现边界

完整案例跑通了 DAPO 训练链路，并记录了 Dynamic Sampling 的真实等待和 token 成本；长上下文正式 run 在第 35 step 手动停止，没有形成可用于比较 DAPO 与 GRPO 效果的完整曲线。

使用该案例时可以确认算法开关、数据流和工程代价。不要用这条未完成 run 声明 DAPO 优于 GRPO，也不要把缩小后的默认配置当作论文规模复现。

## 常见错误

- 不要只记录被保留的有效组；被丢弃的候选已经消耗采样 token 和时间。
- 不要用 shaped reward 判断 Dynamic Sampling 有效组；案例按原始 correctness 判断。
- 不要把 DAPO 简化为非对称 PPO clip；还要处理 Dynamic Sampling、token mean 和 Soft Overlong。
- 不要在 custom loss 内发起网络请求或重新计算 rollout old logprob。
- 不要假设 `asyncio.gather` 会消除补采屏障；补采轮次之间仍有严格依赖。
- 不要在候选预算耗尽后无限补采；允许部分 batch 或显式跳过更新。
