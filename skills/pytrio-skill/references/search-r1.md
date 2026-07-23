# Search-R1 能力说明

当用户要做 Search-R1、Agentic RL、多轮搜索工具调用、NQ / HotpotQA 强化学习，或需要把 tool observation 正确排除在 loss 之外时，先读本文件。

## 本页导航

- [官方实现](#官方实现)
- [核心目标](#核心目标)
- [轨迹与并发](#轨迹与并发)
- [Reward 与 Advantage](#reward-与-advantage)
- [Observation Mask 与 Datum](#observation-mask-与-datum)
- [长轨迹拆批](#长轨迹拆批)
- [训练与评测边界](#训练与评测边界)
- [常见错误](#常见错误)

## 官方实现

Search-R1 是多文件案例，不适合压缩成单个 example：

- 官方案例：https://docs.pytrio.com/docs/example/search-r1
- 官方 Markdown：https://docs.pytrio.cn/docs/content/example/search-r1/content.md
- 完整代码：https://github.com/KMnO4-zx/llm-agent-rl-lab/tree/main/03-search-r1
- 文档核对版本：https://github.com/KMnO4-zx/llm-agent-rl-lab/tree/52c2f1c/03-search-r1

需要生成可运行项目时，先沿用完整代码目录的模块划分，再按用户要求替换数据、搜索后端和 reward；不要把状态机、搜索客户端、训练和评测压进一个脚本。

## 核心目标

模型在一条轨迹中自主决定何时搜索、搜索什么、如何使用 observation，以及何时输出最终答案：

```text
assistant generation
→ search(query)
→ tool observation
→ assistant generation
→ ...
→ Answer: <short answer>
```

训练对象是 Student 的 LoRA 权重。搜索 API 和搜索结果是固定环境，不参与训练；模型学习的是工具调用、证据利用和最终回答策略。

## 推荐项目结构

```text
search-r1/
├── prepare_data.py   # 准备 train / dev / test
├── data.py           # 读取本地数据
├── protocol.py       # tool schema、prompt、tool-call 解析
├── search.py         # 搜索客户端、重试与调用统计
├── rollout.py        # 多轮工具状态机
├── reward.py         # 最终答案与格式 reward
├── train.py          # PyTRIO rollout、Datum、训练与 checkpoint
├── eval.py           # Base Model / checkpoint 统一评测
└── analyse.py        # 汇总评测结果
```

## 轨迹与并发

同一道题的第一轮共享 prompt，可以用一次 `sample_async(..., num_samples=group_size)` 生成整个 group。第一次搜索后，每条轨迹的 query 和 observation 不同，后续必须拆成独立 prompt：

```text
首轮：1 个 shared prompt × group_size
分叉后：group_size 个独立 prompt × num_samples=1
```

单条轨迹内部必须保持：

```text
assistant → search → observation → next assistant
```

只有不同题目或不同已分叉轨迹之间可以并发。不要让下一轮 generation 先于对应的搜索结果。

## Reward 与 Advantage

官方案例只根据最终回答计算 outcome reward：

| 最终结果 | Reward |
|---|---:|
| 格式合法且答案正确 | `1.0` |
| 格式合法但答案错误 | `0.0` |
| 没有合法最终答案 | `-0.1` |

不直接奖励搜索次数或中间 query。每道题的完整 group 全部结束后再计算：

```python
group_mean = sum(item.reward for item in group) / len(group)
for item in group:
    item.advantage = item.reward - group_mean
```

同组 reward 完全相同时 advantage 全为 0，通常跳过该组。不能先拆 micro-batch，再在 micro-batch 内重新计算均值。

## Observation Mask 与 Datum

工具 observation 必须进入后续模型上下文，但不能作为模型动作参与 loss：

```text
system / user / tool observation   → old logprob = 0, advantage = 0
assistant tool call / final answer → rollout old logprob, trajectory advantage
```

构造完整轨迹后，对 tokens、old logprobs 和 advantages 做同一次自回归右移：

```python
input_tokens = full_tokens[:-1]
target_tokens = full_tokens[1:]
old_logprobs = old_logprobs_by_token[1:]
advantages = advantages_by_token[1:]

datum = trio.Datum(
    model_input=trio.ModelInput.from_ints(input_tokens),
    loss_fn_inputs={
        "target_tokens": np.asarray(target_tokens, dtype=np.int64),
        "logprobs": np.asarray(old_logprobs, dtype=np.float32),
        "advantages": np.asarray(advantages, dtype=np.float32),
    },
)
```

四个字段必须严格等长。非零 old logprob 必须来自生成对应 assistant token 时的 Student sampler，不能在更新后重算。

## 长轨迹拆批

先完成整个 logical batch 的 rollout、reward 和 group-relative advantage，再把完整轨迹 Datum 拆成 micro-batch：

```text
完整同题 group
→ reward
→ group-relative advantage
→ 完整轨迹 Datum
→ 多次 forward_backward 累积梯度
→ 一次 optim_step
```

如果远端每次 `forward_backward` 对 micro-batch 样本取均值，不同大小的 micro-batch 要按 `n_k / N` 缩放 advantage，才能保持与完整 logical batch 全局样本均值一致。

## 训练与评测边界

- 训练集只需要问题和参考答案，不需要人工 query 或标准搜索轨迹。
- Base Model 和 checkpoint 必须使用相同工具协议、搜索后端、最大搜索次数和评测集。
- 在线搜索有额度、超时和结果漂移；`search/error_rate` 上升时，不能直接把 reward 下降解释为模型退化。
- 官方案例替换了原论文的本地 Wikipedia 基础设施，复现的是多轮工具环境、结果奖励、组内 advantage、observation mask 和策略更新闭环，不是原论文配置与分数的逐项复刻。

## 建议记录

- `reward/mean`
- `reward/correct`
- `reward/format`
- `rollout/search_calls`
- `rollout/turns`
- `rollout/degenerate_group_rate`
- `train/loss_tokens_per_rollout_batch`
- `search/success_rate`
- `search/error_rate`
- `search/latency`

## 常见错误

- 不要把搜索结果 token 设为非零 advantage。
- 不要训练或宣称训练了搜索后端；PyTRIO 更新的是模型 LoRA。
- 不要用更新后的 Student 重算 rollout old logprobs。
- 不要在拆分后的 micro-batch 内重新计算 group advantage。
- 不要把多轮轨迹当成一次普通 RAG 拼接；工具调用、observation 和下一轮生成之间有严格状态顺序。
- 搜索 payload 必须包含模型可利用的证据内容；只有标题或链接通常不足以替代官方案例的 `title + content + source + url` observation。
