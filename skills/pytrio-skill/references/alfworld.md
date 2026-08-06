# ALFWorld Agentic RL 能力说明

当用户要做 ALFWorld、TextWorld、text-only embodied agent、长轨迹环境交互、同任务 group rollout，或需要把环境 observation 排除在 loss 之外时，先读本文件。

## 本页导航

- [完整案例](#完整案例)
- [复现范围](#复现范围)
- [推荐项目结构](#推荐项目结构)
- [同游戏独立环境](#同游戏独立环境)
- [多轮轨迹与工具协议](#多轮轨迹与工具协议)
- [Reward 与 Advantage](#reward-与-advantage)
- [Datum 与 PPO](#datum-与-ppo)
- [评测、成本与监控](#评测成本与监控)
- [常见错误](#常见错误)

## 完整案例

ALFWorld 包含环境适配、工具协议、长轨迹 rollout、训练、评测和分析，不适合压缩成单个 example：

- 完整代码：https://github.com/KMnO4-zx/agentic-rl-lab/tree/main/08-alfworld
- 本 Skill 核对版本：https://github.com/KMnO4-zx/agentic-rl-lab/tree/906d54e/08-alfworld

运行前按案例锁定的 optional dependencies 安装 ALFWorld / TextWorld，并下载环境数据。第三方环境版本和 Python 版本变化时，重新验证 reset、step 和 worker cleanup。

## 复现范围

该案例使用 ALFWorld 的 text-only 环境，自行加入 GRPO-style group rollout、组相对 advantage 和 PPO 更新，形成 LLM Agentic RL recipe。

ALFWorld 原论文主要研究文字环境与具身环境对齐，并使用 DAgger 等训练方法。该案例不覆盖视觉 BUTLER 系统，也不复刻原论文训练算法。描述成果时使用“基于 ALFWorld 环境的 Agentic RL 案例”。

## 推荐项目结构

```text
alfworld/
├── data.py          # 发现游戏、过滤 split、固定顺序
├── protocol.py      # alfworld_step schema、prompt、解析与 observation
├── environment.py   # 同游戏 K 个独立 TextWorld 状态
├── rollout.py       # 多轮环境交互、轨迹与 reward
├── advantages.py    # 同游戏组相对 advantage
├── train.py         # Datum、PPO、checkpoint 与 SwanLab
├── eval.py          # seen / unseen 固定评测与 JSONL
└── analysis.py      # 汇总 checkpoint 和任务类型结果
```

## 同游戏独立环境

group-relative advantage 要求组内轨迹面对同一个任务，同时保持独立状态：

1. 为一个 `game.tw-pddl` 创建 `group_size` 个环境实例。
2. `reset()` 后校验所有分支的初始 observation 完全一致。
3. 校验每个分支实际加载的 game file 与目标文件一致。
4. 每条轨迹只推进自己的环境状态，禁止分支间共享 action 历史。
5. 结束后显式关闭异步 worker，避免子进程泄漏。

不同游戏可以并行；同一轨迹内部必须等待当前 action 的 environment step 完成，再生成下一轮 action。

## 多轮轨迹与工具协议

模型只获得一个结构化工具，例如 `alfworld_step(action)`。环境返回 observation、可执行动作和终局状态。

- 首轮同游戏分支共享任务与初始 observation，可一次采样整个 group。
- 第一次 action 后，各分支环境状态不同，后续使用独立 prompt。
- 下一轮 prompt 在真实 sampler token 后追加 assistant 结束符和 tool observation，避免历史文本重新 tokenize。
- 每轮保存 prompt tokens、completion tokens、old logprobs、action、observation 和环境结果。
- 同时限制最大环境步数、单轮 assistant tokens 和完整轨迹 tokens。

长轨迹通常包含大量重复 prefix。预算评估需要统计 prefilling、sampling 和 train token，不能只看进入 loss 的 assistant token。

## Reward 与 Advantage

完整案例使用轨迹终局 reward：

```text
reward = 1[won] - 0.1 * invalid_action_count
```

TextWorld 自带 score 只用于记录，不直接替代该自定义 reward。每个游戏的完整 group 结束后计算：

```python
advantage = trajectory.reward - mean(same_game_rewards)
```

组内 reward 全相同时 advantage 全为 0，该组不提供相对训练信号。reward 可能因非法动作惩罚变为负数；不要强行裁到 `[0, 1]`。

## Datum 与 PPO

把一条完整环境轨迹构造成一个右移后的 Datum：

```text
system / user / environment observation → old logprob = 0, advantage = 0
assistant tool call / action             → rollout old logprob, trajectory advantage
```

最后一次工具返回后即使没有新的 assistant turn，也保留真实 observation token，并将训练信号设为 0。每轮 prompt 必须是已有完整 token 序列的前缀扩展。

案例使用 PyTRIO 内置 PPO：

```python
training_client.forward_backward(
    datums,
    loss_fn="ppo",
    loss_fn_config={
        "clip_low_threshold": 0.8,
        "clip_high_threshold": 1.2,
    },
).result()
training_client.optim_step(adam).result()
```

advantage 为 0 或没有 assistant token 的轨迹可以跳过远端训练，但仍要留在 rollout / degenerate 指标中。

## 评测、成本与监控

Base 与 checkpoint 使用相同游戏列表、seen/unseen split、温度、seed、环境步数和轨迹上限。保存逐游戏完整 JSONL，汇总：

- `reward/success_rate` 与各任务类型 success rate；
- `rollout/steps_mean`、`rollout/truncated_rate`；
- `rollout/valid_tool_call_rate`；
- `rollout/admissible_action_rate`、`rollout/invalid_actions_mean`；
- `advantage/degenerate_group_rate`；
- `train/tokens`、`train/loss_tokens`、`train/max_sequence_tokens`；
- `time/update_seconds`。

该案例的一次训练与单 seed 评测给出了正向信号，稳定结论仍需要多 seed、重复评测和更长 checkpoint 曲线。长轨迹成本主要可能来自重复 prefilling；报告预算时分开列出 prefill、sample 和 train。

## 常见错误

- 不要把不同游戏或不同初始状态的轨迹放进同一个 advantage group。
- 不要让多个分支共享同一个可变环境实例。
- 不要把 environment observation token 设为非零 advantage。
- 不要重编码历史 assistant 文本后再拼接下一轮 prompt。
- 不要把 ALFWorld 原论文的 DAgger 训练与本案例的 PPO recipe 混写。
- 不要只统计训练 token；长轨迹的重复 prefix 可能主导实际成本。
- 不要省略环境 worker cleanup 和第三方版本兼容检查。
