# ReTool 能力说明

当用户要做 ReTool、代码解释器 Agent、code-interlaced rollout、数学工具强化学习，或需要让 tool observation 进入上下文但不参与 loss 时，先读本文件。

## 本页导航

- [完整案例](#完整案例)
- [核心目标](#核心目标)
- [推荐项目结构](#推荐项目结构)
- [工具协议与轨迹连续性](#工具协议与轨迹连续性)
- [Reward、Advantage 与训练](#rewardadvantage-与训练)
- [本地执行器的安全边界](#本地执行器的安全边界)
- [评测与实验边界](#评测与实验边界)
- [建议记录](#建议记录)
- [常见错误](#常见错误)

## 完整案例

ReTool 包含工具协议、本地执行器、多轮 rollout、训练、评测和分析，不适合压缩成单个 example：

- 完整代码：https://github.com/KMnO4-zx/agentic-rl-lab/tree/main/05-retool
- 本 Skill 核对版本：https://github.com/KMnO4-zx/agentic-rl-lab/tree/906d54e/05-retool

生成项目时沿用完整目录的模块划分，再替换数据、工具后端、reward 和模型。保留 protocol、sandbox、rollout、train、eval 的清晰边界。

## 核心目标

模型在数学推理中自主决定何时调用代码解释器、写什么代码、怎样利用执行结果，以及何时输出最终答案：

```text
assistant reasoning + tool call
→ code_interpreter(code)
→ stdout / stderr observation
→ next assistant turn
→ ...
→ \boxed{final answer}
```

训练信号只来自最终答案。代码调用次数、代码是否优雅和中间结果都不直接获得 reward，工具使用策略由 outcome reward 驱动。

## 推荐项目结构

```text
retool/
├── prepare_data.py  # 准备训练与评测数据
├── data.py          # 读取本地数学题
├── protocol.py      # 原生 tool schema、prompt、tool-call 解析
├── sandbox.py       # 代码执行与资源限制
├── rollout.py       # 多轮代码交织状态机
├── reward.py        # 最终答案数学等价判定
├── train.py         # Datum、PPO、checkpoint 与 SwanLab
├── eval.py          # text / retool 两种模式统一评测
└── analysis.py      # 汇总 checkpoint 指标
```

## 工具协议与轨迹连续性

- 优先使用模型原生 tool-call chat template；每个 assistant turn 最多调用一次工具，再等待 observation。
- 同一道题首轮可用 `num_samples=group_size` 一次分叉。工具返回后各轨迹上下文不同，后续按独立 prompt 采样。
- 单条轨迹严格执行 `assistant → tool → observation → assistant`；不同轨迹的采样和代码执行可以并发。
- 下一轮 prompt 应在真实 sampler token 后追加 assistant 结束符和 tool observation。不要把历史 assistant 文本 decode 后重新 tokenize；不可逆重编码会破坏 rollout old logprob 与训练 token 的一一对应。
- 对每轮 prompt 做前缀扩展校验。新 prompt 不能以前一轮完整 token 序列为前缀时，停止并排查 chat template 边界。

## Reward、Advantage 与训练

完整案例使用 outcome-only 数学 reward：只检查回答末尾的最后一个 `\boxed{}`，数学等价时为 `+1`，错误或格式非法时为 `-1`。

同题 group 全部结束后再计算：

```python
advantage = reward - mean(group_rewards)
```

完整多轮轨迹只构造一个右移后的 `Datum`：

```text
system / user / tool observation   → old logprob = 0, advantage = 0
assistant reasoning / tool call    → rollout old logprob, trajectory advantage
assistant final answer             → rollout old logprob, trajectory advantage
```

案例使用 PyTRIO 内置 `ppo`，clip 阈值为 `0.8 / 1.28`。先完成整个 logical batch，再按 padding 后的 token 面积拆 micro-batch；每个 micro-batch 的 advantage 按 `n_k / N` 缩放，累计全部 backward 后只调用一次 `optim_step()`。

## 本地执行器的安全边界

完整案例的 `sandbox.py` 使用独立 subprocess，并限制并发、wall-clock、CPU 时间、线程数和输出长度。这些限制用于控制意外资源消耗，不能提供可信的安全隔离。

当前实现仍允许模型生成的代码访问本机文件和网络，并把父进程环境变量传给子进程。执行公开或不可控模型生成代码时：

- 在一次性容器、低权限虚拟机或专用沙箱服务中运行；
- 只传必要的环境变量，移除 PyTRIO、SwanLab、云服务等凭证；
- 禁止挂载个人目录、SSH 配置和项目密钥；
- 设置内存、进程数、文件系统和网络限制；
- 不要在个人工作环境中无人值守运行。

system prompt 中的“不要读写文件”只属于软约束，不能替代操作系统级隔离。

## 评测与实验边界

- Base Model 与 checkpoint 使用同一工具协议、执行器、轨迹预算、sampling 参数和题集。
- 同时提供 text-only 模式，判断收益来自工具策略还是基础模型波动。
- 案例省略 cold-start SFT，依赖所选基座已经能稳定生成合法 tool call；更换基座后先做 smoke test。
- 案例轨迹预算为 8K，原论文使用更长预算；被截断的长推理会直接影响 format 与准确率。
- PyTRIO 内置 PPO 未覆盖案例所参考 recipe 的 dual-clip；报告结果时保留这条实现边界。
- 单次训练和小规模 AIME 评测适合验证闭环，不能直接推出稳定论文收益。

## 建议记录

- `reward/mean`、`reward/correct`、`reward/format`
- `rollout/valid_tool_call_rate`
- `rollout/code_calls`、`rollout/turns`
- `rollout/trajectory_tokens`、`rollout/degenerate_group_rate`
- `sandbox/success_rate`、`sandbox/error_rate`、`sandbox/timeout_rate`
- `sandbox/latency`
- `train/loss_tokens_per_rollout_batch`
- `time/step_seconds`

## 常见错误

- 不要把 stdout/stderr observation 设为非零 advantage。
- 不要在每一轮重新渲染整段历史并替换真实采样 token。
- 不要奖励代码调用次数；模型会学会无意义地反复调用工具。
- 不要让同一轨迹的下一轮生成越过尚未完成的代码执行。
- 不要把本地 subprocess 称为安全沙箱，或在包含凭证的环境中直接执行模型代码。
- 不要在 micro-batch 内重新计算 group advantage，也不要每个 micro-batch 单独做 optimizer step。
