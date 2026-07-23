<div align="center">

<a href="https://pytrio.cn/">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="images/TRIO_LOGO_DARK.svg" />
    <source media="(prefers-color-scheme: light)" srcset="images/TRIO_LOGO.svg" />
    <img src="images/TRIO_LOGO.svg" alt="TRIO" width="360" />
  </picture>
</a>

<h1><a href="https://pytrio.cn/">PyTRIO.skill</a></h1>


> 让 Agent 正确使用 PyTRIO 写 SFT、GRPO、OPD、Search-R1、OPSD、DPO、推理和实验记录代码。

[![PyTRIO 官网](https://img.shields.io/badge/PyTRIO-官网-fc4547)](https://pytrio.cn/)
[![PyTRIO Docs](https://img.shields.io/badge/PyTRIO-官方文档-ff6b57)](https://docs.pytrio.cn/docs)
[![ModelScope Skill](https://img.shields.io/badge/ModelScope-Skill-624aff)](https://www.modelscope.cn/skills/SwanLab/pytrio-skill/summary)
[![Agent Skill](https://img.shields.io/badge/Agent-Skill-7c3aed)](skills/pytrio-skill/SKILL.md)
[![SwanLab](https://img.shields.io/badge/SwanLab-推荐记录工具-1f8f4c)](https://docs.swanlab.cn)
[![License](https://img.shields.io/badge/License-MIT-d4a017)](LICENSE)

没有本地 GPU？没关系。  
PyTRIO 把 LLM 后训练丢到云端执行，你只需要写数据、算法和训练循环。

这个 Skill 会让 Claude Code、Codex 等 Agent 先按任务读取 SFT、GRPO、OPD、Search-R1、OPSD、DPO/custom loss 的本地能力说明，再参考内置示例或官方完整项目生成代码，避免把 PyTorch / HuggingFace 的习惯误套到 PyTRIO 上。

[安装](#安装) · [内容](#内容) · [示例](#示例) · [SwanLab 记录](#swanlab-记录) · [打包](#打包)

</div>

---

## 安装

推荐使用全局安装方式：

```bash
npx skills add SwanHubX/pytrio-skill -g -y
```

也可以在 ModelScope 查看 skill 页面：https://www.modelscope.cn/skills/SwanLab/pytrio-skill/summary

> 注：`-g` 表示安装到当前用户级别，`-y` 表示跳过交互确认。这个方式不会把 skill 写进当前项目目录；它只会把 skill 下载到用户目录下的 `.agents/skills`，再为 Claude Code、Codex 等支持 Agent Skills 的 CLI 创建软链接。同一个用户下的多个 Agent CLI 可以复用这一份 skill。

后续如果 PyTRIO Skill 有更新，执行：

```bash
npx skills update pytrio-skill -g -y
```

使用 Skill 生成或调试代码前，建议先确认 PyTRIO SDK 为最新版。项目没有锁定 PyTRIO 版本时执行：

```bash
python -m pip install --upgrade pytrio
python -c 'from importlib.metadata import version; print(version("pytrio"))'
```

如果项目已经通过 `pyproject.toml`、requirements 或 lockfile 固定版本，应尊重项目依赖，不要强制升级，并以已安装版本的 API 签名为准。

## 内容

```text
skills/
└── pytrio-skill/
    ├── SKILL.md
    ├── references/
    │   ├── doc-index.md
    │   ├── sft.md
    │   ├── grpo.md
    │   ├── opd.md
    │   ├── search-r1.md
    │   ├── opsd.md
    │   ├── dpo.md
    │   └── chat-huanhuan.md
    └── examples/
        ├── quickstart_sft.py
        ├── chat-huanhuan.py
        ├── chat-huanhuan-async.py
        ├── sft-distill-conversation.py
        ├── sft-distill-conversation-async.py
        ├── grpo-gsm8k.py
        ├── grpo-gsm8k-async.py
        ├── opd-deepmath.py
        ├── opd-deepmath-async.py
        └── dpo-hh-rlhf.py
```

| 文件 | 作用 |
|---|---|
| `SKILL.md` | 轻量入口和任务路由，告诉 Agent 应该先读 SFT、GRPO、OPD、Search-R1、OPSD 还是 DPO |
| `references/doc-index.md` | PyTRIO 官方 Markdown 文档和本地示例索引 |
| `references/sft.md` | SFT 数据构造、assistant-only loss mask、同步/异步训练模式 |
| `references/grpo.md` | GRPO rollout、reward、group-relative advantage、`importance_sampling` 训练模式 |
| `references/opd.md` | OPD student rollout、teacher logprob、reverse-KL advantage 训练模式 |
| `references/search-r1.md` | Search-R1 多轮工具状态机、结果 reward、observation mask、长轨迹拆批 |
| `references/opsd.md` | OPSD 同模型 Student / privileged Teacher、sampled-token reverse-KL |
| `references/dpo.md` | DPO chosen/rejected 偏好训练、reference logprob、custom loss |
| `references/chat-huanhuan.md` | Chat-甄嬛案例、同步/异步 SFT、SwanLab 记录模式 |
| `examples/quickstart_sft.py` | 最小 SFT 训练、保存权重、推理示例 |
| `examples/chat-huanhuan.py` | 带 SwanLab 记录的同步 SFT 示例 |
| `examples/chat-huanhuan-async.py` | 带 SwanLab 记录的异步 SFT 示例 |
| `examples/sft-distill-conversation.py` | 多轮对话 SFT 蒸馏示例 |
| `examples/sft-distill-conversation-async.py` | 异步多轮对话 SFT 蒸馏示例 |
| `examples/grpo-gsm8k.py` | 同步 GRPO / GSM8K 示例 |
| `examples/grpo-gsm8k-async.py` | 异步 GRPO / GSM8K 示例 |
| `examples/opd-deepmath.py` | 同步 OPD / DeepMath 示例 |
| `examples/opd-deepmath-async.py` | 异步 OPD / DeepMath 示例 |
| `examples/dpo-hh-rlhf.py` | DPO / HH-RLHF / custom loss 示例 |

## 示例

### SFT

`examples/quickstart_sft.py` 展示最小闭环：

1. 创建 `trio.ServiceClient`
2. 创建 LoRA `TrainingClient`
3. 构造带 prompt mask 的 `trio.Datum`
4. 调用 `forward_backward` 和 `optim_step`
5. 保存推理权重
6. 创建 sampling client 做推理

`examples/chat-huanhuan.py` 是同步版，适合先理解完整 SFT 流程。  
`examples/chat-huanhuan-async.py` 是异步版，适合参考异步提交 batch、异步计算 loss、异步记录 SwanLab 的写法。

`examples/sft-distill-conversation.py` 和异步版展示多轮对话蒸馏：system/user 只作为上下文，assistant 内容和结束标记参与 loss。

新增示例里的相对数据路径默认按运行目录解析，例如 `./datasets/...`；不会把下载数据或 SwanLab 本地日志写进 skill 安装目录。

### GRPO

`examples/grpo-gsm8k.py` 和异步版展示 GSM8K 风格 RLVR：

1. 用当前 student sampler 对同一题采样多个 completion
2. 用 reward 函数打分
3. 计算 group-relative advantage
4. 构造 `importance_sampling` 所需的 `target_tokens`、old `logprobs`、`advantages`
5. 调用 `forward_backward(..., loss_fn="importance_sampling")` 和 `optim_step`

### OPD

`examples/opd-deepmath.py` 和异步版展示 on-policy distillation：

1. student 先对 prompt 采样 completion
2. teacher 对 student 实际生成的 completion 计算 logprob
3. 用 `student_logprob - teacher_logprob` 计算 reverse KL
4. 用 `-kl_coef * reverse_kl` 作为 token-level advantage
5. 通过 `importance_sampling` 更新 student

### Search-R1

`references/search-r1.md` 对齐官方多文件案例，重点说明：

1. 同题首轮共享 prompt、搜索后轨迹分叉的多轮工具状态机
2. 只根据最终答案计算 reward，再在完整同题 group 内计算 advantage
3. tool observation 进入上下文，但使用零 old logprob 和零 advantage 排除在 loss 之外
4. 长轨迹先构造完整 Datum，再拆 micro-batch 累积梯度并只做一次 optimizer step
5. 搜索后端保持固定，训练的是模型 LoRA 的工具使用与回答策略

### OPSD

`references/opsd.md` 对齐官方 On-Policy Self-Distillation 案例：

1. Student 只看问题并生成当前策略 completion
2. 同一个初始模型作为固定 Teacher，通过 privileged prompt 额外看到参考解答
3. Teacher 不重新采样，只对 Student completion 调 `compute_logprobs`
4. 用 sampled-token reverse KL 构造逐 token advantage
5. 参考解答只改变 Teacher 条件分布，不是 Student 的 SFT label

### DPO

`examples/dpo-hh-rlhf.py` 展示 preference training 和 custom loss：

1. 把 HH-RLHF 样本解析成共同 prompt、chosen response、rejected response
2. reference model 计算 chosen/rejected 的参考 logprob
3. 当前 student 通过 `forward_backward_custom` 提供可求导 logprob
4. 本地 torch loss 实现 DPO 公式并返回 metrics
5. PyTRIO 继续负责远端 backward、optimizer step 和权重保存

## SwanLab 记录

SwanLab 是 PyTRIO 训练过程中不可或缺的实验记录工具，也是 PyTRIO 的好搭档。建议训练环境默认安装：

```bash
pip install swanlab
```

训练脚本建议记录：

- SFT：`loss`
- GRPO：`reward`、`frac_degenerate`、`datums`
- OPD：`opd/reverse_kl_mean`、`opd/reverse_kl_std`、completion token 指标
- Search-R1：`reward/correct`、`reward/format`、`rollout/turns`、`search/success_rate`、`search/error_rate`
- OPSD：`trainer/loss_mean`、`opd/reverse_kl_mean`、`opd/reverse_kl_std`、`opd/advantage_mean`
- DPO：`dpo/loss`、`dpo/accuracy`、`dpo/margin`、chosen/rejected reward
- `epoch` / `batch` / `global_step`
- `base_model`
- `dataset_path`
- `lora_rank`
- `learning_rate`
- `max_length`
- `weights_name`

如果 Agent 需要查询 SwanLab 实验、对比曲线、读取指标或写更完整的记录代码，建议同时安装 SwanLab Skill：

```bash
npx skills add SwanHubX/SwanLab-Skill -g -y
```

## 打包

生成发布包：

```bash
make package
```

生成的压缩包会解压成：

```text
SKILL.md
references/
examples/
```

因此可以直接解压到 `.claude/skills/pytrio-skill/`。

## 维护原则

- Skill 内部保持精简，不复制整站文档。
- PyTRIO API 细节以官方 Markdown 文档为准。
- 单文件案例优先放到 `skills/pytrio-skill/examples/`；包含数据、环境、训练和评测的多文件案例保留官方源码目录，并在 `references/` 说明关键实现边界。
- 入口文档保持能力导向，优先保证 Agent 会写 SFT、GRPO、OPD、Search-R1、OPSD 和 DPO/custom loss。
- 如果涉及实验记录和指标查询，优先配合 SwanLab Skill 使用。
