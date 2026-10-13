<div align="center">

<a href="https://pytrio.cn/">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="images/TRIO_LOGO_DARK.png" />
    <source media="(prefers-color-scheme: light)" srcset="images/TRIO_LOGO.png" />
    <img src="images/TRIO_LOGO.png" alt="TRIO"/>
  </picture>
</a>

<h1><a href="https://pytrio.com/">PyTRIO.skill</a></h1>


> 让 Agent 正确使用 PyTRIO 写 SFT、Decision Model 决策模型、多模态推理与训练、Vision GRPO、GRPO、OPD、Search-R1、ReTool、OPSD、DAPO、GSPO、ALFWorld、自定义损失函数、DPO、推理和实验记录代码。

[![PyTRIO 官网](https://img.shields.io/badge/PyTRIO-官网-fc4547)](https://pytrio.cn/)
[![PyTRIO Docs](https://img.shields.io/badge/PyTRIO-官方文档-ff6b57)](https://docs.pytrio.cn/docs)
[![ModelScope Skill](https://img.shields.io/badge/ModelScope-Skill-624aff)](https://www.modelscope.cn/skills/SwanLab/pytrio-skill/summary)
[![Agent Skill](https://img.shields.io/badge/Agent-Skill-7c3aed)](skills/pytrio-skill/SKILL.md)
[![Skill Version](https://img.shields.io/badge/Skill-v0.2.2-7c3aed)](skills/pytrio-skill/SKILL.md)
[![SwanLab](https://img.shields.io/badge/SwanLab-推荐记录工具-1f8f4c)](https://docs.swanlab.cn)
[![License](https://img.shields.io/badge/License-MIT-d4a017)](LICENSE)

没有本地 GPU？没关系。  
PyTRIO 把 LLM 后训练丢到云端执行，你只需要写数据、算法和训练循环。

这个 Skill 会让 Claude Code、Codex 等 Agent 先按任务读取 SFT、Decision Model、多模态与 Vision GRPO、GRPO、OPD、Search-R1、ReTool、OPSD、DAPO、GSPO、ALFWorld、Custom Loss、DPO 的本地能力说明，再参考内置示例或完整项目生成代码，避免把 PyTorch / HuggingFace 的习惯误套到 PyTRIO 上。

Medical OPD、Search-R1、OPSD、ReTool、DAPO、GSPO、ALFWorld、Vision GRPO 和 PyTRIO-Jev 的多文件案例来自 [agentic-rl-lab](https://github.com/KMnO4-zx/agentic-rl-lab)。Skill 保留源码链接与固定 commit 链接，方便在最新版实现和核对版本之间切换。

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
    │   ├── decision-model.md
    │   ├── vision.md
    │   ├── grpo.md
    │   ├── opd.md
    │   ├── search-r1.md
    │   ├── retool.md
    │   ├── opsd.md
    │   ├── dapo.md
    │   ├── gspo.md
    │   ├── alfworld.md
    │   ├── custom-loss.md
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
        ├── vision-grpo.py
        ├── opd-deepmath.py
        ├── opd-deepmath-async.py
        └── dpo-hh-rlhf.py
```

| 文件 | 作用 |
|---|---|
| `SKILL.md` | 轻量入口和任务路由，告诉 Agent 应该先读哪一种训练或 Agentic RL 能力说明 |
| `references/doc-index.md` | PyTRIO 官方 Markdown 文档和本地示例索引 |
| `references/sft.md` | SFT 数据构造、assistant-only loss mask、同步/异步训练模式 |
| `references/decision-model.md` | Decision Model / PyTRIO-Jev、marker 稀疏监督、候选概率推理、温度校准与评测 |
| `references/vision.md` | PyTRIO 0.2.7+ 图文 chunks、`ImageChunk.expected_tokens`、多模态 SFT 与 Vision GRPO 对齐 |
| `references/grpo.md` | GRPO rollout、reward、group-relative advantage、`importance_sampling` 训练模式 |
| `references/opd.md` | OPD student rollout、teacher logprob、reverse-KL advantage、Medical OPD 与多 Teacher 调度 |
| `references/search-r1.md` | Search-R1 多轮工具状态机、结果 reward、observation mask、长轨迹拆批 |
| `references/retool.md` | ReTool 代码解释器轨迹、token 连续性、outcome reward 与本地执行隔离 |
| `references/opsd.md` | OPSD 同模型 Student / privileged Teacher、sampled-token reverse-KL |
| `references/dapo.md` | DAPO Dynamic Sampling、Clip-Higher、token mean、Soft Overlong 与成本口径 |
| `references/gspo.md` | GSPO 序列级重要性比率、序列级裁剪、退化组归一化 |
| `references/alfworld.md` | ALFWorld / TextWorld 独立环境、长轨迹 observation mask、PPO 与评测边界 |
| `references/custom-loss.md` | `forward_backward_custom`、闭包元数据、本地 autograd、异步调用与 surrogate gradient |
| `references/dpo.md` | DPO chosen/rejected 偏好训练、reference logprob、pairwise loss |
| `references/chat-huanhuan.md` | Chat-甄嬛案例、同步/异步 SFT、SwanLab 记录模式 |
| `examples/quickstart_sft.py` | 最小 SFT 训练、保存权重、推理示例 |
| `examples/chat-huanhuan.py` | 带 SwanLab 记录的同步 SFT 示例 |
| `examples/chat-huanhuan-async.py` | 带 SwanLab 记录的异步 SFT 示例 |
| `examples/sft-distill-conversation.py` | 多轮对话 SFT 蒸馏示例 |
| `examples/sft-distill-conversation-async.py` | 异步多轮对话 SFT 蒸馏示例 |
| `examples/grpo-gsm8k.py` | 同步 GRPO / GSM8K 示例 |
| `examples/grpo-gsm8k-async.py` | 异步 GRPO / GSM8K 示例 |
| `examples/vision-grpo.py` | GeoQA 多模态 Vision GRPO 单文件训练示例 |
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

### Decision Model / PyTRIO-Jev

[`references/decision-model.md`](skills/pytrio-skill/references/decision-model.md) 说明固定候选分类决策，适合意图识别、工单路由和分级评分等任务：

1. 用 `state + questions + targets` 描述数据，统一映射 `choice / noul / score` 的候选符号
2. 训练和推理共用模板与 marker 定位，输入保留 JSON 骨架，真实答案仅进入训练 target
3. 使用 `cross_entropy`，只在 marker 前一位置设置非零 loss weight
4. 从 prompt logprobs 读取候选分布，说明 top-k 缺失候选的近似边界
5. 在独立 calibration 划分上拟合温度，分别评测 Accuracy、ECE 和 Brier

数据、模板、异步训练、推理和评测保留完整多文件结构，读取[最新源码](https://github.com/KMnO4-zx/agentic-rl-lab/tree/main/09-pytrio-jev)或[核对版本](https://github.com/KMnO4-zx/agentic-rl-lab/tree/d2ba99ce6964f0525c0c0ab25dcefed50ee50e6b/09-pytrio-jev)。

### 多模态与 Vision GRPO

`references/vision.md` 对齐 PyTRIO 0.2.7 多模态指南与 GeoQA Vision GRPO 完整案例：

1. 按模型 chat template 组织 `EncodedTextChunk + ImageChunk + EncodedTextChunk`
2. 使用匹配远端模型的 image processor 计算 `ImageChunk.expected_tokens`
3. 在多模态 SFT 中对齐 `target_tokens` 与 `weights`，只训练 completion
4. 在 Vision GRPO 中对齐 prompt/image 的零 target、old logprob 与 advantage
5. 校验本地 `ModelInput` 长度、异步返回边界和固定评测集结论

`examples/vision-grpo.py` 基于完整项目的 `train.py`，从 Hugging Face 缓存直接加载 GeoQA train，适合生成或修改训练代码。固定 test、评测和分析仍读取[最新源码](https://github.com/KMnO4-zx/agentic-rl-lab/tree/main/09-vision-grpo)或[核对版本](https://github.com/KMnO4-zx/agentic-rl-lab/tree/cc59b115/09-vision-grpo)。

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

`references/opd.md` 还覆盖 `agentic-rl-lab/02-opd` 中的 Medical SFT → OPD、多 Teacher SAR/IDT 调度、`sampler_weights` 与 `training_state` 边界，以及 held-out 医疗评测限制。

### Search-R1

`references/search-r1.md` 对齐官方多文件案例，重点说明：

1. 同题首轮共享 prompt、搜索后轨迹分叉的多轮工具状态机
2. 只根据最终答案计算 reward，再在完整同题 group 内计算 advantage
3. tool observation 进入上下文，但使用零 old logprob 和零 advantage 排除在 loss 之外
4. 长轨迹先构造完整 Datum，再拆 micro-batch 累积梯度并只做一次 optimizer step
5. 搜索后端保持固定，训练的是模型 LoRA 的工具使用与回答策略

### ReTool

`references/retool.md` 对齐代码解释器 Agent 案例：

1. 使用模型原生 tool-call 协议组织 code-interlaced rollout
2. 延续真实 sampler token 前缀，避免重编码破坏 old logprob 对齐
3. 只给最终答案 outcome reward，代码执行结果进入上下文但不参与 loss
4. 按完整 logical batch 累积梯度，并处理不同 micro-batch 的权重
5. 明确本地 subprocess 的安全边界；运行不可信代码时使用容器或专用沙箱

### OPSD

`references/opsd.md` 对齐官方 On-Policy Self-Distillation 案例：

1. Student 只看问题并生成当前策略 completion
2. 同一个初始模型作为固定 Teacher，通过 privileged prompt 额外看到参考解答
3. Teacher 不重新采样，只对 Student completion 调 `compute_logprobs`
4. 用 sampled-token reverse KL 构造逐 token advantage
5. 参考解答只改变 Teacher 条件分布，不是 Student 的 SFT label

### DAPO

`references/dapo.md` 说明同一训练骨架中的 GRPO / DAPO 机制对照：

1. Dynamic Sampling 按原始 correctness 筛选有效组并限制补采预算
2. Clip-Higher 使用非对称 PPO clip
3. Token-level Policy Gradient Loss 按全部 completion token 归一化
4. Soft Overlong 在长度上限前平滑加入惩罚
5. 候选 token、有效训练 token、refill 等待时间分别记账

### GSPO

`references/gspo.md` 说明 sequence-level policy optimization：

1. 对 completion token 概率比取几何平均，得到一条序列共享的 ratio
2. 对完整 sequence ratio 做一次窄范围裁剪
3. prompt token 不进入 ratio，退化组可以跳过远端计算
4. loss 仍使用原始 rollout batch 分母，避免过滤零目标后放大梯度
5. 将格式、长度和准确率结果分开解释，并保留 matched GRPO 对照边界

### ALFWorld

`references/alfworld.md` 说明 text-only 环境中的长轨迹 Agentic RL：

1. 同一游戏创建多个独立环境实例，保证 group 任务一致且状态互不污染
2. assistant action token 参与 PPO，环境 observation token 只进入上下文
3. 使用终局成功与非法动作惩罚计算同游戏 group-relative advantage
4. 分别统计 prefill、sampling 与 training token 成本
5. Base 与 checkpoint 固定 seen/unseen split、环境预算和评测协议

### Custom Loss

`references/custom-loss.md` 对齐[官网自定义损失函数文档](https://docs.pytrio.com/docs/guide/custom_loss)：

1. `loss_fn(data, logprobs)` 在本地 PyTorch 进程执行并返回 `(loss, metrics)`
2. `Datum.loss_fn_inputs` 只提供与 `model_input` 等长的 `target_tokens`
3. sampling/reference logprob、mask、advantage 和分组信息通过每个 batch 独立的闭包传入
4. 本地 autograd 计算 `dL / dlogprob`，服务端用 surrogate objective 完成模型反向传播
5. 优先使用 `forward_backward_custom_async()`，并正确等待提交与远程完成两层 future

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
- Decision Model：按监督位置归一化的 `loss`、`supervised_tokens`、`trainer/*`；独立评测记录 Accuracy、ECE、Brier、温度和 checkpoint
- Vision GRPO：`reward`、`format_rate`、`degenerate_fraction`、`train_datums`、completion token 指标
- GRPO：`reward`、`frac_degenerate`、`datums`
- OPD：`opd/reverse_kl_mean`、`opd/reverse_kl_std`、completion token 指标
- Search-R1：`reward/correct`、`reward/format`、`rollout/turns`、`search/success_rate`、`search/error_rate`
- ReTool：`reward/correct`、`rollout/code_calls`、`sandbox/success_rate`、`sandbox/timeout_rate`
- OPSD：`trainer/loss_mean`、`opd/reverse_kl_mean`、`opd/reverse_kl_std`、`opd/advantage_mean`
- DAPO：`candidate_groups`、`effective_groups`、`oversample_ratio`、`ppo/clip_fraction`、候选/训练 token
- GSPO：`gspo/seq_ratio_mean`、`gspo/sequence_clip_fraction`、`gspo/normalization_sequences`、有效序列比例
- ALFWorld：`reward/success_rate`、`rollout/steps_mean`、`rollout/invalid_actions_mean`、prefill/sample/train token
- Custom loss：算法 loss/ratio/clip 指标、有效 token、归一化分母、远端与本地计算时间
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

生成的 zip 压缩包会解压成：

```text
SKILL.md
references/
examples/
```

因此可以直接解压到 `.claude/skills/pytrio-skill/`。

### 自动发布

PR 合并进入 `main` 后，[Release workflow](.github/workflows/release-skill.yml) 会确认该次 `main` 更新确实来自已合并的 PR，然后：

1. 从 `skills/pytrio-skill/SKILL.md` 的 `metadata.version` 读取版本号，并校验 README 版本徽章。
2. 调用 `make package` 生成 zip。
3. 校验压缩包结构并排除 Python 缓存文件。
4. 创建 `v<metadata.version>` GitHub Release，上传一个带版本号的 Skill zip 压缩包。

每个需要发布的 PR 都要先更新 `metadata.version` 和 README 版本徽章。对应 tag 已存在时 workflow 会直接失败，避免覆盖已经发布的版本。直接 push 到 `main` 不会发版；需要补发时也可以在 GitHub Actions 页面手动运行这个 workflow。

## 维护原则

- Skill 内部保持精简，不复制整站文档。
- PyTRIO API 细节以官方 Markdown 文档为准。
- 单文件案例优先放到 `skills/pytrio-skill/examples/`；包含数据、环境、训练和评测的多文件案例保留完整源码目录，并在 `references/` 说明关键实现边界。
- 入口文档保持能力导向，优先保证 Agent 能按任务路由到对应训练、Agentic RL 或 custom loss 说明。
- 如果涉及实验记录和指标查询，优先配合 SwanLab Skill 使用。
