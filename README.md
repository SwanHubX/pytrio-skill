<div align="center">

# PyTRIO.skill

> 没有本地 GPU？没关系。把训练丢到云端，你只管写代码。

[![PyTRIO](https://img.shields.io/badge/PyTRIO-0.1.12-blue)](https://pypi.org/project/pytrio/0.1.12/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![Claude Code](https://img.shields.io/badge/Claude%20Code-Skill-blueviolet)](https://claude.ai/code)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

<br>

PyTRIO 是个全新的 SDK，文档还没跟上？<br>
API 签名和 PyTorch / HuggingFace 长得像但处处不同？<br>
Claude 写出来的代码跑不通，因为它根本不知道这个库？<br>

**装上这个 Skill，Claude Code 就能正确使用 PyTRIO 写训练和推理代码。**

<br>

已验证：Claude Opus / Haiku / Kimi 均能基于此 Skill **一次性**写出正确的训练+推理代码。

[安装](#安装) · [它解决什么问题](#它解决什么问题) · [Skill 内容](#skill-内容) · [自动化](#自动化) · [详细安装说明](installation.md)

</div>

---

## 安装

在 Claude Code 中发送：

```
Fetch the installation guide and follow it: https://raw.githubusercontent.com/SwanHubX/pytrio-skill/master/installation.md
```

或手动安装（最新版）：

```bash
mkdir -p .claude/skills/pytrio-skill
curl -sL https://github.com/SwanHubX/pytrio-skill/releases/latest/download/pytrio-skill.tar.gz | tar xz -C .claude/skills/pytrio-skill/
```

安装指定版本（版本号与 PyTRIO SDK 对齐）：

```bash
VERSION="0.1.12"
mkdir -p .claude/skills/pytrio-skill
curl -sL "https://github.com/SwanHubX/pytrio-skill/releases/download/v${VERSION}/pytrio-skill.tar.gz" | tar xz -C .claude/skills/pytrio-skill/
```

---

## 它解决什么问题

[PyTRIO](https://pypi.org/project/pytrio/) 把大模型的前向传播、反向传播、优化器步骤和推理采样全部委托到远端 GPU 集群执行。本地不需要 GPU，只需要网络。

但它太新了 —— AI 编码助手不了解它的 API，写出来的代码全是错的：

| 没有 Skill 时 AI 会写的 | 实际正确写法 |
|---|---|
| `model_input=[1, 2, 3]` | `model_input=ModelInput.from_ints([1, 2, 3])` |
| `max_new_tokens=50` | `max_tokens=50` |
| `result.samples[0]` | `result.sequences[0].text` |
| `forward_result.loss` | `fb_result.metrics.get("loss:sum")` |
| `Qwen/Qwen3-4B` | `Qwen/Qwen3-4B-Instruct-2507` |

**这个 Skill 就是 PyTRIO 的「说明书」**，装上之后 Claude Code 知道每个接口怎么调、有什么坑、最佳实践是什么。

---

## Skill 内容

```
.claude/skills/
├── SKILL.md              — 入口速查（API、PyTorch 概念映射、17 条陷阱清单）
├── references/           — 完整 API 参考
│   ├── 00-overview.md        架构总览 + 环境搭建
│   ├── 01-service-client.md  入口：认证、创建训练/推理客户端
│   ├── 02-training-client.md 训练：forward_backward / optim_step / save
│   ├── 03-sampling-client.md 推理：sample / compute_logprobs
│   ├── 04-rest-client.md     管理：权重列表 / checkpoint / 下载
│   └── 05-data-types.md      数据类型：Datum / ModelInput / TensorData / ...
├── examples/             — 5 个最小可运行示例
│   ├── 01_train_sft.py       SFT 训练
│   ├── 02_inference.py       推理
│   ├── 03_checkpoint_resume.py 断点续训
│   ├── 04_model_management.py  模型管理
│   └── 05_importance_sampling.py GRPO 风格训练
└── best-practices/       — 场景最佳实践
    ├── sft.md                Prompt masking / EOS 追加 / Tokenizer 用法
    └── grpo.md               GRPO/PPO 数据构造 / 自定义 loss
```

---

## 自动化

本项目附带了自动化测试和更新的脚本：

```bash
# 测试 Skill 效果（并行 haiku + kimi + gemini + opencode）
PYTRIO_API_KEY=xxx ./scripts/test-skill.sh

# SDK 更新后自动修改 Skill — 从上游仓库拉取最新代码对比
./scripts/update-skill.sh --upstream --test

# 或手动指定新旧 SDK 目录
./scripts/update-skill.sh --old pytrio-old/ --new pytrio/ --test

# 打包为 zip 分发
./scripts/pack.sh
```

### 测试逻辑

1. 在 `/tmp` 创建隔离环境，安装 Skill + uv 环境 + `trio login`
2. 给 Haiku / Kimi / Gemini / OpenCode 一句简短的任务（模拟真实用户）
3. AI 仅凭 Skill 写代码、运行、输出 `result.json`
4. 产物回传主 session 审查，找出 Skill 缺陷

### 上游更新

`--upstream` 模式会自动从上游仓库拉取最新 SDK 代码并 diff。上游仓库地址在 `scripts/update-skill.sh` 中配置。

---

## 适用版本

| 组件 | 版本 |
|---|---|
| PyTRIO SDK | `0.1.12` |
| 可用模型 | `Qwen/Qwen3-4B-Instruct-2507` |
| Python | `>=3.10` |

---

<div align="center">

MIT License

</div>
