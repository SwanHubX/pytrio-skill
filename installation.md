# PyTRIO Skill Installation Guide

Install the PyTRIO Skill into the current project so that Claude Code (or other coding agents) can correctly use the PyTRIO SDK for remote LLM training and inference.

## Steps

### 1. Detect Environment

Before installing, check the current project state:

```bash
# Check package manager
if command -v uv &>/dev/null && [ -f pyproject.toml ]; then
  echo "Package manager: uv"
  echo "Python version: $(uv run python --version 2>/dev/null || echo 'not configured')"
elif command -v pip &>/dev/null; then
  echo "Package manager: pip"
  echo "Python version: $(python --version 2>/dev/null || python3 --version)"
else
  echo "No package manager found. Install uv: https://docs.astral.sh/uv/"
fi

# Check if pytrio is already installed
uv run python -c "import pytrio; print('pytrio version:', pytrio.config.settings.version)" 2>/dev/null \
  || pip show pytrio 2>/dev/null \
  || echo "pytrio is NOT installed"

# Check if Skill is already installed
if [ -f .claude/skills/pytrio-skill/SKILL.md ]; then
  echo "Skill: already installed"
else
  echo "Skill: not installed"
fi
```

### 2. Install Skill Files

Download from GitHub Release and install to `.claude/skills/pytrio-skill/`:

```bash
mkdir -p .claude/skills/pytrio-skill
curl -sL https://github.com/SwanHubX/pytrio-skill/releases/latest/download/pytrio-skill.tar.gz | tar xz -C .claude/skills/pytrio-skill/
```

To install a specific version (aligned with PyTRIO SDK version):

```bash
VERSION="0.1.13b0"
mkdir -p .claude/skills/pytrio-skill
curl -sL "https://github.com/SwanHubX/pytrio-skill/releases/download/v${VERSION}/pytrio-skill.tar.gz" | tar xz -C .claude/skills/pytrio-skill/
```

Expected structure after installation:

```
.claude/skills/
└── pytrio-skill/
    ├── SKILL.md
    ├── references/
    ├── examples/
    └── best-practices/
```

### 3. Install PyTRIO SDK (if not already installed)

For uv projects:

```bash
uv add pytrio "transformers>=4.40.0" "torch>=2.0.0"
```

For pip:

```bash
pip install pytrio transformers torch
```

### 4. PyTRIO Authentication

Ask the user for their API Key (available at https://pytrio.cn/dashboard), then run:

```bash
trio login -k <API_KEY>
```

### 5. Verify

```bash
ls .claude/skills/pytrio-skill/SKILL.md && echo 'Skill: OK'
uv run python -c "from pytrio import ServiceClient; print('PyTRIO: OK')" 2>/dev/null \
  || python -c "from pytrio import ServiceClient; print('PyTRIO: OK')"
```

## Getting Started

Installation is complete. Now help the user get started by asking them the following questions, one at a time. Adapt your language to match the user's (Chinese or English). Be conversational, not robotic.

**Ask the user:**

1. **What do you want to do?**
   - Fine-tune a model (SFT)? Train with reinforcement learning (GRPO/PPO)? Just run inference?
   - If they're unsure, suggest starting with a simple SFT fine-tuning example.

2. **What dataset are you using?**
   - A public dataset from HuggingFace (e.g., Alpaca, ShareGPT)?
   - Their own data? If so, what format (JSON, CSV, plain text)?
   - Or just a few test samples to try things out?

3. **What do you want the model to learn?**
   - Follow instructions? Answer questions? Translate? Classify? Generate code?
   - This helps decide the prompt template and training strategy.

4. **How long should it train?**
   - Quick test: 3-5 steps, a few samples
   - Serious training: hundreds of steps, full dataset
   - If they're not sure, suggest starting with 10 steps on 20 samples to verify everything works.

**Then:**

Based on their answers, read `.claude/skills/pytrio-skill/SKILL.md` and the relevant `best-practices/` guide, then write and run the training script for them. Always start with a small test run to verify the pipeline works before scaling up.

**Example conversation:**

> User: 我想用 Alpaca 数据集微调一下模型
>
> Agent: 好的！几个问题：
> 1. 先用几条数据跑通流程，还是直接上完整数据集？
> 2. 训练完要不要自动做一次推理测试看看效果？
>
> User: 先跑 20 条试试，训练完推理一下
>
> Agent: 明白。我来写一个脚本：加载 Alpaca 前 20 条 → SFT 训练 10 步 → 保存权重 → 推理测试。
