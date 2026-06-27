<div align="center">

<a href="https://pytrio.cn/">
  <img src="images/TRIO_LOGO.svg" alt="TRIO" width="360" />
</a>

<h1><a href="https://pytrio.cn/">PyTRIO.skill</a></h1>


> 让 Agent 正确使用 PyTRIO 写训练、推理和实验记录代码。

[![PyTRIO 官网](https://img.shields.io/badge/PyTRIO-官网-fc4547)](https://pytrio.cn/)
[![PyTRIO Docs](https://img.shields.io/badge/PyTRIO-官方文档-ff6b57)](https://docs.pytrio.cn/docs)
[![ModelScope Skill](https://img.shields.io/badge/ModelScope-Skill-624aff)](https://www.modelscope.cn/skills/SwanLab/pytrio-skill/summary)
[![Agent Skill](https://img.shields.io/badge/Agent-Skill-7c3aed)](skills/pytrio-skill/SKILL.md)
[![SwanLab](https://img.shields.io/badge/SwanLab-推荐记录工具-1f8f4c)](https://docs.swanlab.cn)
[![License](https://img.shields.io/badge/License-MIT-d4a017)](LICENSE)

没有本地 GPU？没关系。  
PyTRIO 把 LLM 后训练丢到云端执行，你只需要写数据、算法和训练循环。

这个 Skill 会让 Claude Code、Codex 等 Agent 先读取 PyTRIO 官方 Markdown 文档，再参考内置示例生成代码，避免把 PyTorch / HuggingFace 的习惯误套到 PyTRIO 上。

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

手动安装发布包：

```bash
mkdir -p .claude/skills/pytrio-skill
curl -sL https://github.com/SwanHubX/pytrio-skill/releases/latest/download/pytrio-skill.tar.gz | tar xz -C .claude/skills/pytrio-skill/
```

本地开发时也可以直接复制：

```bash
mkdir -p .claude/skills/pytrio-skill
cp -R skills/pytrio-skill/. .claude/skills/pytrio-skill/
```

## 内容

```text
skills/
└── pytrio-skill/
    ├── SKILL.md
    ├── references/
    │   ├── doc-index.md
    │   └── chat-huanhuan.md
    └── examples/
        ├── quickstart_sft.py
        ├── chat-huanhuan.py
        └── chat-huanhuan-async.py
```

| 文件 | 作用 |
|---|---|
| `SKILL.md` | 轻量入口和任务路由，告诉 Agent 应该先读哪些官方文档 |
| `references/doc-index.md` | PyTRIO 官方 Markdown 文档索引 |
| `references/chat-huanhuan.md` | Chat-甄嬛案例、同步/异步 SFT、SwanLab 记录模式 |
| `examples/quickstart_sft.py` | 最小 SFT 训练、保存权重、推理示例 |
| `examples/chat-huanhuan.py` | 带 SwanLab 记录的同步 SFT 示例 |
| `examples/chat-huanhuan-async.py` | 带 SwanLab 记录的异步 SFT 示例 |

## 示例

### 最小 SFT

`examples/quickstart_sft.py` 展示最小闭环：

1. 创建 `trio.ServiceClient`
2. 创建 LoRA `TrainingClient`
3. 构造带 prompt mask 的 `trio.Datum`
4. 调用 `forward_backward` 和 `optim_step`
5. 保存推理权重
6. 创建 sampling client 做推理

### Chat-甄嬛

`examples/chat-huanhuan.py` 是同步版，适合先理解完整 SFT 流程。  
`examples/chat-huanhuan-async.py` 是异步版，适合参考异步提交 batch、异步计算 loss、异步记录 SwanLab 的写法。

## SwanLab 记录

SwanLab 是 PyTRIO 训练过程中不可或缺的实验记录工具，也是 PyTRIO 的好搭档。建议训练环境默认安装：

```bash
pip install swanlab
```

训练脚本建议记录：

- `loss`
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
- 新案例优先放到 `skills/pytrio-skill/examples/`。
- 如果涉及实验记录和指标查询，优先配合 SwanLab Skill 使用。
