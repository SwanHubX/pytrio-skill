# PyTRIO 官方文档索引

写 PyTRIO/TRIO 代码时，先看本地 examples 能不能解决问题；示例覆盖不了或需要确认 API 细节时，再读取官方文档。

## 本地示例代码

后续新增示例时，优先在这里补充索引，方便 Agent 按任务快速找到可参考代码。

| 示例 | 适用场景 | 重点参考 |
|---|---|---|
| `examples/quickstart_sft.py` | 最小 SFT、第一次跑通训练、保存权重后推理 | `Datum` 构造、prompt masking、`forward_backward`、`optim_step`、`save_weights_for_sampler` |
| `examples/chat-huanhuan.py` | 真实角色微调、同步 SFT、SwanLab 记录 | JSON 数据集处理、chat template、逐 batch 记录 loss、训练前后推理对比 |
| `examples/chat-huanhuan-async.py` | 异步 SFT、异步提交 batch、异步记录 SwanLab | `forward_backward_async`、`optim_step_async`、`asyncio.create_task`、后台 loss 计算和日志记录 |
| `examples/sft-distill-conversation.py` | 多轮对话 SFT 蒸馏、conversation-mask | assistant-only loss mask、reasoning 数据过滤、多轮 ChatML 拼接 |
| `examples/sft-distill-conversation-async.py` | 异步多轮对话 SFT 蒸馏 | 异步提交 batch、后台 loss 记录、保存权重后采样对比 |
| `examples/grpo-gsm8k.py` | 同步 GRPO / GSM8K / reward-based RLVR | group rollout、reward、group-relative advantage、`importance_sampling` |
| `examples/grpo-gsm8k-async.py` | 异步 GRPO / GSM8K | 并发 prompt rollout、异步 `importance_sampling` 训练 |
| `examples/opd-deepmath.py` | 同步 OPD / DeepMath / teacher-KL 蒸馏 | student rollout、teacher `compute_logprobs`、reverse-KL advantage |
| `examples/opd-deepmath-async.py` | 异步 OPD / DeepMath | 异步 student 采样、异步 teacher logprob、异步训练更新 |
| `examples/dpo-hh-rlhf.py` | DPO / HH-RLHF / custom loss | chosen/rejected pair、reference logprob、`forward_backward_custom` |

## 能力说明

这些文件是 Agent 写代码前的主入口。示例代码只作为模板，字段语义以这里为准。

| 能力 | 本地说明 |
|---|---|
| SFT | `references/sft.md` |
| GRPO | `references/grpo.md` |
| OPD | `references/opd.md` |
| Search-R1 / Agentic RL | `references/search-r1.md` |
| OPSD / On-Policy Self-Distillation | `references/opsd.md` |
| DPO / custom loss | `references/dpo.md` |
| Chat-甄嬛案例 | `references/chat-huanhuan.md` |

## 阅读建议

- 写 SFT 时，先读 `references/sft.md`，再按场景选择 `quickstart_sft.py`、`chat-huanhuan.py` 或 `sft-distill-conversation.py`。
- 写 GRPO 时，先读 `references/grpo.md`，再参考 `grpo-gsm8k.py` 或异步版。
- 写 OPD 时，先读 `references/opd.md`，再参考 `opd-deepmath.py` 或异步版。
- 写 Search-R1 或多轮搜索工具训练时，先读 `references/search-r1.md`，需要完整项目时再读取其中的官方源码目录。
- 写 OPSD 或 privileged self-distillation 时，先读 `references/opsd.md`，需要完整项目时再读取其中的官方源码目录。
- 写 DPO 或 custom loss 时，先读 `references/dpo.md`，再参考 `dpo-hh-rlhf.py`。
- 写推理时，读取推理、SamplingClient、SamplingParams 和 ModelInput。
- 接入数据集时，读取 HuggingFace datasets 和训练文档。
- 做角色微调、同步/异步 SFT 或 SwanLab 训练记录时，先参考 `examples/chat-huanhuan.py` 与 `examples/chat-huanhuan-async.py`，再读取 Chat-甄嬛说明。
- 训练后要用 OpenAI SDK 部署时，先保存权重，再读取 OpenAI 兼容 API。
- 如果某个页面 404 或看起来过期，打开可视化文档页面，再根据当前导航推导 Markdown 路径。

## URL 规则

大多数文档页面都可以把可视化文档 URL 转成 Markdown 读取：

```text
https://docs.pytrio.cn/docs/<route>
https://docs.pytrio.cn/docs/content/<route>/content.md
```

根页面是：

```text
https://docs.pytrio.cn/docs/content/content.md
```

示例：

```text
https://docs.pytrio.cn/docs/guide/train
https://docs.pytrio.cn/docs/content/guide/train/content.md
```

## 核心页面

| 主题 | Markdown 链接 |
|---|---|
| 什么是 TRIO | https://docs.pytrio.cn/docs/content/content.md |
| 快速开始 | https://docs.pytrio.cn/docs/content/quick-start/content.md |
| 训练 | https://docs.pytrio.cn/docs/content/guide/train/content.md |
| 推理/采样 | https://docs.pytrio.cn/docs/content/guide/sample/content.md |
| 计算 logprobs | https://docs.pytrio.cn/docs/content/advanced/compute_logprobs/content.md |
| 保存权重与继续训练 | https://docs.pytrio.cn/docs/content/guide/resume/content.md |
| 下载权重 | https://docs.pytrio.cn/docs/content/guide/download/content.md |
| 损失函数 | https://docs.pytrio.cn/docs/content/guide/loss_fn/content.md |
| 异步 | https://docs.pytrio.cn/docs/content/guide/async/content.md |
| HuggingFace datasets | https://docs.pytrio.cn/docs/content/advanced/datasets/content.md |
| OpenAI 兼容 API | https://docs.pytrio.cn/docs/content/advanced/openai/content.md |
| 模型列表 | https://docs.pytrio.cn/docs/content/models/content.md |
| 合作与交流 | https://docs.pytrio.cn/docs/content/communication/content.md |

## API 页面

| API | Markdown 链接 |
|---|---|
| `trio.ServiceClient` | https://docs.pytrio.cn/docs/content/api/ServiceClient/content.md |
| `trio.TrainingClient` | https://docs.pytrio.cn/docs/content/api/TrainingClient/content.md |
| `trio.SamplingClient` | https://docs.pytrio.cn/docs/content/api/SamplingClient/content.md |
| `trio.RestClient` | https://docs.pytrio.cn/docs/content/api/RestClient/content.md |
| `trio.Datum` | https://docs.pytrio.cn/docs/content/api/Datum/content.md |
| `trio.ModelInput` | https://docs.pytrio.cn/docs/content/api/ModelInput/content.md |
| `trio.AdamParams` | https://docs.pytrio.cn/docs/content/api/AdamParams/content.md |
| `trio.SamplingParams` | https://docs.pytrio.cn/docs/content/api/SamplingParams/content.md |

## 案例

| 案例 | Markdown 链接 |
|---|---|
| Chat-甄嬛 | https://docs.pytrio.cn/docs/content/example/chat_huanhuan/content.md |
| GSM8K | https://docs.pytrio.cn/docs/content/example/gsm8k/content.md |
| GRPO | https://docs.pytrio.cn/docs/content/example/grpo/content.md |
| On-Policy Distillation | https://docs.pytrio.cn/docs/content/example/opd/content.md |
| Search-R1 | https://docs.pytrio.cn/docs/content/example/search-r1/content.md |
| On-Policy Self-Distillation | https://docs.pytrio.cn/docs/content/example/opsd/content.md |
| DPO | https://docs.pytrio.cn/docs/content/example/dpo/content.md |
