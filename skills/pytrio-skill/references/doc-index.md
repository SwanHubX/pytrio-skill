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
| `examples/vision-grpo.py` | Vision GRPO / GeoQA / 多模态 RLVR | `ImageChunk.expected_tokens`、图文 chunks、异步 group rollout、prompt/image mask |
| `examples/opd-deepmath.py` | 同步 OPD / DeepMath / teacher-KL 蒸馏 | student rollout、teacher `compute_logprobs`、reverse-KL advantage |
| `examples/opd-deepmath-async.py` | 异步 OPD / DeepMath | 异步 student 采样、异步 teacher logprob、异步训练更新 |
| `examples/dpo-hh-rlhf.py` | DPO / HH-RLHF / custom loss | chosen/rejected pair、reference logprob、`forward_backward_custom` |

## 能力说明

这些文件是 Agent 写代码前的主入口。示例代码只作为模板，字段语义以这里为准。

| 能力 | 本地说明 |
|---|---|
| SFT | `references/sft.md` |
| Decision Model / PyTRIO-Jev / 固定候选决策 | `references/decision-model.md` |
| 多模态推理 / 多模态 SFT / Vision GRPO | `references/vision.md` |
| GRPO | `references/grpo.md` |
| OPD | `references/opd.md` |
| Search-R1 / Agentic RL | `references/search-r1.md` |
| ReTool / 代码解释器 Agent | `references/retool.md` |
| OPSD / On-Policy Self-Distillation | `references/opsd.md` |
| DAPO / Dynamic Sampling | `references/dapo.md` |
| GSPO / sequence-level policy optimization | `references/gspo.md` |
| ALFWorld / TextWorld Agentic RL | `references/alfworld.md` |
| Custom loss / `forward_backward_custom` | `references/custom-loss.md` |
| DPO / preference training | `references/dpo.md` |
| Chat-甄嬛案例 | `references/chat-huanhuan.md` |

## 阅读建议

- 写 SFT 时，先读 `references/sft.md`，再按场景选择 `quickstart_sft.py`、`chat-huanhuan.py` 或 `sft-distill-conversation.py`。
- 写 Decision Model、固定候选分类或工单路由训练时，先读 `references/decision-model.md`，再读取完整 PyTRIO-Jev 案例的 `data.py`、共享模板和 `train.py`。
- 做决策概率推理、marker 对齐调试或温度校准时，按 `references/decision-model.md` 核对训练的 `m - 1` 与 prompt logprob 的 `m`，再读取 `inference.py`、`calibrate.py`、`eval.py`；API 细节见 SamplingClient 和 SamplingParams。
- 写图像输入或多模态 SFT 时，先确认 `pytrio>=0.2.7`，再读 `references/vision.md`。
- 写 Vision GRPO 时，先读 `references/vision.md`，再参考 `examples/vision-grpo.py`；固定测试集与评测按其中的完整 GeoQA 项目核对。
- 写 GRPO 时，先读 `references/grpo.md`，再参考 `grpo-gsm8k.py` 或异步版。
- 写 OPD 时，先读 `references/opd.md`，再参考 `opd-deepmath.py` 或异步版；Medical OPD 和多 Teacher 实验按其中的完整案例组织。
- 写 Search-R1 或多轮搜索工具训练时，先读 `references/search-r1.md`，需要完整项目时再读取其中的源码目录。
- 写 ReTool、代码解释器 Agent 或 code-interlaced RL 时，先读 `references/retool.md`，并先确认执行隔离方案。
- 写 OPSD 或 privileged self-distillation 时，先读 `references/opsd.md`，需要完整项目时再读取其中的源码目录。
- 写 DAPO 时，先读 `references/dapo.md`，同时核对 Dynamic Sampling、token mean、Clip-Higher 和 Soft Overlong。
- 写 GSPO 时，先读 `references/gspo.md`，重点核对 sequence ratio、序列级裁剪和原始 batch 分母。
- 写 ALFWorld / TextWorld 长轨迹训练时，先读 `references/alfworld.md`，核对同游戏独立环境、observation mask 和环境清理。
- 写通用 custom loss 时，先读 `references/custom-loss.md`，核对函数签名、闭包元数据和 async 的两次 `await`。
- 写 DPO 时，先读 `references/custom-loss.md` 和 `references/dpo.md`，再参考 `dpo-hh-rlhf.py`。
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
| 多模态 | https://docs.pytrio.com/docs/content/guide/vision/content.md |
| 计算 logprobs | https://docs.pytrio.cn/docs/content/advanced/compute_logprobs/content.md |
| 保存权重与继续训练 | https://docs.pytrio.cn/docs/content/guide/resume/content.md |
| 下载权重 | https://docs.pytrio.cn/docs/content/guide/download/content.md |
| 损失函数 | https://docs.pytrio.cn/docs/content/guide/loss_fn/content.md |
| 自定义损失函数 | https://docs.pytrio.com/docs/content/guide/custom_loss/content.md |
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
| Vision GRPO | https://docs.pytrio.com/docs/content/example/vision-grpo/content.md |
| On-Policy Distillation | https://docs.pytrio.cn/docs/content/example/opd/content.md |
| Search-R1 | https://docs.pytrio.cn/docs/content/example/search-r1/content.md |
| On-Policy Self-Distillation | https://docs.pytrio.cn/docs/content/example/opsd/content.md |
| DPO | https://docs.pytrio.cn/docs/content/example/dpo/content.md |

## 多文件完整案例

以下多文件案例来自 `agentic-rl-lab`，包含数据、环境、训练、评测与分析代码。读取最新实现时使用 `main` 链接；复核字段和实验结论时使用表格中的实际核对版本。

| 案例 | 最新源码 | 本 Skill 核对版本 |
|---|---|---|
| Medical OPD / SAR-OPD / IDT-OPD | https://github.com/KMnO4-zx/agentic-rl-lab/tree/main/02-opd | https://github.com/KMnO4-zx/agentic-rl-lab/tree/906d54e/02-opd |
| Search-R1 | https://github.com/KMnO4-zx/agentic-rl-lab/tree/main/03-search-r1 | https://github.com/KMnO4-zx/agentic-rl-lab/tree/906d54e/03-search-r1 |
| OPSD | https://github.com/KMnO4-zx/agentic-rl-lab/tree/main/04-opsd | https://github.com/KMnO4-zx/agentic-rl-lab/tree/906d54e/04-opsd |
| ReTool | https://github.com/KMnO4-zx/agentic-rl-lab/tree/main/05-retool | https://github.com/KMnO4-zx/agentic-rl-lab/tree/906d54e/05-retool |
| DAPO | https://github.com/KMnO4-zx/agentic-rl-lab/tree/main/06-dapo | https://github.com/KMnO4-zx/agentic-rl-lab/tree/906d54e/06-dapo |
| GSPO | https://github.com/KMnO4-zx/agentic-rl-lab/tree/main/07-gspo | https://github.com/KMnO4-zx/agentic-rl-lab/tree/906d54e/07-gspo |
| ALFWorld Agentic RL | https://github.com/KMnO4-zx/agentic-rl-lab/tree/main/08-alfworld | https://github.com/KMnO4-zx/agentic-rl-lab/tree/906d54e/08-alfworld |
| Vision GRPO / GeoQA | https://github.com/KMnO4-zx/agentic-rl-lab/tree/main/09-vision-grpo | https://github.com/KMnO4-zx/agentic-rl-lab/tree/cc59b115/09-vision-grpo |
| Decision Model / PyTRIO-Jev | https://github.com/KMnO4-zx/agentic-rl-lab/tree/main/09-pytrio-jev | https://github.com/KMnO4-zx/agentic-rl-lab/tree/d2ba99ce6964f0525c0c0ab25dcefed50ee50e6b/09-pytrio-jev |
