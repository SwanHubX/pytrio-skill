# PyTRIO 官方文档索引

写 PyTRIO 代码时，先看本地 examples 能不能解决问题；示例覆盖不了或需要确认 API 细节时，再读取官方文档。

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
- 核对 SDK 版本变更、新特性或 API 调整时，读更新日志（`https://docs.pytrio.com/changelog`）。
- 如果某个页面 404 或看起来过期，先查 `https://docs.pytrio.com/llms.txt` 的页面清单，再按上面的 URL 规则推导 Markdown 路径。

## URL 规则

官方文档域名统一为 `docs.pytrio.com`（`docs.pytrio.cn` 已 301 重定向到 `.com`）。站点根路径和 `/docs` 都会跳转到「什么是 PyTRIO」。

可视化页面 URL：

```text
https://docs.pytrio.com/guide/<route>      # 指南，含 get-started / advanced / resources 子目录
https://docs.pytrio.com/api/<Name>         # API
https://docs.pytrio.com/examples/<name>    # 案例
https://docs.pytrio.com/changelog          # 更新日志（仅可视化页面，无 Markdown）
```

大多数页面可以把可视化 URL 转成 Markdown 读取：`https://docs.pytrio.com/docs/content/<md-route>/content.md`。`<md-route>` 不是页面 URL 的简单镜像，按下面对照表取：

| 可视化页面 | Markdown 路径（省略域名和 `/docs/content/` 前缀） |
|---|---|
| `/guide/get-started/what-is-pytrio` | `get-started/what-is-pytrio/content.md` |
| `/guide/train` | `guide/train/content.md` |
| `/guide/advanced/openai` | `advanced/openai/content.md` |
| `/guide/resources/prefill-cache` | `resources/prefill-cache/content.md` |
| `/api/ServiceClient` | `api/ServiceClient/content.md` |
| `/examples/grpo` | `example/grpo/content.md` |

即：`get-started/`、`advanced/`、`resources/` 子目录在 Markdown 里去掉 `guide/` 前缀；普通指南页保留 `guide/`；案例页 `examples/` 在 Markdown 里是单数 `example/`。

需要全站目录或页面 404 时，读 `https://docs.pytrio.com/llms.txt`（带描述的页面清单）或 `https://docs.pytrio.com/sitemap.xml`。

## 核心页面

### 开始使用

| 主题 | Markdown 链接 |
|---|---|
| 什么是 PyTRIO | https://docs.pytrio.com/docs/content/get-started/what-is-pytrio/content.md |
| 快速开始 | https://docs.pytrio.com/docs/content/get-started/quick-start/content.md |

### 指南

| 主题 | Markdown 链接 |
|---|---|
| 训练 | https://docs.pytrio.com/docs/content/guide/train/content.md |
| 推理/采样 | https://docs.pytrio.com/docs/content/guide/sample/content.md |
| 多模态 | https://docs.pytrio.com/docs/content/guide/vision/content.md |
| 保存权重 | https://docs.pytrio.com/docs/content/guide/save/content.md |
| 继续训练 | https://docs.pytrio.com/docs/content/guide/resume/content.md |
| 下载权重 | https://docs.pytrio.com/docs/content/guide/download/content.md |
| 损失函数 | https://docs.pytrio.com/docs/content/guide/loss_fn/content.md |
| 自定义损失函数 | https://docs.pytrio.com/docs/content/guide/custom_loss/content.md |
| 异步 | https://docs.pytrio.com/docs/content/guide/async/content.md |

### 进阶

| 主题 | Markdown 链接 |
|---|---|
| 环境变量配置 | https://docs.pytrio.com/docs/content/advanced/configuration/content.md |
| OpenAI 兼容 API | https://docs.pytrio.com/docs/content/advanced/openai/content.md |
| HuggingFace datasets | https://docs.pytrio.com/docs/content/advanced/datasets/content.md |
| 计算 logprobs | https://docs.pytrio.com/docs/content/advanced/compute_logprobs/content.md |

### 更多资源

| 主题 | 链接 |
|---|---|
| Prefill、Cache、Sample 与 Train | https://docs.pytrio.com/docs/content/resources/prefill-cache/content.md |
| 时钟周期 | https://docs.pytrio.com/docs/content/resources/clock-cycle/content.md |
| 模型列表 | https://pytrio.cn/home/pricing （外链，无 Markdown） |
| 合作与交流 | https://docs.pytrio.com/docs/content/resources/communication/content.md |
| 更新日志 | https://docs.pytrio.com/changelog （仅可视化页面，无 Markdown） |

## API 页面

| API | Markdown 链接 |
|---|---|
| `trio.ServiceClient` | https://docs.pytrio.com/docs/content/api/ServiceClient/content.md |
| `trio.TrainingClient` | https://docs.pytrio.com/docs/content/api/TrainingClient/content.md |
| `trio.SamplingClient` | https://docs.pytrio.com/docs/content/api/SamplingClient/content.md |
| `trio.RestClient` | https://docs.pytrio.com/docs/content/api/RestClient/content.md |
| `trio.Datum` | https://docs.pytrio.com/docs/content/api/Datum/content.md |
| `trio.ModelInput` | https://docs.pytrio.com/docs/content/api/ModelInput/content.md |
| `trio.AdamParams` | https://docs.pytrio.com/docs/content/api/AdamParams/content.md |
| `trio.SamplingParams` | https://docs.pytrio.com/docs/content/api/SamplingParams/content.md |

## 案例

| 案例 | Markdown 链接 |
|---|---|
| Chat-甄嬛 | https://docs.pytrio.com/docs/content/example/chat_huanhuan/content.md |
| Decision Model / PyTRIO-Jev | https://docs.pytrio.com/docs/content/example/decision-model-jev/content.md |
| GSM8K | https://docs.pytrio.com/docs/content/example/gsm8k/content.md |
| GRPO | https://docs.pytrio.com/docs/content/example/grpo/content.md |
| Vision GRPO | https://docs.pytrio.com/docs/content/example/vision-grpo/content.md |
| On-Policy Distillation | https://docs.pytrio.com/docs/content/example/opd/content.md |
| Search-R1 | https://docs.pytrio.com/docs/content/example/search-r1/content.md |
| On-Policy Self-Distillation | https://docs.pytrio.com/docs/content/example/opsd/content.md |
| DPO | https://docs.pytrio.com/docs/content/example/dpo/content.md |

## 多文件完整案例

以下多文件案例来自 `agentic-rl-lab`，包含数据、环境、训练、评测与分析代码。读取最新实现时使用 `main` 链接（位于 `experiments/` 目录）；复核字段和实验结论时使用表格中的实际核对版本（保留原目录结构）。

| 案例 | 最新源码 | 本 Skill 核对版本 |
|---|---|---|
| Medical OPD / SAR-OPD / IDT-OPD | https://github.com/datawhalechina/agentic-rl-lab/tree/main/experiments/medical-opd | https://github.com/datawhalechina/agentic-rl-lab/tree/906d54e/02-opd |
| Search-R1 | https://github.com/datawhalechina/agentic-rl-lab/tree/main/experiments/search-r1 | https://github.com/datawhalechina/agentic-rl-lab/tree/906d54e/03-search-r1 |
| OPSD | https://github.com/datawhalechina/agentic-rl-lab/tree/main/experiments/opsd | https://github.com/datawhalechina/agentic-rl-lab/tree/906d54e/04-opsd |
| ReTool | https://github.com/datawhalechina/agentic-rl-lab/tree/main/experiments/retool | https://github.com/datawhalechina/agentic-rl-lab/tree/906d54e/05-retool |
| DAPO | https://github.com/datawhalechina/agentic-rl-lab/tree/main/experiments/dapo | https://github.com/datawhalechina/agentic-rl-lab/tree/906d54e/06-dapo |
| GSPO | https://github.com/datawhalechina/agentic-rl-lab/tree/main/experiments/gspo | https://github.com/datawhalechina/agentic-rl-lab/tree/906d54e/07-gspo |
| ALFWorld Agentic RL | https://github.com/datawhalechina/agentic-rl-lab/tree/main/experiments/alfworld | https://github.com/datawhalechina/agentic-rl-lab/tree/906d54e/08-alfworld |
| Vision GRPO / GeoQA | https://github.com/datawhalechina/agentic-rl-lab/tree/main/experiments/vision-grpo | https://github.com/datawhalechina/agentic-rl-lab/tree/cc59b115/09-vision-grpo |
| Decision Model / PyTRIO-Jev | https://github.com/datawhalechina/agentic-rl-lab/tree/main/experiments/pytrio-jev | https://github.com/datawhalechina/agentic-rl-lab/tree/d2ba99ce6964f0525c0c0ab25dcefed50ee50e6b/09-pytrio-jev |
