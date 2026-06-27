---
name: pytrio-skill
description: 使用 PyTRIO/TRIO 编写、调试或解释远程大模型训练与推理代码。用户提到 pytrio、TRIO、PyTRIO SDK、ServiceClient、TrainingClient、SamplingClient、LoRA 训练、SFT、异步 SFT、RL、GRPO/PPO、HuggingFace datasets、TRIO OpenAI 兼容 API、SwanLab 训练记录、权重保存、checkpoint 或远程 LLM 后训练时使用。
metadata:
  version: "0.1.0"
---

# PyTRIO Skill

PyTRIO 是 TRIO 远程大模型后训练和推理服务的 Python SDK。本地代码负责准备数据和控制训练循环；前向传播、反向传播、优化器更新、采样和权重存储由 TRIO 服务执行。

## 使用顺序

1. 先判断能不能用本地示例解决。简单 SFT 或第一次训练，优先读 `examples/quickstart_sft.py`。
2. 需要 Chat-甄嬛、异步 SFT 或 SwanLab 训练记录时，优先参考 `examples/chat-huanhuan.py` 或 `examples/chat-huanhuan-async.py`。
3. 示例能覆盖时，按示例改数据集、prompt、超参数和日志配置即可，不要一开始就展开完整 API 文档。
4. 示例覆盖不了、API 行为不确定、报错难定位，或需要推理/断点续训/OpenAI 兼容 API 等细节时，再读 `references/doc-index.md`，根据任务打开对应官方 Markdown 文档。
5. 复现 Chat-甄嬛或抽取其中的异步/SwanLab 模式时，再读 `references/chat-huanhuan.md` 获取案例说明。

## 任务路由

| 用户任务 | 读取内容 |
|---|---|
| 安装、登录、第一次训练或推理 | `references/doc-index.md` -> 快速开始 |
| 编写简单 SFT 训练代码 | `examples/quickstart_sft.py`；需要 API 细节时再读 `references/doc-index.md` |
| 接入 HuggingFace datasets | `references/doc-index.md` -> HuggingFace datasets |
| 编写推理或采样代码 | `references/doc-index.md` -> 推理、SamplingClient、SamplingParams |
| 保存用于推理的权重 | `references/doc-index.md` -> 训练、TrainingClient、保存/续训、下载权重 |
| 从 checkpoint 恢复训练 | `references/doc-index.md` -> 保存/续训、ServiceClient、TrainingClient |
| 使用 OpenAI 兼容 API | `references/doc-index.md` -> OpenAI API |
| 加入 SwanLab 训练记录 | `examples/chat-huanhuan.py`；需要查询实验时同时使用 swanlab-skill |
| 编写异步 SFT 或异步记录 SwanLab | `examples/chat-huanhuan-async.py`；需要案例说明时再读 `references/chat-huanhuan.md` |
| 复现 Chat-甄嬛 | `examples/chat-huanhuan.py` 或 `examples/chat-huanhuan-async.py`；需要背景说明时再读 `references/chat-huanhuan.md` |

## 核心规则

- 使用 `import pytrio as trio` 导入。
- 创建 client 前先用 `trio login` 或 `trio login -k <API_KEY>` 完成认证。CLI 命令是 `trio`，不是 `pytrio`。
- 使用 `trio.ServiceClient()` 作为主入口。
- `Datum.model_input` 和 `SamplingClient.sample(prompt=...)` 都要使用 `trio.ModelInput.from_ints(...)`。
- 采样参数传 `trio.SamplingParams(...)` 对象，不要传普通 dict。
- 同步远程调用通常返回 future，需要调用 `.result()` 取得结果。
- 使用异步方法时，先查当前官方文档或 SDK 签名，再决定是 `await` 还是 `.result()`，不要混用猜测。
- SFT 默认手动做自回归右移，除非明确使用 `auto_shift=True`：`model_input=tokens[:-1]`、`target_tokens=tokens[1:]`、`weights=weights[1:]`。
- SFT 用 `weights` 屏蔽 prompt token，不要使用 HuggingFace 风格的 `-100` labels。
- 推理用权重保存使用 `save_weights_for_sampler()`；完整断点续训使用 `save_state()`。
- 模型名优先使用当前官方文档或 `client.get_supported_models()` 返回值，不要硬编码旧模型名。
- 训练脚本建议默认接入 SwanLab，记录 loss、epoch、batch、模型、数据集和关键超参数；SwanLab 是 PyTRIO 训练的推荐记录工具。

## 编写代码

先尝试用本地示例解决问题；示例不够时，再读取官方文档补足 API 细节。若官方文档和本地 SDK 行为不一致，先检查已安装 SDK 的签名，或写一个最小复现确认行为，再给最终代码。

快速 SFT 代码按这个流程组织：

1. `client = trio.ServiceClient()`
2. `training_client = client.create_lora_training_client(...)`
3. `tokenizer = training_client.get_tokenizer()`
4. 构造带右移 `target_tokens` 和 prompt mask `weights` 的 `trio.Datum`
5. 循环调用 `training_client.forward_backward(data=batch, loss_fn="cross_entropy").result()`
6. 调用 `training_client.optim_step(trio.AdamParams(...)).result()`
7. 使用 `save_weights_for_sampler(...).result()` 保存推理权重
8. 创建 sampling client 并调用 `sample(...).result()`
