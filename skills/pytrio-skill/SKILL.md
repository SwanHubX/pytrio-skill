---
name: pytrio-skill
description: 使用 PyTRIO/TRIO 编写、调试或解释远程大模型训练与推理代码。用户提到 pytrio、TRIO、PyTRIO SDK、ServiceClient、TrainingClient、SamplingClient、LoRA 训练、SFT、异步 SFT、RL、GRPO、OPD、DPO、custom loss、HuggingFace datasets、TRIO OpenAI 兼容 API、SwanLab 训练记录、权重保存、checkpoint 或远程 LLM 后训练时使用。
metadata:
  version: "0.1.0"
---

# PyTRIO Skill

PyTRIO 是 TRIO 远程大模型后训练和推理服务的 Python SDK。本地代码负责准备数据和控制训练循环；前向传播、反向传播、优化器更新、采样和权重存储由 TRIO 服务执行。

## 安装与更新

安装这个 skill：执行 `npx skills add SwanHubX/pytrio-skill -g -y`，它会安装到当前用户级别的 `.agents/skills` 并供支持 Agent Skills 的 CLI 使用。
更新这个 skill：执行 `npx skills update pytrio-skill -g -y`，更新后重新启动或刷新 Agent 会话即可使用新版本。

## 使用定位

PyTRIO 代码通常由本地 Python 负责数据准备、训练循环和实验记录，由 TRIO 服务负责模型前向、反向、优化器更新和权重保存。

处理训练任务时，先按任务形态选择对应的 `Datum` 构造、loss 和 client 调用方式：

- SFT：监督微调、assistant-only loss mask、同步/异步训练、SwanLab 记录。
- GRPO：student rollout、reward、group-relative advantage、`importance_sampling` 更新。
- OPD：student rollout、teacher logprob、reverse-KL advantage、`importance_sampling` 更新。
- DPO/custom loss：chosen/rejected 偏好数据、reference logprob、本地 torch loss、`forward_backward_custom`。

示例代码提供可改模板；关键字段和常见错误在 `references/` 中说明。

## 使用顺序

1. 先判断任务类型：SFT、GRPO、OPD、DPO/custom loss、推理/保存权重，或 API 调试。
2. 按任务路由读取对应 reference 和 example；不要一开始展开全部官方文档。
3. 示例能覆盖时，按示例替换数据集、prompt、reward、teacher/student 模型、超参数和 SwanLab 配置。
4. 示例覆盖不了、API 行为不确定、报错难定位，或需要 checkpoint/OpenAI 兼容 API 等细节时，再读 `references/doc-index.md`，根据任务打开对应官方 Markdown 文档。
5. 若官方文档和本地 SDK 行为不一致，先检查已安装 SDK 签名或写最小复现，再给最终代码。

## 任务路由

| 用户任务 | 读取内容 |
|---|---|
| 安装、登录、第一次训练或推理 | `references/doc-index.md` -> 快速开始 |
| 编写简单 SFT 训练代码 | `references/sft.md`；`examples/quickstart_sft.py` |
| 编写角色 SFT 或 Chat-甄嬛类微调 | `references/sft.md`；`references/chat-huanhuan.md`；`examples/chat-huanhuan.py` |
| 编写异步 SFT 或异步记录 SwanLab | `references/sft.md`；`examples/chat-huanhuan-async.py` |
| 编写多轮对话 SFT 蒸馏 | `references/sft.md`；`examples/sft-distill-conversation.py` 或 `examples/sft-distill-conversation-async.py` |
| 编写 GRPO / GSM8K / reward-based RLVR | `references/grpo.md`；`examples/grpo-gsm8k.py` 或 `examples/grpo-gsm8k-async.py` |
| 编写 OPD / on-policy distillation / teacher-KL 蒸馏 | `references/opd.md`；`examples/opd-deepmath.py` 或 `examples/opd-deepmath-async.py` |
| 编写 DPO / preference training / custom loss | `references/dpo.md`；`examples/dpo-hh-rlhf.py` |
| 接入 HuggingFace datasets | `references/doc-index.md` -> HuggingFace datasets |
| 编写推理或采样代码 | `references/doc-index.md` -> 推理、SamplingClient、SamplingParams |
| 保存用于推理的权重 | `references/doc-index.md` -> 训练、TrainingClient、保存/续训、下载权重 |
| 从 checkpoint 恢复训练 | `references/doc-index.md` -> 保存/续训、ServiceClient、TrainingClient |
| 使用 OpenAI 兼容 API | `references/doc-index.md` -> OpenAI API |
| 加入 SwanLab 训练记录 | 优先看同类 example；需要查询实验时同时使用 swanlab-skill |

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
- GRPO 和 OPD 通常使用 `loss_fn="importance_sampling"`，`Datum.loss_fn_inputs` 必须包含右移对齐后的 `target_tokens`、旧策略 `logprobs` 和 `advantages`。
- `importance_sampling` 的 prompt/observation 区间不训练时，用 `target_tokens=0`、`logprobs=0.0`、`advantages=0.0` 占位，保证长度和 `model_input` 一致。
- GRPO 的 advantage 来自同一 prompt 的 group 内 reward 相对均值；如果整组 reward 完全相同，advantage 全为 0，通常跳过该组。
- OPD 的 teacher 只对 student 实际采样出来的 completion 计算 logprob；不要用 teacher 自己生成的 completion 替代 student 轨迹。
- DPO 使用 `forward_backward_custom(data, loss_fn)`，不是 `cross_entropy` 或 `importance_sampling`；custom loss 在本地用 torch 写，返回 `(loss, metrics)`。
- DPO batch 保持 `[chosen_0, rejected_0, chosen_1, rejected_1, ...]` 顺序，reference logprobs 要提前计算并通过闭包传给 custom loss。
- 推理用权重保存使用 `save_weights_for_sampler()`；完整断点续训使用 `save_state()`。
- 模型名优先使用当前官方文档或 `client.get_supported_models()` 返回值，不要硬编码旧模型名。
- 训练脚本建议默认接入 SwanLab。SFT 记录 loss；GRPO 记录 reward、degenerate group 比例和 trainer metrics；OPD 记录 reverse KL、completion token 数和 trainer metrics；DPO 记录 loss、accuracy、margin、chosen/rejected reward。

## 代码骨架

先用本地 reference + example 解决问题；示例不够时，再读官方文档补足 API 细节。若官方文档和本地 SDK 行为不一致，先检查已安装 SDK 签名或写最小复现。

SFT 的核心是 assistant-only `weights` 和自回归右移：

```python
tokens = prompt_tokens + completion_tokens
weights = [0.0] * len(prompt_tokens) + [1.0] * len(completion_tokens)
datum = trio.Datum(
    model_input=trio.ModelInput.from_ints(tokens[:-1]),
    loss_fn_inputs={
        "target_tokens": np.asarray(tokens[1:], dtype=np.int32),
        "weights": np.asarray(weights[1:], dtype=np.float32),
    },
)
training_client.forward_backward([datum], loss_fn="cross_entropy").result()
```

GRPO/OPD 都用 `importance_sampling`，区别只在 advantage 来源：

```python
obs_len = len(prompt_tokens) - 1
datum = trio.Datum(
    model_input=trio.ModelInput.from_ints(prompt_tokens + completion_tokens[:-1]),
    loss_fn_inputs={
        "target_tokens": np.asarray([0] * obs_len + completion_tokens, dtype=np.int64),
        "logprobs": np.asarray([0.0] * obs_len + old_logprobs, dtype=np.float32),
        "advantages": np.asarray([0.0] * obs_len + advantages, dtype=np.float32),
    },
)
training_client.forward_backward([datum], loss_fn="importance_sampling").result()
```

- GRPO：`advantages = reward - mean(group_rewards)`，old logprobs 来自同组 rollout 的 student sampler。
- OPD：teacher 对 `prompt + student_completion` 调 `compute_logprobs`，`advantages = -kl_coef * (student_logprobs - teacher_logprobs)`。

DPO/custom loss 用本地 torch loss，不走内置 loss 名称：

```python
def loss_fn(data, logprobs_list):
    # data 顺序为 chosen, rejected；reference_logprobs 由闭包传入。
    loss = compute_dpo_loss(data, logprobs_list, reference_logprobs)
    return loss, {"dpo/loss": float(loss.detach().item())}

training_client.forward_backward_custom([chosen, rejected], loss_fn).result()
```

每次 `forward_backward*` 后调用 `training_client.optim_step(trio.AdamParams(...))`；保存推理权重使用 `save_weights_for_sampler(...).result()`。
