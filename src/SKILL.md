---
name: pytrio
description: Guide for using the PyTRIO SDK — remote LLM training and inference. Trigger when code imports pytrio, or user mentions PyTRIO / TRIO training / TRIO SDK / remote LLM training.
---

# PyTRIO SDK Skill

## PyTRIO 是什么

PyTRIO 是远程 LLM 训练与推理 SDK。本地代码只做数据准备和调度，实际的前向传播、反向传播、优化器步骤和推理采样都在远端 GPU 集群上执行。不需要本地 GPU。

- 安装: `pip install pytrio transformers torch`（或 `uv add pytrio transformers torch`）
- 导入约定: `import pytrio as trio`，所有 API 通过 `trio.xxx` 访问
- Tokenizer: 使用 `train.get_tokenizer()` 或 `sampler.get_tokenizer()`（需安装 modelscope）
- 环境搭建详见 `references/00-overview.md`

## 认证（重要）

使用 PyTRIO 前必须先完成认证，否则所有 API 调用都会失败。

**检查是否已登录：** 查看 `~/.pytrio/config.toml` 是否存在且包含 `api_key` 字段。

**如果未登录，按以下流程操作：**

1. 提示用户提供 API Key（可从 https://pytrio.cn/dashboard 获取）
2. 用户提供 key 后，执行非交互式登录命令：
   ```bash
   trio login -k <用户提供的API_KEY>
   ```
3. 登录成功后凭证保存在 `~/.pytrio/config.toml`，后续 `ServiceClient()` 无需传 api_key

**在代码中也可以直接传入（适合临时使用）：**
```python
client = ServiceClient(api_key="YOUR_API_KEY")
```

注意：CLI 命令是 `trio login`，不是 `pytrio login`。

## PyTRIO ↔ PyTorch 概念速查

| PyTorch | PyTRIO | 说明 |
|---|---|---|
| `model = AutoModel(...)` | `client = ServiceClient(api_key=...)` | 无本地模型对象 |
| `model(input_ids)` | `train.forward(data)` | 远端前向 |
| `loss.backward()` | `train.forward_backward(data)` | 前向+反向合一 |
| `optimizer.step()` | `train.optim_step(AdamParams(...))` | 远端 Adam |
| `optimizer.zero_grad()` | 不需要 | 远端自动管理 |
| `torch.save(...)` | `train.save_state(name)` | 保存到云端 |
| `model.generate(...)` | `sampler.sample(prompt, ...)` | 远端推理 |
| `torch.Tensor` | `TensorData` / plain `list` | 自动互转 |

## 核心 API 速查

### 入口

```python
import pytrio as trio

client = trio.ServiceClient(api_key="...")
models = client.get_supported_models()
```

**模型选择规则：必须使用 `Qwen/Qwen3-4B-Instruct-2507`，不要使用 `Qwen/Qwen3-4B`（Base 模型训练端不可用）。**

### 训练

```python
import pytrio as trio

train = client.create_lora_training_client(
    base_model="Qwen/Qwen3-4B-Instruct-2507",
    rank=32,            # [4, 64]
    train_mlp=True,
    train_attn=True,
    train_unembed=False,
)
tokenizer = train.get_tokenizer()

# tokenizer 注意事项:
#   训练时用 add_special_tokens=False（special tokens 不参与损失计算）
#   推理时用 add_special_tokens=True（默认，模型需要 BOS 起始）
#   如果用 apply_chat_template()，encode 时也应 add_special_tokens=False 避免重复
tokens = tokenizer.encode(text, add_special_tokens=False)

# 构造数据: model_input[i] 预测 target_tokens[i]（auto_shift=False 时需手动偏移）
# 注意: model_input 必须用 ModelInput.from_ints() 包装，不能直接传 list[int]

sample = trio.Datum(
    model_input=trio.ModelInput.from_ints(tokens[:-1]),
    loss_fn_inputs={
        "target_tokens": tokens[1:],       # loss_fn_inputs 的值可以直接传 list
        "weights": [1.0] * (len(tokens) - 1),  # 可选，默认全 1
    },
)

# 训练步骤
fb = train.forward_backward(data=[sample], loss_fn="cross_entropy")
result = fb.result()  # APIFuture -> ForwardBackwardOutput
# result.metrics 的 key 是 "loss:sum"（注意冒号），不是 "loss"
loss = result.metrics.get("loss:sum", 0.0)
train.optim_step(trio.AdamParams(learning_rate=1e-4)).result()
```

### 损失函数

| loss_fn | 必需 loss_fn_inputs | 用途 |
|---|---|---|
| `"cross_entropy"` | `target_tokens`, 可选 `weights` | SFT |
| `"importance_sampling"` | `target_tokens`, `logprobs`, `advantages` | GRPO |
| `"ppo"` | `target_tokens`, `logprobs`, `advantages` | PPO |

自定义 loss（需要本地 torch）:
```python
def my_loss(samples, logprobs_list):
    loss = ...  # torch.Tensor
    return loss, {"metric": loss.item()}
train.forward_backward_custom(data=samples, loss_fn=my_loss)
```

### 保存与恢复

```python
# 保存权重（仅用于推理）
save = train.save_weights_for_sampler("name").result()
# save.path -> 远端路径

# 保存完整 checkpoint（含优化器，用于断点续训）
state = train.save_state("name").result()

# 断点续训
train2 = client.create_training_client_from_state_with_optimizer(state.path)

# 仅加载权重续训（不含优化器动量）
train3 = client.create_training_client_from_state(state.path)
```

### 推理

```python
import pytrio as trio

# 从训练权重推理
sampler = train.create_sampling_client(model_path=save.path)
# 或独立推理
sampler = client.create_sampling_client(base_model="Qwen/Qwen3-4B-Instruct-2507")

tokenizer = sampler.get_tokenizer()
input_ids = tokenizer.encode("Hello")

result = sampler.sample(
    prompt=trio.ModelInput.from_ints(input_ids),  # 必须用 trio.ModelInput 包装
    num_samples=1,
    sampling_params=trio.SamplingParams(max_tokens=100, temperature=0.7),
).result()

text = result.sequences[0].text
logprobs = result.sequences[0].logprobs
```

### 管理

```python
rest = client.create_rest_client()
rest.list_weights()            # 已保存权重
rest.list_training_runs()      # 训练记录
rest.list_checkpoints(run_id)  # 某次训练的 checkpoint
rest.list_user_checkpoints()   # 所有 checkpoint
rest.get_archive_url(id)       # 下载链接
rest.delete_checkpoint(run_id, cp_id)
```

## AdamParams SDK 默认值

注意与 PyTorch 默认值不同：`beta2=0.95`（PyTorch 默认 0.999）、`eps=1e-12`（PyTorch 默认 1e-8）。如需与 PyTorch 行为一致，请显式传入 `AdamParams(beta2=0.999, eps=1e-8)`。

## 错误处理

- `APIFuture.result(timeout=None)` 支持 timeout 参数（秒），超时抛 `TimeoutError`
- 服务端异常抛 `pydantic.ValidationError` 或 `ValueError`
- WebSocket 断连不自动重连，需重建 `ServiceClient`

## 陷阱清单

1. **所有方法返回 `APIFuture`** — 必须调用 `.result()` 获取实际值，否则拿到的是 Future 对象
2. **model_input 和 target_tokens 长度必须一致** — 在 `auto_shift=False`（默认）时，用户负责偏移对齐
3. **`importance_sampling`/`ppo` 需要四个等长字段** — `model_input`, `target_tokens`, `logprobs`, `advantages` 必须等长
4. **`trainable_token_indices` 和 `train_unembed` 互斥** — 不能同时启用
5. **`ignore_eos=True` 时必须设置 `max_tokens`** — 否则抛异常
6. **checkpoint 路径是远端路径** — `save_state()`/`save_weights_for_sampler()` 返回的 path 是云端路径，不是本地文件
7. **`forward_backward_custom` 需要本地 torch** — 自定义 loss 函数在本地计算梯度后传回远端
8. **导入约定 `import pytrio as trio`** — 所有 API 通过 `trio.xxx` 访问（如 `trio.ServiceClient`、`trio.Datum`），不要用 `from pytrio import ...`
9. **登录命令是 `trio login`** — CLI 命令仍然是 `trio`，不是 `pytrio`
10. **所有 loss_fn_inputs 的值会自动转为 TensorData** — 可以直接传 list、numpy array 或 torch tensor
11. **model_input 必须用 `trio.ModelInput.from_ints(list)` 包装** — 不能直接传 `list[int]`，否则 Pydantic 验证报错
12. **prompt 参数同理必须用 `trio.ModelInput.from_ints(list)` 包装** — `sampler.sample(prompt=trio.ModelInput.from_ints(ids), ...)`
13. **tokenizer 使用 `get_tokenizer()`** — `train.get_tokenizer()` 或 `sampler.get_tokenizer()`（需安装 modelscope）
14. **运行时需绕过本地代理** — 如有系统代理，需设置 `no_proxy=pytrio.cn https_proxy="" http_proxy=""` 环境变量
15. **必须使用 Instruct 模型** — `Qwen/Qwen3-4B` (Base) 训练端不可用，必须用 `Qwen/Qwen3-4B-Instruct-2507`
16. **与 HuggingFace API 命名不同** — PyTRIO 的参数名和 HF transformers 有差异，不要混用：
    - 采样参数是 `max_tokens`（不是 `max_new_tokens`）
    - 推理结果是 `result.sequences[0].text`（不是 `result.samples`）
    - 前向传播必须用关键字参数 `train.forward_backward(data=[...])` （不是位置参数）
    - 优化器参数是 `learning_rate`（不是 `lr`）
    - loss_fn_inputs 的 key 是 `target_tokens`（不是 `labels`）
    - 使用 `import pytrio as trio` 而非 `from pytrio import ...`
    - 不要导入内部模块（如 `pytrio.types.forward.encoded_text_chunk`）
17. **model_input 和 target_tokens 必须偏移一位** — `model_input=tokens[:-1]`, `target_tokens=tokens[1:]`，不能传相同的 tokens
18. **loss key 是 `"loss:sum"`（带冒号）** — `fb_result.metrics.get("loss:sum")`，不是 `"loss"`、`"train_loss"` 或 `.loss` 属性
19. **不要用 -100 做 token masking** — HF 用 `-100` 标记忽略的 token，但 PyTRIO 不支持这个机制。PyTRIO 用 `weights` 字段控制哪些 token 参与 loss 计算（0.0=忽略，1.0=计算），见 `best-practices/sft.md`
20. **`sampling_params` 必须传 `trio.SamplingParams` 对象** — 不能传 dict，`sampler.sample(sampling_params=trio.SamplingParams(max_tokens=100))`

## 详细文档

完整 API 文档在本 Skill 的 `references/` 目录下：
- `references/00-overview.md` — 架构总览与概念映射
- `references/01-service-client.md` — ServiceClient API
- `references/02-training-client.md` — TrainingClient API
- `references/03-sampling-client.md` — SamplingClient API
- `references/04-rest-client.md` — RestClient API
- `references/05-data-types.md` — 数据类型参考
- `references/06-debug.md` — 调试指南（分诊流程、错误解码、自检清单）

## 示例代码

可运行的最小闭环示例在本 Skill 的 `examples/` 目录下：
- `examples/01_train_sft.py` — SFT 训练
- `examples/02_inference.py` — 推理
- `examples/03_checkpoint_resume.py` — 断点续训
- `examples/04_model_management.py` — 模型管理
- `examples/05_importance_sampling.py` — GRPO 风格训练

## 场景最佳实践

针对特定训练场景的推荐代码模板和注意事项，在 `best-practices/` 目录下：
- `best-practices/sft.md` — SFT 监督微调：prompt masking（weights 用法）、EOS 追加、tokenizer 选项
- `best-practices/grpo.md` — GRPO/PPO 强化学习：数据构造、典型流程、自定义 loss
