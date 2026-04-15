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

# tokenizer 注意事项（官方示例的标准模式）:
#   序列起始段（prompt）用 add_special_tokens=True   —— 保留 BOS
#   拼接延续段（completion）用 add_special_tokens=False —— 避免重复 BOS
#   推理时 encode 默认 add_special_tokens=True，保留 BOS 作为生成起始
#   apply_chat_template() 已自行管理 special tokens，返回文本再 encode 时也用 False

# 官方标准 SFT 数据构造模式:
prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=True)
prompt_weights = [0] * len(prompt_tokens)
completion_tokens = tokenizer.encode(completion_text, add_special_tokens=False)
completion_weights = [1] * len(completion_tokens)

tokens = prompt_tokens + completion_tokens
weights = prompt_weights + completion_weights

# 手动偏移对齐: model_input[i] 预测 target_tokens[i]
input_tokens = tokens[:-1]
target_tokens = tokens[1:]
weights = weights[1:]  # weights 必须同步偏移

sample = trio.Datum(
    model_input=trio.ModelInput.from_ints(input_tokens),  # 必须用 ModelInput.from_ints() 包装
    loss_fn_inputs={
        "target_tokens": target_tokens,  # 值可直接传 list，会自动转 TensorData
        "weights": weights,              # 可选，默认全 1；SFT 场景强烈建议传
    },
)

# 训练步骤
fb = train.forward_backward(data=[sample], loss_fn="cross_entropy")
result = fb.result()  # APIFuture -> ForwardBackwardOutput
# result.metrics 的 key 是 "loss:sum"（注意冒号），不是 "loss"
# per-token 归一化 loss（两种等价写法）:
#   方式 A: result.metrics["loss:sum"] / sum(all_weights)
#   方式 B: -np.dot(logprobs, weights) / weights.sum()
#          其中 logprobs 从 result.loss_fn_outputs[i]["logprobs"] 拼接
loss_sum = result.metrics.get("loss:sum", 0.0)
train.optim_step(trio.AdamParams(learning_rate=1e-4)).result()
```

### 损失函数

| loss_fn | 必需 loss_fn_inputs | 用途 |
|---|---|---|
| `"cross_entropy"` | `target_tokens`, 可选 `weights` | SFT |
| `"importance_sampling"` | `target_tokens`, `logprobs`, `advantages`，**强烈建议** `weights` | GRPO / 离线 RL |
| `"ppo"` | `target_tokens`, `logprobs`, `advantages`，**强烈建议** `weights` | 在线 RL |

RL 场景的 `weights` 用于对 prompt 部分做 mask（0=ignore，1=计入 loss），不传则 prompt 也会参与策略梯度，通常会跑偏。官方所有 RL 示例都带 `weights`。

`ppo` 可通过 `loss_fn_config` 自定义裁剪阈值（默认 0.2）:
```python
train.forward_backward(
    data=data, loss_fn="ppo",
    loss_fn_config={"clip_low_threshold": 0.9, "clip_high_threshold": 1.1},
)
```

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
21. **`add_special_tokens` 按段区分** — prompt 起始段用 `True`（保留 BOS），completion 拼接段用 `False`（避免重复 BOS）。不要全 True（会多 BOS），也不要全 False（模型失去起始标记，会训偏）
22. **`weights` 必须和 `target_tokens` 同步偏移** — 官方标准模式是先构造 `full_weights`，再 `weights = weights[1:]`，和 `target_tokens = tokens[1:]` 严格对齐
23. **RL 场景（IS/PPO）也要传 `weights`** — 对 prompt 部分 mask 掉，否则 prompt token 也会参与策略梯度
24. **异步版本以 `_async` 结尾** — 高并发 / RL rollout 推荐用异步版，见 `best-practices/async.md`

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
- `best-practices/grpo.md` — GRPO/PPO 强化学习：真实 rollout→reward→forward_backward 闭环
- `best-practices/async.md` — 异步训练：`_async` API、`asyncio.gather`、并发调度

## 部署

- `references/07-openai-compat.md` — 训完模型通过 OpenAI 兼容 API 对外提供服务（`chat.completions` / `completions`）
