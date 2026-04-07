# PyTRIO CodeWiki

> PyTRIO SDK v0.1.12 — 远程 LLM 训练与推理 SDK

## 这是什么

PyTRIO 是一个 Python SDK，将大语言模型的**训练计算（前向传播、反向传播、优化器步骤）**和**推理采样**委托到远端 GPU 集群执行。本地代码只负责数据准备、调度逻辑和结果处理，不需要本地 GPU。

## 环境搭建

### 使用 uv（推荐）

```bash
uv init my-pytrio-project && cd my-pytrio-project
```

在 `pyproject.toml` 中添加依赖：

```toml
[project]
requires-python = ">=3.10"
dependencies = [
    "pytrio>=0.1.12",
    "transformers>=4.40.0",
    "torch>=2.0.0",
]
```

```bash
uv sync
uv run python your_script.py
```

### 使用 pip

```bash
pip install pytrio transformers torch
```

## 认证

使用 PyTRIO 前必须完成认证，获取 API Key: https://pytrio.cn/dashboard

```bash
# 推荐：非交互式登录（凭证自动保存到 ~/.pytrio/config.toml）
trio login -k YOUR_API_KEY

# 或交互式登录
trio login
```

登录后 `trio.ServiceClient()` 无需传 key 即可自动识别。也可以在代码中直接传入：

```python
client = trio.ServiceClient(api_key="your_key")
```

注意：CLI 命令是 `trio login`，不是 `pytrio login`。

## 核心架构

```
用户代码 (本地)
    │
    ├── ServiceClient          ← 入口，建立 WebSocket 连接
    │   ├── create_lora_training_client()  → TrainingClient
    │   ├── create_sampling_client()       → SamplingClient
    │   └── create_rest_client()           → RestClient
    │
    ├── TrainingClient         ← LoRA 微调
    │   ├── forward() / forward_backward()   → 远端计算
    │   ├── forward_backward_custom()        → 自定义 loss
    │   ├── optim_step()                     → 远端优化器
    │   ├── save_state() / save_weights_for_sampler()
    │   └── create_sampling_client()         → 从训练权重推理
    │
    ├── SamplingClient         ← 推理
    │   ├── sample()             → 文本生成
    │   └── compute_logprobs()   → 计算 logprobs
    │
    └── RestClient             ← 管理 API
        ├── list_weights() / list_training_runs() / list_checkpoints()
        ├── get_archive_url() / get_checkpoint_archive_url()
        └── delete_checkpoint()
```

## PyTRIO ↔ PyTorch 概念映射

| PyTorch 概念 | PyTRIO 对应 | 说明 |
|---|---|---|
| `model = AutoModel(...)` | `trio.ServiceClient(api_key=...)` | 模型在远端(如 Qwen/Qwen3-4B)，本地只建立连接 |
| `model.forward(input_ids)` | `train.forward(data)` | 前向传播，远端执行 |
| `loss.backward()` | `train.forward_backward(data)` | 前向+反向一体，远端执行 |
| `optimizer.step()` | `train.optim_step(trio.AdamParams(...))` | 远端执行 Adam 优化 |
| `optimizer.zero_grad()` | （不需要） | 远端自动管理 |
| `torch.save(model.state_dict(), path)` | `train.save_state(name)` | 保存含优化器的完整 checkpoint |
| `model.generate(...)` | `sampler.sample(prompt, ...)` | 远端推理生成 |
| `LoraConfig(r=32, ...)` | `create_lora_training_client(rank=32, ...)` | LoRA 配置通过参数传入 |
| `torch.Tensor` | `TensorData` / plain `list` | 数据传输格式，支持 numpy/torch 互转 |
| `DataLoader` | 用户自行分 batch | PyTRIO 不提供 DataLoader |

## 关键差异

1. **没有本地模型对象** — 不存在 `model` 变量，训练状态在远端维护
2. **forward + backward 通常合一** — `forward_backward()` 是主要方法，单独 `forward()` 用于只做推理不更新梯度
3. **异步通信** — 所有方法返回 `APIFuture[T]`，需要 `.result()` 获取结果
4. **数据格式是 token ids** — 需要自行 tokenize，传入 `Datum` 对象（含 `model_input` + `loss_fn_inputs`）
5. **远端保存** — checkpoint 保存在云端，通过 `RestClient` 管理和下载
