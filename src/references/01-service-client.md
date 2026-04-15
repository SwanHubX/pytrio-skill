# ServiceClient

> 入口类，负责认证、WebSocket 连接、创建训练/推理/管理客户端。

## 导入

```python
import pytrio as trio
```

## 构造

```python
client = trio.ServiceClient(
    api_key: str | None = None,    # API Key，不传则从 ~/.pytrio/config.toml 读取或交互登录
)
```

> **版本变更（0.1.13b0）**：`host` / `use_https` 参数已从构造函数移除。0.1.12 及更早版本仍支持。如需自定义服务地址请改用环境变量或 `pytrio.config.settings`。

构造时自动：
1. 验证 API Key / 触发登录
2. 建立 WebSocket 连接
3. 获取可用模型列表

## 模型校验（0.1.13b0 变更）

0.1.13b0 起，模型校验按 runner type 区分：
- 训练用 `create_lora_training_client(base_model=...)` → 校验 `runner_type="TRAIN"`
- 推理用 `create_sampling_client(base_model=...)` → 校验 `runner_type="SAMPLE"`

同一模型可能只支持其中一种。如果报错 `Model X is not supported for training/sampling`，说明该模型在对应 runner 下不可用。

## 方法

### get_supported_models() -> list[str]

返回当前可用的基础模型列表。

```python
models = client.get_supported_models()
# e.g. ['Qwen/Qwen3-4B', 'Qwen/Qwen3-4B-Instruct-2507']
```

### create_lora_training_client(...) -> TrainingClient

创建 LoRA 微调训练客户端。

```python
train = client.create_lora_training_client(
    base_model: str,                    # 必填，如 'Qwen/Qwen3-4B'
    rank: int = 32,                     # LoRA rank，范围 [4, 64]
    seed: int | None = None,            # 随机种子
    train_mlp: bool = True,             # 训练 MLP 层
    train_attn: bool = True,            # 训练注意力层
    train_unembed: bool = False,        # 训练 lm_head 层
    trainable_token_indices: list[int] | dict[str, list[int]] | None = None,
                                        # 指定 token 的 Embedding 训练
    lora_path: str | None = None,       # 加载已有 LoRA 权重路径
)
```

约束：
- `base_model` 必须在 `get_supported_models()` 返回列表中
- `trainable_token_indices` 和 `train_unembed` 不能同时启用
- `rank` 必须在 [4, 64] 范围内

### create_sampling_client(...) -> SamplingClient

创建独立推理客户端（可加载 LoRA 权重）。

```python
sampler = client.create_sampling_client(
    base_model: str = "",        # 基础模型名
    model_path: str | None = None,  # 可选，LoRA 权重路径
)
```

### create_rest_client() -> RestClient

创建 REST 管理客户端。

```python
rest = client.create_rest_client()
```

### create_training_client_from_state(path: str) -> TrainingClient

从已保存的 checkpoint 恢复训练（仅加载 LoRA 权重，不含优化器状态）。

```python
train = client.create_training_client_from_state("path/to/checkpoint")
```

### create_training_client_from_state_with_optimizer(path: str) -> TrainingClient

从已保存的 checkpoint 恢复训练（包含优化器状态，用于断点续训）。

```python
train = client.create_training_client_from_state_with_optimizer("path/to/checkpoint")
```

## 异步版本

所有方法都有 `_async` 后缀的异步版本，返回 `coroutine`：
- `create_lora_training_client_async(...)`
- `create_sampling_client_async(...)`
- `create_training_client_from_state_async(...)`
- `create_training_client_from_state_with_optimizer_async(...)`
