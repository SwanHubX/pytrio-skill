# RestClient

> REST 管理客户端。用于查询训练历史、管理 checkpoint、下载权重。

## 创建

```python
rest = client.create_rest_client()
```

## 方法

### 模型权重管理

```python
# 列出已保存的模型权重（分页）
rest.list_weights(page=1, page_size=20) -> dict
# 返回: {"data": [...], "pagination": {"total": N, "page": 1, ...}}

# 获取模型下载链接
rest.get_archive_url(checkpoint_id: str) -> dict
# 返回: {"url": "https://...临时链接"}
```

### 训练运行管理

```python
# 列出训练运行（分页）
rest.list_training_runs(limit=20, offset=0) -> dict
# 返回: {"training_runs": [...], "total": N, "has_more": bool}

# 获取单个训练运行详情
rest.get_training_run(training_run_id: str) -> dict
# 返回: {"training_run_id": "...", "base_model": "...", "is_lora": true, "lora_rank": 32}
```

### Checkpoint 管理

```python
# 列出训练运行下的 checkpoint
rest.list_checkpoints(training_run_id: str) -> dict
# 返回: {"checkpoints": [{"checkpoint_id": "...", "checkpoint_type": "sampler", "path": "...", "size_bytes": N}]}

# 获取 checkpoint 下载链接
rest.get_checkpoint_archive_url(training_run_id: str, checkpoint_id: str) -> dict

# 删除 checkpoint
rest.delete_checkpoint(training_run_id: str, checkpoint_id: str) -> None

# 列出用户所有 checkpoint（分页）
rest.list_user_checkpoints(limit=100, offset=0) -> dict
```

### 会话管理

```python
# 列出会话
rest.list_sessions(limit=20, offset=0) -> dict
# 返回: {"sessions": ["session_id_1", "session_id_2", ...]}

# 获取会话详情
rest.get_session(session_id: str) -> dict
# 返回: {"training_run_ids": [...], "sampler_ids": [...]}

# 获取采样器信息
rest.get_sampler(sampler_id: str) -> dict
```

### 其他

```python
# 获取用户信息
rest.get_user_info() -> dict

# 获取 checkpoint 元信息（含 LoRA 配置，用于断点续训）
rest.get_checkpoint_info(path: str) -> dict
```
