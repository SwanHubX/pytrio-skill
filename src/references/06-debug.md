# PyTRIO 调试指南

## 问题分诊流程

遇到问题时，按以下顺序排查：

```
报错了
  ├── 连接阶段失败？ → 1. 网络/认证
  ├── 创建客户端失败？ → 2. 模型/参数
  ├── forward_backward 失败？ → 3. 数据格式
  ├── 结果不对？ → 4. 训练逻辑
  └── 推理无输出？ → 5. 推理配置
```

---

## 1. 网络与认证问题

**症状：** `ConnectionError`、`TimeoutError`、`Login failed`

| 检查项 | 命令 |
|---|---|
| API Key 有效？ | `trio login -k <KEY>` |
| config.toml 完整？ | `cat ~/.pytrio/config.toml`（需含 username, user_id, api_key） |
| 代理干扰？ | `no_proxy=pytrio.cn https_proxy="" http_proxy="" python script.py` |
| 服务可达？ | `curl -s https://pytrio.cn/api/runner/models -H "X-API-KEY: <KEY>"` |

**常见原因：**
- 系统代理（Surge/Clash）劫持 DNS → 设置 `no_proxy=pytrio.cn`
- config.toml 只写了 api_key，缺 username/user_id → 用 `trio login -k` 重新登录
- WebSocket 超时 → 检查 `ws_timeout` 配置（默认 120s）

---

## 2. 客户端创建失败

**症状：** `ValueError: Model xxx is not supported`

```python
# 先检查可用模型
models = client.get_supported_models()
print(models)
# 当前可用: ['Qwen/Qwen3-4B', 'Qwen/Qwen3-4B-Instruct-2507']
# 注意: Qwen/Qwen3-4B (Base) 的训练端不可用，必须用 Instruct-2507
```

| 参数 | 约束 |
|---|---|
| `rank` | 必须在 [4, 64] |
| `train_unembed` + `trainable_token_indices` | 不能同时启用 |

---

## 3. forward_backward 数据问题

**症状：** `ValidationError`、服务端返回 `{'error': '<run_id>'}`

### 错误解码表

| 错误信息 | 原因 | 修复 |
|---|---|---|
| `Input should be a valid dictionary or instance of ModelInput` | model_input 传了 list[int] | 用 `trio.ModelInput.from_ints(tokens)` |
| `input tokens and target tokens must have the same length` | 长度不匹配 | 确保 `len(model_input) == len(target_tokens)` |
| `target_tokens is required` | loss_fn_inputs 缺字段 | cross_entropy 需要 `target_tokens` |
| `logprobs is required` / `advantages is required` | importance_sampling 缺字段 | 需要 `target_tokens` + `logprobs` + `advantages` |
| `{'error': '<some_id>'}` | 服务端 runner 内部错误 | 检查是否用了 Base 模型（改用 Instruct-2507） |
| `Unsupported loss function type: xxx` | loss_fn 拼错 | 可选值: `cross_entropy`, `importance_sampling`, `ppo` |

### 数据构造自检清单

```python
# 1. model_input 包装正确？
assert isinstance(datum.model_input, ModelInput)  # 不是 list[int]

# 2. 长度一致？
assert len(datum.model_input) == len(datum.loss_fn_inputs["target_tokens"])

# 3. 偏移正确？（auto_shift=False 时）
# model_input = tokens[:-1], target_tokens = tokens[1:]

# 4. weights 长度一致？（如果提供了）
if "weights" in datum.loss_fn_inputs:
    assert len(datum.loss_fn_inputs["weights"]) == len(datum.model_input)

# 5. importance_sampling 四字段等长？
if loss_fn == "importance_sampling":
    n = len(datum.model_input)
    assert len(datum.loss_fn_inputs["target_tokens"]) == n
    assert len(datum.loss_fn_inputs["logprobs"]) == n
    assert len(datum.loss_fn_inputs["advantages"]) == n
```

---

## 4. 训练结果问题

**症状：** loss 不下降、loss 为 0、metrics 取不到值

| 症状 | 原因 | 修复 |
|---|---|---|
| `metrics.get("loss")` 返回 0.0 或 None | key 错了 | 用 `metrics.get("loss:sum")`（带冒号） |
| loss 不下降 | 学习率太小或数据太少 | 调大 lr（1e-4 → 5e-4）或增加数据 |
| loss 急剧增大 | 学习率太大 | 调小 lr |
| weights 全 0 导致 loss=0 | prompt masking 覆盖了所有 token | 检查 prompt_len 计算（应为 `max(len(prompt_ids)-1, 0)`） |
| 用了 `-100` 做 masking 但 loss 异常 | PyTRIO 不支持 -100 | 改用 weights 字段（0.0/1.0） |

---

## 5. 推理问题

**症状：** 推理无输出、报错、结果不合理

| 症状 | 原因 | 修复 |
|---|---|---|
| `sampling_params` 报类型错误 | 传了 dict | 用 `trio.SamplingParams(max_tokens=100)` 对象 |
| `max_new_tokens` 不识别 | HF 命名 | PyTRIO 用 `max_tokens` |
| `result.samples` 不存在 | HF 命名 | PyTRIO 用 `result.sequences[0].text` |
| 推理结果和训练前一样 | sampler desync | save_weights 后必须创建新的 SamplingClient |
| 生成不停止 | 没设 max_tokens 且 ignore_eos=True | 设置 `max_tokens` 或去掉 `ignore_eos` |

---

## 6. 环境问题

```bash
# 检查 pytrio 版本
python -c "from pytrio import config; print(config.settings.version)"

# 检查依赖
pip show pytrio transformers torch

# uv 环境下 pytrio 装不上？
# 如果是旧版 beta (0.1.11b0)，需要 [tool.uv] prerelease = "allow"
# 0.1.12+ 是正式版，不再需要此配置
```
