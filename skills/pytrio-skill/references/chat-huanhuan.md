# Chat-甄嬛参考

官方案例页：

```text
https://docs.pytrio.cn/docs/content/example/chat_huanhuan/content.md
```

当用户想复现 Chat-甄嬛角色微调流程，想写同步/异步 SFT，或想要一个带 SwanLab 训练记录的真实 SFT 项目时，使用这个参考。

## 需要读取

1. 先读取上面的官方案例页。
2. 再读 `references/doc-index.md` 里 TrainingClient、SamplingClient、Datum、ModelInput 和 SamplingParams 的链接。
3. 如果用户只要最小训练 demo，用 `examples/quickstart_sft.py`，不要套完整 Chat-甄嬛流程。
4. 如果用户要同步训练和 SwanLab 记录，参考 `examples/chat-huanhuan.py`。
5. 如果用户要异步 SFT、异步提交 batch 或异步记录 SwanLab，参考 `examples/chat-huanhuan-async.py`。

## 案例形态

- 数据集：`dataset/huanhuan.json`
- 数据源：https://github.com/datawhalechina/self-llm/blob/master/dataset/huanhuan.json
- 常见依赖：`pytrio`、`transformers`、`modelscope`、`numpy`、`tqdm`、`swanlab`
- 当前官方文档中的基础模型：`Qwen/Qwen3.5-4B`
- 任务：通过 SFT 训练 LoRA adapter，让模型用目标角色风格回答
- 记录：SwanLab 用于记录 loss、epoch、batch、模型、数据集路径、LoRA rank、学习率等配置，是 PyTRIO 训练的推荐默认记录工具

## 数据处理模式

数据集是 JSON 数组。每条数据通常包含：

- `instruction`
- `input`
- `output`

当 `input` 为空时，用户输入只用 `instruction`；否则把 `instruction` 和 `input` 用换行拼接。assistant 的训练目标是 `output`。

使用 chat template 做 SFT 时：

1. 用 system prompt 和 user message 构造 prompt messages。
2. tokenizer 支持时，调用 `tokenizer.apply_chat_template(..., tokenize=False, add_generation_prompt=True, enable_thinking=False)`。
3. prompt 编码使用 `add_special_tokens=False`，因为 chat template 已经管理特殊 token。
4. assistant completion 编码也使用 `add_special_tokens=False`。
5. tokenizer 有 EOS 时显式追加 EOS。
6. prompt 部分 `weights` 设为 `0`，assistant/EOS 部分 `weights` 设为 `1`。
7. 三个数组一起右移对齐：`input_tokens=tokens[:-1]`、`target_tokens=tokens[1:]`、`loss_weights=weights[1:]`。

## 训练模式

- 先把样本批量预处理成 `trio.Datum`。
- 用 `training_client.forward_backward(data=batch, loss_fn="cross_entropy").result()`。
- 用 `training_client.optim_step(trio.AdamParams(learning_rate=...)).result()`。
- 用返回的 logprobs 和 batch 中相同的 loss weights 计算 per-token loss。
- 使用 SwanLab 时，用稳定的 `global_step` 逐 batch 记录日志，并在 `swanlab.init(..., config={...})` 中写入关键超参数，便于复现实验。
- 用 `save_weights_for_sampler(name=...)` 保存推理权重，再用 `model_path=saved.path` 创建 sampling client。

## 异步训练与 SwanLab

异步版示例展示了三件事：

- 用 `create_lora_training_client_async`、`forward_backward_async`、`optim_step_async` 异步提交训练任务。
- 用 `asyncio.create_task(...)` 让 loss 计算和 `swanlab.log(...)` 尽早开始执行，避免把日志全部堆到 epoch 末尾。
- 每个 epoch 结束前 `await asyncio.gather(...)`，确保后台日志任务完成后再进入下一轮或保存权重。

写异步 SFT 时，不要只收集 coroutine 对象而不调度；需要立即创建 task，SwanLab 曲线才会在训练过程中持续更新。

## SwanLab Skill

如果任务涉及查询 SwanLab 实验、读取指标、对比多次训练、画 loss 曲线或补全更复杂的 SwanLab 记录代码，建议同时使用 SwanLab Skill。PyTRIO Skill 负责训练 API 和数据构造，SwanLab Skill 负责实验追踪和指标查询，两者配合使用。

## 使用原则

除非用户明确要求完整脚本，否则不要把官方案例整篇复制进回答。多数场景只提取相关模式，并把路径、模型名、超参数和日志配置保留为可配置项。
