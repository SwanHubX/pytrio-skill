# 多模态与 Vision GRPO 能力说明

当用户要做图像输入、多模态推理、多模态 SFT、Vision GRPO、视觉 RLVR、`ImageChunk`，或需要把图文 prompt 与 completion loss 严格对齐时，先读本文件。

## 本页导航

- [版本与资料](#版本与资料)
- [图文输入结构](#图文输入结构)
- [ImageChunk 与视觉 token](#imagechunk-与视觉-token)
- [多模态 SFT 对齐](#多模态-sft-对齐)
- [Vision GRPO 对齐](#vision-grpo-对齐)
- [异步调用边界](#异步调用边界)
- [GeoQA 完整案例](#geoqa-完整案例)
- [评测与监控](#评测与监控)
- [常见错误](#常见错误)

## 版本与资料

PyTRIO 从 `0.2.7` 开始支持本文件中的图像输入、多模态推理与训练接口。先检查项目实际安装或锁定的版本；低于 `0.2.7` 时不要生成 `ImageChunk` 代码并声称可运行。

- 官方多模态指南：https://docs.pytrio.com/docs/content/guide/vision/content.md
- 官方 Vision GRPO 案例：https://docs.pytrio.com/docs/content/example/vision-grpo/content.md
- Skill 单文件训练示例：`examples/vision-grpo.py`
- GeoQA 最新源码：https://github.com/KMnO4-zx/agentic-rl-lab/tree/main/09-vision-grpo
- 本 Skill 核对版本：https://github.com/KMnO4-zx/agentic-rl-lab/tree/cc59b115/09-vision-grpo

多模态指南覆盖图像推理和 LaTeX OCR SFT。Vision GRPO 案例覆盖 GeoQA 数据准备、异步 group rollout、`importance_sampling` 更新、checkpoint 和固定集评测。

## 图文输入结构

纯文本输入可以使用 `ModelInput.from_ints()`。图文输入要按模型实际阅读顺序显式组合 chunks：

```python
prompt = trio.ModelInput(
    chunks=[
        trio.types.EncodedTextChunk(tokens=before_image_tokens),
        trio.ImageChunk(
            data=image_bytes,
            format="png",
            expected_tokens=image_tokens,
        ),
        trio.types.EncodedTextChunk(tokens=after_image_tokens),
    ]
)
```

优先让模型自己的 chat template 生成完整 prompt，再沿该模型的图片占位符拆分文本并插入真实 `ImageChunk`。Qwen3.5 的案例使用 `<|image_pad|>`；这不是所有视觉模型的通用协议。

chat template 已加入对话、视觉和角色特殊 token。拆分后的文本再次编码时使用：

```python
tokenizer.encode(text, add_special_tokens=False)
```

否则会重复添加特殊 token，破坏 prompt 与远端模型的格式。

## ImageChunk 与视觉 token

`ImageChunk` 的字段边界：

- `data`：图片原始二进制字节，不传路径字符串或 Base64 文本。
- `format`：与字节内容一致，使用 `png` 或 `jpeg`。
- `expected_tokens`：远端视觉编码器将产生的 token 数。

纯推理可以只传 `data` 和 `format`。多模态训练、Vision GRPO，以及任何依赖 `len(ModelInput)` 的本地对齐都要提供正确的 `expected_tokens`。使用与远端基模匹配的 image processor 计算，不要写死：

```python
patches = image_processor.get_number_of_image_patches(
    image.height,
    image.width,
    images_kwargs={},
)
expected_tokens = patches // int(image_processor.merge_size) ** 2
```

图片存在透明通道时，先合成到明确背景再转 RGB；同时让 `format` 与重新编码后的字节一致。采样后校验：

```python
if response.input_tokens != len(prompt):
    raise ValueError("图文 prompt 的本地与远端 token 长度不一致")
```

校验失败时停止训练，检查 image processor、缩放规则、图片编码和 chunk 顺序。

## 多模态 SFT 对齐

多模态 SFT 仍使用 `cross_entropy` 和 assistant-only `weights`。差异在于 `model_input` 必须保留图文 chunks：

```text
model_input   = 完整图文 prompt + completion[:-1]
target_tokens = prompt/image 区间填 0 + 完整 completion
weights       = prompt/image 区间填 0 + completion 区间填 1
```

设 `prompt_length = len(ModelInput(chunks=prompt_chunks))`，第一个 completion token 由 prompt 最后一个位置预测，因此写入起点是：

```python
start = prompt_length - 1
target_tokens[start : start + len(completion)] = completion
weights[start : start + len(completion)] = 1.0
```

不要把图片展平成普通 token ids，不要用 `-100` 代替 loss mask，也不要漏掉 completion 的结束标记。

## Vision GRPO 对齐

Vision GRPO 沿用标准 GRPO 的当前策略 rollout、reward 和组相对 advantage。图文 prompt 只提供上下文，策略梯度只落在 completion：

| 字段 | prompt / image 区间 | completion 区间 |
|---|---|---|
| `model_input` | 完整图文 chunks | `completion[:-1]` |
| `target_tokens` | `0` | 完整 completion tokens |
| `logprobs` | `0.0` | rollout 返回的 old logprobs |
| `advantages` | `0.0` | 当前 completion 的组相对 advantage |

核心构造：

```python
model_input = trio.ModelInput(
    chunks=[
        *prompt_chunks,
        trio.types.EncodedTextChunk(tokens=completion_tokens[:-1]),
    ]
)
observation_length = prompt_length - 1
datum = trio.Datum(
    model_input=model_input,
    loss_fn_inputs={
        "target_tokens": np.asarray(
            [0] * observation_length + completion_tokens,
            dtype=np.int64,
        ),
        "logprobs": np.asarray(
            [0.0] * observation_length + old_logprobs,
            dtype=np.float32,
        ),
        "advantages": np.asarray(
            [0.0] * observation_length
            + [group_advantage] * len(completion_tokens),
            dtype=np.float32,
        ),
    },
)
```

completion tokens、old logprobs 和 completion 区间 advantages 必须严格等长。old logprobs 必须来自生成该轨迹的当前 sampler；参数更新后重新计算的 logprob 不能替代它。

同题 reward 使用同一组内的均值计算：

```python
advantage = reward - mean(group_rewards)
```

整组 reward 完全相同时 advantage 全为零，跳过该组的 backward，同时保留 degenerate group 指标。

## 异步调用边界

Vision GRPO 案例在同一个当前 sampler 上并发不同题目的 rollout：

- `sample_async()` 一次 `await` 直接返回 `SampleResponse`，不要 `.result()` 或再次 `await`。
- `save_weights_and_get_sampling_client_async()` 一次 `await` 直接返回 sampler。
- `forward_backward_async()` 和 `optim_step_async()` 第一次 `await` 返回 `APIFuture`，再 `await` 该 future 才等待远端任务完成。

先完成同一个 logical batch 的 rollout、reward 和 advantage，再提交 backward。不要在部分题目尚未完成时更新模型，避免一个 group 或 batch 混入不同 policy 版本。

## GeoQA 完整案例

`examples/vision-grpo.py` 将官方 `train.py` 整理成单文件训练示例：它通过 Hugging Face 缓存读取 GeoQA parquet、筛选原始 train，并保留图文 rollout、reward、Datum、更新和 checkpoint 主链路。

固定 test、Base/checkpoint 评测和分析继续使用完整的 `09-vision-grpo` 多文件项目：

```text
09-vision-grpo/
├── download-dataset.py   # 下载 GeoQA，生成 train 与固定 test
├── train.py              # 图文 group rollout、reward、更新与 checkpoint
├── eval.py               # 单个 Base 或 checkpoint 的异步评测
├── analysis.py           # 汇总固定实验结果
├── start.md              # 启动命令
└── readme.md             # 原理、结果与边界
```

案例使用 `Qwen/Qwen3.5-4B`、GeoQA、二元 boxed-choice reward、`importance_sampling` 和 `enable_thinking=False`。关闭模板自带 thinking 不等于禁止在 prompt 中要求简短推理。

数据准备保留 3,503 条原始 train，并从 759 条原始 test 中按 `seed=42` 固定 100 条评测；dev 不进入本次实验。严格复现时锁定 Hugging Face dataset revision。

## 评测与监控

训练至少记录：

- `reward` 与 `format_rate`
- `degenerate_fraction` 与 `train_datums`
- `rollout/completion_tokens_mean`
- `trainer/*`，尤其 `loss_mean`
- 图像 token、本地/远端 prompt 长度不一致次数
- step wall-clock、sampling token 和 train token

评测一次只创建一个 sampler：不传 `model_path` 评测 Base，传 `trio://.../sampler_weights/...` 评测一个 checkpoint。固定相同数据、seed、temperature、max tokens 和解析器后再比较。

案例在固定 100 题上记录 Base `Accuracy 71.0% / Format rate 75.0%`，step-100 为 `87.0% / 91.0%`。该结果只覆盖这 100 条、单次训练和一个视觉模型，不能写成完整 759 条 test benchmark 或多 seed 结论。

## 常见错误

- 不要在 `pytrio<0.2.7` 上生成多模态代码并假设 `ImageChunk` 可用。
- 不要把图片路径、PIL 对象或 Base64 字符串直接放进 `ImageChunk.data`。
- 不要把 Qwen3.5 的特殊 token、图片占位符和 image processor 规则硬编码给其他模型。
- 不要省略训练时的 `expected_tokens`，也不要忽略本地与远端 input token 校验。
- 不要从图文 prompt 中丢掉图片 chunk 后再构造训练 `Datum`。
- 不要让 prompt/image token 获得非零 target、old logprob、weight 或 advantage。
- 不要使用更新后的 policy logprob 替代 rollout old logprob。
- 不要把全部正确或全部错误的 group 当作有效相对训练信号。
- 不要用在线 batch reward 代替固定评测集，也不要把固定 100 题结果外推成完整 benchmark。
