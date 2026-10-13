# Decision Model（PyTRIO-Jev）能力说明

当用户要用 PyTRIO 训练固定候选决策模型、复现 PyTRIO-Jev、做意图分类或工单路由，或需要决策位置的稀疏监督、候选概率推理和温度校准时，先读本文件。通用生成式 SFT 的数据与训练循环见 `references/sft.md`。

## 本页导航

- [完整案例与版本](#完整案例与版本)
- [数据与候选符号](#数据与候选符号)
- [模板与 marker 定位](#模板与-marker-定位)
- [Datum 与 Loss](#datum-与-loss)
- [异步训练与权重](#异步训练与权重)
- [候选概率推理](#候选概率推理)
- [校准与评测](#校准与评测)
- [运行顺序](#运行顺序)
- [监控与常见错误](#监控与常见错误)

## 完整案例与版本

- 最新源码：https://github.com/KMnO4-zx/agentic-rl-lab/tree/main/09-pytrio-jev
- 本 Skill 核对版本：https://github.com/KMnO4-zx/agentic-rl-lab/tree/d2ba99ce6964f0525c0c0ab25dcefed50ee50e6b/09-pytrio-jev

核对案例使用 `Qwen/Qwen3.5-4B`、`pytrio==0.2.9` 和 LoRA masked SFT；完整项目要求 Python >= 3.13。复现时按项目根目录的依赖与 lockfile 安装；迁移到其他项目时检查实际 SDK 的 prompt logprob 接口和 tokenizer，不能据此推定所有基模都兼容。

该案例的数据编译、模板、训练与推理共同定义决策协议，生成项目时沿用完整目录：

```text
09-pytrio-jev/
├── 00-prepare-data.py         # 数据转换、切分与 ToolACE 评测请求排除
├── data.py                   # canonical record、候选映射、Encoder、Datum
├── templates/qwen3_5_jev.jinja # 训练与推理共用的本地模板
├── train.py                  # 异步 cross_entropy、SwanLab、推理权重
├── inference.py              # prompt logprob、候选分布、业务标签映射
├── calibrate.py              # calibration 上拟合温度
├── eval.py                   # Accuracy、ECE、Brier
├── analysis.py               # 汇总评测 JSON
└── start.md                  # 完整命令清单
```

## 数据与候选符号

一条记录包含共享的 `state`、一个或多个 `questions`，训练时另有 `targets`。例如：

```json
{
  "id": "ticket-1",
  "state": {"message": "耳机被重复扣款，请尽快退还多扣的金额。"},
  "questions": {
    "team": {
      "type": "choice",
      "instructions": "应该转给哪个团队？",
      "criteria": {"billing": "支付与退款", "delivery": "物流", "technical": "技术支持"}
    },
    "urgency": {
      "type": "score",
      "instructions": "选择处理优先级。",
      "criteria": ["低：一般咨询", "中：需要跟进", "高：资金问题"]
    },
    "refund_requested": {"type": "noul", "instructions": "客户是否明确要求退款？"}
  },
  "targets": {
    "team": {"label": "billing"},
    "urgency": {"label": "2"},
    "refund_requested": {"label": "yes"}
  }
}
```

三种题型都先映射到 `A-Z a-z 0-9` 候选符号，再映射回业务值：

| 题型 | 候选顺序与标签 | 推理输出 |
|---|---|---|
| `choice` | 按 `criteria` 字典插入顺序映射符号 | 各业务选项概率、argmax 标签 |
| `noul` | 固定 `no → A`、`yes → B`；合法标签可由布尔值或 0/1 归一化 | `p_yes`、yes/no 标签 |
| `score` | 列表按从 0 开始的索引编号；字典沿用其键与顺序 | 各等级概率、argmax 标签、概率加权期望分 |

`score` 的业务值必须可转为整数，才能沿用案例的期望分计算。现有符号表最多容纳 62 个候选；改编时先校验每个字段候选非空、未超上限、标签属于候选，且所有候选符号都编码为单 token。候选映射须贯穿训练、推理与评测。

案例使用硬标签。`targets` 只用于构造监督，不能写进输入 messages；assistant 输入中每个字段的值都填 marker。多个字段共享一条输入序列，但字段顺序仍会改变因果上下文，不承诺顺序不影响概率。

## 模板与 marker 定位

`data.Encoder` 在本地读取 `templates/qwen3_5_jev.jinja`。该模板覆盖单轮纯文本 `system + user + assistant`，保留 system message 和固定空 think 块。训练、推理都使用同一模板渲染，再以 `add_special_tokens=False` 编码并构造 `ModelInput`。

核对案例从现有词表选取 `decision` 作为单 token marker，并检查编码长度；无需扩展远端词表。不要把 Qwen 的 marker ID 或模板硬编码到其他模型，也不要把本地 `add_special_tokens()` 当成远端词表扩展。

定位 marker 时先校验完整输入与 assistant 前缀一致，再限定到 JSON 骨架区域。字段名也可能叫 `decision`，需跳过 key 中的命中，仅记录 value 的 marker；state 或 instructions 中同名文本不能被当成监督位置。复用 `Encoder` 的定位逻辑，并校验每个字段恰有一个有效位置。

## Datum 与 Loss

设完整输入为 `ids`，某字段 marker 的零基索引为 `m`：

- `model_input = ids[:-1]`；
- `target_tokens` 先取 `ids[1:]`，再把 `target_tokens[m - 1]` 替换为正确候选符号；
- 仅 `weights[m - 1] = 1`，其他位置为 0；
- 使用 `loss_fn="cross_entropy"`，每条记录的监督 token 数等于字段数。

下面片段在完整案例中使用，`encoder` 是绑定当前模型 tokenizer 的 `data.Encoder`，`record` 是上述格式：

```python
import numpy as np
import pytrio as trio

encoded = encoder.encode(record, include_targets=True)
ids = encoded["ids"]
target_tokens = list(ids[1:])
weights = [0.0] * len(target_tokens)
for field, symbol in encoded["targets"].items():
    symbol_ids = encoder.tokenizer.encode(symbol, add_special_tokens=False)
    if len(symbol_ids) != 1:
        raise ValueError("候选符号必须是单 token")
    position = encoded["markers"][field] - 1
    target_tokens[position] = symbol_ids[0]
    weights[position] = 1.0

datum = trio.Datum(
    model_input=trio.ModelInput.from_ints(ids[:-1]),
    loss_fn_inputs={
        "target_tokens": np.asarray(target_tokens, dtype=np.int64),
        "weights": np.asarray(weights, dtype=np.float32),
    },
)
```

输入骨架始终保留 marker，真实答案仅替换训练 target。JSON 标点、字段名、marker 本身和结束符均不额外获得 loss weight；未监督位置保留合法 token ID，用零 `weights` 屏蔽。

## 异步训练与权重

`train.py` 用 `forward_backward_async(datums, "cross_entropy")`、`optim_step_async(adam)` 提交操作，再由 `asyncio.create_task(...)` 消费两次提交返回的 futures 并记录日志。调用边界沿用 `references/sft.md`；保存 checkpoint 和结束训练前收齐 pending tasks，避免遗漏远端错误或日志。

每个 batch 的 `supervised_tokens = sum(weights)`，记录的 `loss = loss_sum / supervised_tokens`，分母使用实际监督位置数。监督位置为零时应排查数据，不能直接除零；整体输入仍有 prefill / training 成本，少量监督 token 不代表只处理这些 token。

案例按 step 和 epoch 结束保存 `sampler_weights`，供推理、校准和评测使用；需要恢复 optimizer 状态时另用完整 `training_state`。核对版本会丢弃不足一个 batch 的尾部，数据少于 batch size 时需调整 batch size，避免零步训练。

## 候选概率推理

`inference.py` 使用同一 Encoder，以 `include_targets=False` 构造完整的 marker 骨架。下面的 `sampling_client` 使用与 Encoder 对应的 base model 和可选 LoRA 权重：

```python
encoded = encoder.encode(record, include_targets=False)
response = sampling_client.sample(
    prompt=trio.ModelInput.from_ints(encoded["ids"]),
    num_samples=1,
    sampling_params=trio.SamplingParams(max_tokens=1),
    include_prompt_logprobs=True,
    topk_prompt_logprobs=max(len(opts) for opts in encoded["options"].values()),
).result()
```

决策读取 prompt 内的分布，额外生成的 1 个 token 不用于决策。先校验 `response.input_tokens == len(encoded["ids"])`，再读取每个 marker 的 `response.topk_prompt_logprobs[m]`。此处返回数组按被预测的 prompt token 索引；训练数组经过右移，监督位置是 `m - 1`。

只在本字段候选符号上计算 `softmax(candidate_logprobs / T)`，然后恢复业务标签、`p_yes` 或期望分。这里的 `T` 是本地概率校准温度，与采样参数的 temperature 分开。

**top-k 存在近似边界：** 返回的是全词表 top-k，令 K 等于候选数不能保证候选全部出现。核对实现对缺失候选使用 `min(topk.values()) - 10` 作为 logprob 近似。改编时应检查候选覆盖情况，按服务支持范围调整 K 或明确缺失处理；缺失未解决时不能宣称得到精确的完整候选分布。归一化后的概率也需要独立校准验证。

## 校准与评测

`00-prepare-data.py` 在 Typed Decisions 的 train split 内按案例、按 workflow 划分 train/dev/calibration；ToolACE 先排除评测请求，再采样训练记录。Jevbench 和独立 test 留作评测。更换数据时按完整记录划分，避免同一 state 的不同字段跨集合，并核对跨集合重复样本。

需要选择 checkpoint 时使用 dev；当前案例保留了该划分，未提供自动选优流程。温度只在 calibration 上拟合，最终协议冻结后评测 test。`calibrate.py` 在 `log T` 上做黄金分割搜索以最小化 NLL；校准文件记录权重路径，使用前核对它与待评权重匹配。固定同一组候选得分时，正温度缩放保持 argmax 不变，不应把重新请求后的准确率变化直接归因于温度。

`eval.py` 按字段统计 Accuracy、10-bin ECE 和 Brier；五项基准的平均准确率采用各 suite 等权平均。Base / checkpoint 对照保持数据、候选顺序、模板和推理协议一致，分别报告准确率与校准质量。

上游记录来自单次 LoRA 训练，Typed Decisions 使用同 workflow 的 specialist 设置，与零样本对照条件不同；Jevbench-Hard 的准确率还出现下降。引用结果时保留这些条件，loss 降低或概率归一化均不能替代泛化与校准评测。

## 运行顺序

将完整项目检出到用户工作目录，按项目依赖安装并完成 `trio login` 后，在 `09-pytrio-jev/` 中运行；数据和评测输出写入该案例目录的 `datasets/`、`outputs/`：

```bash
uv run python 00-prepare-data.py
uv run python train.py --epochs 2 --batch-size 16 --save-every 60 --swanlab-mode online
uv run python inference.py --weights trio://...
uv run python calibrate.py --weights trio://...
uv run python eval.py --weights trio://... --calibration outputs/calibration.json --tag trained
uv run python analysis.py
```

将 `trio://...` 替换为选定 checkpoint 的实际路径，校准与评测使用同一权重。迁移到业务数据时保留 canonical record、候选映射、模板和位置对齐，再替换数据准备与评测部分。

## 监控与常见错误

- 训练记录 `loss`、`supervised_tokens`、`epoch`、global step、`trainer/*`；配置记录模型、SDK 版本、LoRA rank、学习率、数据集和模板版本。
- 推理与评测记录候选覆盖率、Accuracy、ECE、Brier、温度、checkpoint 和数据划分；候选覆盖率需在改编时补充，当前案例未自动上报该指标。
- 不要给整段 assistant 骨架设置 `weights=1`，也不要把 marker 自身当作目标答案。
- 不要把训练的 `m - 1` 与 prompt logprob 返回数组的 `m` 混用。
- 不要将 noul 直接当成预测 Yes/No token，或将 score 的业务数字当成训练符号；先读取候选映射。
- 不要在输入中填入真实答案，也不要在 test 上拟合温度或选择 checkpoint。
- 更换 tokenizer、模板或服务入口后，重新校验单 token 候选、marker 定位和本地/远端长度；不能只替换模型名。
