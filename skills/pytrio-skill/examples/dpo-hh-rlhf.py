"""同步版 DPO demo：ModelScope HH-RLHF + PyTRIO + SwanLab。

实现逻辑是标准 DPO preference training：
1. 每条 HH-RLHF 样本包含 chosen / rejected 两段完整对话；
2. 解析出共同 prompt，以及最终的 chosen / rejected assistant 回复；
3. reference model 计算两条回复的参考 logprob；
4. 当前 student 通过 forward_backward_custom 拿到可求导 logprob；
5. 用 DPO loss 更新 LoRA：
   -log sigmoid(beta * ((log pi_chosen - log ref_chosen)
                        - (log pi_rejected - log ref_rejected))).

小成本试跑：
python examples/dpo-hh-rlhf.py \
    --steps 100 \
    --batch-size 16 \
    --sample-size 5000 \
    --max-length 1024 \
    --base-model Qwen/Qwen3.5-4B \
    --swanlab-mode disabled

注意：PyTRIO 的 forward_backward_custom 需要本地安装 torch。
如果当前环境没有 torch，请先安装 torch。
"""

from __future__ import annotations

import argparse
import random
import re
import shutil
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from datasets import Dataset, concatenate_datasets, load_dataset
import numpy as np
import pytrio as trio
import swanlab
import torch
import torch.nn.functional as F


# 数据默认缓存到运行目录下；首次从 ModelScope 下载，后续直接复用本地文件。
DEFAULT_DATA_DIR = Path("datasets") / "hh-rlhf"
DEFAULT_LOG_DIR = Path("swanlog")
DEFAULT_SUBSETS = ["helpful-base"]
# red-team-attempts 不是 preference modeling 数据，所以不放进可训练子集白名单。
VALID_SUBSETS = {
    "helpful-base",
    "helpful-online",
    "helpful-rejection-sampled",
    "harmless-base",
}
# HH-RLHF 原始行是 "\n\nHuman:" / "\n\nAssistant:" 拼起来的一整段文本。
TURN_RE = re.compile(r"\n\n(Human|Assistant):")


@dataclass(frozen=True)
class PreferencePair:
    """一条 DPO 偏好样本：共同 prompt + chosen/rejected 回复。"""

    prompt_messages: list[dict[str, str]]
    chosen_response: str
    rejected_response: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PyTRIO 同步版 DPO / HH-RLHF")

    parser.add_argument("--dataset-repo", default="Anthropic/hh-rlhf")
    parser.add_argument("--dataset-revision", default="master")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--dataset-subsets",
        nargs="+",
        default=DEFAULT_SUBSETS,
        help=(
            "HH-RLHF 子集，可选 helpful-base / helpful-online / "
            "helpful-rejection-sampled / harmless-base"
        ),
    )
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--sample-size", type=int, default=1000, help="随机抽样数量；<=0 表示全用")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--base-model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--lora-rank", type=int, default=32)

    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--dpo-beta", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--save-weights-name", default="dpo-hh-rlhf-qwen35-4b-sync")

    parser.add_argument("--swanlab", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--swanlab-project", default="trio-case")
    parser.add_argument("--swanlab-name", default="dpo-qwen35-4b-sync")
    parser.add_argument("--swanlab-workspace", default=None)
    parser.add_argument(
        "--swanlab-mode",
        choices=["online", "local", "offline", "disabled"],
        default=None,
    )
    args = parser.parse_args()

    unknown_subsets = sorted(set(args.dataset_subsets) - VALID_SUBSETS)
    if unknown_subsets:
        raise ValueError(f"Unsupported HH-RLHF subsets: {unknown_subsets}")
    for name in ("steps", "batch_size"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be >= 1")
    if args.max_length < 8:
        raise ValueError("--max-length must be >= 8")
    if args.dpo_beta <= 0:
        raise ValueError("--dpo-beta must be > 0")
    return args


def modelscope_file_url(repo: str, revision: str, file_path: str) -> str:
    """拼出 ModelScope 单文件下载接口 URL。"""
    namespace, name = repo.split("/", 1)
    # FilePath 通过 query 参数传入，urllib 会把 helpful-base/train.jsonl.gz 里的斜杠编码好。
    query = urllib.parse.urlencode(
        {
            "Source": "SDK",
            "Revision": revision,
            "FilePath": file_path,
            "View": "False",
        }
    )
    return f"https://www.modelscope.cn/api/v1/datasets/{namespace}/{name}/repo?{query}"


def download_if_needed(url: str, local_path: Path, force: bool) -> None:
    if local_path.exists() and not force:
        return

    # 先下载到 .tmp，再替换目标文件，避免中断时留下半截 gzip。
    local_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = local_path.with_suffix(local_path.suffix + ".tmp")
    print(f"Downloading {url}\n  -> {local_path}")
    request = urllib.request.Request(url, headers={"User-Agent": "pytrio-skill-dpo-demo"})
    with urllib.request.urlopen(request) as response, tmp_path.open("wb") as output:
        shutil.copyfileobj(response, output)
    tmp_path.replace(local_path)


def load_hh_rlhf(args: argparse.Namespace) -> Dataset:
    """从 ModelScope 下载 HH-RLHF jsonl.gz 到本地，然后用 datasets 读取。"""
    datasets: list[Dataset] = []
    for subset in args.dataset_subsets:
        # 原始 train/test 由文件路径决定；datasets 这里只是读取本地 jsonl.gz。
        remote_path = f"{subset}/{args.split}.jsonl.gz"
        local_path = args.dataset_dir / remote_path
        download_if_needed(
            modelscope_file_url(args.dataset_repo, args.dataset_revision, remote_path),
            local_path,
            args.force_download,
        )
        dataset = load_dataset(
            "json",
            data_files=str(local_path),
            split="train",
            cache_dir=str(args.dataset_dir / ".datasets_cache"),
        )
        if not isinstance(dataset, Dataset):
            raise TypeError(f"Expected Dataset, got {type(dataset)!r}")
        datasets.append(dataset)

    dataset = datasets[0] if len(datasets) == 1 else concatenate_datasets(datasets)
    required = {"chosen", "rejected"}
    if not required.issubset(dataset.column_names):
        raise ValueError(f"HH-RLHF must contain {required}, got {dataset.column_names}")

    dataset = dataset.shuffle(seed=args.seed)
    if args.sample_size > 0:
        dataset = dataset.select(range(min(args.sample_size, len(dataset))))
    return dataset


def parse_hh_transcript(text: str) -> list[dict[str, str]] | None:
    """把 HH-RLHF 的 `Human:` / `Assistant:` transcript 解析成 chat messages。"""
    matches = list(TURN_RE.finditer(text))
    if not matches:
        return None

    messages: list[dict[str, str]] = []
    for index, match in enumerate(matches):
        role = "user" if match.group(1) == "Human" else "assistant"
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        content = text[start:end].strip()
        if not content:
            return None
        messages.append({"role": role, "content": content})
    return messages


def row_to_preference_pair(row: dict[str, str]) -> PreferencePair | None:
    """从 chosen/rejected 完整对话中取共同 prompt 和最终两条 assistant 回复。

    DPO 比较的是同一个 prompt 下两个回复谁更好，所以这里会检查前文一致。
    """
    chosen_turns = parse_hh_transcript(row["chosen"])
    rejected_turns = parse_hh_transcript(row["rejected"])
    if not chosen_turns or not rejected_turns:
        return None
    if chosen_turns[-1]["role"] != "assistant" or rejected_turns[-1]["role"] != "assistant":
        return None

    # 去掉最后一条 assistant 回复后，剩下的必须是 chosen/rejected 共享的 prompt。
    chosen_prompt = chosen_turns[:-1]
    rejected_prompt = rejected_turns[:-1]
    if chosen_prompt != rejected_prompt:
        return None
    if not chosen_prompt or chosen_prompt[-1]["role"] != "user":
        return None

    # 最后一轮 assistant 回复才是 DPO 要比较的对象；空回复或完全相同的回复没有偏好信号。
    chosen_response = chosen_turns[-1]["content"].strip()
    rejected_response = rejected_turns[-1]["content"].strip()
    if not chosen_response or not rejected_response or chosen_response == rejected_response:
        return None

    return PreferencePair(
        prompt_messages=chosen_prompt,
        chosen_response=chosen_response,
        rejected_response=rejected_response,
    )


def encode_messages(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool,
    enable_thinking: bool,
) -> list[int]:
    # add_generation_prompt=True 用于 prompt；False 用于已经包含 assistant 回复的完整对话。
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=enable_thinking,
    )
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if not tokens:
        raise ValueError("Encoded chat template is empty")
    return list(tokens)


def build_completion_datum(
    tokenizer: Any,
    prompt_messages: list[dict[str, str]],
    response: str,
    *,
    max_length: int,
    enable_thinking: bool,
) -> trio.Datum | None:
    """构造只在 assistant 回复 token 上有权重的 cross-entropy Datum。"""
    # prompt_tokens 应停在 assistant 开始生成的位置。
    prompt_tokens = encode_messages(
        tokenizer,
        prompt_messages,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    # full_tokens 是 prompt + 某个候选回复，用来做自回归右移。
    full_messages = [*prompt_messages, {"role": "assistant", "content": response}]
    full_tokens = encode_messages(
        tokenizer,
        full_messages,
        add_generation_prompt=False,
        enable_thinking=enable_thinking,
    )

    if full_tokens[: len(prompt_tokens)] != prompt_tokens:
        raise ValueError("Prompt tokens are not a prefix of full tokens")

    completion_len = len(full_tokens) - len(prompt_tokens)
    # completion_len <= 0 表示没有 assistant 回复；len(full_tokens) > max_length 表示超长。
    if completion_len <= 0 or len(full_tokens) > max_length:
        return None

    # prompt 部分权重为 0，response 部分权重为 1；右移后用 weights[1:] 对齐 target。
    token_weights = [0.0] * len(prompt_tokens) + [1.0] * completion_len
    input_ids = full_tokens[:-1]
    target_ids = full_tokens[1:]
    weights = token_weights[1:]

    # DPO loss 只在 response token 上计算，所以 input_ids/target_ids/weights 长度必须一致。
    if not (len(input_ids) == len(target_ids) == len(weights)):
        raise ValueError("DPO datum fields must have the same length")

    return trio.Datum(
        model_input=trio.ModelInput.from_ints(input_ids),
        loss_fn_inputs={
            "target_tokens": np.asarray(target_ids, dtype=np.int64),
            "weights": np.asarray(weights, dtype=np.float32),
        },
    )


def build_pair_datums(
    tokenizer: Any,
    pair: PreferencePair,
    *,
    max_length: int,
    enable_thinking: bool,
) -> list[trio.Datum] | None:
    chosen = build_completion_datum(
        tokenizer,
        pair.prompt_messages,
        pair.chosen_response,
        max_length=max_length,
        enable_thinking=enable_thinking,
    )
    rejected = build_completion_datum(
        tokenizer,
        pair.prompt_messages,
        pair.rejected_response,
        max_length=max_length,
        enable_thinking=enable_thinking,
    )
    if chosen is None or rejected is None:
        return None
    return [chosen, rejected]


def build_step_batch(
    dataset: Dataset,
    tokenizer: Any,
    step: int,
    args: argparse.Namespace,
) -> list[trio.Datum]:
    """为一个 step 构造 chosen/rejected 交错排列的 batch。"""
    datums: list[trio.Datum] = []
    start = step * args.batch_size

    for index in range(start, start + args.batch_size):
        # 用取模回绕，允许 steps 超过数据集可切出的 batch 数，方便小样本试跑。
        row = dataset[index % len(dataset)]
        pair = row_to_preference_pair(row)
        if pair is None:
            continue

        pair_datums = build_pair_datums(
            tokenizer,
            pair,
            max_length=args.max_length,
            enable_thinking=args.enable_thinking,
        )
        if pair_datums is None:
            continue

        # DPO loss 按偶数/奇数下标配对：chosen_0, rejected_0, chosen_1, rejected_1...
        datums.extend(pair_datums)
    return datums


def datum_full_tokens(datum: trio.Datum) -> list[int]:
    """从右移后的 Datum 还原完整 token 序列，用于 reference compute_logprobs。"""
    target_tokens = list(datum.loss_fn_inputs["target_tokens"].data)
    if not target_tokens:
        raise ValueError("Datum target_tokens cannot be empty")
    return datum.model_input.to_ints() + [int(target_tokens[-1])]


def compute_reference_logprob_seqs(reference_client: Any, datums: list[trio.Datum]) -> list[list[float]]:
    """reference 对 batch 中每条序列计算与 target 对齐的 logprob。"""
    # compute_logprobs 需要完整序列；返回后丢掉第一个无前文条件的 token logprob。
    futures = [
        reference_client.compute_logprobs(trio.ModelInput.from_ints(datum_full_tokens(datum)))
        for datum in datums
    ]
    ref_logprobs: list[list[float]] = []
    for datum, future in zip(datums, futures, strict=True):
        values = future.result()
        shifted = values[1:]
        if len(shifted) != len(datum.model_input) or any(value is None for value in shifted):
            raise ValueError("Invalid reference logprobs for DPO datum")
        ref_logprobs.append([float(value) for value in shifted])
    return ref_logprobs


def compute_dpo_loss(
    chosen_logprobs: list[Any],
    rejected_logprobs: list[Any],
    chosen_ref_logprobs: list[Any],
    rejected_ref_logprobs: list[Any],
    dpo_beta: float,
) -> tuple[Any, dict[str, float]]:
    """标准 DPO loss 公式。"""
    chosen_log_ratio = torch.stack(
        [lp - rlp for lp, rlp in zip(chosen_logprobs, chosen_ref_logprobs, strict=True)]
    )
    rejected_log_ratio = torch.stack(
        [lp - rlp for lp, rlp in zip(rejected_logprobs, rejected_ref_logprobs, strict=True)]
    )

    losses = -F.logsigmoid(dpo_beta * (chosen_log_ratio - rejected_log_ratio))
    loss = losses.mean()

    chosen_rewards = dpo_beta * chosen_log_ratio.detach()
    rejected_rewards = dpo_beta * rejected_log_ratio.detach()
    metrics = {
        "dpo/loss": float(loss.detach().item()),
        "dpo/accuracy": float((chosen_log_ratio > rejected_log_ratio).float().mean().item()),
        "dpo/margin": float((chosen_rewards - rejected_rewards).mean().item()),
        "dpo/chosen_reward": float(chosen_rewards.mean().item()),
        "dpo/rejected_reward": float(rejected_rewards.mean().item()),
    }
    return loss, metrics


def make_dpo_loss_fn(
    reference_logprobs: list[list[float]],
    dpo_beta: float,
) -> Callable[[list[trio.Datum], list[Any]], tuple[Any, dict[str, float]]]:
    """创建 PyTRIO forward_backward_custom 使用的本地 DPO loss。

    loss = -log sigmoid(beta * ((log pi_chosen - log ref_chosen)
                                - (log pi_rejected - log ref_rejected)))
    """

    def dpo_loss_fn(data: list[trio.Datum], logprobs_list: list[Any]) -> tuple[Any, dict[str, float]]:
        if len(data) != len(logprobs_list) or len(data) != len(reference_logprobs):
            raise ValueError("DPO loss got mismatched data/logprob lengths")
        if len(data) % 2 != 0:
            raise ValueError("DPO loss requires chosen/rejected pairs")

        # 每个值都是 response token 上加权求和后的序列 logprob。
        chosen_logprobs = []
        rejected_logprobs = []
        chosen_ref_logprobs = []
        rejected_ref_logprobs = []

        for index in range(0, len(data), 2):
            chosen_datum = data[index]
            rejected_datum = data[index + 1]
            chosen_seq = logprobs_list[index].float()
            rejected_seq = logprobs_list[index + 1].float()
            device = chosen_seq.device

            chosen_weights = torch.as_tensor(
                chosen_datum.loss_fn_inputs["weights"].data,
                dtype=torch.float32,
                device=device,
            )
            rejected_weights = torch.as_tensor(
                rejected_datum.loss_fn_inputs["weights"].data,
                dtype=torch.float32,
                device=rejected_seq.device,
            )
            chosen_ref_seq = torch.as_tensor(
                reference_logprobs[index],
                dtype=torch.float32,
                device=device,
            )
            rejected_ref_seq = torch.as_tensor(
                reference_logprobs[index + 1],
                dtype=torch.float32,
                device=rejected_seq.device,
            )

            chosen_logprobs.append(torch.dot(chosen_seq, chosen_weights))
            rejected_logprobs.append(torch.dot(rejected_seq, rejected_weights))
            chosen_ref_logprobs.append(torch.dot(chosen_ref_seq, chosen_weights))
            rejected_ref_logprobs.append(torch.dot(rejected_ref_seq, rejected_weights))

        return compute_dpo_loss(
            chosen_logprobs=chosen_logprobs,
            rejected_logprobs=rejected_logprobs,
            chosen_ref_logprobs=chosen_ref_logprobs,
            rejected_ref_logprobs=rejected_ref_logprobs,
            dpo_beta=dpo_beta,
        )

    return dpo_loss_fn


def start_swanlab(args: argparse.Namespace, dataset_size: int):
    if not args.swanlab:
        return None
    config = vars(args).copy()
    config["dataset_dir"] = str(args.dataset_dir)
    config["dataset_size"] = dataset_size
    return swanlab.init(
        project=args.swanlab_project,
        name=args.swanlab_name,
        workspace=args.swanlab_workspace,
        mode=args.swanlab_mode,
        config=config,
        tags=["TRIO", "DPO", "HH-RLHF", "ModelScope"],
        log_dir=str(DEFAULT_LOG_DIR),
    )


def main(args: argparse.Namespace) -> None:
    """同步 DPO 训练主入口。"""
    random.seed(args.seed)
    np.random.seed(args.seed)

    print("Loading HH-RLHF dataset from ModelScope...")
    dataset = load_hh_rlhf(args)
    print(f"Loaded {len(dataset)} HH-RLHF rows from {args.dataset_subsets}")

    print("Creating PyTRIO clients...")
    service_client = trio.ServiceClient()
    training_client = service_client.create_lora_training_client(
        base_model=args.base_model,
        rank=args.lora_rank,
        seed=args.seed,
    )
    tokenizer = training_client.get_tokenizer()

    # 本 demo 不从 checkpoint 恢复；DPO reference 直接用未训练的 base model。
    reference_client = service_client.create_sampling_client(base_model=args.base_model)

    adam = trio.AdamParams(
        learning_rate=args.learning_rate,
        beta1=args.beta1,
        beta2=args.beta2,
        eps=args.adam_eps,
    )
    run = start_swanlab(args, len(dataset))

    try:
        for step in range(args.steps):
            step_start = time.time()
            data = build_step_batch(dataset, tokenizer, step, args)
            if not data:
                raise RuntimeError(
                    "No valid DPO pairs were built. Increase --max-length or --sample-size."
                )

            # reference logprobs 先算好，再作为闭包输入本地 DPO loss。
            reference_logprobs = compute_reference_logprob_seqs(
                reference_client,
                data,
            )
            fwd_bwd = training_client.forward_backward_custom(
                data,
                make_dpo_loss_fn(reference_logprobs, args.dpo_beta),
            )
            optim = training_client.optim_step(adam)
            fwd_bwd_result = fwd_bwd.result()
            optim.result()

            elapsed = time.time() - step_start
            metrics = {
                "data/pairs": len(data) // 2,
                "data/datums": len(data),
                "data/tokens": sum(datum.model_input.length for datum in data),
                "train/learning_rate": args.learning_rate,
                "time/step_elapsed_time": elapsed,
            }
            # 后端指标加 trainer/ 前缀，本地 DPO 指标保持 dpo/ 前缀。
            metrics.update(
                {
                    key if key.startswith("dpo/") else f"trainer/{key}": float(value)
                    for key, value in dict(fwd_bwd_result.metrics).items()
                }
            )
            if run is not None:
                swanlab.log(metrics, step=step)

            print(
                f"step {step:03d}/{args.steps} | pairs {metrics['data/pairs']} | "
                f"dpo_loss {metrics.get('dpo/loss', float('nan')):.4f} | "
                f"acc {metrics.get('dpo/accuracy', float('nan')):.3f} | "
                f"time {elapsed:.2f}s"
            )

        save_result = training_client.save_weights_for_sampler(args.save_weights_name).result()
        print(f"Saved weights: {save_result.path}")
        if run is not None:
            swanlab.log({"save/weights_path": swanlab.Text(save_result.path)}, step=args.steps)
    finally:
        if run is not None:
            swanlab.finish()
        training_client.close()


if __name__ == "__main__":
    start = time.time()
    main(parse_args())
    print("#" * 50)
    print("# all done")
    print(f"# train cost {time.time() - start:.2f}s")
    print("#" * 50)
