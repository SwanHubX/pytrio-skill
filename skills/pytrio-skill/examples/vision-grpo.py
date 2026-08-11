"""PyTRIO 0.2.7 Vision GRPO / GeoQA 单文件训练示例。

代码基于：
https://github.com/KMnO4-zx/agentic-rl-lab/blob/main/09-vision-grpo/train.py

与完整项目的区别：本文件会把 GeoQA parquet 下载到 Hugging Face 缓存并直接
筛选原始 train，因此不依赖当前目录中的 datasets 文件夹。固定 test、评测和分析
仍以完整项目为准。

环境：
    pip install "pytrio>=0.2.7" "datasets>=5.0.0" \
        huggingface_hub numpy pillow swanlab tqdm transformers torch torchvision
    trio login

低成本试跑（会调用远端服务并产生费用）：
    python examples/vision-grpo.py \
        --steps 1 \
        --batch-size 1 \
        --group-size 4 \
        --max-tokens 256 \
        --save-every 0 \
        --no-save-weights \
        --swanlab-mode disabled \
        --show-samples

参考配置：
    python examples/vision-grpo.py \
        --steps 100 \
        --batch-size 8 \
        --group-size 8 \
        --max-tokens 1024 \
        --save-every 25 \
        --swanlab-mode online
"""

from __future__ import annotations

import argparse
import asyncio
import io
import re
import time
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import pytrio as trio
import swanlab
from datasets import Dataset, load_dataset
from huggingface_hub import snapshot_download
from PIL import Image
from tqdm.asyncio import tqdm_asyncio
from transformers import AutoImageProcessor

DATASET_ID = "hz2475/geoQA"
DEFAULT_MODEL = "Qwen/Qwen3.5-4B"
IMAGE_PAD_TOKEN = "<|image_pad|>"
CHOICE_LETTERS = "ABCD"
BOXED_CHOICE_PATTERN = re.compile(
    r"\\boxed\s*\{\s*([A-D])\s*\}",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class RolloutSample:
    """一条 completion 及其 rollout old logprobs 和组相对 advantage。"""

    tokens: list[int]
    logprobs: list[float]
    text: str
    predicted_choice: str | None
    reward: float
    advantage: float


@dataclass(frozen=True)
class RolloutGroup:
    """同一道 GeoQA 题目的图文 prompt 与一组 completion。"""

    prompt_chunks: list[Any]
    prompt_length: int
    samples: list[RolloutSample]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PyTRIO Vision GRPO / GeoQA demo")
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument(
        "--dataset-revision",
        default="main",
        help="GeoQA revision；严格复现时传入固定 commit",
    )
    parser.add_argument(
        "--dataset-cache-dir",
        type=Path,
        help="可选的 Hugging Face 缓存目录",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="忽略已有 Hugging Face 缓存并重新下载 GeoQA parquet",
    )
    parser.add_argument("--base-model", default=DEFAULT_MODEL)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="0 表示使用全部 GeoQA train",
    )
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=4e-5)
    parser.add_argument(
        "--swanlab-mode",
        choices=("online", "local", "offline", "disabled"),
        default="online",
    )
    parser.add_argument("--swanlab-project", default="pytrio-vision-grpo")
    parser.add_argument(
        "--experiment-name",
        default="vision-grpo-qwen35-4b-geoqa",
    )
    parser.add_argument(
        "--weights-name",
        default="vision-grpo-qwen35-4b-geoqa",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=10,
        help="每隔多少个 step 保存一次，0 表示只在训练结束时保存",
    )
    parser.add_argument(
        "--save-weights",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--show-samples", action="store_true")
    return parser.parse_args()


def load_geoqa_train(
    dataset_id: str,
    revision: str,
    cache_dir: Path | None,
    force_download: bool,
    seed: int,
    max_samples: int,
) -> Dataset:
    """从 Hugging Face 缓存读取 GeoQA，并只保留原始 train split。"""
    snapshot_path = Path(
        snapshot_download(
            repo_id=dataset_id,
            repo_type="dataset",
            revision=revision,
            cache_dir=cache_dir,
            allow_patterns=["data/*.parquet"],
            force_download=force_download,
        )
    )
    parquet_files = sorted((snapshot_path / "data").glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(
            f"GeoQA snapshot 中没有 data/*.parquet：{snapshot_path}"
        )

    dataset = load_dataset(
        "parquet",
        data_files={"train": [str(path) for path in parquet_files]},
        split="train",
    )
    if not isinstance(dataset, Dataset):
        raise TypeError(f"期望 datasets.Dataset，实际得到 {type(dataset)!r}")
    if "original_split" not in dataset.column_names:
        raise ValueError("GeoQA 数据缺少 original_split 字段")

    train_data = dataset.filter(
        lambda split: split == "train",
        input_columns=["original_split"],
        desc="提取 GeoQA train",
    ).remove_columns("original_split")
    if len(train_data) == 0:
        raise ValueError("GeoQA 原始 train split 为空")

    train_data = train_data.shuffle(seed=seed)
    if max_samples > 0:
        train_data = train_data.select(range(min(max_samples, len(train_data))))
    return train_data


def pick_batch(dataset: Dataset, step: int, batch_size: int) -> Dataset:
    """按 step 顺序取 batch，走完数据后从头继续。"""
    start = step * batch_size
    indices = [(start + offset) % len(dataset) for offset in range(batch_size)]
    return dataset.select(indices)


def encode_image(image: Image.Image, image_processor: Any) -> trio.ImageChunk:
    """把 PIL 图片编码成 PyTRIO ImageChunk，并计算视觉 token 数。"""
    chunk_format = "jpeg" if image.format in {"JPG", "JPEG"} else "png"

    # 透明图层直接转 RGB 会变黑；先合成白色背景，保持几何图可读。
    rgba = image.convert("RGBA")
    background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    image = Image.alpha_composite(background, rgba).convert("RGB")

    buffer = io.BytesIO()
    image.save(buffer, format=chunk_format.upper())
    patches = image_processor.get_number_of_image_patches(
        image.height,
        image.width,
        images_kwargs={},
    )
    expected_tokens = patches // int(image_processor.merge_size) ** 2
    if expected_tokens <= 0:
        raise ValueError(f"视觉 token 数必须为正数，实际为 {expected_tokens}")

    return trio.ImageChunk(
        data=buffer.getvalue(),
        format=chunk_format,
        expected_tokens=expected_tokens,
    )


def format_question(subject: str, choices: list[str]) -> str:
    """把题目和四个选项整理成可验证的选择题 prompt。"""
    choice_lines = "\n".join(
        f"{letter}. {choice}"
        for letter, choice in zip(CHOICE_LETTERS, choices, strict=True)
    )
    return (
        "请根据图片解答下面的几何选择题。\n"
        f"题目：{subject.strip()}\n"
        f"选项：\n{choice_lines}\n"
        "请先进行简单逻辑推理思考，再给出最终答案。"
        "最终选项格式必须是 \\boxed{A}、\\boxed{B}、\\boxed{C} 或 \\boxed{D}。"
    )


def build_prompt_chunks(
    tokenizer: Any,
    image_processor: Any,
    image: Image.Image,
    subject: str,
    choices: list[str],
) -> list[Any]:
    """先渲染模型 chat template，再用真实 ImageChunk 替换图片占位符。"""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": format_question(subject, choices)},
                {"type": "image", "image": "geoqa"},
            ],
        }
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    parts = prompt.split(IMAGE_PAD_TOKEN)
    if len(parts) != 2:
        raise ValueError(f"期望 1 个 {IMAGE_PAD_TOKEN}，实际得到 {len(parts) - 1} 个")
    before_image, after_image = parts
    return [
        trio.types.EncodedTextChunk(
            tokens=tokenizer.encode(before_image, add_special_tokens=False)
        ),
        encode_image(image, image_processor),
        trio.types.EncodedTextChunk(
            tokens=tokenizer.encode(after_image, add_special_tokens=False)
        ),
    ]


def extract_choice(text: str) -> str | None:
    """提取回答中最后一个合法的 boxed 选项。"""
    matches = BOXED_CHOICE_PATTERN.findall(text)
    return matches[-1].upper() if matches else None


async def run_rollout_group(
    sampling_client: Any,
    tokenizer: Any,
    prompt_chunks: list[Any],
    gold_choice: str,
    sampling_params: trio.SamplingParams,
    group_size: int,
) -> RolloutGroup:
    """异步采样同一道题的一组回答，并计算组相对 advantage。"""
    prompt = trio.ModelInput(chunks=prompt_chunks)
    prompt_length = len(prompt)
    response = await sampling_client.sample_async(
        prompt=prompt,
        num_samples=group_size,
        sampling_params=sampling_params,
        return_text=True,
    )
    if response.input_tokens != prompt_length:
        raise ValueError(
            f"图文 prompt 长度不一致：local={prompt_length}, "
            f"remote={response.input_tokens}"
        )
    if len(response.sequences) != group_size:
        raise ValueError(
            f"采样数量不一致：expected={group_size}, actual={len(response.sequences)}"
        )

    raw_samples: list[tuple[list[int], list[float], str, str | None, float]] = []
    rewards: list[float] = []
    for sequence in response.sequences:
        tokens = list(sequence.tokens)
        logprobs = [float(value) for value in sequence.logprobs]
        if len(tokens) != len(logprobs):
            raise ValueError("生成 token 与 logprob 长度不一致")
        text = sequence.text or tokenizer.decode(tokens, skip_special_tokens=True)
        predicted_choice = extract_choice(text)
        reward = float(predicted_choice == gold_choice)
        rewards.append(reward)
        raw_samples.append((tokens, logprobs, text, predicted_choice, reward))

    mean_reward = sum(rewards) / len(rewards)
    samples = [
        RolloutSample(
            tokens=tokens,
            logprobs=logprobs,
            text=text,
            predicted_choice=predicted_choice,
            reward=reward,
            advantage=reward - mean_reward,
        )
        for tokens, logprobs, text, predicted_choice, reward in raw_samples
    ]
    return RolloutGroup(prompt_chunks, prompt_length, samples)


def build_grpo_datum(group: RolloutGroup, sample: RolloutSample) -> trio.Datum:
    """把图文 prompt 和一条 completion 对齐成 importance_sampling Datum。"""
    if not sample.tokens:
        raise ValueError("不能用空 completion 构造 GRPO Datum")
    if len(sample.tokens) != len(sample.logprobs):
        raise ValueError("completion token 与 old logprob 必须等长")

    model_input = trio.ModelInput(
        chunks=[
            *group.prompt_chunks,
            trio.types.EncodedTextChunk(tokens=sample.tokens[:-1]),
        ]
    )
    observation_length = group.prompt_length - 1
    target_tokens = np.asarray(
        [0] * observation_length + sample.tokens,
        dtype=np.int64,
    )
    padded_logprobs = np.asarray(
        [0.0] * observation_length + sample.logprobs,
        dtype=np.float32,
    )
    padded_advantages = np.asarray(
        [0.0] * observation_length + [sample.advantage] * len(sample.tokens),
        dtype=np.float32,
    )
    if not (
        len(model_input)
        == len(target_tokens)
        == len(padded_logprobs)
        == len(padded_advantages)
    ):
        raise ValueError("Vision GRPO Datum 的所有字段必须严格等长")

    return trio.Datum(
        model_input=model_input,
        loss_fn_inputs={
            "target_tokens": target_tokens,
            "logprobs": padded_logprobs,
            "advantages": padded_advantages,
        },
    )


def init_swanlab(args: argparse.Namespace, dataset_size: int) -> Any:
    """记录 Vision GRPO 的配置、reward、格式和训练指标。"""
    return swanlab.init(
        mode=args.swanlab_mode,
        project=args.swanlab_project,
        experiment_name=args.experiment_name,
        config={
            "algorithm": "vision-grpo",
            "dataset": args.dataset_id,
            "dataset_revision": args.dataset_revision,
            "dataset_size": dataset_size,
            "base_model": args.base_model,
            "pytrio_version": version("pytrio"),
            "enable_thinking": False,
            "lora_rank": args.lora_rank,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "group_size": args.group_size,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "learning_rate": args.learning_rate,
            "save_every": args.save_every,
        },
    )


async def save_checkpoint(
    training_client: trio.TrainingClient,
    weights_name: str,
    step: int,
) -> None:
    """同时保存推理用 sampler weights 和可续训的完整 state。"""
    prefix = f"{weights_name}-step-{step}"
    sampler_future = await training_client.save_weights_for_sampler_async(
        name=f"{prefix}-sampler"
    )
    state_future = await training_client.save_state_async(name=f"{prefix}-state")
    sampler_weights, training_state = await asyncio.gather(
        sampler_future,
        state_future,
    )
    print(f"Sampler 权重：{sampler_weights.path}")
    print(f"State 权重：{training_state.path}")


async def main(args: argparse.Namespace) -> None:
    if args.steps <= 0:
        raise ValueError("--steps 必须大于 0")
    if args.batch_size <= 0 or args.group_size <= 0:
        raise ValueError("--batch-size 和 --group-size 必须大于 0")
    if args.max_tokens <= 0:
        raise ValueError("--max-tokens 必须大于 0")

    cache_dir = (
        args.dataset_cache_dir.expanduser().resolve()
        if args.dataset_cache_dir is not None
        else None
    )
    train_data = load_geoqa_train(
        dataset_id=args.dataset_id,
        revision=args.dataset_revision,
        cache_dir=cache_dir,
        force_download=args.force_download,
        seed=args.seed,
        max_samples=args.max_samples,
    )
    print(f"加载 GeoQA train 数据：{len(train_data)} 条")
    print(f"PyTRIO：{version('pytrio')}")

    service_client = trio.ServiceClient()
    training_client = await service_client.create_lora_training_client_async(
        base_model=args.base_model,
        rank=args.lora_rank,
        seed=args.seed,
    )
    tokenizer = training_client.get_tokenizer()
    image_processor = AutoImageProcessor.from_pretrained(
        args.base_model,
        backend="pil",
    )
    sampling_params = trio.SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        stop="<|im_end|>",
    )
    adam_params = trio.AdamParams(learning_rate=args.learning_rate)
    swanlab_run = init_swanlab(args, len(train_data))
    last_saved_step = 0

    try:
        for step in range(args.steps):
            batch_rows = list(pick_batch(train_data, step, args.batch_size))

            # 每个 step 都从更新前的同一版 LoRA 创建 sampler，保证 rollout on-policy。
            sampling_client = (
                await training_client.save_weights_and_get_sampling_client_async()
            )
            datums: list[trio.Datum] = []
            all_samples: list[RolloutSample] = []
            prompt_rewards: list[float] = []
            degenerate_groups = 0

            # 不同题目可以并发；同一道题的一组 completion 在一次请求中返回。
            rollout_groups = await tqdm_asyncio.gather(
                *(
                    run_rollout_group(
                        sampling_client=sampling_client,
                        tokenizer=tokenizer,
                        prompt_chunks=build_prompt_chunks(
                            tokenizer=tokenizer,
                            image_processor=image_processor,
                            image=row["image"],
                            subject=str(row["subject"]),
                            choices=[str(choice) for choice in row["choices"]],
                        ),
                        gold_choice=CHOICE_LETTERS[int(row["label"])],
                        sampling_params=sampling_params,
                        group_size=args.group_size,
                    )
                    for row in batch_rows
                ),
                desc=f"Step {step + 1}/{args.steps} rollout",
                unit="题",
            )

            for row, group in zip(batch_rows, rollout_groups, strict=True):
                gold_choice = CHOICE_LETTERS[int(row["label"])]
                all_samples.extend(group.samples)
                rewards = [sample.reward for sample in group.samples]
                prompt_rewards.append(sum(rewards) / len(rewards))

                if args.show_samples:
                    print(f"\nGeoQA id={row['id']} gold={gold_choice}")
                    for index, sample in enumerate(group.samples):
                        print(
                            f"  sample={index} predicted={sample.predicted_choice} "
                            f"reward={sample.reward:.0f} text={sample.text!r}"
                        )

                # 全对或全错时组相对 advantage 全为零，不提交无效 backward。
                if len(set(rewards)) == 1:
                    degenerate_groups += 1
                    continue
                datums.extend(
                    build_grpo_datum(group, sample)
                    for sample in group.samples
                    if sample.tokens
                )

            mean_output_tokens = sum(
                len(sample.tokens) for sample in all_samples
            ) / len(all_samples)
            tqdm_asyncio.write(
                f"本 batch 平均输出长度：{mean_output_tokens:.1f} tokens"
            )

            trainer_metrics: dict[str, float] = {}
            if datums:
                forward_backward = await training_client.forward_backward_async(
                    datums,
                    loss_fn="importance_sampling",
                )
                optim_step = await training_client.optim_step_async(adam_params)
                result = await forward_backward
                await optim_step
                trainer_metrics = {
                    key: float(value) for key, value in result.metrics.items()
                }

            mean_reward = sum(prompt_rewards) / len(prompt_rewards)
            format_rate = sum(
                sample.predicted_choice is not None for sample in all_samples
            ) / len(all_samples)
            degenerate_fraction = degenerate_groups / len(prompt_rewards)
            metrics = {
                "reward": mean_reward,
                "format_rate": format_rate,
                "degenerate_fraction": degenerate_fraction,
                "train_datums": len(datums),
                "rollout/completion_tokens_mean": mean_output_tokens,
                **{f"trainer/{key}": value for key, value in trainer_metrics.items()},
            }
            swanlab.log(metrics, step=step)

            loss_mean = trainer_metrics.get("loss_mean")
            loss_text = "n/a" if loss_mean is None else f"{loss_mean:.4f}"
            print(
                f"Step {step + 1}/{args.steps} | reward={mean_reward:.3f} | "
                f"format={format_rate:.1%} | "
                f"degenerate={degenerate_fraction:.1%} | "
                f"datums={len(datums)} | loss_mean={loss_text}",
                flush=True,
            )

            current_step = step + 1
            if (
                args.save_weights
                and args.save_every > 0
                and current_step % args.save_every == 0
            ):
                await save_checkpoint(training_client, args.weights_name, current_step)
                last_saved_step = current_step

        if args.save_weights and last_saved_step != args.steps:
            await save_checkpoint(training_client, args.weights_name, args.steps)
    finally:
        swanlab_run.finish()


if __name__ == "__main__":
    start_time = time.perf_counter()
    asyncio.run(main(parse_args()))
    print(f"训练耗时：{time.perf_counter() - start_time:.2f}s")
