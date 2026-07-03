import os
from pathlib import Path
import random
import time
from typing import Any

from datasets import DatasetDict, load_from_disk
import numpy as np
import pytrio as trio
import swanlab
from tqdm import tqdm


# 基础训练配置：按需替换模型、数据集和 LoRA 权重名称。
BASE_MODEL = os.getenv("TRIO_BASE_MODEL", "Qwen/Qwen3.5-4B")
DATASET_ID = "angrygiraffe/claude-opus-4.6-4.7-reasoning-8.7k"
DATASET_PATH = Path("datasets/claude-opus-4.6-4.7-reasoning-8.7k")
NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", "1"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "4"))
LORA_RANK = int(os.getenv("LORA_RANK", "32"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "1e-4"))
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "4096"))
SEED = int(os.getenv("SEED", "42"))
MAX_TRAIN_EXAMPLES = int(os.getenv("MAX_TRAIN_EXAMPLES", "0"))

# SwanLab 配置支持通过环境变量覆盖，方便复用同一份脚本跑多组实验。
SWANLAB_PROJECT = os.getenv("SWANLAB_PROJECT", "trio-case")
SWANLAB_EXPERIMENT_NAME = os.getenv(
    "SWANLAB_EXPERIMENT_NAME",
    "claude-opus47-conversation-mask-qwen35-4b",
)
SWANLAB_MODE = os.getenv("SWANLAB_MODE")
WEIGHTS_NAME = os.getenv("TRIO_WEIGHTS_NAME", SWANLAB_EXPERIMENT_NAME)


def resolve_local_path(path: Path) -> Path:
    return path if path.is_absolute() else Path.cwd() / path


def build_prompt_text(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    # 把已有上下文拼成完整 ChatML，并停在助手消息开头，供推理继续生成。
    # 推理时停在助手起始标记处，让模型自己生成 "<think>" 和后续回答。
    history_text = "".join(build_message_text(message) for message in messages)
    return history_text + "<|im_start|>assistant\n"


def build_message_text(message: dict[str, str]) -> str:
    # 把一条 system/user/assistant 消息转换成 Qwen 使用的 ChatML 片段。
    return f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"


# 加载数据集
def load_examples(dataset_path: Path) -> list[dict[str, Any]]:
    # 每条原始多轮对话保留为一条训练样本，不再按助手轮次拆分。
    dataset = load_from_disk(str(dataset_path))
    raw_examples = dataset["train"] if isinstance(dataset, DatasetDict) else dataset
    examples: list[dict[str, Any]] = []

    for row_index, item in enumerate(raw_examples):
        messages = [
            {
                "role": str(message.get("role", "")).strip(),
                "content": str(message.get("content", "")).strip(),
            }
            for message in item["messages"]
        ]
        messages = [message for message in messages if message["role"] and message["content"]]
        assistant_count = sum(1 for message in messages if message["role"] == "assistant")
        if assistant_count == 0:
            continue
        # 统计这条对话里有多少个助手回复带完整思考标签。
        thinking_count = sum(
            1
            for message in messages
            if message["role"] == "assistant"
            and "<think>" in message["content"]
            and "</think>" in message["content"]
        )
        # conversation-mask 版本按完整对话训练；只要某个助手回复没有 reasoning，整条对话就跳过。
        # 这样保留下来的每个 loss 位置都来自带 <think>...</think> 的助手内容。
        if thinking_count != assistant_count:
            continue

        examples.append(
            {
                "messages": messages,
                "row_index": row_index,
                "category": item.get("category", ""),
                "source_model": item.get("model", ""),
                "assistant_count": assistant_count,
                "thinking_count": thinking_count,
            }
        )

    if not examples:
        raise ValueError(f"No valid conversation examples found in {dataset_path}")

    random.Random(SEED).shuffle(examples)
    if MAX_TRAIN_EXAMPLES > 0:
        examples = examples[:MAX_TRAIN_EXAMPLES]

    return examples


def encode_with_weight(tokenizer, text: str, weight: int) -> tuple[list[int], list[int]]:
    tokens = tokenizer.encode(text, add_special_tokens=False)
    return tokens, [weight] * len(tokens)


def build_tokens_and_weights(example: dict[str, Any], tokenizer) -> tuple[list[int], list[int], bool]:
    tokens: list[int] = []
    weights: list[int] = []

    for message in example["messages"]:
        if message["role"] == "assistant":
            # 助手起始标记只是格式上下文，不参与损失计算。
            prefix_tokens, prefix_weights = encode_with_weight(
                tokenizer,
                "<|im_start|>assistant\n",
                0,
            )
            # 按数据集里的原始助手回复监督；不额外给无思考样本补空的 "<think></think>"。
            # 助手回复内容和结束标记都参与损失计算。
            completion_tokens, completion_weights = encode_with_weight(
                tokenizer,
                message["content"].strip() + "<|im_end|>\n",
                1,
            )
            tokens.extend(prefix_tokens)
            weights.extend(prefix_weights)
            tokens.extend(completion_tokens)
            weights.extend(completion_weights)
        else:
            # 系统消息和用户消息只作为上下文，不参与损失计算。
            context_tokens, context_weights = encode_with_weight(
                tokenizer,
                build_message_text(message),
                0,
            )
            tokens.extend(context_tokens)
            weights.extend(context_weights)

    was_truncated = len(tokens) > MAX_LENGTH
    if len(tokens) > MAX_LENGTH:
        tokens = tokens[:MAX_LENGTH]
        weights = weights[:MAX_LENGTH]

    return tokens, weights, was_truncated


def build_datum(example: dict[str, Any], tokenizer) -> trio.Datum | None:
    tokens, weights, _ = build_tokens_and_weights(example, tokenizer)

    # 过滤右移后无效，或截断后没有任何助手回复参与损失计算的样本。
    if len(tokens) < 2 or sum(weights) == 0:
        return None

    # 自回归训练需要右移一位：输入词元预测目标词元，损失权重对齐目标词元。
    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]
    loss_weights = weights[1:]

    return trio.Datum(
        model_input=trio.ModelInput.from_ints(tokens=input_tokens),
        loss_fn_inputs={
            "weights": np.asarray(loss_weights, dtype=np.float32),
            "target_tokens": np.asarray(target_tokens, dtype=np.int32),
        },
    )


def count_tokens_and_weights(example: dict[str, Any], tokenizer) -> tuple[int, int, bool]:
    # 仅用于数据预览：统计截断后的总词元数、参与损失计算的词元数和是否被截断。
    tokens, weights, was_truncated = build_tokens_and_weights(example, tokenizer)
    return len(tokens), sum(weights), was_truncated


def print_examples_preview(examples: list[dict[str, Any]], tokenizer, limit: int = 3) -> None:
    actual_token_lengths = []
    actual_loss_token_lengths = []
    truncated_count = 0

    # 统计会完整扫一遍过滤后的对话，口径和 build_datum 保持一致。
    for example in tqdm(examples, desc="统计样本长度", unit="sample"):
        actual_tokens, actual_loss_tokens, was_truncated = count_tokens_and_weights(example, tokenizer)
        # 这里的可训练样本定义和 build_datum 一样：右移后有效，且至少有一个 loss token。
        if actual_tokens >= 2 and actual_loss_tokens > 0:
            actual_token_lengths.append(actual_tokens)
            actual_loss_token_lengths.append(actual_loss_tokens)
            truncated_count += int(was_truncated)

    print("\n数据集统计")
    print(f"过滤后样本数: {len(examples)}")
    print(f"实际参与训练样本数: {len(actual_token_lengths)}")
    print(f"最大长度限制: {MAX_LENGTH}")
    if actual_token_lengths:
        print(
            "总 token 数: "
            f"max={max(actual_token_lengths)}, "
            f"avg={np.mean(actual_token_lengths):.1f}, "
            f"median={np.median(actual_token_lengths):.1f}"
        )
        print(
            "loss token 数: "
            f"max={max(actual_loss_token_lengths)}, "
            f"avg={np.mean(actual_loss_token_lengths):.1f}, "
            f"median={np.median(actual_loss_token_lengths):.1f}"
        )
        print(f"被截断样本数: {truncated_count}")

    print("\n数据处理预览")
    for index, example in enumerate(examples[:limit], start=1):
        roles = [message["role"] for message in example["messages"]]
        actual_tokens, actual_loss_tokens, was_truncated = count_tokens_and_weights(example, tokenizer)
        first_user = next(
            (message["content"] for message in example["messages"] if message["role"] == "user"),
            "",
        )
        first_assistant = next(
            (message["content"] for message in example["messages"] if message["role"] == "assistant"),
            "",
        )
        thinking_count = sum(
            1
            for message in example["messages"]
            if message["role"] == "assistant"
            and "<think>" in message["content"]
            and "</think>" in message["content"]
        )

        print(f"\n样本 {index}")
        print(f"row_index: {example['row_index']}")
        print(f"category: {example['category']}, source_model: {example['source_model']}")
        print(f"roles: {roles}")
        print(f"assistant_count: {example['assistant_count']}, thinking_count: {thinking_count}")
        print(
            f"actual_tokens: {actual_tokens}, actual_loss_tokens: {actual_loss_tokens}, "
            f"truncated: {was_truncated}"
        )
        print(f"first_user: {' '.join(first_user.split())[:180]}")
        print(f"first_assistant: {' '.join(first_assistant.split())[:260]}")


def evaluate_client(client, tokenizer, prompts: list[str], title: str) -> None:
    # 训练前后都用同一组提示词测试，便于观察 LoRA 微调带来的变化。
    print(f"\n{title}")
    stop_tokens = [tokenizer.eos_token] if tokenizer.eos_token else ["<|im_end|>"]
    params = trio.SamplingParams(max_tokens=512, temperature=0.2, stop=stop_tokens)

    for prompt in prompts:
        messages = [
            {
                "role": "system",
                "content": "You are a helpful AI assistant. Show your thinking in <think> tags when useful.",
            },
            {"role": "user", "content": prompt},
        ]
        prompt_text = build_prompt_text(tokenizer, messages)
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        # 同步 sample 返回 APIFuture，必须 .result() 后才是 SampleResponse。
        future = client.sample(
            prompt=trio.ModelInput.from_ints(prompt_ids),
            sampling_params=params,
            num_samples=1,
        )
        result = future.result()
        print(f"User: {prompt}")
        print(f"Assistant: {result.sequences[0].text.strip()}\n")


def main() -> None:
    # 相对路径按运行目录解析，避免把数据写入或读取到 skill 安装目录。
    dataset_path = resolve_local_path(DATASET_PATH)
    examples = load_examples(dataset_path)
    print(f"Loaded {len(examples)} conversation training examples from {dataset_path}")

    # 创建 PyTrio 服务客户端，并基于指定基座模型创建 LoRA 训练客户端。
    service_client = trio.ServiceClient()
    training_client = service_client.create_lora_training_client(
        base_model=BASE_MODEL,
        rank=LORA_RANK,
    )

    print("Loading tokenizer...")
    tokenizer = training_client.get_tokenizer()
    print("Tokenizer ready")
    print_examples_preview(examples, tokenizer)

    # 预先把原始对话样本转换成 PyTrio 训练所需的 Datum。
    processed_examples = []
    for example in tqdm(examples, desc="处理训练样本", unit="sample"):
        datum = build_datum(example, tokenizer)
        if datum is not None:
            processed_examples.append(datum)
    if not processed_examples:
        raise ValueError("No valid training examples after tokenization")

    print("Start training")
    # 计算每个 epoch 的训练步数和总步数，便于进度条显示和 SwanLab 日志记录。
    steps_per_epoch = (len(processed_examples) + BATCH_SIZE - 1) // BATCH_SIZE
    total_steps = NUM_EPOCHS * steps_per_epoch

    # 把关键超参数写入 SwanLab，便于后续复现实验。
    swanlab_init_kwargs = {
        "project": SWANLAB_PROJECT,
        "name": SWANLAB_EXPERIMENT_NAME,
        "config": {
            "base_model": BASE_MODEL,
            "dataset_id": DATASET_ID,
            "dataset_path": str(dataset_path),
            "weights_name": WEIGHTS_NAME,
            "num_epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "lora_rank": LORA_RANK,
            "learning_rate": LEARNING_RATE,
            "max_length": MAX_LENGTH,
            "num_examples": len(processed_examples),
            "steps_per_epoch": steps_per_epoch,
            "total_steps": total_steps,
            "loss_policy": "one datum per conversation; assistant contents use weight=1; system/user use weight=0",
        },
    }
    if SWANLAB_MODE:
        swanlab_init_kwargs["mode"] = SWANLAB_MODE

    swanlab_run = swanlab.init(**swanlab_init_kwargs)

    progress_bar = tqdm(total=total_steps, desc="SFT Training", unit="batch")
    for epoch in range(NUM_EPOCHS):
        for start in range(0, len(processed_examples), BATCH_SIZE):
            batch = processed_examples[start:start + BATCH_SIZE]
            batch_index = start // BATCH_SIZE
            global_step = epoch * steps_per_epoch + batch_index

            # 提交训练任务，进行前向和反向传播，并更新优化器参数。
            fwdbwd_future = training_client.forward_backward(batch, "cross_entropy")
            optim_future = training_client.optim_step(trio.AdamParams(learning_rate=LEARNING_RATE))
            fwdbwd_result = fwdbwd_future.result()
            optim_future.result()

            # PyTrio 返回每个词元的对数概率，这里按损失权重求加权平均损失。
            logprobs = np.concatenate(
                [output["logprobs"].tolist() for output in fwdbwd_result.loss_fn_outputs]
            )
            weights = np.concatenate(
                [example.loss_fn_inputs["weights"].tolist() for example in batch]
            )
            loss = -np.dot(logprobs, weights) / weights.sum()
            swanlab.log(
                {
                    "loss": float(loss),
                    "epoch": epoch + 1,
                    "batch": batch_index + 1,
                },
                step=global_step,
            )
            progress_bar.update(1)
            progress_bar.set_postfix(epoch=f"{epoch + 1}/{NUM_EPOCHS}", loss=f"{loss:.4f}")

    progress_bar.close()
    print("Saving LoRA weights...")

    # 保存 LoRA 权重，并拿到带 LoRA 权重的采样客户端用于效果测试。
    sft_weights_future = training_client.save_weights_for_sampler(name=WEIGHTS_NAME)
    sft_weights = sft_weights_future.result()
    # 未训练前的基座模型采样客户端，用于对比训练前后的效果。
    base_sampling_client = service_client.create_sampling_client(base_model=BASE_MODEL)
    # 训练后带 LoRA 权重的采样客户端，用于对比训练前后的效果。
    tuned_sampling_client = service_client.create_sampling_client(
        base_model=BASE_MODEL,
        model_path=sft_weights.path,
    )
    # 测试提示词列表，便于观察 LoRA 微调带来的变化。
    test_prompts = [
        "A triangle has sides 5, 12, and 13. Is it a right triangle? What is its area?",
        "Explain why Docker containers should handle SIGTERM.",
        "I have a flaky async test around a cache TTL. How should I test it?",
    ]

    # 训练前后都用同一组提示词测试，便于观察 LoRA 微调带来的变化。
    evaluate_client(base_sampling_client, tokenizer, test_prompts, title="Base model responses")
    evaluate_client(tuned_sampling_client, tokenizer, test_prompts, title="Fine-tuned model responses")

    print(f"Saved weights name: {WEIGHTS_NAME}, weights path: {sft_weights.path}")
    swanlab_run.finish()


if __name__ == "__main__":
    start_main_time = time.time()
    main()
    end_main_time = time.time()
    print("#" * 50)
    print("# all done")
    print(f"# train cost {end_main_time - start_main_time:.2f}s")
    print("#" * 50)
