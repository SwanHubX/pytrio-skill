import json
import os
from pathlib import Path
import time

import numpy as np
import pytrio as trio
import swanlab
from tqdm import tqdm

# 基础训练配置：按需替换模型、数据集和 LoRA 权重名称。
BASE_MODEL = "Qwen/Qwen3.5-4B"
DATASET_PATH = Path("dataset/huanhuan.json")
NUM_EPOCHS = 3
BATCH_SIZE = 16
LORA_RANK = 32
LEARNING_RATE = 1e-4
MAX_LENGTH = 1024
SYSTEM_PROMPT = "现在你要扮演皇帝身边的女人--甄嬛"

# SwanLab 配置支持通过环境变量覆盖，方便复用同一份脚本跑多组实验。
SWANLAB_PROJECT = os.getenv("SWANLAB_PROJECT", "trio-case")
SWANLAB_EXPERIMENT_NAME = os.getenv("SWANLAB_EXPERIMENT_NAME", "chat-huanhuan-qwen35-4b")
WEIGHTS_NAME = os.getenv("TRIO_WEIGHTS_NAME", SWANLAB_EXPERIMENT_NAME)


# 加载数据集
def load_examples(dataset_path: Path) -> list[dict[str, str]]:
    # 数据集是 JSON 数组，每条样本包含 instruction/input/output 三个字段。
    raw_examples = json.loads(dataset_path.read_text(encoding="utf-8"))
    examples: list[dict[str, str]] = []

    for item in raw_examples:
        instruction = item.get("instruction", "").strip()
        input_text = item.get("input", "").strip()
        output_text = item.get("output", "").strip()

        if not instruction or not output_text:
            continue

        # input 为空时只使用 instruction；否则把 instruction 和 input 合并成用户输入。
        user_text = instruction if not input_text else f"{instruction}\n{input_text}"
        examples.append({"user": user_text, "assistant": output_text})

    if not examples:
        raise ValueError(f"没有在 {dataset_path} 中找到有效训练样本")

    return examples


def build_datum(example: dict[str, str], tokenizer) -> trio.Datum:
    # system prompt 用于固定角色设定，user 内容来自数据集里的 instruction/input。
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": example["user"]},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    # prompt 部分不参与 loss，等价于常见 SFT 代码里 labels 使用 -100。
    prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
    prompt_weights = [0] * len(prompt_tokens)

    # assistant 回复才是模型需要学习的目标，因此 loss 权重为 1。
    completion_tokens = tokenizer.encode(example["assistant"], add_special_tokens=False)
    completion_weights = [1] * len(completion_tokens)

    # 显式补上 EOS，让模型学习在回答结束处停止。
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is not None:
        completion_tokens = completion_tokens + [eos_token_id]
        completion_weights = completion_weights + [1]

    tokens = prompt_tokens + completion_tokens
    weights = prompt_weights + completion_weights
    if len(tokens) > MAX_LENGTH:
        # 超长样本直接截断，保持 tokens 和 weights 对齐。
        tokens = tokens[:MAX_LENGTH]
        weights = weights[:MAX_LENGTH]

    # 自回归训练需要右移一位：input 预测 target，loss_weights 对齐 target。
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


def evaluate_client(client, tokenizer, prompts: list[str], title: str) -> None:
    # 训练前后都用同一组 prompt 测试，便于观察 LoRA 微调带来的变化。
    print(f"\n{title}")
    stop_tokens = [tokenizer.eos_token] if tokenizer.eos_token else ["<|im_end|>"]
    params = trio.SamplingParams(max_tokens=80, temperature=0.0, stop=stop_tokens)

    for prompt in prompts:
        # 推理时也保留同一个 system prompt，保证训练和测试输入格式一致。
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        future = client.sample(
            prompt=trio.ModelInput.from_ints(prompt_ids),
            sampling_params=params,
            num_samples=1,
        )
        result = future.result()
        print(f"用户：{prompt}")
        print(f"助手：{result.sequences[0].text.strip()}\n")


def main() -> None:
    # 使用脚本所在目录拼接数据路径，避免从其他工作目录运行时找不到数据集。
    dataset_path = Path(__file__).resolve().parent / DATASET_PATH
    examples = load_examples(dataset_path)
    print(f"已从 {dataset_path} 加载 {len(examples)} 条训练样本")

    # 创建 PyTrio 服务客户端，并基于指定基座模型创建 LoRA 训练客户端。
    service_client = trio.ServiceClient()
    training_client = service_client.create_lora_training_client(
        base_model=BASE_MODEL,
        rank=LORA_RANK,
    )

    print("正在加载 tokenizer...")
    tokenizer = training_client.get_tokenizer()
    print("tokenizer 已就绪")

    # 预先把原始文本样本转换成 PyTrio 训练所需的 Datum。
    processed_examples = [build_datum(example, tokenizer) for example in examples]

    print("开始训练")
    # 计算每个 epoch 的训练步数和总步数，便于进度条显示和 SwanLab 日志记录。
    steps_per_epoch = (len(processed_examples) + BATCH_SIZE - 1) // BATCH_SIZE
    total_steps = NUM_EPOCHS * steps_per_epoch

    # 把关键超参数写入 SwanLab，便于后续复现实验。
    swanlab_init_kwargs = {
        "project": SWANLAB_PROJECT,
        "experiment_name": SWANLAB_EXPERIMENT_NAME,
        "config": {
            "base_model": BASE_MODEL,
            "dataset_path": str(DATASET_PATH),
            "weights_name": WEIGHTS_NAME,
            "num_epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "lora_rank": LORA_RANK,
            "learning_rate": LEARNING_RATE,
            "max_length": MAX_LENGTH,
            "system_prompt": SYSTEM_PROMPT,
            "num_examples": len(processed_examples),
            "steps_per_epoch": steps_per_epoch,
            "total_steps": total_steps,
        },
    }

    swanlab_run = swanlab.init(**swanlab_init_kwargs)

    progress_bar = tqdm(total=total_steps, desc="SFT 训练", unit="batch")
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

            # PyTrio 返回每个 token 的 logprob，这里按 loss 权重求加权平均 loss。
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
    print("正在保存 LoRA 权重...")

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
    # 测试 prompt 列表，便于观察 LoRA 微调带来的变化。
    test_prompts = [
        "你是谁？",
        "介绍一下你自己。",
        "朕今天偶感风寒，你觉得我该如何调养身体？",
    ]

    # 训练前后都用同一组 prompt 测试，便于观察 LoRA 微调带来的变化。
    evaluate_client(base_sampling_client, tokenizer, test_prompts, title="基础模型回答")
    evaluate_client(tuned_sampling_client, tokenizer, test_prompts, title="微调后模型回答")

    print(f"已保存权重名称：{WEIGHTS_NAME}，权重路径：{sft_weights.path}")
    swanlab_run.finish()


if __name__ == "__main__":
    start_main_time = time.time()
    main()
    end_main_time = time.time()
    print("#" * 50)
    print("# 全部完成")
    print(f"# 训练耗时 {end_main_time - start_main_time:.2f}s")
    print("#" * 50)
