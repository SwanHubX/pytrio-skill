"""
PyTRIO 最小训练示例 — SFT (Supervised Fine-Tuning)

场景：使用交叉熵损失对 Qwen3-4B 做 LoRA 微调
闭环：数据准备 → 前向+反向 → 优化器步骤 → 保存权重

采用官方标准数据构造模式：
  - prompt 起始段 add_special_tokens=True
  - completion 拼接段 add_special_tokens=False
  - 先构造 full_weights，再与 target_tokens 同步偏移
"""

import numpy as np
import pytrio as trio

# ========== 1. 初始化 ==========
client = trio.ServiceClient(api_key="YOUR_API_KEY")
print("可用模型:", client.get_supported_models())

train = client.create_lora_training_client(
    base_model="Qwen/Qwen3-4B-Instruct-2507",
    rank=32,
)

tokenizer = train.get_tokenizer()

# ========== 2. 准备训练数据 ==========
examples = [
    {"input": "what is trio", "output": "trio is emotionmachine's AI Infra product."},
    {"input": "tell me about trio", "output": "trio is an AI infra product from emotionmachine."},
]


def process_example(example: dict) -> trio.Datum:
    prompt = f"Question: {example['input']}\nAnswer:"

    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    prompt_weights = [0] * len(prompt_tokens)

    completion_tokens = tokenizer.encode(f" {example['output']}\n\n", add_special_tokens=False)
    completion_weights = [1] * len(completion_tokens)

    tokens = prompt_tokens + completion_tokens
    weights = prompt_weights + completion_weights

    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]
    weights = weights[1:]

    return trio.Datum(
        model_input=trio.ModelInput.from_ints(tokens=input_tokens),
        loss_fn_inputs=dict(weights=weights, target_tokens=target_tokens),
    )


processed = [process_example(ex) for ex in examples]

# ========== 3. 训练循环 ==========
for step in range(3):
    fb_future = train.forward_backward(processed, "cross_entropy")
    opt_future = train.optim_step(trio.AdamParams(learning_rate=1e-4))
    fb_result = fb_future.result()
    opt_future.result()

    # per-token loss（从 loss_fn_outputs 算）
    logprobs = np.concatenate([o["logprobs"].tolist() for o in fb_result.loss_fn_outputs])
    weights = np.concatenate([ex.loss_fn_inputs["weights"].tolist() for ex in processed])
    loss = -np.dot(logprobs, weights) / weights.sum()
    print(f"Step {step} | loss per token: {loss:.4f}")

# ========== 4. 保存权重 ==========
save = train.save_weights_for_sampler("sft_example").result()
print(f"权重已保存: {save.path}")
