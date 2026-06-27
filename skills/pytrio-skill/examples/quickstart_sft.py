"""最小 PyTRIO SFT 快速开始示例。

运行前准备：
    pip install pytrio transformers modelscope numpy
    trio login

这个脚本使用三条玩具样本做 SFT，保存用于推理的 LoRA 权重，
并对比基础模型和微调后模型的回答。
"""

import pytrio as trio
import numpy as np

# 1. 与 TRIO 建立连接
service_client = trio.ServiceClient()

# 2. 创建 1 个训练客户端
base_model = "Qwen/Qwen3.6-27B"
training_client = service_client.create_lora_training_client(
    base_model=base_model,
    rank=32,
)

# 3. 数据集：让 LLM 答对什么是 TRIO
SYSTEM_PROMPT = "You are a helpful assistant that answers questions about TRIO."
examples = [
    [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "what is trio"},
        {"role": "assistant", "content": "trio is emotionmachine's AI Infra products."}
    ],
    [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "can you explain what trio is"},
        {"role": "assistant", "content": "trio is an AI infra product developed by emotionmachine."}
    ],
    [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "tell me about trio"},
        {"role": "assistant", "content": "trio is a product from emotionmachine that provides AI Infra capabilities."}
    ]
]

# 4. 获取 Tokenizer
print("Loading tokenizer...")
tokenizer = training_client.get_tokenizer()
print("Tokenizer finish")

# 5. 处理数据集，转换为训练需要的格式
def process_example(messages: list[dict[str, str]], tokenizer) -> trio.Datum:
    prompt_messages = messages[:-1]
    completion = messages[-1]["content"]

    prompt = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)

    # 将 prompt tokens 的权重设为 0，避免它们在训练时参与 loss 计算。
    # 模型只学习预测 completion tokens。
    prompt_weights = [0] * len(prompt_tokens)

    completion_tokens = tokenizer.encode(completion, add_special_tokens=False)
    completion_weights = [1] * len(completion_tokens)

    # 给 completion tokens 和 weights 追加 EOS token
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is not None:
        completion_tokens = completion_tokens + [eos_token_id]
        completion_weights = completion_weights + [1]

    tokens = prompt_tokens + completion_tokens
    weights = prompt_weights + completion_weights

    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]
    loss_weights = weights[1:]

    # 转换为 TRIO 训练需要的格式
    return trio.Datum(
        model_input=trio.ModelInput.from_ints(tokens=input_tokens),
        loss_fn_inputs={
            "weights": np.asarray(loss_weights, dtype=np.float32),
            "target_tokens": np.asarray(target_tokens, dtype=np.int32),
        },
    )

processed_examples = [process_example(ex, tokenizer) for ex in examples]

# 6. 训练
print("Start Training")
for iter in range(15):
    fwdbwd_future = training_client.forward_backward(processed_examples, "cross_entropy")  # 前向反向计算
    optim_future = training_client.optim_step(trio.AdamParams(learning_rate=1e-4))  # Adam 优化器更新

    fwdbwd_result = fwdbwd_future.result()
    optim_result = optim_future.result()

    logprobs = np.concatenate([output['logprobs'].tolist() for output in fwdbwd_result.loss_fn_outputs])
    weights = np.concatenate([example.loss_fn_inputs['weights'].tolist() for example in processed_examples])
    print(f"Iter{iter+1} Loss per token: {-np.dot(logprobs, weights) / weights.sum():.4f}")

# 保存训练后的权重
sft_weights = training_client.save_weights_for_sampler(name="what-is-trio")

# 7. 推理与评估
print("Start Sampling")
sampling_base_client = service_client.create_sampling_client(base_model=base_model)
sampling_sft_client = service_client.create_sampling_client(
    base_model=base_model,
    model_path=sft_weights.result().path,
)

prompt_messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "what is trio"},
]
prompt_text = tokenizer.apply_chat_template(
    prompt_messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)
prompt = trio.ModelInput.from_ints(tokenizer.encode(prompt_text, add_special_tokens=False))
params = trio.SamplingParams(max_tokens=20, temperature=0.0)

future_base = sampling_base_client.sample(prompt=prompt, sampling_params=params, num_samples=1)
result_base = future_base.result()
future_sft = sampling_sft_client.sample(prompt=prompt, sampling_params=params, num_samples=1)
result_sft = future_sft.result()

print("Base Responses:")
print(f"{repr(result_base.sequences[0].text)}")

print("SFT Responses:")
print(f"{repr(result_sft.sequences[0].text)}")
