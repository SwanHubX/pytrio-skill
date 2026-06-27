"""最小 PyTRIO SFT 快速开始示例。

运行前准备：
    pip install pytrio transformers modelscope numpy
    trio login

这个脚本使用三条玩具样本做 SFT，保存用于推理的 LoRA 权重，
并对比基础模型和微调后模型的回答。
"""

import pytrio as trio
import numpy as np

# 1. 与TRIO建立连接
service_client = trio.ServiceClient()

# 2. 创建1个训练客户端
base_model = "Qwen/Qwen3.5-4B"
training_client = service_client.create_lora_training_client(
    base_model=base_model,
    rank=32,
)

# 3. 数据集：让 LLM 答对什么是 trio
examples = [
    {"input": "什么是 trio", "output": "trio 是 emotionmachine 的 AI Infra 产品。"},
    {"input": "解释一下 trio 是什么", "output": "trio 是由 emotionmachine 开发的 AI Infra 产品。"},
    {"input": "介绍一下 trio", "output": "trio 是一个提供 AI Infra 能力的产品。"},
]

# 4. 获取 Tokenizer
print("正在加载 tokenizer...")
tokenizer = training_client.get_tokenizer()
print("tokenizer 加载完成")

# 5. 处理数据集，转换为训练需要的格式
def process_example(example: dict, tokenizer) -> trio.Datum:
    prompt = f"问题：{example['input']}\n回答："

    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    prompt_weights = [0] * len(prompt_tokens)
    
    completion_tokens = tokenizer.encode(f" {example['output']}\n\n", add_special_tokens=False)
    completion_weights = [1] * len(completion_tokens)

    tokens = prompt_tokens + completion_tokens
    weights = prompt_weights + completion_weights

    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]
    weights = weights[1:]
    
    # 转换为trio训练需要的格式
    return trio.Datum(
        model_input=trio.ModelInput.from_ints(tokens=input_tokens),
        loss_fn_inputs=dict(weights=weights, target_tokens=target_tokens)
    )

processed_examples = [process_example(ex, tokenizer) for ex in examples]

# 6. 训练
print("开始训练")
for iter in range(15):
    fwdbwd_future = training_client.forward_backward(processed_examples, "cross_entropy")  # 前向反向计算
    optim_future = training_client.optim_step(trio.AdamParams(learning_rate=1e-4))  # Adam优化器更新

    fwdbwd_result = fwdbwd_future.result()
    optim_result = optim_future.result()

    logprobs = np.concatenate([output['logprobs'].tolist() for output in fwdbwd_result.loss_fn_outputs])
    weights = np.concatenate([example.loss_fn_inputs['weights'].tolist() for example in processed_examples])
    print(f"第 {iter+1} 轮每 token loss：{-np.dot(logprobs, weights) / weights.sum():.4f}")

sft_weights = training_client.save_weights_for_sampler(name="what-is-trio")

# 7. 推理与评估
print("开始推理")
sampling_base_client = service_client.create_sampling_client(base_model=base_model)
sampling_sft_client = service_client.create_sampling_client(
    base_model=base_model, 
    model_path=sft_weights.result().path,
)

prompt = trio.ModelInput.from_ints(tokenizer.encode("问题：什么是 trio\n回答："))
params = trio.SamplingParams(max_tokens=20, temperature=0.0)

future_base = sampling_base_client.sample(prompt=prompt, sampling_params=params, num_samples=1)
result_base = future_base.result()
future_sft = sampling_sft_client.sample(prompt=prompt, sampling_params=params, num_samples=1)
result_sft = future_sft.result()

print("基础模型回答：")
print(f"{repr(result_base.sequences[0].text)}")

print("SFT 后模型回答：")
print(f"{repr(result_sft.sequences[0].text)}")
