"""
PyTRIO 最小推理示例

场景：使用基础模型（或 LoRA 权重）生成文本
闭环：创建采样客户端 → 编码 prompt → 采样生成 → 解码输出
"""

import pytrio as trio

# ========== 1. 初始化 ==========
client = trio.ServiceClient(api_key="YOUR_API_KEY")

# 方式 A: 基础模型推理（无 LoRA）
sampler = client.create_sampling_client(base_model="Qwen/Qwen3-4B-Instruct-2507")

# 方式 B: 加载 LoRA 权重推理（需要先有保存的权重路径）
# sampler = client.create_sampling_client(
#     base_model="Qwen/Qwen3-4B-Instruct-2507",
#     model_path="path/from/save_weights_for_sampler",
# )

# ========== 2. 获取 tokenizer ==========
tokenizer = sampler.get_tokenizer()

# ========== 3. 采样生成 ==========
prompt = "今天天气怎么样？"
input_ids = tokenizer.encode(prompt)

params = trio.SamplingParams(
    max_tokens=100,
    temperature=0.7,
    top_p=0.9,
)

future = sampler.sample(
    prompt=trio.ModelInput.from_ints(input_ids),
    num_samples=1,
    sampling_params=params,
)
result = future.result()

# 解码输出
for i, seq in enumerate(result.sequences):
    print(f"--- 生成 {i} ---")
    print(f"文本: {seq.text}")
    print(f"Token 数: {len(seq.tokens)}")
    print(f"停止原因: {seq.stop_reason}")

print(f"耗时: {result.elapsed_time}ms")

# ========== 4. 计算 logprobs ==========
lp_future = sampler.compute_logprobs(prompt=trio.ModelInput.from_ints(input_ids))
lp_result = lp_future.result()
print(f"Prompt logprobs: {lp_result['prompt_logprobs']}")
