"""
PyTRIO 最小训练示例 — SFT (Supervised Fine-Tuning)

场景：使用交叉熵损失对 Qwen3-4B 做 LoRA 微调
闭环：数据准备 → 前向+反向 → 优化器步骤 → 保存权重
"""

import pytrio as trio

# ========== 1. 初始化 ==========
client = trio.ServiceClient(api_key="YOUR_API_KEY")
print("可用模型:", client.get_supported_models())

train = client.create_lora_training_client(
    base_model="Qwen/Qwen3-4B-Instruct-2507",
    rank=32,
    train_mlp=True,
    train_attn=True,
)

# 获取 tokenizer 用于数据编码
tokenizer = train.get_tokenizer()

# ========== 2. 准备训练数据 ==========
text = "Hello, world! This is a test."
tokens = tokenizer.encode(text, add_special_tokens=False)

# 手动对齐：input[i] 预测 target[i]
model_input = tokens[:-1]
target_tokens = tokens[1:]

sample = trio.Datum(
    model_input=trio.ModelInput.from_ints(model_input),
    loss_fn_inputs={
        "target_tokens": target_tokens,
        # weights 可选，默认全 1.0
    },
)

# ========== 3. 训练循环 ==========
for step in range(3):
    # 前向 + 反向（返回 APIFuture）
    fb_future = train.forward_backward(
        data=[sample],
        loss_fn="cross_entropy",
    )
    fb_result = fb_future.result()
    print(f"Step {step} metrics:", fb_result.metrics)

    # 优化器步骤
    opt_future = train.optim_step(trio.AdamParams(
        learning_rate=1e-4,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
    ))
    opt_result = opt_future.result()
    print(f"Step {step} optim:", opt_result.metrics)

# ========== 4. 保存权重 ==========
save_future = train.save_weights_for_sampler("sft_example")
save_result = save_future.result()
print(f"权重已保存到: {save_result.path}")
print(f"文件大小: {save_result.size} bytes")
