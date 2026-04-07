"""
PyTRIO 断点续训示例

场景：从之前保存的 checkpoint 恢复训练，继续优化
闭环：保存 checkpoint → 恢复训练 → 继续训练 → 再次保存
"""

import pytrio as trio

# ========== 1. 初始阶段训练 ==========
client = trio.ServiceClient(api_key="YOUR_API_KEY")

train = client.create_lora_training_client(
    base_model="Qwen/Qwen3-4B-Instruct-2507",
    rank=32,
)

tokenizer = train.get_tokenizer()

# 准备一条简单的训练数据
text = "The quick brown fox jumps over the lazy dog."
tokens = tokenizer.encode(text, add_special_tokens=False)
sample = trio.Datum(
    model_input=trio.ModelInput.from_ints(tokens[:-1]),
    loss_fn_inputs={"target_tokens": tokens[1:]},
)

# 训练 2 步
for step in range(2):
    fb = train.forward_backward(data=[sample], loss_fn="cross_entropy")
    fb.result()
    train.optim_step(trio.AdamParams(learning_rate=1e-4)).result()
    print(f"初始训练 Step {step} 完成")

# ========== 2. 保存完整 checkpoint（含优化器状态） ==========
state_future = train.save_state("checkpoint_step2")
state_result = state_future.result()
checkpoint_path = state_result.path
print(f"Checkpoint 已保存: {checkpoint_path}")

# ========== 3. 断点续训：加载 checkpoint + 优化器状态 ==========
# 注意：这里创建了新的 TrainingClient
train_resumed = client.create_training_client_from_state_with_optimizer(checkpoint_path)

# 继续训练 3 步
for step in range(3):
    fb = train_resumed.forward_backward(data=[sample], loss_fn="cross_entropy")
    fb.result()
    train_resumed.optim_step(trio.AdamParams(learning_rate=1e-4)).result()
    print(f"续训 Step {step} 完成")

# ========== 4. 保存最终权重用于推理 ==========
final = train_resumed.save_weights_for_sampler("resumed_final")
final_result = final.result()
print(f"最终权重: {final_result.path}")

# ========== 备注 ==========
# create_training_client_from_state(path)         — 仅加载权重，不含优化器（用于继续微调，不保留动量）
# create_training_client_from_state_with_optimizer(path) — 加载权重+优化器状态（完整断点续训）
