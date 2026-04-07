"""
PyTRIO 重要性采样训练示例

场景：使用 importance_sampling loss 做 GRPO/PPO 风格训练
闭环：准备含 logprobs/advantages 的数据 → 前向+反向 → 优化
"""

import pytrio as trio

# ========== 1. 初始化 ==========
client = trio.ServiceClient(api_key="YOUR_API_KEY")

train = client.create_lora_training_client(
    base_model="Qwen/Qwen3-4B-Instruct-2507",
    rank=32,
)

# ========== 2. 准备 importance sampling 数据 ==========
# 在 GRPO/PPO 场景中，你需要：
# - target_tokens: 生成的 token 序列
# - logprobs: 参考策略（旧策略）对这些 token 的 logprobs
# - advantages: 每个 token 位置的优势值（来自 reward model）

sample = trio.Datum(
    model_input=trio.ModelInput.from_ints([100, 200, 300, 400]),  # 输入 tokens
    loss_fn_inputs={
        "target_tokens": [200, 300, 400, 500],  # 目标 tokens
        "logprobs": [-0.5, -0.3, -0.8, -0.2],   # 旧策略 logprobs
        "advantages": [0.8, 0.5, -0.2, 0.3],    # 优势值
    },
)

# 所有 loss_fn_inputs 的值长度必须与 model_input 一致

# ========== 3. 训练 ==========
for step in range(3):
    fb_future = train.forward_backward(
        data=[sample, sample],  # 可以传多个样本作为 batch
        loss_fn="importance_sampling",  # 或 "ppo"（当前实现相同）
    )
    fb_result = fb_future.result()
    print(f"Step {step} metrics:", fb_result.metrics)

    opt_future = train.optim_step(trio.AdamParams(learning_rate=1e-4))
    opt_result = opt_future.result()

# ========== 4. 保存 ==========
save = train.save_weights_for_sampler("is_example")
print(f"保存路径: {save.result().path}")
