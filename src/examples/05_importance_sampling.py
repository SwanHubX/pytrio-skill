"""
PyTRIO 重要性采样训练示例（真实 RL 闭环）

场景：让 Qwen3-4B 用格式正确地回答简单算术题
闭环：sample（rollout）→ reward → process_rollout → forward_backward(importance_sampling) → optim_step

对照官方 docs/guide/train.md RL 示例简化而来。
"""

import re

import numpy as np
import pytrio as trio

# ========== 1. 初始化 ==========
client = trio.ServiceClient(api_key="YOUR_API_KEY")

base_model = "Qwen/Qwen3-4B-Instruct-2507"
train = client.create_lora_training_client(base_model=base_model, rank=32)
tokenizer = train.get_tokenizer()

# ========== 2. 数据与奖励函数 ==========
dataset = [
    ("What is 2 + 3?", 5),
    ("What is 7 - 4?", 3),
    ("What is 6 * 8?", 48),
    ("What is 12 / 3?", 4),
]


def parse_number(text: str):
    m = re.fullmatch(r"-?\d+(?:\.\d+)?", text.strip())
    return float(m.group()) if m else None


def compute_reward(text: str, gold: float) -> float:
    pred = parse_number(text)
    if pred is None:
        return -1.0  # 格式错误
    if abs(pred - gold) < 1e-6:
        return 2.0   # 答对
    return -0.5      # 答错


def to_np(x):
    return np.array(x.tolist() if hasattr(x, "tolist") else x, dtype=float)


# ========== 3. Rollout → Datum 构造 ==========
def process_rollout(prompt_tokens, completion_tokens, completion_logprobs, reward):
    tokens = prompt_tokens + completion_tokens

    prompt_weights = [0] * len(prompt_tokens)
    completion_weights = [1] * len(completion_tokens)
    weights = prompt_weights + completion_weights

    old_logprobs = [0.0] * len(prompt_tokens) + list(completion_logprobs)
    old_logprobs = old_logprobs[: len(tokens)]
    old_logprobs += [0.0] * (len(tokens) - len(old_logprobs))

    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]
    weights = weights[1:]
    old_logprobs = old_logprobs[1:]
    advantages = [reward] * (len(tokens) - 1)

    return trio.Datum(
        model_input=trio.ModelInput.from_ints(tokens=input_tokens),
        loss_fn_inputs=dict(
            weights=weights,
            target_tokens=target_tokens,
            logprobs=old_logprobs,
            advantages=advantages,
        ),
    )


# ========== 4. RL 主循环 ==========
for iteration in range(5):
    # 4a. 拉取当前权重做 rollout
    sampler = train.save_weights_and_get_sampling_client(name=f"rl-iter{iteration}")

    rollouts = []
    rewards = []
    for question, gold in dataset:
        prompt_text = (
            f"Question: {question}\nReturn only the final numeric answer.\nAnswer:"
        )
        prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=True)

        sample_result = sampler.sample(
            prompt=trio.ModelInput.from_ints(prompt_tokens),
            num_samples=4,
            sampling_params=trio.SamplingParams(max_tokens=8, temperature=0.7),
        ).result()

        for seq in sample_result.sequences:
            reward = compute_reward(seq.text, float(gold))
            rewards.append(reward)

            completion_tokens = tokenizer.encode(seq.text, add_special_tokens=False)
            if not completion_tokens:
                continue

            rollouts.append(
                process_rollout(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    completion_logprobs=seq.logprobs,
                    reward=reward,
                )
            )

    print(f"Iter {iteration} | mean reward: {np.mean(rewards):.4f} | samples: {len(rollouts)}")

    # 4b. 用 rollouts 做一次策略更新
    fb_result = train.forward_backward(rollouts, "importance_sampling").result()
    train.optim_step(trio.AdamParams(learning_rate=1e-5)).result()

    # 4c. 监控 IS loss
    logprobs = np.concatenate([to_np(o["logprobs"]) for o in fb_result.loss_fn_outputs])
    weights = np.concatenate([to_np(ex.loss_fn_inputs["weights"]) for ex in rollouts])
    old_logprobs = np.concatenate([to_np(ex.loss_fn_inputs["logprobs"]) for ex in rollouts])
    advantages = np.concatenate([to_np(ex.loss_fn_inputs["advantages"]) for ex in rollouts])

    mask = weights > 0
    loss = -np.sum(np.exp(logprobs[mask] - old_logprobs[mask]) * advantages[mask]) / mask.sum()
    print(f"Iter {iteration} IS loss: {loss:.4f}")

# ========== 5. 保存最终权重 ==========
final = train.save_weights_for_sampler("rl_final").result()
print(f"最终权重: {final.path}")
