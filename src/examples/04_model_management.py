"""
PyTRIO 模型管理示例

场景：通过 RestClient 查询和管理训练历史、checkpoint、模型权重
闭环：列出模型 → 列出训练记录 → 列出 checkpoint → 获取下载链接
"""

import pytrio as trio

# ========== 1. 初始化 ==========
client = trio.ServiceClient(api_key="YOUR_API_KEY")

# 列出可用基础模型
print("可用模型:", client.get_supported_models())

# 创建 REST 管理客户端
rest = client.create_rest_client()

# ========== 2. 查看用户信息 ==========
user_info = rest.get_user_info()
print(f"用户: {user_info['username']} (ID: {user_info['user_id']})")

# ========== 3. 列出已保存的模型权重 ==========
weights = rest.list_weights(page=1, page_size=10)
print(f"\n已保存的权重 (共 {weights['pagination']['total']} 个):")
for w in weights["data"]:
    print(f"  - {w['cuid']} | 路径: {w['path']} | 类型: {w['type']}")

# ========== 4. 列出训练运行记录 ==========
runs = rest.list_training_runs(limit=5)
print(f"\n训练记录 (共 {runs['total']} 条):")
for run in runs["training_runs"]:
    print(f"  - {run['training_run_id']} | 模型: {run['base_model']} | LoRA: {run['is_lora']}")

# ========== 5. 列出某次训练的 checkpoint ==========
if runs["training_runs"]:
    run_id = runs["training_runs"][0]["training_run_id"]
    checkpoints = rest.list_checkpoints(run_id)
    print(f"\n训练 {run_id} 的 checkpoint:")
    for cp in checkpoints["checkpoints"]:
        print(f"  - {cp['checkpoint_id']} | 类型: {cp['checkpoint_type']} | 大小: {cp['size_bytes']} bytes")

# ========== 6. 列出用户所有 checkpoint ==========
all_cps = rest.list_user_checkpoints(limit=10)
print(f"\n所有 checkpoint (共 {all_cps['total']} 个):")
for cp in all_cps["checkpoints"]:
    print(f"  - {cp['checkpoint_id']} | {cp['path']}")

# ========== 7. 获取模型下载链接（如果有权重） ==========
if weights["data"]:
    first_weight = weights["data"][0]
    archive = rest.get_archive_url(first_weight["cuid"])
    print(f"\n下载链接: {archive['url'][:80]}...")

# ========== 8. 会话管理 ==========
sessions = rest.list_sessions(limit=5)
print(f"\n会话列表: {sessions.get('sessions', [])}")
