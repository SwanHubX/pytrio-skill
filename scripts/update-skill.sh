#!/bin/bash
# =================================================================
# PyTRIO Skill 自动更新（Git 驱动）
#
# 用法:
#   # 从上游仓库自动拉取最新 SDK 并对比当前版本
#   ./scripts/update-skill.sh --upstream
#
#   # 手动指定新旧 SDK 目录
#   ./scripts/update-skill.sh --old pytrio-old/ --new pytrio/
#
#   # 更新后自动测试
#   ./scripts/update-skill.sh --upstream --test
#
# 上游仓库:
#   上游仓库地址在 UPSTREAM_REPO 变量中配置
#   SDK 路径: trio/src/pytrio/
#
# Git 流程:
#   1. 在 skill/update-<ts> 分支上工作
#   2. diff 旧→新 SDK → Claude 修改 src/ 下的 Skill → commit
#   3. (--test) 自动调用 test-skill.sh → commit 报告
#   4. 提示 merge 或回滚
# =================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SKILL_DIR="$PROJECT_ROOT/src"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

UPSTREAM_REPO="${PYTRIO_UPSTREAM_REPO:-}"  # 通过环境变量传入上游仓库地址
UPSTREAM_SDK_PATH="trio/src/pytrio"

OLD_DIR=""
NEW_DIR=""
USE_UPSTREAM=false
RUN_TEST=false
PYTRIO_API_KEY="${PYTRIO_API_KEY:-}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --old) OLD_DIR="$2"; shift 2 ;;
        --new) NEW_DIR="$2"; shift 2 ;;
        --upstream) USE_UPSTREAM=true; shift ;;
        --test) RUN_TEST=true; shift ;;
        --pytrio-key) PYTRIO_API_KEY="$2"; shift 2 ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

cd "$PROJECT_ROOT"

# ----- 上游模式：自动拉取 -----
if [[ "$USE_UPSTREAM" == true ]]; then
    if [[ -z "$UPSTREAM_REPO" ]]; then
        echo "错误: 需要设置 PYTRIO_UPSTREAM_REPO 环境变量指定上游仓库地址"
        echo "用法: PYTRIO_UPSTREAM_REPO=https://github.com/xxx/yyy.git ./scripts/update-skill.sh --upstream"
        exit 1
    fi
    echo "[准备] 从上游拉取 SDK ($UPSTREAM_REPO)..."

    UPSTREAM_DIR="/tmp/pytrio-upstream-$TIMESTAMP"
    mkdir -p "$UPSTREAM_DIR"

    git clone --depth 1 "$UPSTREAM_REPO" "$UPSTREAM_DIR/new" 2>&1 | tail -2
    NEW_DIR="$UPSTREAM_DIR/new/$UPSTREAM_SDK_PATH"

    if [[ ! -d "$NEW_DIR" ]]; then
        echo "错误: 上游仓库中找不到 $UPSTREAM_SDK_PATH"
        rm -rf "$UPSTREAM_DIR"
        exit 1
    fi

    # 旧版本：用项目中已有的 pytrio/ 目录（如果有的话）
    if [[ -d "$PROJECT_ROOT/pytrio/src/pytrio" ]]; then
        OLD_DIR="$PROJECT_ROOT/pytrio/src/pytrio"
        echo "  旧版本: $OLD_DIR (本地)"
    elif [[ -d "$PROJECT_ROOT/pytrio" ]]; then
        OLD_DIR="$PROJECT_ROOT/pytrio"
        echo "  旧版本: $OLD_DIR (本地)"
    else
        echo "  旧版本: 无（首次对比，将使用空目录）"
        OLD_DIR="/tmp/pytrio-empty-$$"
        mkdir -p "$OLD_DIR"
    fi

    echo "  新版本: $NEW_DIR (上游 master)"
fi

if [[ -z "$NEW_DIR" || -z "$OLD_DIR" ]]; then
    echo "用法:"
    echo "  ./scripts/update-skill.sh --upstream [--test]"
    echo "  ./scripts/update-skill.sh --old <旧SDK> --new <新SDK> [--test]"
    exit 1
fi

echo "========================================"
echo "PyTRIO Skill 自动更新 ($TIMESTAMP)"
echo "========================================"
echo "旧: $OLD_DIR"
echo "新: $NEW_DIR"

# ----- Step 1: Git 准备 -----
echo "[1/4] Git 准备..."

if ! git diff --quiet -- src/ 2>/dev/null; then
    git add src/
    git commit -m "chore: auto-save skill before update ($TIMESTAMP)"
fi

BRANCH_NAME="skill/update-$TIMESTAMP"
ORIGINAL_BRANCH=$(git branch --show-current)
git checkout -b "$BRANCH_NAME"
echo "  分支: $BRANCH_NAME (基于 $ORIGINAL_BRANCH)"

# ----- Step 2: SDK diff -----
echo "[2/4] 计算 SDK diff..."

DIFF_FILE="/tmp/pytrio-sdk-diff-$TIMESTAMP.patch"
diff -rN \
    --exclude='__pycache__' --exclude='*.pyc' \
    --exclude='*.egg-info' --exclude='.git' \
    "$OLD_DIR" "$NEW_DIR" \
    > "$DIFF_FILE" 2>/dev/null || true

DIFF_LINES=$(wc -l < "$DIFF_FILE" | tr -d ' ')
echo "  diff: $DIFF_LINES 行"

if [[ "$DIFF_LINES" -eq 0 ]]; then
    echo "  无变更。"
    git checkout "$ORIGINAL_BRANCH"
    git branch -d "$BRANCH_NAME"
    exit 0
fi

DIFF_CONTENT=$(head -c 100000 "$DIFF_FILE")

# ----- Step 3: Claude 更新 Skill -----
echo "[3/4] Claude 更新 Skill..."

claude --dangerously-skip-permissions -p "你是 PyTRIO SDK 的 Skill 维护者。SDK 更新了，根据 diff 修改 Skill。

## Skill 位置: $SKILL_DIR/
- SKILL.md, docs/, examples/, best-practices/

## SDK 上游仓库
$UPSTREAM_REPO — SDK 在 $UPSTREAM_SDK_PATH/

## SDK Diff
\`\`\`diff
$DIFF_CONTENT
\`\`\`

## 任务
1. 读当前 Skill，分析 diff 中的 API 变更
2. 对应修改 Skill 文件，不改无关内容
3. 创建 update-summary.json:
{\"changes\": [...], \"files_modified\": [...], \"breaking\": [...]}" \
    --output-format text \
    > /tmp/skill-update-$TIMESTAMP.log 2>&1 || true

git add src/ update-summary.json 2>/dev/null
if ! git diff --cached --quiet; then
    SUMMARY=$(cat update-summary.json 2>/dev/null | head -c 500 || echo "auto-update")
    git commit -m "feat: update skill for SDK changes

$SUMMARY"
    rm -f update-summary.json
    echo "  已提交"
else
    echo "  无需修改"
fi

# ----- Step 4: 可选测试 -----
if [[ "$RUN_TEST" == true ]]; then
    echo "[4/4] 自动测试..."
    if [[ -n "$PYTRIO_API_KEY" ]]; then
        "$SCRIPT_DIR/test-skill.sh" --pytrio-key "$PYTRIO_API_KEY" || true
        if [[ -d "$PROJECT_ROOT/test-reports" ]]; then
            git add test-reports/
            git commit -m "test: skill validation ($TIMESTAMP)" 2>/dev/null || true
        fi
    else
        echo "  跳过：未设置 PYTRIO_API_KEY"
    fi
else
    echo "[4/4] 跳过测试（加 --test 启用）"
fi

echo ""
echo "========================================"
echo "完成"
echo "========================================"
echo "查看: git diff $ORIGINAL_BRANCH...$BRANCH_NAME -- src/"
echo "合并: git checkout $ORIGINAL_BRANCH && git merge $BRANCH_NAME"
echo "回滚: git checkout $ORIGINAL_BRANCH && git branch -D $BRANCH_NAME"
