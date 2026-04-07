#!/bin/bash
# =================================================================
# PyTRIO Skill 多模型测试 + evals 校验
#
# 用法:
#   PYTRIO_API_KEY=xxx ./scripts/test-skill.sh
#   ./scripts/test-skill.sh --pytrio-key KEY
#   ./scripts/test-skill.sh --pytrio-key KEY --task "自定义任务"
#   ./scripts/test-skill.sh --pytrio-key KEY --cli claude  # 只跑单个
#
# 产物:
#   test-reports/
#     ├── report_<cli>_<ts>.md       — 完整报告（含 evals 校验结果）
#     ├── code_<cli>_<ts>/           — 生成的代码
#     └── result_<cli>_<ts>.json     — AI 自评
# =================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SKILL_SOURCE="$PROJECT_ROOT/src"
EVALS_FILE="$SCRIPT_DIR/evals/evals.json"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
REPORT_DIR="$PROJECT_ROOT/test-reports"

TASK="用 pytrio 写一个训练+推理的脚本，3条 Alpaca 样本做 SFT 训练 3 步，然后用训练好的权重推理一下，写在一个文件里跑通就行"
PYTRIO_API_KEY="${PYTRIO_API_KEY:-}"
SINGLE_CLI=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --task) TASK="$2"; shift 2 ;;
        --pytrio-key) PYTRIO_API_KEY="$2"; shift 2 ;;
        --cli) SINGLE_CLI="$2"; shift 2 ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

if [[ -z "$PYTRIO_API_KEY" ]]; then
    echo "错误: 需要 PYTRIO_API_KEY 环境变量或 --pytrio-key"
    exit 1
fi

mkdir -p "$REPORT_DIR"

# =================================================================
# Evals 校验函数：对生成的代码做 expectations 检查
# =================================================================
run_evals_check() {
    local code_dir="$1"
    local report_file="$2"

    if [[ ! -f "$EVALS_FILE" ]]; then
        echo "## Evals 校验" >> "$report_file"
        echo "evals.json 不存在，跳过" >> "$report_file"
        return
    fi

    # 合并所有 py 文件为一个文本用于检查
    local all_code=""
    for f in "$code_dir"/*.py; do
        [[ -f "$f" ]] || continue
        all_code+="$(cat "$f")"$'\n'
    done

    if [[ -z "$all_code" ]]; then
        echo "## Evals 校验" >> "$report_file"
        echo "无 .py 文件，跳过" >> "$report_file"
        return
    fi

    local total=0
    local passed=0
    local failed=0
    local results=""

    # 只检查 eval id=1 的 expectations（SFT 训练+推理，和默认 TASK 匹配）
    # 用 python 解析 JSON，逐条 grep 检查
    while IFS='|' read -r exp; do
        [[ -z "$exp" ]] && continue
        total=$((total + 1))

        local check_passed=false

        case "$exp" in
            *"Qwen/Qwen3-4B-Instruct-2507"*)
                echo "$all_code" | grep -q "Instruct-2507\|Instruct" && check_passed=true ;;
            *"ModelInput.from_ints"*)
                echo "$all_code" | grep -q "ModelInput.from_ints" && check_passed=true ;;
            *"偏移对齐"*|*"tokens\[:-1\]"*)
                # 两种正确写法：手动偏移 tokens[:-1]/[1:] 或 auto_shift=True
                (echo "$all_code" | grep -q '\[:-1\]' && echo "$all_code" | grep -q '\[1:\]') && check_passed=true
                echo "$all_code" | grep -q 'auto_shift.*True\|auto_shift=True' && check_passed=true ;;
            *"关键字参数"*|*"data="*)
                echo "$all_code" | grep -q 'data=' && check_passed=true ;;
            *".result()"*)
                echo "$all_code" | grep -q '\.result()' && check_passed=true ;;
            *"loss:sum"*)
                echo "$all_code" | grep -q 'loss:sum' && check_passed=true ;;
            *"SamplingParams"*"对象"*|*"SamplingParams"*"dict"*)
                echo "$all_code" | grep -q 'SamplingParams(' && check_passed=true ;;
            *"max_new_tokens"*)
                ! echo "$all_code" | grep -q 'max_new_tokens' && check_passed=true ;;
            *"result.sequences"*|*"result.samples"*)
                echo "$all_code" | grep -q '\.sequences' && check_passed=true ;;
            *"learning_rate"*"lr"*)
                echo "$all_code" | grep -q 'learning_rate' && check_passed=true ;;
            *"target_tokens"*"labels"*)
                echo "$all_code" | grep -q 'target_tokens' && ! echo "$all_code" | grep -q '"labels"' && check_passed=true ;;
            *)
                # 无法自动检查的 expectation，标记为跳过
                results+="  - ⏭ $exp (无法自动检查)"$'\n'
                total=$((total - 1))
                continue ;;
        esac

        if [[ "$check_passed" == true ]]; then
            passed=$((passed + 1))
            results+="  - ✅ $exp"$'\n'
        else
            failed=$((failed + 1))
            results+="  - ❌ $exp"$'\n'
        fi
    done < <(python3 -c "
import json, sys
with open('$EVALS_FILE') as f:
    data = json.load(f)
# 取 eval id=1 的 expectations（默认 SFT 任务）
for e in data['evals']:
    if e['id'] == 1:
        for exp in e['expectations']:
            print(exp)
        break
" 2>/dev/null)

    {
        echo ""
        echo "## Evals 校验 (eval #1: SFT 训练+推理)"
        echo ""
        echo "通过: $passed / $total | 失败: $failed"
        echo ""
        echo "$results"
    } >> "$report_file"
}

# =================================================================
# 单个测试用例的执行函数
# =================================================================
run_test_case() {
    local cli_label="$1"
    local cli_cmd="$2"
    local test_dir="/tmp/pytrio-skill-test-${cli_label}-$TIMESTAMP"

    echo "[$cli_label] 开始..."

    # 搭建隔离环境
    mkdir -p "$test_dir/.claude/skills"
    cp -r "$SKILL_SOURCE"/* "$test_dir/.claude/skills/"

    cd "$test_dir"
    uv init --no-readme 2>/dev/null
    cat > pyproject.toml << 'TOML'
[project]
name = "pytrio-skill-test"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "pytrio>=0.1.12",
    "transformers>=4.40.0",
    "torch>=2.0.0",
]
TOML
    uv sync 2>&1 | tail -1

    # PyTRIO 认证
    VIRTUAL_ENV="" no_proxy="pytrio.cn" https_proxy="" http_proxy="" all_proxy="" \
        uv run trio login -k "$PYTRIO_API_KEY" 2>&1 | tail -1

    # CLAUDE.md 引导
    cat > "$test_dir/CLAUDE.md" << 'MDEOF'
# 项目说明

本项目使用 PyTRIO SDK 进行远程 LLM 训练和推理。
完整文档在 `.claude/skills/` 下，使用前请先阅读 `.claude/skills/SKILL.md`。

环境：`uv run python xxx.py`，已认证，运行时需 `no_proxy=pytrio.cn https_proxy="" http_proxy="" all_proxy=""`
MDEOF

    local full_prompt="$TASK

写完跑一下确认能通过，然后写个 result.json 记录下结果：成功还是失败，创建了哪些文件，遇到了什么问题"

    # 调用 CLI
    cd "$test_dir"
    VIRTUAL_ENV="" no_proxy="pytrio.cn" https_proxy="" http_proxy="" all_proxy="" \
        $cli_cmd "$full_prompt" \
        > "$test_dir/cli_output.log" 2>&1 || true

    # 收集产物
    local code_dir="$REPORT_DIR/code_${cli_label}_$TIMESTAMP"
    mkdir -p "$code_dir"
    cp "$test_dir"/*.py "$code_dir/" 2>/dev/null || true
    cp "$test_dir/result.json" "$REPORT_DIR/result_${cli_label}_$TIMESTAMP.json" 2>/dev/null || true

    # 生成报告
    local report_file="$REPORT_DIR/report_${cli_label}_$TIMESTAMP.md"
    {
        echo "# 测试报告 ($cli_label @ $TIMESTAMP)"
        echo "- CLI: $cli_cmd"
        echo "- 任务: $TASK"
        echo ""
        echo "## result.json"
        echo '```json'
        cat "$test_dir/result.json" 2>/dev/null || echo '{"error": "未生成"}'
        echo '```'
        echo ""
        echo "## 生成的代码"
        for f in "$test_dir"/*.py; do
            [[ -f "$f" ]] || continue
            echo "### $(basename "$f")"
            echo '```python'
            cat "$f"
            echo '```'
            echo ""
        done
        echo "## CLI 输出（末尾）"
        echo '```'
        tail -80 "$test_dir/cli_output.log" 2>/dev/null || echo "(无)"
        echo '```'
    } > "$report_file"

    # 运行 evals 校验
    run_evals_check "$code_dir" "$report_file"

    echo "[$cli_label] 完成 → $report_file"
}

# =================================================================
# 主流程
# =================================================================
echo "========================================"
echo "PyTRIO Skill 测试 ($TIMESTAMP)"
echo "========================================"

if [[ -n "$SINGLE_CLI" ]]; then
    case "$SINGLE_CLI" in
        haiku)    run_test_case "haiku" "claude --dangerously-skip-permissions --model haiku -p" ;;
        kimi)     run_test_case "kimi" "kimi --yolo --thinking -p" ;;
        gemini)   run_test_case "gemini" "gemini --yolo -m gemini-3-flash-preview -p" ;;
        opencode) run_test_case "opencode" "$HOME/.opencode/bin/opencode run" ;;
        claude)   run_test_case "claude" "claude --dangerously-skip-permissions -p" ;;
        *)        run_test_case "$SINGLE_CLI" "$SINGLE_CLI -p" ;;
    esac
else
    echo "检测可用 CLI 并启动测试..."
    echo ""
    PIDS=()

    if command -v claude &>/dev/null; then
        run_test_case "haiku" "claude --dangerously-skip-permissions --model haiku -p" &
        PIDS+=($!)
        echo "  [haiku] 已启动"
    fi

    if command -v kimi &>/dev/null; then
        run_test_case "kimi" "kimi --yolo --thinking -p" &
        PIDS+=($!)
        echo "  [kimi] 已启动"
    fi

    if command -v gemini &>/dev/null; then
        run_test_case "gemini" "gemini --yolo -m gemini-3-flash-preview -p" &
        PIDS+=($!)
        echo "  [gemini] 已启动"
    fi

    if [[ -x "$HOME/.opencode/bin/opencode" ]]; then
        run_test_case "opencode" "$HOME/.opencode/bin/opencode run" &
        PIDS+=($!)
        echo "  [opencode] 已启动（无 yolo 模式，可能因权限拒绝失败）"
    else
        echo "  [opencode] 未安装，跳过"
    fi

    echo ""

    for pid in "${PIDS[@]}"; do
        wait "$pid" 2>/dev/null || true
    done
    echo "所有测试已完成"
fi

# =================================================================
# 汇总 evals 结果
# =================================================================
echo ""
echo "========================================"
echo "全部测试完成"
echo "========================================"
echo "报告目录: $REPORT_DIR/"
echo ""

for report in "$REPORT_DIR"/report_*_$TIMESTAMP.md; do
    [[ -f "$report" ]] || continue
    cli_name=$(basename "$report" | sed "s/report_//;s/_${TIMESTAMP}.md//")
    echo "--- $cli_name ---"

    # 提取 evals 校验摘要
    if grep -q "Evals 校验" "$report" 2>/dev/null; then
        grep "通过:" "$report" 2>/dev/null || echo "  (无校验结果)"
        # 只显示失败项
        grep "❌" "$report" 2>/dev/null | head -5 || echo "  全部通过"
    else
        echo "  (无校验)"
    fi
    echo ""
done
