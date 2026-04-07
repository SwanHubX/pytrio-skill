#!/bin/bash
# 打包 src/ 为 release 用的 tarball 和 zip
# 解压后直接是 SKILL.md, docs/, examples/ 等（不带外层目录）
# 用户执行: tar xz -C .claude/skills/pytrio-skill/ 即可
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DIST_DIR="$PROJECT_ROOT/dist"

mkdir -p "$DIST_DIR"
rm -f "$DIST_DIR/pytrio-skill.tar.gz" "$DIST_DIR/pytrio-skill.zip"

cd "$PROJECT_ROOT/src"
tar czf "$DIST_DIR/pytrio-skill.tar.gz" .
zip -r "$DIST_DIR/pytrio-skill.zip" .

echo "打包完成:"
ls -lh "$DIST_DIR"/pytrio-skill.*
echo ""
echo "发布: gh release create vX.Y.Z dist/pytrio-skill.tar.gz dist/pytrio-skill.zip"
