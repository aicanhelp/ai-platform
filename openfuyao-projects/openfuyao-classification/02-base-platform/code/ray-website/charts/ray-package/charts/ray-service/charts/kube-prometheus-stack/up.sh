#!/bin/bash

# 用法: ./script.sh [目标目录，默认当前目录]
target_dir="${1:-./charts}"

# 检查目录是否存在
if [[ ! -d "$target_dir" ]]; then
  echo "错误：目录不存在 $target_dir"
  exit 1
fi

# 递归查找所有 YAML 文件
find "$target_dir" -type f -name "*.yaml" -print0 | while IFS= read -r -d '' file; do
  echo "处理文件: $file"
  
  # 判断操作系统
  if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS 特殊处理
    sed -i '' '1i\
{{- if eq (include "prometheus.installed" .) "false" }}
' "$file"
    sed -i '' '$a\
{{- end }}
' "$file"
  else
    # Linux 处理
    sed -i '1i{{- if eq (include "prometheus.installed" .) "false" }}' "$file"
    sed -i '$a{{- end }}' "$file"
  fi
done

echo "所有 YAML 文件处理完成！"

