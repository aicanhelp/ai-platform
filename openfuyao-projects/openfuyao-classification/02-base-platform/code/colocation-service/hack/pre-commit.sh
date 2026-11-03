#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# openFuyao is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

SENSITIVE_PATTERNS=(
  "client-key-data:"
  "BEGIN RSA PRIVATE KEY"
  "BEGIN EC PRIVATE KEY"
  "BEGIN DSA PRIVATE KEY"
  "BEGIN OPENSSH PRIVATE KEY"
  "BEGIN PRIVATE KEY"
)

FOUND=0

for pattern in "${SENSITIVE_PATTERNS[@]}"; do
  MATCHES=$(git diff --cached --diff-filter=ACM -U0 | grep -n "$pattern" 2>/dev/null)
  if [ -n "$MATCHES" ]; then
    echo "ERROR: Sensitive content detected in staged files:"
    echo "$MATCHES"
    echo "Pattern matched: $pattern"
    echo "Please remove sensitive data before committing."
    FOUND=1
  fi
done

KUBECONFIG_FILES=$(git diff --cached --name-only --diff-filter=ACM | grep -E '\.(conf|kubeconfig)$' 2>/dev/null)
if [ -n "$KUBECONFIG_FILES" ]; then
  for f in $KUBECONFIG_FILES; do
    if git diff --cached --diff-filter=ACM -U0 "$f" | grep -q "client-key-data\|BEGIN.*PRIVATE KEY"; then
      echo "ERROR: Kubeconfig file with private key detected: $f"
      FOUND=1
    fi
  done
fi

if [ $FOUND -eq 1 ]; then
  echo "Commit rejected. Use 'git commit --no-verify' to bypass (NOT recommended)."
  exit 1
fi

exit 0
