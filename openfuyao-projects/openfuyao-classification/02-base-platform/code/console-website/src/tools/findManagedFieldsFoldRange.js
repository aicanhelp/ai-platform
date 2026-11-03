/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2. */

/**
 * 计算 Kubernetes resource YAML 中 metadata.managedFields 对应文档区间（用于 CodeMirror fold）。
 * @param {string} text
 * @returns {{ from: number, to: number } | null}
 */
export function findManagedFieldsFoldRange(text) {
  if (text == null || typeof text !== 'string') {
    return null;
  }
  const lines = text.split(/\r?\n/);
  const keyPattern = /^(\s*)managedFields:\s*(?:#.*)?$/;
  let startLine = -1;
  let keyIndent = -1;

  for (let i = 0; i < lines.length; i++) {
    const m = lines[i].match(keyPattern);
    if (m) {
      startLine = i;
      keyIndent = m[1].length;
      break;
    }
  }
  if (startLine < 0) {
    return null;
  }

  // 从下一行起，遇到与 managedFields 同级或更外层的「兄弟键」则结束（排除以 - 开头的 YAML 列表项）
  let endLine = lines.length;
  for (let i = startLine + 1; i < lines.length; i++) {
    const raw = lines[i];
    if (!raw.trim()) {
      continue;
    }
    const m = raw.match(/^(\s*)/);
    const ind = m[1].length;
    if (ind > keyIndent) {
      continue;
    }
    const afterWs = raw.slice(ind);
    if (afterWs.startsWith('-')) {
      continue;
    }
    if (/^[\w.-]+:\s*/.test(afterWs)) {
      endLine = i;
      break;
    }
  }

  // 已找到下一个兄弟键时：不把紧邻其上的空行算进折叠区（否则容易看起来像多折了一行）
  let endExclusive = endLine;
  if (endExclusive < lines.length) {
    while (endExclusive > startLine + 1 && !(lines[endExclusive - 1] || '').trim()) {
      endExclusive -= 1;
    }
  }

  const lineStartOffset = (lineIdx) => {
    if (lineIdx <= 0) {
      return 0;
    }
    let pos = 0;
    for (let L = 0; L < lineIdx; L++) {
      const n = text.indexOf('\n', pos);
      if (n === -1) {
        return text.length;
      }
      pos = n + 1;
    }
    return pos;
  };

  const from = lineStartOffset(startLine);
  const to = endExclusive >= lines.length ? text.length : lineStartOffset(endExclusive);
  if (to <= from) {
    return null;
  }
  return { from, to: to - 1 };
}
