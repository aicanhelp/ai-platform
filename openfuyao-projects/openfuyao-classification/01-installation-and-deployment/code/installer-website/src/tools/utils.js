/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
/**
 * 处理async/await错误捕获
 */
export async function requestResource(fn) {
  try {
    let response = await fn;
    return [null, response];
  } catch (err) {
    return [err, null];
  }
}

export function sorterFirstAlphabet(a = '--', b = '--') {
  if (typeof a === 'number' && typeof b === 'number') {
    return a - b;
  }
  return a.toString().localeCompare(b.toString());
}

/**
 * 将403整体抽出
 */
export function forbiddenMsg(message, errorMsg, intl) {
  const id = errorMsg.response.data.message.includes('User') ? 'error.forbidden' : 'error.serverRefused';
  return message.error(intl.formatMessage({ id }));
}

export function disabledHandle(status) {
  let isClick = false;
  if (status.includes('ing') || status === 'ScalingWorkerNodesUp' || status === 'ScalingWorkerNodesDown') {
    isClick = true;
  }
  return isClick;
}

export function extractIPv4(text) {
  const ipv4Regex =
    /\b(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b/g;
  return text.match(ipv4Regex) || [];
}

export function compareVersions(a, b) {
  // 移除开头的'v'并分割版本号
  if (typeof a !== 'string' || typeof b !== 'string') {
    return true;
  }
  const v1 = a.replace(/^v/, '').split('.');
  const v2 = b.replace(/^v/, '').split('.');

  const maxLength = Math.max(v1.length, v2.length);

  for (let i = 0; i < maxLength; i++) {
    const part1 = v1[i] || '0';
    const part2 = v2[i] || '0';

    const num1 = parseInt(part1, 10);
    const num2 = parseInt(part2, 10);

    if (num1 > num2) return true;
    if (num1 < num2) return false;
  }

  return false;
}

/** Use backend-computed isAdminCluster from cluster list API. */
export function isAdminCluster(cluster) {
  return !!cluster?.isAdminCluster;
}

function formatStatusText(prefix, statusVal, intl) {
  if (!statusVal) {
    return '';
  }
  const id = `status.${prefix}.${statusVal}`;
  if (!intl) {
    return statusVal;
  }
  const text = intl.formatMessage({ id });
  return text === id ? statusVal : text;
}

export function clusterStatusValueToText(statusVal, intl) {
  return formatStatusText('cluster', statusVal, intl);
}

export function nodeStatusValueToText(statusVal, intl) {
  return formatStatusText('node', statusVal, intl);
}
