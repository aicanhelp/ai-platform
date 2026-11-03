/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import { GET } from '@/api/request';
import qs from 'query-string';

// 统一公共前缀
function commonApiPrefix() {
  const clusterStore = sessionStorage?.cluster;
  const clusterPrefix = clusterStore ? `/clusters/${clusterStore}` : '';
  let paramObj = {
    prometheusUrl: `${clusterPrefix}/rest/scheduling`,
    apiUrl: `${clusterPrefix}/rest/scheduling/v1`,
    workloadUrl: `/`,
    rulesUrl: `/`,
  };
  return paramObj;
}

const pretty = true; // 规范格式数据

export function getNodeList() {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().apiUrl}/nodeList`,
  });
  return GET(url);
}

export function getCR(resourceName) {
  const encodedResourceName = encodeURIComponent(resourceName);
  const url = `${commonApiPrefix().apiUrl}/cr/${encodedResourceName}`;
  return GET(url);
}

export function getPodByUUID(podUUID) {
  const encodedPodUUID = encodeURIComponent(podUUID);
  const url = `${commonApiPrefix().apiUrl}/uuidToName/${encodedPodUUID}`;
  return GET(url);
}

export function getExpressQuery(rawQuery) {
  const encodedQuery = encodeURIComponent(rawQuery);
  const url = `${commonApiPrefix().prometheusUrl}/api/v1/query?query=${encodedQuery}`;
  return GET(url);
}

// 读取指定的自定义资源实例
export function getSpecifiedCustomResourceInstance(nodeName) {
  const url = `${commonApiPrefix().nodeUrl}/apis/nodeinfo.volcano.sh/v1alpha1/numatopologies/${nodeName}`;
  return GET(url);
}
export function normalizeKeys(obj) {
  if (obj && typeof obj === 'object' && !Array.isArray(obj)) {
    return Object.keys(obj).reduce((acc, key) => {
      const normalizedKey = key.replace(/\s+/g, '').toLowerCase(); // 移除空格并转换为小写
      acc[normalizedKey] = normalizeKeys(obj[key]);
      return acc;
    }, {});
  } else if (Array.isArray(obj)) {
    return obj.map((item) => normalizeKeys(item));
  } else {
    return obj;
  }
}
