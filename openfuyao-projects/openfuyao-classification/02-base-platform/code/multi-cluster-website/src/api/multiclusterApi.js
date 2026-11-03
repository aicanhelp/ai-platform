/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import { GET, POST, DELETE, PATCH } from '@/api/request';
import qs from 'query-string';

// 统一公共前缀
function multiclusterApi() {
  let paramObj = `/rest/multicluster/v1beta1/`;
  return paramObj;
}

/**
 * 获取多集群列表
 */
export function getClusters() {
  const url = qs.stringifyUrl({ url: `${multiclusterApi()}resources/clusters` });
  return GET(url);
}

/**
 * 创建集群
 */
export function creatCluster(data) {
  const url = qs.stringifyUrl({ url: `${multiclusterApi()}clusters` });
  return POST(url, data, { 'Content-Type': 'application/json' });
}

/**
 * 删除集群
 */
export function deleteCluster(data) {
  const url = qs.stringifyUrl({ url: `${multiclusterApi()}clusters/${data}` });
  return DELETE(url);
}

/**
 * 编辑标签
 */
export function editClusterLabel(name, data) {
  const url = qs.stringifyUrl({ url: `${multiclusterApi()}clusters/${name}` });
  return PATCH(url, data, { 'Content-Type': 'application/json' });
}

/**
 * 获取凭证
 */
export function getCertificate(clusterTag) {
  const url = qs.stringifyUrl({ url: `${multiclusterApi()}clusters/${clusterTag}/kubeconfig` });
  return GET(url);
}
