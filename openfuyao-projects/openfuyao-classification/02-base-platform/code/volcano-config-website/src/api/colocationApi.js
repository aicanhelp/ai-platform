/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import { GET, POST, DELETE } from '@/api/request';
import qs from 'query-string';

// 统一公共前缀
function commonApiPrefix() {
  let paramObj = {
    nodeUrl: `/rest/colocation/v1`,
    workloadUrl: `/`,
    rulesUrl: `/`,
  };
  return paramObj;
}
/**
 * 获取混部节点管理-列表数据
 */
export function getNodeListData() {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().nodeUrl}/colocation-nodes`,
  });
  return GET(url);
}

/**
 * 获取混部节点管理-列表数据
 */
export function getNonColocationNodeListData() {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().nodeUrl}/non-colocation-nodes`,
  });
  return GET(url);
}

/**
 * 获取混部节点管理-顶部节点及pod总数数据
 */
export function getNodesAndPodsCount() {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().nodeUrl}/colocation-stats`,
  });
  return GET(url);
}

/**
 * 混部节点管理-移除单条数据
 */
export function deleteColocationNodesById(id) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().nodeUrl}/colocation-nodes/${id}`,
  });
  return DELETE(url);
}

/**
 * 混部节点管理-添加混部节点
 */
export function addColocationNodes(data) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().nodeUrl}/colocation-nodes`,
  });
  return POST(url, data);
}
