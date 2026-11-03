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
  const clusterStore = sessionStorage?.cluster;
  const clusterPrefix = clusterStore ? `/clusters/${clusterStore}` : '';
  let paramObj = {
    commonUrl: `${clusterPrefix}/rest/ray/rest/v1`,
    grafanaUrl: `${clusterPrefix}/rest/ray/prometheus/api/v1`,
    lokiUrl: `${clusterPrefix}/rest/ray/loki/api/v1`,
  };
  return paramObj;
}
/**
 *  概览接口
 */
export function getOverviewNum() {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().commonUrl}/overview/all`,
  });
  return GET(url);
}

/**
 *  cluster接口
 */
export function getClusterList() {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().commonUrl}/raycluster/list`,
  });
  return GET(url);
}

// 启动cluster
export function restartClusterList(namespace, name) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().commonUrl}/raycluster/restart/${namespace}/${name}`,
  });
  return POST(url);
}

// 终止cluster
export function stopClusterList(namespace, name) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().commonUrl}/raycluster/stop/${namespace}/${name}`,
  });
  return POST(url);
}

// 删除cluster
export function deleteClusterList(namespace, name) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().commonUrl}/raycluster/delete/${namespace}/${name}`,
  });
  return DELETE(url);
}

// 获取cluster详情
export function getClusterDetail(namespace, name) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().commonUrl}/raycluster/get/${namespace}/${name}`,
  });
  return GET(url);
}

// 创建cluster
export function addCluster(data) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().commonUrl}/raycluster/create`,
  });
  return POST(url, data);
}

/**
 *  service接口
 */
export function getServiceList() {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().commonUrl}/rayservice/list`,
  });
  return GET(url);
}

// 启动service接口
export function restartServiceList(namespace, name) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().commonUrl}/rayservice/restart/${namespace}/${name}`,
  });
  return POST(url);
}

// 终止service接口
export function stopServiceList(namespace, name) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().commonUrl}/rayservice/stop/${namespace}/${name}`,
  });
  return POST(url);
}

// 删除service接口
export function deleteServiceList(namespace, name) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().commonUrl}/rayservice/delete/${namespace}/${name}`,
  });
  return DELETE(url);
}

// 获取service详情
export function getServiceDetail(namespace, name) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().commonUrl}/rayservice/get/${namespace}/${name}`,
  });
  return GET(url);
}

// 创建service
export function addService(data) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().commonUrl}/rayservice/create`,
  });
  return POST(url, data);
}

/**
 *  job接口
 */
export function getJobList() {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().commonUrl}/rayjob/list`,
  });
  return GET(url);
}

// 启动Job接口
export function restartJobList(namespace, name) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().commonUrl}/rayjob/restart/${namespace}/${name}`,
  });
  return POST(url);
}

// 终止Job接口
export function stopJobList(namespace, name) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().commonUrl}/rayjob/stop/${namespace}/${name}`,
  });
  return POST(url);
}

// 删除Job接口
export function deleteJobList(namespace, name) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().commonUrl}/rayjob/delete/${namespace}/${name}`,
  });
  return DELETE(url);
}

// 获取Job详情
export function getJobDetail(namespace, name) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().commonUrl}/rayjob/get/${namespace}/${name}`,
  });
  return GET(url);
}

// 创建Job
export function addJob(data) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().commonUrl}/rayjob/create`,
  });
  return POST(url, data);
}

// 查询普罗表达式数据
export function getExpressQuery(rawQuery, step, start, end) {
  const encodedQuery = encodeURIComponent(rawQuery);
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().grafanaUrl}/query_range?query=${encodedQuery}`,
    query: {
      step,
      start,
      end,
    },
  });
  return GET(url);
}

export function getLokiQuery(rawQuery, step, start, end, limit) {
  const encodedQuery = encodeURIComponent(rawQuery);
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().lokiUrl}/query_range?query=${encodedQuery}`,
    query: {
      step,
      start,
      end,
      limit,
    },
  });
  return GET(url);
}
