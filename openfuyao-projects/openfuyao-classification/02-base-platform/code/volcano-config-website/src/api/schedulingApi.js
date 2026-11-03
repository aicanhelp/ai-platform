/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import { GET, POST, PUT, DELETE, PATCH } from '@/api/request';
import qs from 'query-string';

// 统一公共前缀
function commonApiPrefix() {
  const clusterStore = sessionStorage?.cluster;
  const clusterPrefix = clusterStore ? `/clusters/${clusterStore}` : '';
  let paramObj = {
    schedulingUrl: `${clusterPrefix}/rest/scheduling`,
    priorityClassesUrl: `${clusterPrefix}/rest/scheduling/apis/scheduling.k8s.io/v1`,
    podAffintyUrl: '/api/kubernetes/apis/resource.sop.huawei.com/v1alpha1/namespaces/tcs/oenumas/podafi',
  };
  return paramObj;
}

// 查询NUMA亲和性调度策略
export function getNumaAffinitySchedulingStatus() {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().schedulingUrl}/v1/numaaware`,
  });
  return GET(url);
}

// 修改NUMA亲和性调度策略
export function updateNumaAffinitySchedulingStatus(data) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().schedulingUrl}/v1/numaaware`,
  });
  return PUT(url, data, { 'Content-Type': 'application/json' });
}

// 查询Binpack亲和性调度策略
export function getBinpackSchedulingStatus() {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().schedulingUrl}/v1/binpack`,
  });
  return GET(url);
}

// 修改NUMA亲和性调度策略
export function updateBinpackSchedulingStatus(data) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().schedulingUrl}/v1/binpack`,
  });
  return PUT(url, data, { 'Content-Type': 'application/json' });
}

// 查询公平调度策略
export function getFairSchedulingStatus() {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().schedulingUrl}/v1/drf`,
  });
  return GET(url);
}

// 修改公平调度策略
export function updateFairSchedulingStatus(data) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().schedulingUrl}/v1/drf`,
  });
  return PUT(url, data, { 'Content-Type': 'application/json' });
}

// 查询组调度策略
export function getGroupSchedulingStatus() {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().schedulingUrl}/v1/gang`,
  });
  return GET(url);
}

// 修改组调度策略
export function updateGroupSchedulingStatus(data) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().schedulingUrl}/v1/gang`,
  });
  return PUT(url, data, { 'Content-Type': 'application/json' });
}

// 查询优先级调度策略
export function getPrioritySchedulingStatus() {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().schedulingUrl}/v1/priority`,
  });
  return GET(url);
}

// 修改优先级调度策略
export function updatePrioritySchedulingStatus(data) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().schedulingUrl}/v1/priority`,
  });
  return PUT(url, data, { 'Content-Type': 'application/json' });
}

// 获取优先级队列列表
export function getPriorityClassesList(name) {
  const baseUrl = `${commonApiPrefix().priorityClassesUrl}/priorityclasses`;
  const url = name ? `${baseUrl}?fieldSelector=metadata.name=${name}` : baseUrl;
  return GET(url);
}

// 读取指定的优先级队列
export function getSpecifiedPriorityClasses(name) {
  const url = `${commonApiPrefix().priorityClassesUrl}/priorityclasses/${name}`;
  return GET(url);
}

// 创建一个优先级队列
export function createSpecifiedPriorityClasses(data) {
  const url = `${commonApiPrefix().priorityClassesUrl}/priorityclasses`;
  return POST(url, data, { 'Content-Type': 'application/json' });
}

export function editAnnotationsOrLabels(name, data) {
  const url = `${commonApiPrefix().priorityClassesUrl}/priorityclasses/${name}`;
  return PATCH(url, data, { 'Content-Type': 'application/json-patch+json' });
}

// 更新指定的优先级队列
export function updateSpecifiedPriorityClasses(name, data) {
  const url = `${commonApiPrefix().priorityClassesUrl}/priorityclasses/${name}`;
  return PUT(url, data, { 'Content-Type': 'application/json' });
}

export function deleteSpecifiedPriorityClasses(name) {
  const url = `${commonApiPrefix().priorityClassesUrl}/priorityclasses/${name}`;
  return DELETE(url);
}

// numaDistance
export function getNumaDistanceSchedulingStatus() {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().schedulingUrl}/v1/numaDistance`,
  });
  return GET(url);
}

export function updateNumaDistanceSchedulingStatus(data) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().schedulingUrl}/v1/numaDistance`,
  });
  return PUT(url, data, { 'Content-Type': 'application/json' });
}

export function getPodStatus() {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().schedulingUrl}/v1/numaFast`,
  });
  return GET(url);
}

export function updatePodStatus(data) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().schedulingUrl}/v1/numaFast`,
  });
  return PUT(url, data, { 'Content-Type': 'application/json' });
}

export function getBalanceStatus() {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().schedulingUrl}/v1/balance`,
  });
  return GET(url);
}

export function updateBalanceStatus(data) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().schedulingUrl}/v1/balance`,
  });
  return PUT(url, data, { 'Content-Type': 'application/json' });
}

export function getPodAffinityList() {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().podAffintyUrl}`,
  });
  return GET(url);
}
