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
import { solveParams } from '@/tools/utils';

// 统一公共前缀
function commonApiPrefix() {
  const clusterStore = sessionStorage?.cluster;
  const clusterPrefix = clusterStore ? `/clusters/${clusterStore}` : '';
  let paramObj = {
    nodeUrl: `${clusterPrefix}/rest/colocation/v1`,
    workloadUrl: `${clusterPrefix}/rest/colocation/v1`,
    rulesUrl: `${clusterPrefix}/rest/colocation/v1`,
    namespaceUrl: `${clusterPrefix}/rest/colocation/v1`,
  };
  return paramObj;
};
/**
 * 获取混部节点管理-列表数据
 */
export function getNodeListData() {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().nodeUrl}/colocation-configs`,
  });
  return GET(url);
}

/**
 * 获取非混部节点管理-列表数据
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

/**
 * 获取混部工作负载-pod-列表数据
 */
export function getColocationPodsListData() {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().workloadUrl}/colocation-pods`,
  });
  return GET(url);
}

/**
 * 获取非混部工作负载-pod-列表数据
 */
export function getNonColocationPodsListData() {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().workloadUrl}/non-colocation-pods`,
  });
  return GET(url);
}

/**
 * 混部工作负载-pod-添加混部pod
 */
export function addColocationPods(data) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().workloadUrl}/colocation-pods`,
  });
  return POST(url, data);
}

/**
 * 混部工作负载-pod-修改为在线状态
 */
export function updatePodsExpectStatusToOnline(id) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().workloadUrl}/colocation-pods/${id}/action/colocation-online`,
  });
  return POST(url);
}

/**
 * 混部工作负载-pod-修改为离线状态
 */
export function updatePodsExpectStatusToOffline(id) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().workloadUrl}/colocation-pods/${id}/action/colocation-offline`,
  });
  return POST(url);
}

/**
 * 混部工作负载-pod-移除单条数据
 */
export function deleteColocationPodsById(id) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().workloadUrl}/colocation-pods/${id}`,
  });
  return DELETE(url);
}


/**
 * 获取混部工作负载-deployment-列表数据
 */
export function getColocationDeploymentsListData() {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().workloadUrl}/colocation-deployments`,
  });
  return GET(url);
}

/**
 * 获取非混部工作负载-deployment-列表数据
 */
export function getNonColocationDeploymentsListData() {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().workloadUrl}/non-colocation-deployments`,
  });
  return GET(url);
}

/**
 * 混部工作负载-deployment-添加混部deployment
 */
export function addColocationDeployments(data) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().workloadUrl}/colocation-deployments`,
  });
  return POST(url, data);
}

/**
 * 混部工作负载-deployment-修改为在线状态
 */
export function updateDeploymentsExpectStatusToOnline(id) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().workloadUrl}/colocation-deployments/${id}/action/colocation-online`,
  });
  return POST(url);
}

/**
 * 混部工作负载-deployment-修改为离线状态
 */
export function updateDeploymentsExpectStatusToOffline(id) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().workloadUrl}/colocation-deployments/${id}/action/colocation-offline`,
  });
  return POST(url);
}

/**
 * 混部工作负载-deployment-移除单条数据
 */
export function deleteColocationDeploymentsById(id) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().workloadUrl}/colocation-deployments/${id}`,
  });
  return DELETE(url);
}

/**
 * 获取混部工作负载-statefulSet-列表数据
 */
export function getColocationStatefulSetsListData() {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().workloadUrl}/colocation-statefulsets`,
  });
  return GET(url);
}

/**
 * 获取非混部工作负载-statefulSet-列表数据
 */
export function getNonColocationStatefulSetsListData() {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().workloadUrl}/non-colocation-statefulsets`,
  });
  return GET(url);
}

/**
 * 混部工作负载-statefulSet-添加混部statefulSet
 */
export function addColocationStatefulSets(data) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().workloadUrl}/colocation-statefulsets`,
  });
  return POST(url, data);
}

/**
 * 混部工作负载-statefulSet-修改为在线状态
 */
export function updateStatefulSetsExpectStatusToOnline(id) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().workloadUrl}/colocation-statefulsets/${id}/action/colocation-online`,
  });
  return POST(url);
}

/**
 * 混部工作负载-statefulSet-修改为离线状态
 */
export function updateStatefulSetsExpectStatusToOffline(id) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().workloadUrl}/colocation-statefulsets/${id}/action/colocation-offline`,
  });
  return POST(url);
}

/**
 * 混部工作负载-statefulSet-移除单条数据
 */
export function deleteColocationStatefulSetsById(id) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().workloadUrl}/colocation-statefulsets/${id}`,
  });
  return DELETE(url);
}

/**
 * 获取混部工作负载-DaemonSet-列表数据
 */
export function getColocationDaemonSetsListData() {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().workloadUrl}/colocation-daemonsets`,
  });
  return GET(url);
}

/**
 * 获取非混部工作负载-DaemonSet-列表数据
 */
export function getNonColocationDaemonSetsListData() {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().workloadUrl}/non-colocation-daemonsets`,
  });
  return GET(url);
}

/**
 * 混部工作负载-DaemonSet-添加混部DaemonSet
 */
export function addColocationDaemonSets(data) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().workloadUrl}/colocation-daemonsets`,
  });
  return POST(url, data);
}

/**
 * 混部工作负载-DaemonSet-修改为在线状态
 */
export function updateDaemonSetsExpectStatusToOnline(id) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().workloadUrl}/colocation-daemonsets/${id}/action/colocation-online`,
  });
  return POST(url);
}

/**
 * 混部工作负载-DaemonSet-修改为离线状态
 */
export function updateDaemonSetsExpectStatusToOffline(id) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().workloadUrl}/colocation-daemonsets/${id}/action/colocation-offline`,
  });
  return POST(url);
}

/**
 * 混部工作负载-DaemonSet-移除单条数据
 */
export function deleteColocationDaemonSetsById(id) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().workloadUrl}/colocation-daemonsets/${id}`,
  });
  return DELETE(url);
}

/**
 * 获取混部工作负载-Job-列表数据
 */
export function getColocationJobsListData() {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().workloadUrl}/colocation-jobs`,
  });
  return GET(url);
}

/**
 * 获取非混部工作负载-Job-列表数据
 */
export function getNonColocationJobsListData() {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().workloadUrl}/non-colocation-jobs`,
  });
  return GET(url);
}

/**
 * 混部工作负载-Job-添加混部Job
 */
export function addColocationJobs(data) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().workloadUrl}/colocation-jobs`,
  });
  return POST(url, data);
}

/**
 * 混部工作负载-Job-修改为在线状态
 */
export function updateJobsExpectStatusToOnline(id) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().workloadUrl}/colocation-jobs/${id}/action/colocation-online`,
  });
  return POST(url);
}

/**
 * 混部工作负载-Job-修改为离线状态
 */
export function updateJobsExpectStatusToOffline(id) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().workloadUrl}/colocation-jobs/${id}/action/colocation-offline`,
  });
  return POST(url);
}

/**
 * 混部工作负载-Job-移除单条数据
 */
export function deleteColocationJobsById(id) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().workloadUrl}/colocation-jobs/${id}`,
  });
  return DELETE(url);
}

/**
 * 获取混部工作负载-cronJob-列表数据
 */
export function getColocationCronJobsListData() {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().workloadUrl}/colocation-cronjobs`,
  });
  return GET(url);
}

/**
 * 获取非混部工作负载-cronJob-列表数据
 */
export function getNonColocationCronJobsListData() {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().workloadUrl}/non-colocation-cronjobs`,
  });
  return GET(url);
}

/**
 * 混部工作负载-cronJob-添加混部cronJob
 */
export function addColocationCronJobs(data) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().workloadUrl}/colocation-cronjobs`,
  });
  return POST(url, data);
}

/**
 * 混部工作负载-cronJob-修改为在线状态
 */
export function updateCronJobsExpectStatusToOnline(id) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().workloadUrl}/colocation-cronjobs/${id}/action/colocation-online`,
  });
  return POST(url);
}

/**
 * 混部工作负载-cronJob-修改为离线状态
 */
export function updateCronJobsExpectStatusToOffline(id) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().workloadUrl}/colocation-cronjobs/${id}/action/colocation-offline`,
  });
  return POST(url);
}

/**
 * 混部工作负载-cronJob-移除单条数据
 */
export function deleteColocationCronJobsById(id) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().workloadUrl}/colocation-cronjobs/${id}`,
  });
  return DELETE(url);
}

/**
 * 获取混部规则管理-列表数据
 */
export function getColocationRulesListData() {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().rulesUrl}/colocation-rules`,
  });
  return GET(url);
}

/**
 * 获取混部规则管理-创建规则
 */
export function createColocationRule(data) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().rulesUrl}/colocation-rules`,
  });
  return POST(url, data);
}

/**
 * 获取混部规则管理-更新规则
 */
export function updateColocationRule(name, data) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().rulesUrl}/colocation-rules/${name}`,
  });
  return PUT(url, data);
}

/**
 * 获取混部规则管理-删除规则
 */
export function deleteColocationRule(name) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().rulesUrl}/colocation-rules/${name}`,
  });
  return DELETE(url);
}

/**
 * 获取命名空间
 */
export function getNamespaceList() {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().namespaceUrl}/namespaces`,
  });
  return GET(url);
}

/**
 * 节点列表页设置为超卖
 */
export function setOneNodeOversold(nodeId, data) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().nodeUrl}/colocation-nodes/${nodeId}?oversubscription-status=${data}`,
  });
  return PATCH(url, { 'Content-Type': 'application/json' });
}

/**
 * 概览页超卖cpu/memory资源top5统计
 */
export function getResourcesTop5() {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().nodeUrl}/oversubscription-stats`,
  });
  return GET(url);
}

/**
 * 概览页获取cpu及内存使用率
 */
export function getCpuAndMemoryUsage() {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().nodeUrl}/colocation-metrics`,
  });
  return GET(url);
}

export function modifyNodeConfig(data) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().nodeUrl}/colocation-configs`,
  });
  return POST(url, data);
}

export function modifyColocationStatus(name, status) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().nodeUrl}/colocation-nodes/${name}`,
    query: { 'colocation-status': status },
  });
  return PATCH(url);
}

export function getColocationNodeCount() {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().nodeUrl}/colocation-status`,
  });
  return GET(url);
}

export function getColocationCluster() {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().nodeUrl}/colocation-metrics/cluster`,
  });
  return GET(url);
}

export function getColocationNode(node) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().nodeUrl}/colocation-metrics/nodes/${node}`,
  });
  return GET(url);
}