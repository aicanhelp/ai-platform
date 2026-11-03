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

function monitorPrefix() {
  const clusterStore = sessionStorage?.cluster;
  const clusterPrefix = clusterStore ? `/clusters/${clusterStore}` : '';
  return `${clusterPrefix}/rest/monitoring-dashboard/v1beta1`;
}

/**
 * 获取dashboard列表
 */
export function getMonitorDashboardList() {
  return GET(`${monitorPrefix()}/dashboards`);
}

/**
 * 创建单个Dashboard资源
 */
export function createEachDashboard(data) {
  return POST(`${monitorPrefix()}/dashboards`, data);
}

/**
 * 获取单个Dashboard资源
 */
export function getDashboardResource(dashboardID, start, end, step) {
  const url = qs.stringifyUrl({
    url: `${monitorPrefix()}/dashboards/${dashboardID}`,
    query: {
      start,
      end,
      step,
    },
  });
  return GET(url);
}

/**
 * 修改单个dashboard资源
 */
export function updateEachDashboard(dashboardID, data) {
  const url = qs.stringifyUrl({
    url: `${monitorPrefix()}/dashboards/${dashboardID}`,
  });
  return PATCH(url, data);
}

/**
 * 删除单个dashboard资源
 */
export function deleteDashboard(dashboardID) {
  return DELETE(`${monitorPrefix()}/dashboards/${dashboardID}`);
}

/**
 * 创建单个widget资源
 */
export function createWidgetResource(data) {
  return POST(`${monitorPrefix()}/widgets`, data);
}

/**
 * 修改单个Widget资源
 */
export function updateWidgetResource(widgetID, data) {
  return PATCH(`${monitorPrefix()}/widgets/${widgetID}`, data);
}

/**
 * 删除单个widget资源
 */
export function deleteWidgetReousrce(widgetID) {
  return DELETE(`${monitorPrefix()}/widgets/${widgetID}`);
}

/**
 * 获取指标列表
 */
export function getIndicatorsList(dataSource) {
  const url = qs.stringifyUrl({
    url: `${monitorPrefix()}/widgets/indicators`,
    query: {
      dataSource,
    },
  });
  return GET(url);
}

/**
 * 根据指标名称获取实例列表
 */
export function getInstanceList(displayName, dataSource) {
  const url = qs.stringifyUrl({
    url: `${monitorPrefix()}/widgets/indicators/${displayName}`,
    query: {
      dataSource,
    },
  });
  return GET(url);
}

/**
 * 获取实例列表中每个实例的数据
 */
export function getInstanceEachData(dataSource, start, end, step, instances) {
  const url = qs.stringifyUrl({
    url: `${monitorPrefix()}/widgets/indicators/instances`,
    query: {
      dataSource,
      start,
      end,
      step,
    },
  });
  return POST(url, instances);
}
