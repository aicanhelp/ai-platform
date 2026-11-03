/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
export const ResponseCode = {
  OK: 200,
  Created: 201,
  Accepted: 202,
  NoContent: 204,
  BadRequest: 400,
  UnAuthorized: 401,
  Forbidden: 403,
  NotFound: 404,
  Conflict: 409,
  InternalServerError: 500,
  BadGateway: 502,
  GatewayTimeout: 504,
};

// 容器平台路由前缀
export const containerRouterPrefix = 'monitoring-dashboard';
export const containerStylePrefix = 'monitoring-dashboard'; // 样式前缀

export const DEFAULT_PAGE_SIZE = 10;
export const DEFAULT_CURRENT_PAGE = 1;

export const stepList = {
  '10m': 2,
  '30m': 6,
  '1h': 14,
  '3h': 72,
  '6h': 86,
  '1d': 345,
  '3d': 1035,
  '7d': 2419,
  '14d': 4838,
}; // 基于prometheus 5min-1s

export const timePeriodOptions = [
  {
    label: 'timePeriod.last10Minutes',
    value: '10m',
  },
  {
    label: 'timePeriod.last30Minutes',
    value: '30m',
  },
  {
    label: 'timePeriod.last1Hour',
    value: '1h',
  },
  {
    label: 'timePeriod.last3Hours',
    value: '3h',
  },
  {
    label: 'timePeriod.last6Hours',
    value: '6h',
  },
  {
    label: 'timePeriod.last1Day',
    value: '1d',
  },
  {
    label: 'timePeriod.last3Days',
    value: '3d',
  },
  {
    label: 'timePeriod.last7Days',
    value: '7d',
  },
  {
    label: 'timePeriod.last14Days',
    value: '14d',
  },
];

export const refreshTimeOptions = [
  {
    label: 'refreshTime.noAutoRefresh',
    value: 0,
  },
  {
    label: 'refreshTime.interval15s',
    value: 15,
  },
  {
    label: 'refreshTime.interval30s',
    value: 30,
  },
  {
    label: 'refreshTime.interval1m',
    value: 60,
  },
  {
    label: 'refreshTime.interval5m',
    value: 300,
  },
  {
    label: 'refreshTime.interval1d',
    value: 86400,
  },
];

export const componentTypeOptions = [
  {
    label: 'componentType.lineChart',
    value: 'line',
  },
  {
    label: 'componentType.stackedAreaChart',
    value: 'area',
  },
  {
    label: 'componentType.number',
    value: 'number',
  },
  {
    label: 'componentType.gauge',
    value: 'dashboard',
  },
  {
    label: 'componentType.donut',
    value: 'dount',
  },
  {
    label: 'componentType.barChart',
    value: 'bar',
  },
];

export const widgetTypeList = {
  line: '0',
  area: '1',
  number: '2',
  dashboard: '3',
  dount: '4',
  bar: '5',
};
export const reserveWidgetType = {
  0: 'line',
  1: 'area',
  2: 'number',
  3: 'dashboard',
  4: 'dount',
  5: 'bar',
};

// 业务流code
export const BusinessCode = {
  DuplicateName: 403,
};
