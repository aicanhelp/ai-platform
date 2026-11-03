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
export const containerRouterPrefix = 'logging-website';
export const containerStylePrefix = 'logging-website';
// 日志下拉框选项logging
export const levelOptions = [
  {
    value: 'error',
    label: 'error',
  },
  {
    value: 'warning',
    label: 'warning',
  },
  {
    value: 'info',
    label: 'info',
  },
  {
    value: 'debug',
    label: 'debug',
  },
  {
    value: 'critical',
    label: 'critical',
  },
];
