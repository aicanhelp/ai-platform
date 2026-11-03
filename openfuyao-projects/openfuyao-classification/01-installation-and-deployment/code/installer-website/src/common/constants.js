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
export const containerRouterPrefix = 'cluster';
export const containerStylePrefix = 'cluster'; // 样式前缀

export const DEFAULT_PAGE_SIZE = 10;
export const DEFAULT_CURRENT_PAGE = 1;

// 集群生命周期管理状态
export const clusterStatus = {
  Healthy: 'healthy',
  Unhealthy: 'unhealthy',
  Creating: 'creating',
  Deploying: 'deploying',
  ScalingWorkerNodesUp: 'scalingWorkerNodesUp',
  ScalingWorkerNodesDown: 'scalingWorkerNodesDown',
  ScalingMasterNodesUp: 'scalingMasterNodesUp',
  ScalingMasterNodesDown: 'scalingMasterNodesDown',
  Upgrading: 'upgrading',
  DeployFailed: 'deployFailed',
  ScaleFailed: 'scaleFailed',
  UpgradeFailed: 'upgradeFailed',
  Deleting: 'deleting',
  DeleteFailed: 'deleteFailed',
  Checking: 'checking',
  Paused: 'paused',
  PauseFailed: 'pauseFailed',
  DryRun: 'dryRun',
  DryRunFailed: 'dryRunFailed',
};

export const nodeStatus = {
  Ready: 'healthy',
  NotReady: 'unhealthy',
  Unknown: 'unknown',
  // in-progress
  Pending: 'deploying',
  Provisioned: 'checking',
  Initializing: 'deploying',
  BootStrapping: 'deploying',
  Upgrading: 'upgrading',
  Deleting: 'deleting',
  Managing: 'checking',
  // failed
  Failed: 'deployFailed',
  InitFailed: 'deployFailed',
  BootStrapFailed: 'deployFailed',
  UpgradeFailed: 'upgradeFailed',
  DeleteFailed: 'deleteFailed',
  ManageFailed: 'deployFailed',
};

export const documentUrl = `https://docs.openfuyao.com/docs/%E7%94%A8%E6%88%B7%E6%8C%87%E5%8D%97/%E9%9B%86%E7%BE%A4%E7%94%9F%E5%91%BD%E5%91%A8%E6%9C%9F%E7%AE%A1%E7%90%86#%E5%8D%87%E7%BA%A7%E9%9B%86%E7%BE%A4`;

export const ADDONLIST = {
  NONE: 'none',
  CLUSTERAPI: 'cluster-api',
  OPENFUYAOSYSTEMCONTROLLER: 'openfuyao-system-controller',
  KUBEPROXY: 'kubeproxy',
};

export const DEFAULT_REPO = {
  imageRepo: 'cr.openfuyao.cn',
  helmRepo: 'https://helm.openfuyao.cn/_core',
};
