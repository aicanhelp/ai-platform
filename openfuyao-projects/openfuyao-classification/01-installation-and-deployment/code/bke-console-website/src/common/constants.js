/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
export const DEFAULT_PAGE_SIZE = 10;
export const DEFAULT_CURRENT_PAGE = 1;

export const ResponseCode = {
  OK: 200,
  Created: 201,
  Accepted: 202,
  NoContent: 204,
  Found: 302,
  BadRequest: 400,
  UnAuthorized: 401,
  Forbidden: 403,
  NotFound: 404,
  Conflict: 409,
  InternalServerError: 500,
  BadGateway: 502,
  GatewayTimeout: 504,
};

export const workloadFilterOptions = ['pending', 'running', 'succeeded', 'failed', 'unknown'];

export const podContainerOptions = ['Waiting', 'Running', 'Terminated', 'Unknown'];
// 节点状态
export const nodeStatusOptions = {
  normal: '正常',
  failed: '异常',
};

// 节点类型
export const nodeTypeOptions = {
  workerNode: '工作节点',
  manageNode: '管理节点',
};

export const namespaceStatusOptions = {
  active: 'Active',
  terminating: 'Terminating',
};

// helm状态
export const manageStatusFilterOptions = ['部署成功', '部署失败', '处理中', '卸载中'];

// helm turbo
export const manageTurboFilterOptions = ['true', 'false'];

export const ReleaseStatus = {
  Current: '运行中',
  Failed: '失败',
  InProgress: '创建中',
  Terminating: '删除中',
};

// deployment状态
export const deploymentStatus = {
  Active: 'active',
  Updating: 'updating',
  Failed: 'failed',
};

// job
export const jobStatus = {
  Complete: 'complete',
  Active: 'active',
  Failed: 'failed',
};

// cronJob
export const cronJobStatus = {
  Completed: 'completed',
  Active: 'active',
};

export const timePeriodOptions = [
  {
    label: '近10分钟',
    value: '10m',
  },
  {
    label: '近30分钟',
    value: '30m',
  },
  {
    label: '近1小时',
    value: '1h',
  },
  {
    label: '近3小时',
    value: '3h',
  },
  {
    label: '近6小时',
    value: '6h',
  },
  {
    label: '近1天',
    value: '1d',
  },
  {
    label: '近3天',
    value: '3d',
  },
  {
    label: '近7天',
    value: '7d',
  },
  {
    label: '近14天',
    value: '14d',
  },
];

export const refreshTimeOptions = [
  {
    label: '不自动刷新',
    value: 0,
  },
  {
    label: '刷新间隔 15秒',
    value: 15,
  },
  {
    label: '刷新间隔 30秒',
    value: 30,
  },
  {
    label: '刷新间隔 1分钟',
    value: 60,
  },
  {
    label: '刷新间隔 5分钟',
    value: 300,
  },
  {
    label: '刷新间隔 1天',
    value: 86400,
  },
];

export const controllMonitorOptions = [
  {
    label: 'ETCD',
    value: 'etcd',
  },
  {
    label: 'kube-apiserver',
    value: 'kube-apiserver',
  },
  {
    label: 'kube-scheduler',
    value: 'kube-scheduler',
  },
  {
    label: 'kube-controller-manager',
    value: 'kube-controller-manager',
  },
];

export const nodeMonitorOptions = [
  {
    label: 'kubelet',
    value: 'kubelet',
  },
  {
    label: 'kube-proxy',
    value: 'kube-proxy',
  },
];

export const typeOptions = [
  {
    label: 'deployment',
    value: 'deployment',
  },
  {
    label: 'StatefulSet',
    value: 'statefulset',
  },
  {
    label: 'DaemonSet',
    value: 'daemonset',
  },
];

export const monitorGoalFilterOptions = [
  {
    label: 'up',
    value: 'up',
  },
  {
    label: 'down',
    value: 'down',
  },
  {
    label: 'unknown',
    value: 'unknown',
  },
];

export const alarmStatusOptions = [
  {
    label: '全部',
    value: '',
  },
  {
    label: '未触发(inactive)',
    value: 'inactive',
  },
  {
    label: '待定(pending)',
    value: 'pending',
  },
  {
    label: '触发(firing)',
    value: 'firing',
  },
];

export const alarmLevelOptions = [
  {
    label: '全部',
    value: '',
  },
  {
    label: '严重',
    value: 'critical',
  },
  {
    label: '提示',
    value: 'info',
  },
  {
    label: '警告',
    value: 'warning',
  },
];


export const alarmStatusEx = {
  inactive: '未触发',
  firing: '触发',
  pending: '待定',
};

export const alarmLevelEx = {
  critical: '严重',
  info: '提示',
  warning: '警告',
};

export const stepList = {
  '10m': '2s',
  '30m': '6s',
  '1h': '14s',
  '3h': '72s',
  '6h': '86s',
  '1d': '345s',
  '3d': '1035s',
  '7d': '2419s',
  '14d': '4838s',
}; // 基于prometheus 5min-1s

export const disabledModifyMonitorServiceCr = [
  'alertmanager-main',
  'blackbox-exporter',
  'coredns', 'etcd',
  'kube-apiserver',
  'kube-controller-manager',
  'kube-proxy',
  'kube-scheduler',
  'kube-state-metrics',
  'kubelet',
  'node-exporter',
  'prometheus-k8s',
  'prometheus-operator',
]; // 禁止修改的serviceminitor实例

export const eventLevelOptions = [
  {
    label: '全部',
    value: '',
  },
  {
    label: '警告',
    value: 'Warning',
  },
  {
    label: '正常',
    value: 'Normal',
  },
];

export const silentOptions = [
  { label: '分', value: 'm' },
  { label: '时', value: 'h' },
  { label: '天', value: 'd' },
];
// 终端类型
export const terminalType = {
  cluster: '集群',
  container: '容器',
};

/** 在线文档根路径（中文，与 docs.openfuyao.cn 文档中心默认版本一致） */
export const DOC_SITE_ROOT = 'https://docs.openfuyao.cn/zh/docs/v26.03';

export const docAddress = [
  {
    path: '/container_platform/cluster/',
    address: `${DOC_SITE_ROOT}/user_guide/cluster_installation_deployment/cluster_lifecycle_management.html`,
  },
  {
    path: '/user_manage/user',
    address: `${DOC_SITE_ROOT}/user_guide/user_management.html#查看用户信息`,
  },
  {
    path: '/user_manage/role',
    address: `${DOC_SITE_ROOT}/user_guide/user_management.html#查看平台级角色列表`,
  },
  {
    path: '/setting/userinfo',
    address: `${DOC_SITE_ROOT}/user_guide/user_management.html#使用用户视角的用户管理`,
  },
  {
    path: '/',
    address: `${DOC_SITE_ROOT}`,
  },
];

// 扩展组件前缀匹配
export const expandComponent = {
  Logging: 'logging',
  ComputingPowerEngine: 'computing-power-engine',
  MonitoringDashboard: 'monitoring-dashboard',
  Colocation: 'colocation',
  Volcano: 'scheduling',
  Cluster: 'cluster',
  Ray: 'ray',
};
