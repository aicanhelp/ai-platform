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

export const ADMIN_USERNAME = 'admin';

export const PASSWORD_PATTERN = /^(?=.*[0-9])(?=.*[A-Za-z])(?=.*[`~!@#$%^&*()\-_=+\\|[{}\];:'",<.>/?])[A-Za-z0-9`~!@#$%^&*()\-_=+\\|[{}\];:'",<.>/?]+$/;

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
  ServiceUnavailable: 503,
  GatewayTimeout: 504,
};

export const podStatus = ['pending', 'running', 'succeeded', 'failed', 'unknown'];
export const workloadStatus = ['active', 'updating', 'failed'];
export const jobStatus = ['complete', 'active', 'failed'];
export const cronJobStatus = ['completed', 'active', 'suspended'];
export const podContainerOptions = ['Waiting', 'Running', 'Terminated', 'Unknown'];
// 节点状态 - stable keys for i18n (use getNodeStatusLabel(key) in components)
export const nodeStatusOptions = {
  normal: 'normal',
  failed: 'failed',
};

// 节点类型 - stable keys for i18n (use getNodeTypeLabel(key) in components)
export const nodeTypeOptions = {
  workerNode: 'workerNode',
  manageNode: 'manageNode',
};

export const namespaceStatusOptions = {
  active: 'Active',
  terminating: 'Terminating',
};

// helm状态 - stable keys for i18n
export const manageStatusFilterOptions = ['deployed', 'failed', 'pending', 'uninstalling'];

// helm turbo
export const manageTurboFilterOptions = ['true', 'false'];

// Helm resource release status - stable keys for i18n
export const ReleaseStatus = {
  Current: 'current',
  Failed: 'failed',
  InProgress: 'inProgress',
  Terminating: 'terminating',
};

// deployment状态
export const deploymentStatus = {
  Active: 'active',
  Updating: 'updating',
  Failed: 'failed',
};

// Time period options - stable keys for i18n
export const timePeriodOptionsKeys = [
  '10m', '30m', '1h', '3h', '6h', '1d', '3d', '7d', '14d',
];

// Refresh time options - stable keys for i18n
export const refreshTimeOptionsKeys = [
  '0', '15', '30', '60', '300', '86400',
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
    label: 'Deployment',
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

// Alert status options - stable keys for i18n
export const alertStatusOptions = [
  { label: 'all', value: '' },
  { label: 'inactive', value: 'inactive' },
  { label: 'pending', value: 'pending' },
  { label: 'firing', value: 'firing' },
];

// Alert level options - stable keys for i18n
export const alertLevelOptions = [
  { label: 'all', value: '' },
  { label: 'critical', value: 'critical' },
  { label: 'info', value: 'info' },
  { label: 'warning', value: 'warning' },
];

export const alertStatusEx = {
  inactive: 'inactive',
  firing: 'firing',
  pending: 'pending',
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

// Event level options - stable keys for i18n
export const eventLevelOptions = [
  { label: 'all', value: '' },
  { label: 'warning', value: 'Warning' },
  { label: 'normal', value: 'Normal' },
];

// Silent options - stable keys for i18n
export const silentOptions = [
  { label: 'm', value: 'm' },
  { label: 'h', value: 'h' },
  { label: 'd', value: 'd' },
];

/** 在线文档根路径（中文，与 docs.openfuyao.cn 文档中心默认版本一致） */
export const DOC_SITE_ROOT = 'https://docs.openfuyao.cn/zh/docs/v26.03';

/** 对外参考链接（kubectl、用户指南入口等） */
export const publicLink = {
  kubectlLink: 'https://kubernetes.io/zh-cn/docs/reference/kubectl/',
  helpWordLink: `${DOC_SITE_ROOT}/user_guide/getting_started.html`,
};

export const docAddress = [
  {
    path: '/container_platform/overview',
    address: `${DOC_SITE_ROOT}/user_guide/summary.html`,
  },
  {
    path: '/container_platform/appMarket/appOverview',
    address: `${DOC_SITE_ROOT}/user_guide/application_market.html#概览`,
  },
  {
    path: '/container_platform/appMarket/marketCategory',
    address: `${DOC_SITE_ROOT}/user_guide/application_market.html#使用应用列表`,
  },
  {
    path: '/container_platform/appMarket/stash',
    address: `${DOC_SITE_ROOT}/user_guide/application_market.html#使用仓库配置`,
  },
  {
    path: '/container_platform/applicationManageHelm',
    address: `${DOC_SITE_ROOT}/user_guide/application_management.html`,
  },
  {
    path: '/container_platform/extendManage',
    address: `${DOC_SITE_ROOT}/user_guide/extension_management.html`,
  },
  {
    path: '/container_platform/workload/pod',
    address: `${DOC_SITE_ROOT}/user_guide/resource_management/workloads.html#使用pod`,
  },
  {
    path: '/container_platform/workload/deployment',
    address: `${DOC_SITE_ROOT}/user_guide/resource_management/workloads.html#后续操作`,
  },
  {
    path: '/container_platform/workload/statefulSet',
    address: `${DOC_SITE_ROOT}/user_guide/resource_management/workloads.html#后续操作`,
  },
  {
    path: '/container_platform/workload/daemonSet',
    address: `${DOC_SITE_ROOT}/user_guide/resource_management/workloads.html#后续操作`,
  },
  {
    path: '/container_platform/workload/job',
    address: `${DOC_SITE_ROOT}/user_guide/resource_management/workloads.html#后续操作`,
  },
  {
    path: '/container_platform/workload/cronJob',
    address: `${DOC_SITE_ROOT}/user_guide/resource_management/workloads.html#后续操作`,
  },
  {
    path: '/container_platform/network/service',
    address: `${DOC_SITE_ROOT}/user_guide/resource_management/networking.html#使用service`,
  },
  {
    path: '/container_platform/network/ingress',
    address: `${DOC_SITE_ROOT}/user_guide/resource_management/networking.html#使用ingress`,
  },
  {
    path: '/container_platform/storage/pv',
    address: `${DOC_SITE_ROOT}/user_guide/resource_management/storage.html#使用pv`,
  },
  {
    path: '/container_platform/storage/pvc',
    address: `${DOC_SITE_ROOT}/user_guide/resource_management/storage.html#使用pvc`,
  },
  {
    path: '/container_platform/storage/sc',
    address: `${DOC_SITE_ROOT}/user_guide/resource_management/storage.html#使用sc`,
  },
  {
    path: '/container_platform/nodeManage',
    address: `${DOC_SITE_ROOT}/user_guide/resource_management/nodes.html`,
  },
  {
    path: '/container_platform/configuration/configMap',
    address: `${DOC_SITE_ROOT}/user_guide/resource_management/configmap_and_secrets.html#使用configmap`,
  },
  {
    path: '/container_platform/configuration/secret',
    address: `${DOC_SITE_ROOT}/user_guide/resource_management/configmap_and_secrets.html#使用secret`,
  },
  {
    path: '/container_platform/namespace/namespaceManage',
    address: `${DOC_SITE_ROOT}/user_guide/resource_management/namespaces.html#使用namespace`,
  },
  {
    path: '/container_platform/namespace/limitRange',
    address: `${DOC_SITE_ROOT}/user_guide/resource_management/namespaces.html#使用limitrange`,
  },
  {
    path: '/container_platform/namespace/resourceQuota',
    address: `${DOC_SITE_ROOT}/user_guide/resource_management/namespaces.html#使用resourcequota`,
  },
  {
    path: '/container_platform/customResourceDefinition',
    address: `${DOC_SITE_ROOT}/user_guide/resource_management/custom_resources.html`,
  },
  {
    path: '/container_platform/colocation/ColocationOverview',
    address: `${DOC_SITE_ROOT}/user_guide/computing_power_optimization_center/colocation.html#查看概览`,
  },
  {
    path: '/container_platform/colocation/ColocationNodeManagement',
    address: `${DOC_SITE_ROOT}/user_guide/computing_power_optimization_center/colocation.html#打开或关闭节点的混部标签`,
  },
  {
    path: '/container_platform/colocation/ColocationWorkloadManagement',
    address: `${DOC_SITE_ROOT}/user_guide/resource_management/workloads.html`,
  },
  {
    path: '/container_platform/colocation/ColocationRulesManagement',
    address: `${DOC_SITE_ROOT}/user_guide/computing_power_optimization_center/colocation.html#使用混部策略参数配置`,
  },
  {
    path: '/container_platform/scheduling/NumaOverview',
    address: `${DOC_SITE_ROOT}/user_guide/computing_power_optimization_center/numa-aware_scheduling_user_guide.html#查看概览页`,
  },
  {
    path: '/container_platform/scheduling/AffinityPolicyConfiguration',
    address: `${DOC_SITE_ROOT}/user_guide/computing_power_optimization_center/numa-aware_scheduling_user_guide.html#使用亲和策略配置`,
  },
  {
    path: '/container_platform/scheduling/numaMonitor',
    address: `${DOC_SITE_ROOT}/user_guide/computing_power_optimization_center/numa-aware_scheduling_user_guide.html#使用集群numa监控`,
  },
  {
    path: '/container_platform/monitor/monitorDashboard',
    address: `${DOC_SITE_ROOT}/user_guide/observability_center/monitoring.html#使用监控看板`,
  },
  {
    path: '/container_platform/monitor/monitorGoalManage',
    address: `${DOC_SITE_ROOT}/user_guide/observability_center/monitoring.html#使用监控目标`,
  },
  {
    path: '/container_platform/monitor/monitorRuleManage',
    address: `${DOC_SITE_ROOT}/user_guide/observability_center/monitoring.html#告警规则`,
  },
  {
    path: '/container_platform/monitoring-dashboard/dashboard',
    address: `${DOC_SITE_ROOT}/user_guide/observability_center/custom_monitoring_dashboard.html`,
  },
  {
    path: '/container_platform/logging/logSearch',
    address: `${DOC_SITE_ROOT}/user_guide/observability_center/logs.html#使用日志查询`,
  },
  {
    path: '/container_platform/logging/logSet',
    address: `${DOC_SITE_ROOT}/user_guide/observability_center/logs.html#使用日志配置`,
  },
  {
    path: '/container_platform/alarm/alarmIndex',
    address: `${DOC_SITE_ROOT}/user_guide/observability_center/alerts.html#使用当前告警`,
  },
  {
    path: '/container_platform/alarm/silentAlarm',
    address: `${DOC_SITE_ROOT}/user_guide/observability_center/alerts.html#使用静默告警`,
  },
  {
    path: '/container_platform/event',
    address: `${DOC_SITE_ROOT}/user_guide/observability_center/events.html`,
  },
  {
    path: '/container_platform/clusterUser',
    address: `${DOC_SITE_ROOT}/user_guide/user_management.html#分配集群角色`,
  },
  {
    path: '/container_platform/clusterMember',
    address: `${DOC_SITE_ROOT}/user_guide/user_management.html#查看集群成员`,
  },
  {
    path: '/container_platform/userManage/serviceAccount',
    address: `${DOC_SITE_ROOT}/user_guide/permission_management/rbac_management.html#使用服务账号`,
  },
  {
    path: '/container_platform/userManage/role',
    address: `${DOC_SITE_ROOT}/user_guide/permission_management/rbac_management.html#使用角色`,
  },
  {
    path: '/container_platform/userManage/roleBinding',
    address: `${DOC_SITE_ROOT}/user_guide/permission_management/rbac_management.html#使用角色绑定`,
  },
  {
    path: '/multicluster',
    address: `${DOC_SITE_ROOT}/user_guide/multi_cluster_management.html`,
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
    path: '/container_platform/cluster',
    address: `${DOC_SITE_ROOT}/user_guide/cluster_installation_deployment/cluster_lifecycle_management.html`,
  },
  {
    path: '/container_platform/cluster/clusterManage',
    address: `${DOC_SITE_ROOT}/user_guide/cluster_installation_deployment/cluster_lifecycle_management.html#查看集群`,
  },
  {
    path: '/container_platform/ray',
    address: `${DOC_SITE_ROOT}/user_guide/computing_power_optimization_center/openfuyao_ray.html`,
  },
  {
    path: '/',
    address: `${DOC_SITE_ROOT}/quick_start.html`,
  },
];
// 终端ws前缀
export function getWsPrefix() {
  let clusterName = sessionStorage.getItem('cluster');
  let paramObj = {
    terminalUrl: `/clusters/${clusterName}/rest/webterminal/v1`,
  };
  return paramObj;
}

// 扩展组件前缀匹配
export const expandComponent = {
  Logging: 'logging',
  ComputingPowerEngine: 'computing-power-engine',
  MonitoringDashboard: 'monitoring-dashboard',
  Colocation: 'colocation',
  Volcano: 'scheduling',
  Cluster: 'cluster',
  Ray: 'ray',
  MisManagement: 'mis-management',
};
