export const NUMA_MONITOR_PATH = '/container_platform/scheduling/numaMonitor';

export const NUMA_MONITOR_TABLE_HEADERS = [
  '节点名称',
  'NUMA节点总数',
  'CPU总数',
  '内存总数',
  'CPU分配率',
  '内存分配率',
  '拓扑策略',
  '容器数',
  '跨NUMA容器数',
];

export const NUMA_POLICY_PATH = '/container_platform/scheduling/numaAffinity';

export const TEST_SSH_ADDRESS = process.env.TEST_NODE2_IP ?? '';
export const TEST_SSH_USERNAME = process.env.TEST_NODE2_USER ?? 'root';
export const TEST_SSH_PASSWORD = process.env.TEST_NODE2_PASSWORD ?? '';

export const TEST_KUBELET_CONFIG_PATH = '/var/lib/kubelet/config.yaml';
export const TEST_KUBELET_SERVICE = 'kubelet';

export const NUMA_RELEASE_NAME = 'numa-affinity-package';
export const NUMA_RELEASE_REPO = 'oci://cr.openfuyao.cn/charts/numa-affinity-package';
export const NUMA_RELEASE_VERSION = '0.0.0-latest';
export const NUMA_RELEASE_NAMESPACE = 'default';

export const RUNTIME_POD_AFFINITY_RPM_KEYWORDS = ['oeaware', 'netrela', 'nriplugin'];
