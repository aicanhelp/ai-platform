/** APIs and Resource URLs */
export const COLOCATION_CONFIG_CONFIGMAP =
  '/clusters/host/api/kubernetes/api/v1/namespaces/openfuyao-colocation/configmaps/colocation-config';
export const VOLCANO_CONFIG_CONFIGMAP =
  '/clusters/host/api/kubernetes/api/v1/namespaces/volcano-system/configmaps/volcano-scheduler-configmap';
export const COLOCATION_NODES = '/clusters/host/rest/colocation/v1/colocation-nodes';
export const COLOCATION_CONFIGS = '/clusters/host/rest/colocation/v1/colocation-configs';
export const NODE_PATH = '/clusters/host/api/kubernetes/api/v1/nodes/';

/** Page URIs */
export const COLOCATION_OVERVIEW_PAGE_URI = '/container_platform/colocation/ColocationOverview';
export const COLOCATION_CONFIG_PAGE_URI = '/container_platform/colocation/ColocationConfiguration';
export const COLOCATION_MONITOR_PAGE_URI = '/container_platform/colocation/ColocationMonitor';
export const WORKLOAD_DEPLOYMENT_PAGE_URI = '/container_platform/workload/pod';

export const FUYAO_BASE_URL = process.env.TEST_FUYAOURL ?? '';
export const FUYAO_USERNAME = process.env.TEST_USERNAME ?? '';
export const FUYAO_PASSWORD = process.env.TEST_PASSWORD ?? '';

export const TEST_SSH_ADDRESS = process.env.TEST_NODE2_IP ?? '';
export const TEST_SSH_USERNAME = process.env.TEST_NODE2_USER ?? 'root';
export const TEST_SSH_PASSWORD = process.env.TEST_NODE2_PASSWORD ?? '';

export const NUMA_RELEASE_NAME = 'numa-affinity-package';
export const NUMA_RELEASE_REPO = 'oci://cr.openfuyao.cn/charts/numa-affinity-package';
export const NUMA_RELEASE_VERSION = '0.0.0-latest';
export const NUMA_RELEASE_NAMESPACE = 'default';

export const COLOCATION_RELEASE_NAME = 'colocation-package';
export const COLOCATION_RELEASE_REPO = 'oci://cr.openfuyao.cn/charts/colocation-package';
export const COLOCATION_RELEASE_VERSION = '0.0.0-latest';
export const COLOCATION_RELEASE_NAMESPACE = 'default';
