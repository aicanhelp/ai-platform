export default {
  // Common
  'common.loading': 'Loading...',
  'common.confirm': 'OK',
  'common.cancel': 'Cancel',
  'common.delete': 'Delete',
  'common.create': 'Create',
  'common.edit': 'Edit',
  'common.search': 'Search',
  'common.refresh': 'Refresh',
  'common.name': 'Name',
  'common.status': 'Status',
  'common.action': 'Operation',
  'common.noData': 'No Data',
  'common.save': 'Save',
  'common.reset': 'Reset',
  'common.success': 'Success',
  'common.failed': 'Failed',
  'common.warning': 'Warning',
  'common.tip': 'Tip',
  'common.yes': 'Yes',
  'common.no': 'No',
  'common.enable': 'Enable',
  'common.disable': 'Disable',
  'common.close': 'Close',
  'common.back': 'Back',
  'common.submit': 'Submit',
  'common.select': 'Please select',
  'common.input': 'Please input',
  'common.type': 'Type',
  'common.description': 'Description',
  'common.time': 'Time',
  'common.total': 'Total',

  // Colocation
  'colocation.title': 'Colocation',
  'colocation.overview': 'Overview',
  'colocation.cluster': 'Cluster',
  'colocation.node': 'Node',
  'colocation.resource': 'Resource',
  'colocation.utilization': 'Utilization',
  'colocation.cpuUsage': 'CPU Usage',
  'colocation.memoryUsage': 'Memory Usage',
  'colocation.onlineService': 'Online Service',
  'colocation.offlineTask': 'Offline Task',
  'colocation.sla': 'SLA Guarantee',
  'colocation.strategy': 'Colocation Strategy',
  'colocation.config': 'Colocation Strategy Config',

  // Workload types
  'workload.pod': 'Pod',
  'workload.deployment': 'Deployment',
  'workload.statefulset': 'StatefulSet',
  'workload.daemonset': 'DaemonSet',
  'workload.job': 'Job',
  'workload.cronjob': 'CronJob',
  'workload.workload': 'Workload',

  // Workload status
  'workload.status.online': 'Online',
  'workload.status.offline': 'Offline',
  'workload.status.oversold': 'Oversold',

  // Table column titles
  'table.column.podName': 'Pod Name',
  'table.column.workloadName': 'Workload Name',
  'table.column.namespace': 'Namespace',
  'table.column.colocationNode': 'Colocation Node',
  'table.column.podIP': 'Pod IP',
  'table.column.businessAttr': 'Business Attribute',
  'table.column.expectedBusinessAttr': 'Expected Business Attribute',
  'table.column.createTime': 'Create Time',
  'table.column.scheduleTime': 'Schedule Time',
  'table.column.nodeName': 'Node Name',
  'table.column.ipAddress': 'IP Address',
  'table.column.osVersion': 'OS Version',
  'table.column.colocationCapability': 'Colocation Capability',
  'table.column.enableColocation': 'Enable Colocation',

  // Actions
  'action.changeToOffline': 'Change to Offline',
  'action.changeToOversold': 'Change to Oversold',
  'action.setNonColocation': 'Set as Non-Colocation',
  'action.setNonColocationWorkload': 'Set as Non-Colocation Workload',

  // Messages
  'message.nonColocationSetSuccess': 'Non-colocation set successfully',
  'message.nonColocationSetFailed': 'Failed to set non-colocation',
  'message.apiError': 'API error',
  'message.modifySuccess': 'Modified successfully',
  'message.modifyFailed': 'Failed to modify',
  'message.selectAtLeastOneWorkload': 'Please select at least 1 workload',
  'message.maxTenWorkloads': 'You can add up to 10 workloads at a time',
  'message.addColocationSuccess': 'Colocation {type} added successfully',
  'message.addColocationFailed': 'Failed to add colocation {type}',
  'message.selectedWorkloads': '{count} workloads selected',
  'message.confirmSetNonColocation':
    'Are you sure you want to set the colocation workload {name} as a non-colocation workload?',
  'message.changeToState': 'Change to {state}',
  'message.confirmChangeState': 'After confirmation, your colocation attribute will be changed to {state} state',
  'message.qosTip':
    'Online workloads have higher QoS levels than offline workloads, and oversold workloads use oversold resources.',
  'message.higherControllerTip': 'Please go to other higher-level controllers for management',
  'message.colocationOnlyOnColocationNode': 'Colocation applications can only be deployed on colocation nodes',
  'message.dataFetchFailed': 'Failed to fetch data',
  'message.maxTenNodes': 'You can add up to 10 nodes at a time',
  'message.selectAtLeastOneNode': 'Please select at least 1 node',
  'message.selectedNodes': '{count} nodes selected',

  // Search placeholders
  'search.podName': 'Search Pod name',
  'search.workloadName': 'Search workload name',
  'search.byLabel': 'Search by label  e.g.: key:value',
  'search.node': 'Search node',

  // Buttons
  'button.addColocation': 'Add Colocation {type}',
  'button.colocationParamConfig': 'Colocation Policy Parameter Configuration',
  'button.ok': 'OK',

  // Pagination
  'pagination.total': 'Total {total} items',

  // Configuration page
  'config.title': 'Colocation Policy Configuration',
  'config.fullSupport': 'Full Support',
  'config.partialSupport': 'Partial Support',
  'config.unavailable': 'Unavailable',
  'config.requiredFeatures': 'Required Features:',
  'config.advancedFeatures': 'Advanced Features:',
  'config.paramConfigTitle': 'Colocation Policy Parameter Configuration',
  'config.paramConfigTip':
    'This configuration will be applied to all nodes with colocation enabled, and does not currently support specific configuration for each colocation node',
  'config.validationFailed': 'Colocation configuration parameter validation failed',
  'config.modifySuccess': 'Colocation strategy configuration modified successfully',
  'config.modifyFailed': 'Failed to modify colocation strategy configuration',
  'config.inputValue': 'Please enter a value',
  'config.inputIntegerRange': 'Please enter an integer between {min}-{max}',
  'config.inputFloatRange': 'Please enter a float between {min}-{max}',
  'config.maxDecimals': 'Up to {max} decimal places',
  'config.nodeFeatureNotSupported': 'Node does not support required features: {features}',
  'config.nodeAdvancedFeatureNotSupported': 'Node does not support enabled advanced features: {features}',
  'config.colocationNodeNotSupportFeature': 'Colocation-enabled nodes do not support this advanced feature: ',
  'config.nodeEnabled': 'enabled',
  'config.nodeDisabled': 'disabled',
  'config.nodeColocationEnabled': 'Node {name} colocation {status}',
  'config.modifyNodeStatusFailed': 'Failed to modify node {name} colocation status, {error}',

  // Feature support status
  'config.feature.cpuPreemption': 'CPU Preemption',
  'config.feature.memoryPreemption': 'Memory Preemption',
  'config.feature.quotaTurbo': 'CPU Elastic Throttling',
  'config.feature.dynMemory': 'Memory Async Tiered Reclaim',
  'config.feature.dynCache': 'Memory Bandwidth & L3 Cache Limit',
  'config.feature.psiDetection': 'PSI Detection',
  'config.feature.detecting': 'Detecting',
  'config.feature.missing_cpu_qos_level': 'Kernel missing cpu.qos_level interface',
  'config.feature.missing_memory_qos_level': 'Kernel missing memory.qos_level interface',
  'config.feature.missing_cpu_cfs': 'Kernel missing CPU.CFS control interface',
  'config.feature.missing_memory_high_async_ratio': 'Kernel missing memory.high_async_ratio interface',
  'config.feature.unsupported_rdt_mpam': 'Hardware does not support RDT/MPAM',
  'config.feature.unsupported_psi': 'Kernel does not support PSI',

  // Volcano scheduling
  'config.volcano.title': 'Load-aware Scheduling',
  'config.volcano.tip':
    'openFuyao senses the CPU and Memory load of cluster nodes, and schedules Pods to nodes with lower loads to achieve node load balancing and avoid application or node failures caused by high load on a single node',
  'config.volcano.thresholdSwitch': 'Real Load Scheduling Threshold Switch',
  'config.volcano.cpuThreshold': 'Node CPU Real Load Scheduling Threshold (%)',
  'config.volcano.cpuThresholdTip':
    'After the CPU real utilization of online business on the node exceeds this threshold, newly delivered workloads will not be scheduled to the current node, and the workloads already running on the node will not be affected',
  'config.volcano.memoryThreshold': 'Node Memory Real Load Scheduling Threshold (%)',
  'config.volcano.memoryThresholdTip':
    'After the memory real utilization of online business on the node exceeds this threshold, newly delivered workloads will not be scheduled to the current node, and the workloads already running on the node will not be affected',

  // Rubik
  'config.rubik.title': 'Offline Load Watermark Eviction',
  'config.rubik.cpuEvictionThreshold': 'CPU Eviction Watermark (%)',
  'config.rubik.cpuEvictionTip':
    'When the actual CPU usage of all Pods on the node exceeds this threshold, offline job eviction is triggered until it returns below the watermark',
  'config.rubik.memoryEvictionThreshold': 'Memory Eviction Watermark (%)',
  'config.rubik.memoryEvictionTip':
    'When the actual memory usage of all Pods on the node exceeds this threshold, offline job eviction is triggered until it returns below the watermark',

  // QuotaTurbo
  'config.quotaTurbo.title': 'CPU Elastic Throttling',
  'config.quotaTurbo.tip':
    'Implemented through the CFS bandwidth control capability provided by the Linux kernel, while ensuring the stability and security of the whole machine load level and not affecting the operation of other businesses, it allows business containers to adaptively adjust CPU limits through a dual-water-level mechanism, alleviates CPU resource bottlenecks, and improves business operation performance',
  'config.quotaTurbo.highWaterMark': 'CPU Load High Water Mark',
  'config.quotaTurbo.highWaterMarkTip':
    'After the CPU real utilization of the node exceeds this threshold, newly delivered workloads will not be scheduled to the current node, and the workloads already running on the node will not be affected',
  'config.quotaTurbo.alarmWaterMark': 'CPU Load Alarm Water Mark',
  'config.quotaTurbo.alarmWaterMarkTip':
    'The colocation engine determines the Quota configuration for pods with quotaTurbo enabled based on the relationship between the actual CPU load of the whole machine and the high water mark/alarm water mark:\n1. If the whole machine CPU load is higher than the alarm water mark, all pods with quotaTurbo enabled whose actual Quota exceeds the configured limit value will be quickly called back Quota to the limit value;\n2. If the whole machine CPU load is lower than the alarm water mark, all pods with quotaTurbo enabled and whose CPU is suppressed during the current cycle will be slowly increased Quota above the configured limit value;\n3. If the whole machine CPU load is higher than the high water mark, and the pod with quotaTurbo enabled during the current cycle has not been suppressed, the pod will be slowly called back to the limit value',

  // DynMemory
  'config.dynMemory.title': 'Memory Asynchronous Tiered Reclaim',
  'config.dynMemory.tip':
    'Based on the kernel cgroup dynamic water-level fast-suppression and slow-recovery strategy, when the whole machine memory usage is high, it asynchronously reclaims offline container memory in the background without blocking online business; when the whole machine memory usage is low, it slowly increases offline container memory',

  // DynCache
  'config.dynCache.title': 'Memory Access Bandwidth and L3 Cache Limit',
  'config.dynCache.tip':
    'Reduce the interference of offline business on online business by limiting the use of memory access bandwidth L3Cache by offline business',
  'config.dynCache.memoryBandWatermark': 'Memory Access Bandwidth Water Mark',
  'config.dynCache.memoryBandTip':
    'Represents different degrees of limitation on offline business memory access bandwidth, for example, when low: 20%, medium: 30%, high: 50%, it means the colocation engine will dynamically limit offline Pods to only use 20%, 30%, 50% of memory access bandwidth based on the cache miss metrics of actual online business Pods',
  'config.dynCache.low': 'Low',
  'config.dynCache.mid': 'Medium',
  'config.dynCache.high': 'High',
  'config.dynCache.l3CacheWatermark': 'L3Cache Watermark',
  'config.dynCache.l3CacheTip':
    'Represents different degrees of limitation on offline business L3Cache, for example, when low: 20%, medium: 30%, high: 50%, it means the colocation engine will dynamically limit offline Pods to only use 20%, 30%, 50% of L3 Cache based on the L3 miss metrics of actual online business Pods',

  // PSI
  'config.psi.title': 'PSI-based Pod Eviction',
  'config.psi.tip':
    'By monitoring the PSI metrics of online Pods (CPU, memory, IO some avg10 metrics, representing the average blocking time ratio of any online Pod within 10 seconds), when any interference metric reaches the threshold, offline Pods are evicted according to the policy',
  'config.psi.resourceLabel': 'Detection Resource Type',
  'config.psi.cpu': 'CPU',
  'config.psi.memory': 'Memory',
  'config.psi.io': 'IO',
  'config.psi.threshold': 'Interference Eviction Threshold',
  'config.psi.thresholdTip':
    'Users can choose to monitor CPU, memory, and IO resources and set corresponding thresholds. If the blocking ratio exceeds the threshold, rubik will evict offline Pods according to a certain policy and release corresponding resources. If online Pods have high CPU and memory utilization, rubik will evict the offline workload that currently consumes the most CPU/memory resources. If offline workloads have high I/O, rubik will choose to evict the offline workload that consumes the most CPU resources',

  // Input placeholders
  'placeholder.input0to100': 'Please enter 0~100',
  'placeholder.input5to100': 'Please enter 5.0~100.0',

  // Overview page
  'overview.title': 'Colocation',
  'overview.clusterMonitor': 'Cluster Monitor',
  'overview.cpuUsage': 'CPU Usage',
  'overview.memoryUsage': 'Memory Usage',
  'overview.onlineWorkload': 'Online Workload',
  'overview.offlineWorkload': 'Offline Workload',
  'overview.nodeDistribution': 'Node Distribution',
  'overview.totalNodes': 'Total Nodes',
  'overview.colocationNodes': 'Colocation Nodes',
  'overview.nonColocationNodes': 'Non-Colocation Nodes',
  'overview.workloadDistribution': 'Workload Distribution',
  'overview.total': 'Total',
  'overview.unit': '',
  'overview.resourceManagement': 'Resource Management',
  'overview.workloadManagement': 'Workload Management',
  'overview.resourceAllocation': 'Resource Allocation',
  'overview.resourceRequest': 'Resource Request',
  'overview.resourceLimit': 'Resource Limit',
  'overview.usageInstructions': 'Usage Instructions',
  'overview.step1Title': 'View Colocation Data',
  'overview.step1Description': 'View cluster colocation and oversold node related data and metrics',
  'overview.step2Title': 'Configure Colocation',
  'overview.step2Description': 'Configure cluster colocation and oversold nodes',
  'overview.step3Title': 'Configure Colocation Rules',
  'overview.step3Description': 'Configure cluster colocation and oversold auto-matching rules',
  'overview.step4Title': 'Deploy Workloads',
  'overview.step4Description': 'Newly deployed workloads matching rules will be set with specific attribute',

  'workload.management': 'Workload Management',

  // Rule page
  'rule.title': 'Colocation Rules',
  'rule.create': 'Create Rule',
  'rule.edit': 'Edit Rule',
  'rule.delete': 'Delete Rule',
  'rule.name': 'Rule Name',
  'rule.type': 'Rule Type',
  'rule.description': 'Rule Description',
  'rule.action': 'Operation',
  'rule.priority': 'Priority',
  'rule.label': 'Label',
  'rule.value': 'Value',
  'rule.operator': 'Operator',
  'rule.match': 'Match',
  'rule.exact': 'Exact',
  'rule.detail': 'Rule Detail',
  'rule.labelSelector': 'Label Selector',
  'rule.colocationType': 'Colocation Type',
  'rule.online': 'Online',
  'rule.offline': 'Offline',
  'rule.oversold': 'Oversold',
  'rule.confirmDelete': 'Are you sure you want to delete this rule?',
  'rule.deleteSuccess': 'Rule deleted successfully',
  'rule.deleteFailed': 'Failed to delete rule',
  'rule.createSuccess': 'Rule created successfully',
  'rule.createFailed': 'Failed to create rule',
  'rule.updateSuccess': 'Rule updated successfully',
  'rule.updateFailed': 'Failed to update rule',
  'rule.matchCount': 'Matching Rule Count',
  'rule.typeMismatchCannotEdit': 'Cannot edit due to type mismatch',
  'rule.typeMismatchCannotView': 'Cannot view due to type mismatch',
  'rule.maxLimitReached': 'Rule limit is 100',
  'rule.searchName': 'Search rule name',
  'rule.tipMessage':
    'After setting rules, newly deployed workloads matching the rules will automatically have the relevant colocation attribute.',
  'rule.labelSelectorMode': 'Label Selector Mode',
  'rule.exactMatchMode': 'Exact Match Mode',
  'rule.matchingRules': 'Matching Rules',
  'rule.deleteTitle': 'Delete Colocation Rule',
  'rule.confirmDeleteByName': 'Are you sure you want to delete rule {name}?',
  'rule.onlineRule': 'Online Rules',
  'rule.offlineRule': 'Offline Rules',
  'rule.oversoldRule': 'Oversold Rules',
  'rule.management': 'Rule Management',
  'rule.maxTenItems': 'Maximum 10 items allowed, cannot add more when exceeding 10',
  'rule.exactMatch': 'Exact Match',
  'rule.namespaceRequired': 'Namespace is required',
  'rule.workloadTypeRequired': 'Workload type is required',
  'rule.workloadNameRequired': 'Workload name is required',
  'rule.keyRequired': 'Key is required',
  'rule.key': 'Key',
  'rule.valueRequired': 'Value is required',
  'rule.addLabel': 'Add Label',

  // Node management
  'node.title': 'Colocation Node Management',
  'node.name': 'Node Name',
  'node.status': 'Node Status',
  'node.ipAddress': 'IP Address',
  'node.osVersion': 'OS Version',
  'node.colocationStatus': 'Colocation Status',
  'node.enable': 'Enable',
  'node.disable': 'Disable',
  'node.enabled': 'Enabled',
  'node.disabled': 'Disabled',
  'node.ready': 'Ready',
  'node.notReady': 'NotReady',
  'node.status.normal': 'Normal',
  'node.status.abnormal': 'Abnormal',
  'node.setOversold': 'Set as Oversold',
  'node.setNonColocation': 'Set as Non-Colocation Node',
  'node.setSuccess': 'Set successfully',
  'node.setFailed': 'Failed to set',
  'node.addSuccess': 'Added successfully',
  'node.addFailed': 'Failed to add',
  'node.management': 'Node Management',
  'node.tipMessage':
    'Colocation nodes can only deploy offline colocation workloads. Oversold nodes support oversold node resources. Oversold nodes must be colocation nodes.',
  'node.searchName': 'Search node name',
  'node.addColocation': 'Add Colocation Node',
  'node.confirmDelete': 'Are you sure you want to set node {name} as a non-colocation node?',

  // Charts
  'chart.noData': 'No Data',
  'chart.time': 'Time',
  'chart.value': 'Value',
  'chart.percentage': 'Percentage',

  // Menu
  'menu.overview': 'Overview',
  'menu.workloadManagement': 'Workload Management',
  'menu.nodeManagement': 'Node Management',
  'menu.ruleManagement': 'Rule Management',
  'menu.configManagement': 'Config Management',
  'menu.monitor': 'Colocation Monitoring',

  // Navbar
  'navbar.title': 'Colocation',
  'navbar.cluster': 'Cluster',
  'navbar.logout': 'Logout',

  // Table modal
  'tableModal.title': 'Table Settings',
  'tableModal.columnDisplay': 'Column Display',
  'tableModal.selectAll': 'Select All',
  'tableModal.reset': 'Reset',

  // Empty data
  'empty.noData': 'No Data',

  // Monitor pages
  'monitor.clusterLevelColocationMonitor': 'Cluster-Level Colocation Monitoring',
  'monitor.nodeLevelColocationMonitor': 'Node-Level Colocation Monitoring',
  'monitor.colocation': 'Colocation',
  'monitor.monitor': 'Colocation Monitoring',
  'monitor.normal': 'Normal',
  'monitor.colocationNodeInfo': 'Colocation Node Info',
  'monitor.totalNodes': 'Total Nodes',
  'monitor.colocationLoadInfo': 'Colocation Load Info',
  'monitor.totalPods': 'Total Number of Pods',
  'monitor.clusterCpuUsage': 'Cluster CPU Resource Actual Usage (Core)',
  'monitor.clusterMemoryUsage': 'Cluster Memory Resource Actual Usage (MiB)',
  'monitor.clusterOversoldCpuRequest': 'Cluster Oversold CPU Request (Core)',
  'monitor.clusterOversoldMemoryRequest': 'Cluster Oversold Memory Request (MiB)',
  'monitor.clusterNormalCpuRequest': 'Cluster General CPU Request (Core)',
  'monitor.clusterNormalMemoryRequest': 'Cluster General Memory Request (MiB)',
  'monitor.nonColocationPods': 'Non-Colocation Pod Count',
  'monitor.hlsPods': 'HLS Level Pod Count',
  'monitor.lsPods': 'LS Level Pod Count',
  'monitor.bePods': 'BE Level Pod Count',
  'monitor.nodeCpuActualUsage': 'Node CPU Resource Actual Usage (Core)',
  'monitor.nodeMemoryActualUsage': 'Node Memory Resource Actual Usage (MiB)',
  'monitor.nodeOversoldCpuRequest': 'Node Oversold CPU Request (Core)',
  'monitor.nodeOversoldMemoryRequest': 'Node Oversold Memory Request (MiB)',
  'monitor.nodeNormalCpuRequest': 'Node General CPU Request (Core)',
  'monitor.nodeNormalMemoryRequest': 'Node General Memory Request (MiB)',

  // Overview page additional
  'overview.envPreparation': 'Environment Preparation',
  'overview.envPreparationDesc': 'Modify kubelet and containerd configuration to enable colocation feature on nodes.',
  'overview.colocationStrategyConfig': 'Colocation Policy Configuration',
  'overview.colocationStrategyConfigDesc':
    'Configure colocation nodes to ensure corresponding nodes enable colocation feature; configure colocation QoS guarantee, scheduling, and eviction related parameters.',
  'overview.configureColocationStrategy': 'Configure Colocation Strategy >',
  'overview.workloadDeployment': 'Workload Deployment',
  'overview.workloadDeploymentDesc': 'Deploy workloads in resource management and use colocation capability.',
  'overview.deployWorkload': 'Deploy Workload >',
  'overview.colocationMonitor': 'Colocation Monitoring',
  'overview.colocationMonitorDesc':
    'View the number of colocation nodes and pods in the cluster; as well as the usage of normal CPU/memory resources and oversold CPU/memory resources.',
  'overview.viewColocationMonitor': 'View Colocation Monitor >',
  'overview.colocation': 'Colocation',
  'overview.overview': 'Overview',
  'overview.colocationDesc':
    'Supports hybrid deployment of online/offline workloads, guarantees scheduling of online workloads and suppression of offline workloads during peak usage, while enabling offline workloads to use oversold resources during off-peak periods to improve cluster resource utilization.',
  'overview.normal': 'Normal',
  'overview.normalNode': 'Normal Node',
  'overview.colocationNode': 'Colocation Node',
  'overview.oversoldNode': 'Oversold Node',
  'overview.nodes': 'Nodes',
  'overview.selectNode': 'Please select node',
  'overview.resourceUsage': 'Resource Usage',
  'overview.cpuUsagePercent': 'CPU Usage(%)',
  'overview.memoryUsagePercent': 'Memory Usage(%)',
  'overview.legend': 'Legend',
  'overview.resourceStatistics': 'Resource Statistics',
  'overview.oversoldAvailable': 'Oversold Available',
  'overview.oversoldUsed': 'Oversold Used',
  'overview.oversoldAgentDeploying': 'Oversold Agent Deploying',
  'overview.oversoldCpuTop5': 'Oversold CPU Top5',
  'overview.oversoldMemoryTop5': 'Oversold Memory Top5',

  // Extra prompts
  'message.understandRisk': 'I understand this operation may cause some services to be temporarily unavailable',
  'message.caution': 'Please proceed with caution',
  'message.understandRiskForce':
    'I understand this operation may cause some services to be temporarily unavailable, force execute',

  // Monitor page - node info
  'monitor.nodesWithColocationEnabled': 'Nodes with Colocation Enabled',
  'monitor.nodesWithoutColocationEnabled': 'Nodes without Colocation Enabled',

  // Monitor metric names
  'metric.physicalResourceTotal': 'Total Physical Resources',
  'metric.nonColocationPodUsage': 'Non-Colocation Pod Usage',
  'metric.hlsPodUsage': 'HLS-Level Pod Usage',
  'metric.lsPodUsage': 'LS-Level Pod Usage',
  'metric.bePodUsage': 'BE-Level Pod Usage',
  'metric.oversoldResourceTotal': 'Total Oversold Resources',
  'metric.oversoldResourceRequested': 'Oversold Resources Requested',
  'metric.hlsPodResourceRequest': 'HLS-Level Pod Resource Request',
  'metric.lsPodResourceRequest': 'LS-Level Pod Resource Request',
  'metric.nonColocationPodResourceRequest': 'Non-Colocation Pod Resource Request',
  'metric.generalResourceTotal': 'Total General Resources',

  // Permission error messages
  'message.operationFailedNoPermission':
    'Operation failed, current user does not have permission. Please contact the administrator to add permissions!',
  'message.requestRejectedByServer': 'Request rejected by the server. Please consider modifying the request!',
};
