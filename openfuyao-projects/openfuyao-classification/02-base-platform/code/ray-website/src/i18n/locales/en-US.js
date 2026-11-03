export default {
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
  'common.pleaseEnter': 'Please enter',
  'common.type': 'Type',
  'common.creator': 'Creator',
  'common.createTime': 'Created At',
  'common.updateTime': 'Updated At',
  'common.start': 'Start',
  'common.stop': 'Stop',
  'common.total': 'Total {total} items',
  'common.confirmDelete': 'Confirm delete?',
  'common.save': 'Save',
  'common.back': 'Back',
  'common.view': 'View',
  'common.inputRequired': 'Please enter!',
  'common.enterName': 'Enter name',
  'common.enterValue': 'Enter value',

  // Shared table column titles
  'table.nodeType': 'Node Type',
  'table.nodeConfigName': 'Node Config Name',
  'table.npuCard': 'NPU (Card)',
  'table.cpuCore': 'CPU (Core)',
  'table.memoryG': 'Memory (G)',
  'table.memory': 'Memory',
  'table.maxReplicas': 'Max Replicas',
  'table.minReplicas': 'Min Replicas',
  'table.npuType': 'NPU Type',
  'table.shouldBeNumber': '{field} should be a number!',

  // Ray cluster
  'ray.cluster': 'RayCluster',
  'ray.cluster.name': 'Cluster Name',
  'ray.cluster.ip': 'Cluster IP',
  'ray.cluster.ipAddress': 'Cluster IP Address',
  'ray.cluster.addConfig': 'Create Config',
  'ray.cluster.uploadConfig': 'Upload Config',
  'ray.cluster.rayVersion': 'Ray Version',
  'ray.cluster.gcsPort': 'GCS Port',
  'ray.cluster.dashboardPort': 'Dashboard Port',
  'ray.cluster.create': 'Create Cluster',
  'ray.cluster.createAndStart': 'Create and Start',
  'ray.cluster.createTemplate': 'Create Template',
  'ray.cluster.configTitle': 'Ray Cluster Config',
  'ray.cluster.headConfig': 'Head Node Config',
  'ray.cluster.workerConfig': 'Worker Node Config',
  'ray.cluster.allowHeadWorkload': 'Allow head node to process workload',
  'ray.cluster.allowHeadWorkloadShort': 'Allow head node to process workload',
  'ray.cluster.headNote':
    "Note: CPU and Memory are required and must be numbers > 0. NPU card and NPU type must both be empty or both filled with a value other than '--'",
  'ray.cluster.workerNote':
    "Note: Node config name, CPU, and memory are required and must be numbers > 0. NPU card and NPU type must both be empty or both filled with a value other than '--'",
  'ray.cluster.addNodeConfig': 'Add Node Config',
  'ray.cluster.deleteTitle': 'Delete RayCluster',
  'ray.cluster.deleteWarning': 'Deleted RayCluster cannot be recovered. Please proceed with caution.',
  'ray.cluster.deleteConfirm': 'Are you sure to delete {name}{extra} under RayCluster?',
  'ray.cluster.listError': 'Failed to get RayCluster list',
  'ray.cluster.searchPlaceholder': 'Enter name',
  'ray.cluster.createBtn': 'Create',
  'ray.cluster.startSuccess':
    '{name} started successfully. If resource status has not changed, please refresh manually!',
  'ray.cluster.startFail': '{name} failed to start',
  'ray.cluster.stopSuccess':
    '{name} stopped successfully. If resource status has not changed, please refresh manually!',
  'ray.cluster.stopFail': '{name} failed to stop',
  'ray.cluster.deleteSuccess':
    '{name} deleted successfully. If resource status has not changed, please refresh manually!',
  'ray.cluster.deleteFail': '{name} failed to delete',

  // Ray job
  'ray.job': 'RayJob',
  'ray.job.name': 'Job Name',
  'ray.job.id': 'Job ID',
  'ray.job.clusterName': 'Cluster Name',
  'ray.job.basicInfo': 'Basic Info',
  'ray.job.programEntry': 'Program Entry',
  'ray.job.submitMode': 'Job Submit Mode',
  'ray.job.entryCpuCount': 'Program Entry CPU Count',
  'ray.job.entryResource': 'Program Entry Resources',
  'ray.job.addResource': 'Add Resource Config',
  'ray.job.rayClusterConfig': 'Ray Cluster Config',
  'ray.job.headConfig': 'Head Node Config',
  'ray.job.workerConfig': 'Worker Node Config',
  'ray.job.allowHeadWorkload': 'Allow head node to process workload',
  'ray.job.headNote':
    "Note: CPU and memory are required and must be numbers > 0. NPU card and NPU type must both be empty or both filled with a value other than '--'",
  'ray.job.workerNote':
    "Note: Node config name, CPU, and memory are required and must be numbers > 0. NPU card and NPU type must both be empty or both filled with a value other than '--'",
  'ray.job.addNodeConfig': 'Add Node Config',
  'ray.job.create': 'Create Job',
  'ray.job.createAndStart': 'Create and Start',
  'ray.job.createTemplate': 'Create Template',
  'ray.job.formError': 'Form has errors, please re-fill!',
  'ray.job.createSuccess': 'Created successfully!',
  'ray.job.createFail': 'Failed to create',
  'ray.job.listError': 'Failed to get RayJob list',
  'ray.job.deleteTitle': 'Delete RayJob',
  'ray.job.deleteWarning': 'Deleted RayJob cannot be recovered. Please proceed with caution.',
  'ray.job.deleteConfirm': 'Are you sure to delete {name}{extra} under RayJob?',
  'ray.job.startSuccess': '{name} started successfully. If resource status has not changed, please refresh manually!',
  'ray.job.startFail': '{name} failed to start',
  'ray.job.stopSuccess': '{name} stopped successfully. If resource status has not changed, please refresh manually!',
  'ray.job.stopFail': '{name} failed to stop',
  'ray.job.deleteSuccess': '{name} deleted successfully. If resource status has not changed, please refresh manually!',
  'ray.job.deleteFail': '{name} failed to delete',
  'ray.job.detailError': 'Failed to get {name} details',
  'ray.job.addConfig': 'Create Config',
  'ray.job.uploadConfig': 'Upload Config',
  'ray.job.clusterIpAddress': 'Cluster IP Address',
  'ray.job.searchPlaceholder': 'Enter name',
  'ray.job.createBtn': 'Create',
  'ray.job.deleteTitle': 'Delete RayJob',
  'ray.job.deleteWarning': 'Deleted RayJob cannot be recovered. Please proceed with caution.',
  'ray.job.deleteConfirm': 'Are you sure to delete {name}{extra} under RayJob?',
  'ray.job.listError': 'Failed to get RayJob list',
  'ray.job.entrypointMode': 'Job Submit Mode:',
  'ray.job.entrypointLabel': 'Program Entry:',
  'ray.job.entrypointCpu': 'Program Entry CPU Count:',

  // Ray service
  'ray.service': 'RayService',
  'ray.service.name': 'Service Name',
  'ray.service.basicInfo': 'Basic Info',
  'ray.service.rayClusterConfig': 'Ray Cluster Config',
  'ray.service.headConfig': 'Head Node Config',
  'ray.service.workerConfig': 'Worker Node Config',
  'ray.service.allowHeadWorkload': 'Allow head node to process workload',
  'ray.service.headNote':
    "Note: CPU and memory are required and must be numbers > 0. NPU card and NPU type must both be empty or both filled with a value other than '--'",
  'ray.service.workerNote':
    "Note: Node config name, CPU, and memory are required and must be numbers > 0. NPU card and NPU type must both be empty or both filled with a value other than '--'",
  'ray.service.addNodeConfig': 'Add Node Config',
  'ray.service.create': 'Create Service',
  'ray.service.createAndStart': 'Create and Start',
  'ray.service.createTemplate': 'Create Template',
  'ray.service.serveConfig': 'Serve Config',
  'ray.service.serviceName': 'Ray Service Name',
  'ray.service.enterServiceName': 'Please enter service name!',
  'ray.service.copy': 'Copy',
  'ray.service.copySuccess': 'Copied successfully!',
  'ray.service.enterServeConfig': 'Please fill in serve config!',
  'ray.service.formError': 'Form has errors, please re-fill!',
  'ray.service.createSuccess': 'Created successfully!',
  'ray.service.createFail': 'Failed to create',
  'ray.service.listError': 'Failed to get RayService list',
  'ray.service.deleteTitle': 'Delete RayService',
  'ray.service.deleteWarning': 'Deleted RayService cannot be recovered. Please proceed with caution.',
  'ray.service.deleteConfirm': 'Are you sure to delete {name}{extra} under RayService?',
  'ray.service.startSuccess':
    '{name} started successfully. If resource status has not changed, please refresh manually!',
  'ray.service.startFail': '{name} failed to start',
  'ray.service.stopSuccess':
    '{name} stopped successfully. If resource status has not changed, please refresh manually!',
  'ray.service.stopFail': '{name} failed to stop',
  'ray.service.deleteSuccess':
    '{name} deleted successfully. If resource status has not changed, please refresh manually!',
  'ray.service.deleteFail': '{name} failed to delete',
  'ray.service.detailError': 'Failed to get {name} details',
  'ray.service.addConfig': 'Create Config',
  'ray.service.uploadConfig': 'Upload Config',
  'ray.service.searchPlaceholder': 'Enter name',
  'ray.service.createBtn': 'Create',
  'ray.service.listError': 'Failed to get RayService list',

  // Detail pages shared
  'detail.monitorGoTo': 'For monitoring details, please go to',
  'detail.viewAt': 'to view',
  'detail.basicInfo': 'Basic Info',
  'detail.algorithmFramework': 'Algorithm Framework',
  'detail.imageName': 'Image Name:',
  'detail.imageVersion': 'Image Version:',
  'detail.envVar': 'Environment Variables',
  'detail.entryResource': 'Program Entry Resources',
  'detail.headNodeConfig': 'Head Node Config',
  'detail.workerNodeConfig': 'Worker Node Config',

  // Image fields
  'image.name': 'Image Name',
  'image.version': 'Image Version',
  'image.enterName': 'Please enter image name!',
  'image.enterVersion': 'Please enter image version!',
  'image.enterNameFirst': 'Please enter first',
  'image.enterVersionFirst': 'Please enter image name first',

  // Environment variables
  'env.title': 'Environment Variables',
  'env.add': 'Add Environment Variable',
  'env.enterName': 'Enter name',
  'env.enterValue': 'Enter value',

  // Volumes
  'volumes.title': 'volumes',
  'volumeMounts.title': 'volumeMounts',

  // Status labels
  'status.unknown': 'Unknown',
  'status.running': 'Running',
  'status.notRunning': 'Not Running',
  'status.failed': 'Failed',
  'status.unhealthy': 'Unhealthy',
  'status.restarting': 'Restarting',
  'status.deploying': 'Deploying',
  'status.initializing': 'Initializing',
  'status.paused': 'Paused',
  'status.retrying': 'Retrying',
  'status.completed': 'Completed',

  // Filter labels
  'filter.instance': 'Instance',
  'filter.template': 'Template',

  // Time period options
  'time.last10min': 'Last 10 min',
  'time.last30min': 'Last 30 min',
  'time.last1h': 'Last 1 hour',
  'time.last3h': 'Last 3 hours',
  'time.last6h': 'Last 6 hours',
  'time.last1d': 'Last 1 day',
  'time.last3d': 'Last 3 days',
  'time.last7d': 'Last 7 days',
  'time.last14d': 'Last 14 days',

  // Chart types
  'chart.managedCpu': 'Managed CPU',
  'chart.cpuUsage': 'CPU Usage',
  'chart.managedMemory': 'Managed Memory',
  'chart.memoryUsage': 'Memory Usage',
  'chart.managedGpu': 'Managed GPU',
  'chart.gpuUsage': 'GPU Usage',
  'chart.gramUsage': 'GRAM Usage',
  'chart.managedNpu': 'Managed NPU',
  'chart.npuUsage': 'NPU Usage',
  'chart.hbmUsage': 'HBM Usage',

  // Loki metric types
  'loki.gcsStateTotal': 'GCS Event Pressure (state total)',
  'loki.gcsStateActive': 'GCS Event Pressure (state active)',
  'loki.gcsQueueTime': 'GCS Event Pressure (queue time)',
  'loki.gcsHealthyCheck': 'Node Health Check (GCS)',
  'loki.gcsRayLet': 'Node Health Check (Raylet)',
  'loki.jobGet': 'Job Event Pressure (Job Get)',
  'loki.jobPost': 'Job Event Pressure (Job Post)',
  'loki.jobDelete': 'Job Event Pressure (Job Delete)',
  'loki.jobLogGet': 'Job Event Pressure (Log Get)',
  'loki.jobLogDelete': 'Job Event Pressure (Log Delete)',
  'loki.dashboardNode': 'Dashboard Pressure',
  'loki.dashboardLog': 'Dashboard Pressure',
  'loki.dashboardJob': 'Dashboard Pressure',

  // Upload component
  'upload.title': 'Upload Config',
  'upload.cancelText': 'Cancel',
  'upload.confirmText': 'Upload and Deploy',
  'upload.selectFile': 'Select File',
  'upload.downloadTemplate': 'Download {type} Template',
  'upload.sizeLimit': 'Max file size 1M, supports yaml format.',
  'upload.yamlContent': 'yaml config content',
  'upload.uploadSuccess': 'Uploaded successfully!',
  'upload.uploadFail': 'Failed to upload',

  // AddYaml component
  'yaml.addTitle': 'Add yaml Config',
  'yaml.editTitle': 'Edit yaml Config',
  'yaml.name': 'Config Name',
  'yaml.content': 'yaml Config Content',
  'yaml.nameRequired': 'Please enter a name!',
  'yaml.contentRequired': 'Please enter yaml content!',
  'yaml.createSuccess': 'Created successfully!',
  'yaml.createFail': 'Failed to create',

  // DeleteInfoModal component
  'delete.confirmCheck': 'I acknowledge the above impact and confirm deletion',
  'delete.multipleItems': ' and multiple items',

  // QuickStart component
  'quickstart.title': 'Quick Start',
  'quickstart.step1': '1. Create Cluster',
  'quickstart.step2': '2. Submit Job',
  'quickstart.step3': '3. Deploy Service',
  'quickstart.step1Desc': 'First create a Ray cluster and configure Head and Worker node specs.',
  'quickstart.step2Desc': 'Submit a Ray Job on the cluster.',
  'quickstart.step3Desc': 'Deploy a Ray Serve service to provide inference capabilities.',
  'quickstart.learnMore': 'Learn More',

  // NavBar
  'nav.overview': 'Overview',

  // Empty area
  'empty.noData': 'No Data',

  // Overview
  'overview.title': 'openFuyao Ray Overview',
  'overview.timeRange': 'Time Range',
  'overview.clusterOverview': 'Cluster Overview',
  'overview.apiOverview': 'API Overview',

  // DataDisplayList
  'dataDisplay.key': 'Key',
  'dataDisplay.value': 'Value',

  // Log
  'log.title': 'Logs',

  // Yaml view
  'yaml.viewTitle': 'Yaml Config',

  // Detail common
  'detail.noData': 'None',
  'detail.none': '--',

  // Log pages
  'log.info': 'Log Info',
  'log.export': 'Export',
  'log.noLog': 'No logs available',
  'log.exportSuccess': 'Exported successfully',
  'log.detailError': 'Failed to get {name} details',

  // Yaml pages
  'yaml.title': 'Yaml Config',
  'yaml.export': 'Export',
  'yaml.exportSuccess': 'Exported successfully',

  // AddConfig pages
  'config.createTitle': 'Create Config',
  'config.name': 'Config Name',
  'config.content': 'Config Content',
  'config.nameRequired': 'Please enter a name!',
  'config.contentRequired': 'Please enter config content!',
  'config.createSuccess': 'Created successfully!',
  'config.createFail': 'Failed to create',

  // AddYaml pages
  'yaml.createTitle': 'Create yaml Config',

  // Detail wrapper pages
  'detail.tab.detail': 'Detail',
  'detail.tab.log': 'Logs',
  'detail.tab.yaml': 'Yaml',
  'detail.tab.monitor': 'Monitor',

  // AddConfig mode
  'config.formCreate': 'Form Create',
  'config.yamlCreate': 'YAML Create',

  // Detail wrapper
  'detail.operation': 'Actions',
  'detail.title': 'Detail',
  'detail.clusterHealth': 'Cluster Health',

  // AddYaml success/fail
  'yaml.createSuccessMsg': '{type} resource YAML created successfully!',
  'yaml.createFailMsg': '{type} resource YAML failed to create',

  // Monitor
  'monitor.title': 'Monitor',

  // Monitor sub-sections
  'monitor.clusterHealth': 'Cluster Health',
  'monitor.gcsPressure': 'GCS Event Pressure',
  'monitor.gcsStateTotalDesc': 'GCS Pending Events Total',
  'monitor.gcsStateActiveDesc': 'GCS Active Events Count',
  'monitor.gcsQueueTimeDesc': 'GCS Queue Time',
  'monitor.nodeHealth': 'Node Health Check',
  'monitor.gcsHealthCheckDesc': 'GCS Health Check Duration',
  'monitor.rayletHealthCheckDesc': 'Raylet Health Check Duration',
  'monitor.jobPressure': 'Job Event Pressure',
  'monitor.jobGetDesc': 'Job Get API Duration',
  'monitor.jobPostDesc': 'Job Post API Duration',
  'monitor.jobDeleteDesc': 'Job Delete API Duration',
  'monitor.jobLogGetDesc': 'Job Log Get API Duration',
  'monitor.jobLogDeleteDesc': 'Job Log Delete API Duration',
  'monitor.dashboardPressure': 'Dashboard Pressure',
  'monitor.dashboardDesc': 'Dashboard API Duration',
  'monitor.configMapError':
    'This feature requires creating the ConfigMap resource fluentbit-config, and the ConfigMap and raycluster resources must be in the same namespace. Please refer to the user guide for configuration.',
  'monitor.metricFail': 'Failed to get {name} metric!',

  // Charts
  'chart.clusterMonitor': 'Cluster Monitor',
  'chart.cpuUsageTooltip':
    '"CPU Usage" represents the ratio of current CPU usage to managed CPU, not the entire server CPU.',
  'chart.memoryUsageTooltip':
    '"Memory Usage" represents the ratio of current memory usage to managed memory, not the entire server memory.',
  'chart.gpuUsageTooltip':
    '"GPU Usage" represents the ratio of current GPU usage to managed GPU, not the entire server GPU.',
  'chart.npuUsageTooltip':
    '"NPU Usage" represents the ratio of current NPU usage to managed NPU, not the entire server NPU.',

  // Overview
  'overview.activeRayClusters': 'Active RayClusters',
  'overview.activeRayService': 'Active RayService',
  'overview.activeRayJobs': 'Active RayJobs',
  'overview.activeRayCluster': 'Active Ray Clusters',
  'overview.quickStart': 'Quick Start',
  'overview.welcome': 'Welcome to openFuyao Ray - Manage your Ray clusters and jobs',
  'overview.fetchDataFail': 'Failed to get overview data',

  // Upload component internal
  'upload.yamlParseSuccess': 'YAML file parsed successfully!',
  'upload.yamlParseFail': 'Failed to parse YAML file, please check YAML format!',
  'upload.downloadSuccess': 'Downloaded successfully',
  'upload.selectYamlFile': 'Please select a YAML file',
  'common.export': 'Export',
  'common.exportSuccess': 'Exported successfully',
  'common.copy': 'Copy',
  'common.copySuccess': 'Copied successfully!',
  'common.ok': 'OK',
  'common.close': 'Close',
  'common.legend': 'Legend',
  'delete.understandIrreversible': 'I understand this {action} action is irreversible',
  'quickstart.welcome': 'Welcome to openFuyao-Ray',
  'quickstart.whatIsRay': 'What is Ray',
  'quickstart.howToUseRay': 'How to use Ray',
  'quickstart.rayDescription':
    "Ray is an open-source unified framework for scaling ML/AI and Python applications. It provides a compute layer for parallel processing, so you don't need to be a distributed systems expert.",
  'quickstart.installInstruction': 'To run this example, install RayCore:',
  'quickstart.usageInstruction':
    'Import Ray and initialize it with ray.init(). Then decorate your function with @ray.remote to declare that you want to run this function remotely. Finally, invoke the function using .remote() instead of calling it normally.\nThis remote call produces a future, a Ray object reference, which you can then retrieve using ray.get().',
};
