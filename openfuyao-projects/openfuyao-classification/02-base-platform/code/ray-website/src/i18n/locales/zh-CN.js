export default {
  'common.loading': '加载中...',
  'common.confirm': '确认',
  'common.cancel': '取消',
  'common.delete': '删除',
  'common.create': '创建',
  'common.edit': '编辑',
  'common.search': '搜索',
  'common.refresh': '刷新',
  'common.name': '名称',
  'common.status': '状态',
  'common.action': '操作',
  'common.noData': '暂无数据',
  'common.pleaseEnter': '请输入',
  'common.type': '类型',
  'common.creator': '创建人',
  'common.createTime': '创建时间',
  'common.updateTime': '更新时间',
  'common.start': '启动',
  'common.stop': '终止',
  'common.total': '共{total}条',
  'common.confirmDelete': '确认删除?',
  'common.save': '保存',
  'common.back': '返回',
  'common.view': '查看',
  'common.inputRequired': '请输入!',
  'common.enterName': '请输入名称',
  'common.enterValue': '请输入值',

  // Shared table column titles
  'table.nodeType': '节点类型',
  'table.nodeConfigName': '节点配置名称',
  'table.npuCard': 'NPU （卡）',
  'table.cpuCore': 'CPU (Core)',
  'table.memoryG': '内存 (G)',
  'table.memory': '内存',
  'table.maxReplicas': '最大副本数',
  'table.minReplicas': '最小副本数',
  'table.npuType': 'NPU类型',
  'table.shouldBeNumber': '{field}类型应为数字！',

  // Ray cluster
  'ray.cluster': 'Ray集群',
  'ray.cluster.name': '集群名称',
  'ray.cluster.ip': '集群IP',
  'ray.cluster.ipAddress': '集群IP地址',
  'ray.cluster.addConfig': '创建配置',
  'ray.cluster.uploadConfig': '上传配置',
  'ray.cluster.rayVersion': 'Ray版本',
  'ray.cluster.gcsPort': 'GCS端口',
  'ray.cluster.dashboardPort': 'Dashboard端口',
  'ray.cluster.create': '创建集群',
  'ray.cluster.createAndStart': '创建并启动',
  'ray.cluster.createTemplate': '创建模板',
  'ray.cluster.configTitle': 'Ray集群配置',
  'ray.cluster.headConfig': 'head节点规格配置',
  'ray.cluster.workerConfig': 'worker节点规格配置',
  'ray.cluster.allowHeadWorkload': '允许头节点处理工作负载',
  'ray.cluster.allowHeadWorkloadShort': '允许head节点处理工作负载',
  'ray.cluster.headNote':
    "注：CPU、内存为必填项且CPU、内存为大于0的数字，NPU卡和NPU类型必须同时不填或者同时填入且值不为'--'",
  'ray.cluster.workerNote':
    "注：节点配置名称、CPU、内存为必填项且CPU、内存为大于0的数字，NPU卡和NPU类型必须同时不填或者同时填入且值不为'--'",
  'ray.cluster.addNodeConfig': '添加节点配置',
  'ray.cluster.deleteTitle': '删除RayCluster',
  'ray.cluster.deleteWarning': '删除RayCluster后将无法恢复，请谨慎操作。',
  'ray.cluster.deleteConfirm': '确定删除RayCluster下{name}{extra} 吗？',
  'ray.cluster.listError': '获取RayCluster列表失败',
  'ray.cluster.searchPlaceholder': '请输入名称',
  'ray.cluster.createBtn': '创建',
  'ray.cluster.startSuccess': '{name}启动成功，如果资源状态仍未变化，请稍后手动刷新！',
  'ray.cluster.startFail': '{name}启动失败',
  'ray.cluster.stopSuccess': '{name}终止成功，如果资源状态仍未变化，请稍后手动刷新！',
  'ray.cluster.stopFail': '{name}终止失败',
  'ray.cluster.deleteSuccess': '{name}删除成功，如果资源状态仍未变化，请稍后手动刷新！',
  'ray.cluster.deleteFail': '{name}删除失败',

  // Ray job
  'ray.job': 'RayJob',
  'ray.job.name': 'Job名称',
  'ray.job.id': 'Job ID',
  'ray.job.clusterName': '集群名称',
  'ray.job.basicInfo': '基本信息',
  'ray.job.programEntry': '程序入口',
  'ray.job.submitMode': 'Job提交模式',
  'ray.job.entryCpuCount': '程序入口Cpu数量',
  'ray.job.entryResource': '程序入口资源',
  'ray.job.addResource': '添加资源配置',
  'ray.job.rayClusterConfig': 'Ray集群配置',
  'ray.job.headConfig': 'head节点规格配置',
  'ray.job.workerConfig': 'worker节点规格配置',
  'ray.job.allowHeadWorkload': '允许头节点处理工作负载',
  'ray.job.headNote':
    "注：CPU、内存为必填项且CPU、内存为大于0的数字，NPU卡和NPU类型必须同时不填或者同时填入且值不为'--'",
  'ray.job.workerNote':
    "注：节点配置名称、CPU、内存为必填项且CPU、内存为大于0的数字，NPU卡和NPU类型必须同时不填或者同时填入且值不为'--'",
  'ray.job.addNodeConfig': '添加节点配置',
  'ray.job.create': '创建Job',
  'ray.job.createAndStart': '创建并启动',
  'ray.job.createTemplate': '创建模板',
  'ray.job.formError': '表单格式有问题，请重新填写！',
  'ray.job.createSuccess': '创建成功！',
  'ray.job.createFail': '创建失败',
  'ray.job.listError': '获取RayJob列表失败',
  'ray.job.deleteTitle': '删除RayJob',
  'ray.job.deleteWarning': '删除RayJob后将无法恢复，请谨慎操作。',
  'ray.job.deleteConfirm': '确定删除RayJob下{name}{extra} 吗？',
  'ray.job.startSuccess': '{name}启动成功，如果资源状态仍未变化，请稍后手动刷新！',
  'ray.job.startFail': '{name}启动失败',
  'ray.job.stopSuccess': '{name}终止成功，如果资源状态仍未变化，请稍后手动刷新！',
  'ray.job.stopFail': '{name}终止失败',
  'ray.job.deleteSuccess': '{name}删除成功，如果资源状态仍未变化，请稍后手动刷新！',
  'ray.job.deleteFail': '{name}删除失败',
  'ray.job.detailError': '获取{name}详情失败',
  'ray.job.addConfig': '创建配置',
  'ray.job.uploadConfig': '上传配置',
  'ray.job.clusterIpAddress': '集群IP地址',
  'ray.job.searchPlaceholder': '请输入名称',
  'ray.job.createBtn': '创建',
  'ray.job.deleteTitle': '删除RayJob',
  'ray.job.deleteWarning': '删除RayJob后将无法恢复，请谨慎操作。',
  'ray.job.deleteConfirm': '确定删除RayJob下{name}{extra} 吗？',
  'ray.job.listError': '获取RayJob列表失败',
  'ray.job.entrypointMode': 'Job提交模式：',
  'ray.job.entrypointLabel': '程序入口：',
  'ray.job.entrypointCpu': '程序入口CPU数量：',

  // Ray service
  'ray.service': 'RayService',
  'ray.service.name': '服务名称',
  'ray.service.basicInfo': '基本信息',
  'ray.service.rayClusterConfig': 'Ray集群配置',
  'ray.service.headConfig': 'head节点规格配置',
  'ray.service.workerConfig': 'worker节点规格配置',
  'ray.service.allowHeadWorkload': '允许头节点处理工作负载',
  'ray.service.headNote':
    "注：CPU、内存为必填项且CPU、内存为大于0的数字，NPU卡和NPU类型必须同时不填或者同时填入且值不为'--'",
  'ray.service.workerNote':
    "注：节点配置名称、CPU、内存为必填项且CPU、内存为大于0的数字，NPU卡和NPU类型必须同时不填或者同时填入且值不为'--'",
  'ray.service.addNodeConfig': '添加节点配置',
  'ray.service.create': '创建服务',
  'ray.service.createAndStart': '创建并启动',
  'ray.service.createTemplate': '创建模板',
  'ray.service.serveConfig': 'Serve配置',
  'ray.service.serviceName': 'Ray服务名称',
  'ray.service.enterServiceName': '请输入服务名称!',
  'ray.service.copy': '复制',
  'ray.service.copySuccess': '复制成功！',
  'ray.service.enterServeConfig': '请填写serve配置！',
  'ray.service.formError': '表单格式有问题，请重新填写！',
  'ray.service.createSuccess': '创建成功！',
  'ray.service.createFail': '创建失败',
  'ray.service.listError': '获取RayService列表失败',
  'ray.service.deleteTitle': '删除RayService',
  'ray.service.deleteWarning': '删除RayService后将无法恢复，请谨慎操作。',
  'ray.service.deleteConfirm': '确定删除RayService下{name}{extra} 吗？',
  'ray.service.startSuccess': '{name}启动成功，如果资源状态仍未变化，请稍后手动刷新！',
  'ray.service.startFail': '{name}启动失败',
  'ray.service.stopSuccess': '{name}终止成功，如果资源状态仍未变化，请稍后手动刷新！',
  'ray.service.stopFail': '{name}终止失败',
  'ray.service.deleteSuccess': '{name}删除成功，如果资源状态仍未变化，请稍后手动刷新！',
  'ray.service.deleteFail': '{name}删除失败',
  'ray.service.detailError': '获取{name}详情失败',
  'ray.service.addConfig': '创建配置',
  'ray.service.uploadConfig': '上传配置',
  'ray.service.searchPlaceholder': '请输入名称',
  'ray.service.createBtn': '创建',
  'ray.service.listError': '获取RayService列表失败',

  // Detail pages shared
  'detail.monitorGoTo': '监控详情请前往',
  'detail.viewAt': '查看',
  'detail.basicInfo': '基本信息',
  'detail.algorithmFramework': '算法框架',
  'detail.imageName': '镜像名称：',
  'detail.imageVersion': '镜像版本：',
  'detail.envVar': '环境变量',
  'detail.entryResource': '程序入口资源',
  'detail.headNodeConfig': 'head节点规格配置',
  'detail.workerNodeConfig': 'worker节点规格配置',

  // Image fields
  'image.name': '镜像名称',
  'image.version': '镜像版本',
  'image.enterName': '请输入镜像名称!',
  'image.enterVersion': '请输入镜像版本!',
  'image.enterNameFirst': '请先输入',
  'image.enterVersionFirst': '请先输入镜像名称',

  // Environment variables
  'env.title': '环境变量',
  'env.add': '添加环境变量',
  'env.enterName': '请输入名称',
  'env.enterValue': '请输入值',

  // Volumes
  'volumes.title': 'volumes',
  'volumeMounts.title': 'volumeMounts',

  // Status labels
  'status.unknown': '未知',
  'status.running': '运行中',
  'status.notRunning': '未运行',
  'status.failed': '失败',
  'status.unhealthy': '不健康',
  'status.restarting': '重启中',
  'status.deploying': '服务部署中',
  'status.initializing': '初始化',
  'status.paused': '暂停中',
  'status.retrying': '重试中',
  'status.completed': '完成',

  // Filter labels
  'filter.instance': '实例',
  'filter.template': '模板',

  // Time period options
  'time.last10min': '近10分钟',
  'time.last30min': '近30分钟',
  'time.last1h': '近1小时',
  'time.last3h': '近3小时',
  'time.last6h': '近6小时',
  'time.last1d': '近1天',
  'time.last3d': '近3天',
  'time.last7d': '近7天',
  'time.last14d': '近14天',

  // Chart types
  'chart.managedCpu': '托管CPU',
  'chart.cpuUsage': 'CPU使用率',
  'chart.managedMemory': '托管内存',
  'chart.memoryUsage': '内存使用率',
  'chart.managedGpu': '托管GPU',
  'chart.gpuUsage': 'GPU使用率',
  'chart.gramUsage': 'GRAM使用率',
  'chart.managedNpu': '托管NPU',
  'chart.npuUsage': 'NPU使用率',
  'chart.hbmUsage': 'HBM使用率',

  // Loki metric types
  'loki.gcsStateTotal': 'GCS事件压力（state total）',
  'loki.gcsStateActive': 'GCS事件压力（state active）',
  'loki.gcsQueueTime': 'GCS事件压力（queue time）',
  'loki.gcsHealthyCheck': '节点健康检查（GCS）',
  'loki.gcsRayLet': '节点健康检查（Raylet）',
  'loki.jobGet': 'Job事件压力（Job Get）',
  'loki.jobPost': 'Job事件压力（Job Post）',
  'loki.jobDelete': 'Job事件压力（Job Delete）',
  'loki.jobLogGet': 'Job事件压力（Log Get）',
  'loki.jobLogDelete': 'Job事件压力（Log Delete）',
  'loki.dashboardNode': 'Dashboard压力',
  'loki.dashboardLog': 'Dashboard压力',
  'loki.dashboardJob': 'Dashboard压力',

  // Upload component
  'upload.title': '上传配置',
  'upload.cancelText': '取消',
  'upload.confirmText': '上传并部署',
  'upload.selectFile': '选择文件',
  'upload.downloadTemplate': '下载{type}模版',
  'upload.sizeLimit': '文件最大限制1M，支持上传yaml格式。',
  'upload.yamlContent': 'yaml配置内容',
  'upload.uploadSuccess': '上传成功！',
  'upload.uploadFail': '上传失败',

  // AddYaml component
  'yaml.addTitle': '添加yaml配置',
  'yaml.editTitle': '编辑yaml配置',
  'yaml.name': '配置名称',
  'yaml.content': 'yaml配置内容',
  'yaml.nameRequired': '请输入名称!',
  'yaml.contentRequired': '请输入yaml内容!',
  'yaml.createSuccess': '创建成功！',
  'yaml.createFail': '创建失败',

  // DeleteInfoModal component
  'delete.confirmCheck': '我已知晓上述影响并确认删除',
  'delete.multipleItems': '等多条数据',

  // QuickStart component
  'quickstart.title': '快速入门',
  'quickstart.step1': '1. 创建集群',
  'quickstart.step2': '2. 提交任务',
  'quickstart.step3': '3. 部署服务',
  'quickstart.step1Desc': '首先创建Ray集群，配置Head和Worker节点规格。',
  'quickstart.step2Desc': '在集群上提交Ray Job任务。',
  'quickstart.step3Desc': '部署Ray Serve服务对外提供推理能力。',
  'quickstart.learnMore': '了解更多',

  // NavBar
  'nav.overview': '总览',

  // Empty area
  'empty.noData': '暂无数据',

  // Overview
  'overview.title': 'openFuyao Ray 总览',
  'overview.timeRange': '时间范围',
  'overview.clusterOverview': '集群总览',
  'overview.apiOverview': 'API总览',

  // DataDisplayList
  'dataDisplay.key': '键',
  'dataDisplay.value': '值',

  // Log
  'log.title': '日志',

  // Yaml view
  'yaml.viewTitle': 'Yaml配置',

  // Detail common
  'detail.noData': '无',
  'detail.none': '--',

  // Log pages
  'log.info': '日志信息',
  'log.export': '导出',
  'log.noLog': '暂无日志',
  'log.exportSuccess': '导出成功',
  'log.detailError': '获取{name}详情失败',

  // Yaml pages
  'yaml.title': 'Yaml配置',
  'yaml.export': '导出',
  'yaml.exportSuccess': '导出成功',

  // AddConfig pages
  'config.createTitle': '创建配置',
  'config.name': '配置名称',
  'config.content': '配置内容',
  'config.nameRequired': '请输入名称!',
  'config.contentRequired': '请输入配置内容!',
  'config.createSuccess': '创建成功！',
  'config.createFail': '创建失败',

  // AddYaml pages
  'yaml.createTitle': '创建yaml配置',

  // Detail wrapper pages
  'detail.tab.detail': '详情',
  'detail.tab.log': '日志',
  'detail.tab.yaml': 'Yaml',
  'detail.tab.monitor': '监控',

  // AddConfig mode
  'config.formCreate': '表单创建',
  'config.yamlCreate': 'YAML创建',

  // Detail wrapper
  'detail.operation': '操作',
  'detail.title': '详情',
  'detail.clusterHealth': '集群健康信息',

  // AddYaml success/fail
  'yaml.createSuccessMsg': '{type}资源YAML创建成功！',
  'yaml.createFailMsg': '{type}资源YAML创建失败',

  // Monitor
  'monitor.title': '监控',

  // Monitor sub-sections
  'monitor.clusterHealth': '集群健康信息',
  'monitor.gcsPressure': 'GCS事件压力',
  'monitor.gcsStateTotalDesc': 'GCS待处理事件总数',
  'monitor.gcsStateActiveDesc': 'GCS正在处理事件数量',
  'monitor.gcsQueueTimeDesc': 'GCS时间排队用时',
  'monitor.nodeHealth': '节点健康检查',
  'monitor.gcsHealthCheckDesc': 'GCS健康检查用时',
  'monitor.rayletHealthCheckDesc': 'raylet健康检查用时',
  'monitor.jobPressure': 'Job事件压力',
  'monitor.jobGetDesc': '获取任务接口用时',
  'monitor.jobPostDesc': '提交任务接口用时',
  'monitor.jobDeleteDesc': '删除任务接口用时',
  'monitor.jobLogGetDesc': '获取任务日志接口用时',
  'monitor.jobLogDeleteDesc': '删除任务日志接口用时',
  'monitor.dashboardPressure': 'Dashboard压力',
  'monitor.dashboardDesc': 'dashboard接口用时',
  'monitor.configMapError':
    '该功能需要先创建相关ConfigMap资源fluentbit-config,且ConfigMap和raycluster资源在相同的namespace下，请参考用户指南进行配置',
  'monitor.metricFail': '获取{name}指标失败！',

  // Charts
  'chart.clusterMonitor': '集群监控',
  'chart.cpuUsageTooltip': '"CPU使用率"表示当前使用CPU占托管CPU的比率，而非整个服务器CPU的比率。',
  'chart.memoryUsageTooltip': '"内存使用率"表示当前使用内存占托管内存的比率，而非整个服务器内存的比率。',
  'chart.gpuUsageTooltip': '"GPU使用率"表示当前使用GPU占托管GPU的比率，而非整个服务器GPU的比率。',
  'chart.npuUsageTooltip': '"NPU使用率"表示当前使用NPU占托管NPU的比率，而非整个服务器NPU的比率。',

  // Overview
  'overview.activeRayClusters': '活跃RayClusters',
  'overview.activeRayService': '活跃RayService',
  'overview.activeRayJobs': '活跃RayJobs',
  'overview.activeRayCluster': '活跃Ray集群',
  'overview.quickStart': '快速开始',
  'overview.welcome': '欢迎使用openFuyao Ray管理您的Ray集群与作业',
  'overview.fetchDataFail': '获取概览数据失败',

  // Upload component internal
  'upload.yamlParseSuccess': 'YAML文件解析成功！',
  'upload.yamlParseFail': '解析YAML文件失败,请检查YAML格式！',
  'upload.downloadSuccess': '下载成功',
  'upload.selectYamlFile': '请选择YAML文件',
  'common.export': '导出',
  'common.exportSuccess': '导出成功',
  'common.copy': '复制',
  'common.copySuccess': '复制成功！',
  'common.ok': '确定',
  'common.close': '关闭',
  'common.legend': '图例',
  'delete.understandIrreversible': '已了解{action}操作无法恢复',
  'quickstart.welcome': '欢迎使用openFuyao-Ray',
  'quickstart.whatIsRay': '什么是Ray',
  'quickstart.howToUseRay': '如何使用Ray',
  'quickstart.rayDescription':
    'Ray是一个开源的统一框架，用于扩展机器学习AI和Python应用。它为并行处理提供了计算层，因此您无需成为分布式系统专家。',
  'quickstart.installInstruction': '要运行此实例，请安装RayCore：',
  'quickstart.usageInstruction':
    '导入Ray并使用ray.init()对其进行初始化。然后用@ray.remote装饰函数，声明你想远程运行这个函数。最后，使用.remote()调用函数，而不是正常调用它。\n这个远程调用会产生一个future，一个Ray对象引用，然后你可以使用ray.get来获取它。',
};
