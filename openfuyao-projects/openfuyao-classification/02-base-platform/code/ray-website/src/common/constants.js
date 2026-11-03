/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
export const DEFAULT_CURRENT_PAGE = 1;
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
export const containerRouterPrefix = 'ray';
export const containerStylePrefix = 'ray-website'; // 样式前缀
export const getFilterTypeList = (intl) => [
  {
    text: intl.formatMessage({ id: 'filter.instance' }),
    value: '实例',
  },
  {
    text: intl.formatMessage({ id: 'filter.template' }),
    value: '模板',
  },
];
export const getTimePeriodOptions = (intl) => [
  {
    label: intl.formatMessage({ id: 'time.last10min' }),
    value: '10m',
  },
  {
    label: intl.formatMessage({ id: 'time.last30min' }),
    value: '30m',
  },
  {
    label: intl.formatMessage({ id: 'time.last1h' }),
    value: '1h',
  },
  {
    label: intl.formatMessage({ id: 'time.last3h' }),
    value: '3h',
  },
  {
    label: intl.formatMessage({ id: 'time.last6h' }),
    value: '6h',
  },
  {
    label: intl.formatMessage({ id: 'time.last1d' }),
    value: '1d',
  },
  {
    label: intl.formatMessage({ id: 'time.last3d' }),
    value: '3d',
  },
  {
    label: intl.formatMessage({ id: 'time.last7d' }),
    value: '7d',
  },
  {
    label: intl.formatMessage({ id: 'time.last14d' }),
    value: '14d',
  },
];
export const getFilterClusterStatusList = (intl) => [
  {
    text: intl.formatMessage({ id: 'status.unknown' }),
    value: '未知',
  },
  {
    text: intl.formatMessage({ id: 'status.running' }),
    value: '运行中',
  },
  {
    text: intl.formatMessage({ id: 'status.notRunning' }),
    value: '未运行',
  },
  {
    text: intl.formatMessage({ id: 'status.failed' }),
    value: '失败',
  },
  {
    text: intl.formatMessage({ id: 'status.unhealthy' }),
    value: '不健康',
  },
];
export const getFilterServiceStatusList = (intl) => [
  {
    text: intl.formatMessage({ id: 'status.unknown' }),
    value: '未知',
  },
  {
    text: intl.formatMessage({ id: 'status.running' }),
    value: '运行中',
  },
  {
    text: intl.formatMessage({ id: 'status.notRunning' }),
    value: '未运行',
  },
  {
    text: intl.formatMessage({ id: 'status.restarting' }),
    value: '重启中',
  },
  {
    text: intl.formatMessage({ id: 'status.failed' }),
    value: '失败',
  },
  {
    text: intl.formatMessage({ id: 'status.deploying' }),
    value: '服务部署中',
  },
];
export const getFilterJobStatusList = (intl) => [
  {
    text: intl.formatMessage({ id: 'status.unknown' }),
    value: '未知',
  },
  {
    text: intl.formatMessage({ id: 'status.initializing' }),
    value: '初始化',
  },
  {
    text: intl.formatMessage({ id: 'status.running' }),
    value: '运行中',
  },
  {
    text: intl.formatMessage({ id: 'status.notRunning' }),
    value: '未运行',
  },
  {
    text: intl.formatMessage({ id: 'status.paused' }),
    value: '暂停中',
  },
  {
    text: intl.formatMessage({ id: 'status.retrying' }),
    value: '重试中',
  },
  {
    text: intl.formatMessage({ id: 'status.failed' }),
    value: '失败',
  },
  {
    text: intl.formatMessage({ id: 'status.completed' }),
    value: '完成',
  },
];
// nputype
export const npuTypeOption = [
  {
    label: 'huawei.com/Ascend910',
    value: 'huawei.com/Ascend910',
  },
  {
    label: 'huawei.com/Ascend310',
    value: 'huawei.com/Ascend310',
  },
  {
    label: 'huawei.com/Ascend310P',
    value: 'huawei.com/Ascend310P',
  },
];

export const addFormDefault = {
  apiVersion: 'ray.io/v1',
  kind: 'RayCluster',
  metadata: {
    name: '',
    namespace: 'vcjob',
  },
  spec: {
    rayVersion: '',
    autoscalerOptions: {
      idleTimeoutSeconds: 120,
      imagePullPolicy: 'IfNotPresent',
      resources: {
        limits: {
          cpu: '500m',
          memory: '512Mi',
        },
        requests: {
          cpu: '500m',
          memory: '512Mi',
        },
      },
      securityContext: {},
      upscalingMode: 'Default',
    },
    enableInTreeAutoscaling: true,
    headGroupSpec: {
      serviceType: 'NodePort',
      rayStartParams: {
        'dashboard-host': '0.0.0.0',
        'dashboard-port': '8265',
        'num-cpus': '0',
      },
      template: {
        spec: {
          containers: [
            {
              image: '',
              lifecycle: {
                preStop: {
                  exec: {
                    command: ['/bin/sh', '-c', 'ray stop'],
                  },
                },
              },
              name: '',
              ports: [
                {
                  containerPort: 6379,
                  name: 'gcs',
                  protocol: 'TCP',
                },
                {
                  containerPort: 8265,
                  name: 'dashboard',
                  protocol: 'TCP',
                },
                {
                  containerPort: 10001,
                  name: 'client',
                  protocol: 'TCP',
                },
              ],
              resources: {
                limits: {
                  cpu: '',
                  memory: '',
                },
                requests: {
                  cpu: '',
                  memory: '',
                },
              },
              env: [],
            },
          ],
        },
      },
    },
    workerGroupSpecs: [],
  },
};

export const addFormWorkGroupData = {
  groupName: 'small-group',
  maxReplicas: 0,
  minReplicas: 0,
  rayStartParams: {},
  replicas: 0,
  template: {
    spec: {
      containers: [
        {
          image: '',
          lifecycle: {
            preStop: {
              exec: {
                command: ['/bin/sh', '-c', 'ray stop'],
              },
            },
          },
          resources: {
            limits: {
              cpu: '',
              memory: '',
            },
            requests: {
              cpu: '',
              memory: '',
            },
          },
        },
      ],
    },
  },
};

export const serviceAddFormDefault = {
  apiVersion: 'ray.io/v1',
  kind: 'RayService',
  metadata: {
    name: 'rayservice-2',
  },
  spec: {
    serveConfigV2: null,
    rayClusterConfig: {
      rayVersion: '',
      headGroupSpec: {
        serviceType: 'NodePort',
        rayStartParams: {
          'num-cpus': '0',
        },
        template: {
          spec: {
            containers: [
              {
                name: '',
                image: '',
                resources: {
                  limits: {
                    cpu: '',
                    memory: '',
                  },
                  requests: {
                    cpu: '',
                    memory: '',
                  },
                },
                ports: [
                  {
                    containerPort: 6379,
                    name: 'gcs',
                  },
                  {
                    containerPort: 8265,
                    name: 'dashboard',
                  },
                  {
                    containerPort: 10001,
                    name: 'client',
                  },
                  {
                    containerPort: 8000,
                    name: 'serve',
                  },
                ],
              },
            ],
          },
        },
      },
      workerGroupSpecs: [],
    },
  },
};

export const serviceAddFormWorkGroupData = {
  replicas: 1,
  minReplicas: '',
  maxReplicas: '',
  groupName: 'small-group',
  rayStartParams: {},
  template: {
    spec: {
      containers: [
        {
          name: '',
          image: '',
          resources: {
            limits: {
              cpu: '',
              memory: '',
            },
            requests: {
              cpu: '',
              memory: '',
            },
          },
        },
      ],
    },
  },
};

export const jobAddFormDefault = {
  apiVersion: 'ray.io/v1',
  kind: 'RayJob',
  metadata: {
    name: '',
  },
  spec: {
    submissionMode: '',
    entrypoint: '',
    rayClusterSpec: {
      rayVersion: '',
      headGroupSpec: {
        serviceType: 'NodePort',
        rayStartParams: {},
        template: {
          spec: {
            containers: [
              {
                name: '',
                image: '',
                ports: [
                  {
                    containerPort: 6379,
                    name: 'gcs',
                  },
                  {
                    containerPort: 8265,
                    name: 'dashboard',
                  },
                  {
                    containerPort: 10001,
                    name: 'client',
                  },
                ],
                resources: {
                  limits: {
                    cpu: '',
                  },
                  requests: {
                    cpu: '',
                  },
                },
              },
            ],
          },
        },
      },
    },
  },
};

export const jobAddFormWorkGroupData = {
  replicas: 0,
  minReplicas: 0,
  maxReplicas: 0,
  groupName: 'small-group',
  rayStartParams: {},
  template: {
    spec: {
      containers: [
        {
          name: '',
          image: '',
          resources: {
            limits: {
              cpu: '',
            },
            requests: {
              cpu: '',
            },
          },
        },
      ],
    },
  },
};

export const clusterSample = `
apiVersion: ray.io/v1
kind: RayCluster
metadata:
  name: raycluster-sample-1
  namespace: vcjob
spec:
  rayVersion: '2.9.0'
  enableInTreeAutoscaling: true
  autoscalerOptions:
    upscalingMode: Default
    idleTimeoutSeconds: 120
    imagePullPolicy: IfNotPresent
    securityContext: {}
    env: []
    envFrom: []
    resources:
      limits:
        cpu: "500m"
        memory: "512Mi"
      requests:
        cpu: "500m"
        memory: "512Mi"
  headGroupSpec:
    serviceType: 'NodePort'
    rayStartParams:
      num-cpus: "0"
      dashboard-host: '0.0.0.0'
      dashboard-port: '8265'
    template:
      spec:
        containers:
        - name: 'head'
          image: docker.io/rayproject/ray:2.41.0
          ports:
          - containerPort: 6379
            name: gcs
          - containerPort: 8265
            name: dashboard
          - containerPort: 10001
            name: client
          lifecycle:
            preStop:
              exec:
                command: ["/bin/sh","-c","ray stop"]
          resources:
            requests:
              cpu: "300m"
              memory: "1000Mi"
            limits:
              cpu: "300m"
              memory: "1000Mi"
  workerGroupSpecs:
  - replicas: 0
    minReplicas: 0
    maxReplicas: 8
    groupName: small-group
    rayStartParams: {}
    template:
      spec:
        containers:
        - name: ray-worker
          image: docker.io/rayproject/ray:2.41.0
          lifecycle:
            preStop:
              exec:
                command: ["/bin/sh","-c","ray stop"]
          resources:
            limits:
              cpu: "1"
              memory: "4G"
            requests:
              cpu: "1"
              memory: "4G"

`;

export const jobSample = `
apiVersion: ray.io/v1
kind: RayJob
metadata:
  name: rayjob-sample
spec:
  entrypoint: python -c "print('Hello Kubeary RayJob!', end=' ')"
  rayClusterSpec:
    rayVersion: '2.41.0'
    headGroupSpec:
      serviceType: 'NodePort'
      rayStartParams: {}
      template:
        spec:
          containers:
            - name: 'head'
              image: docker.io/rayproject/ray:2.41.0
              ports:
                - containerPort: 6379
                  name: gcs
                - containerPort: 8265
                  name: dashboard
                - containerPort: 10001
                  name: client
              resources:
                limits:
                  cpu: "1"
                requests:
                  cpu: "200m"
    workerGroupSpecs:
      - replicas: 0
        minReplicas: 0
        maxReplicas: 5
        groupName: small-group
        rayStartParams: {}
        template:
          spec:
            containers:
              - name: ray-worker
                image: docker.io/rayproject/ray:2.41.0
                resources:
                  limits:
                    cpu: "1"
                  requests:
                    cpu: "200m"

`;

export const serviceSample = `
apiVersion: ray.io/v1
kind: RayService
metadata:
  name: rayservice-sample
  namespace: vcjob
spec:
  serveConfigV2: |
    applications:
      - name: math_app
        import_path: conditional_dag.serve_dag
        route_prefix: /calc
        runtime_env:
          working_dir: "https://github.com/ray-project/test_dag/archive/78b4a5da38796123d9f9ffff59bab2792a043e95.zip"
        deployments:
          - name: Adder
            num_replicas: 1
            user_config:
              increment: 3
            ray_actor_options:
              num_cpus: 0.1
          - name: Multiplier
            num_replicas: 1
            user_config:
              factor: 5
            ray_actor_options:
              num_cpus: 0.1
          - name: Router
            num_replicas: 1
  rayClusterConfig:
    rayVersion: '2.41.0'
    headGroupSpec:
      serviceType: 'NodePort'
      rayStartParams: {}
      template:
        spec:
          containers:
            - name: 'head'
              image: docker.io/rayproject/ray:2.41.0
              resources:
                limits:
                  cpu: 1
                  memory: 2G
                requests:
                  cpu: 1
                  memory: 2G
              ports:
                - containerPort: 6379
                  name: gcs
                - containerPort: 8265
                  name: dashboard
                - containerPort: 10001
                  name: client
                - containerPort: 8000
                  name: serve
    workerGroupSpecs:
      - replicas: 1
        minReplicas: 1
        maxReplicas: 5
        groupName: small-group
        rayStartParams: {}
        template:
          spec:
            containers:
              - name: ray-worker
                image: docker.io/rayproject/ray:2.41.0
                resources:
                  limits:
                    cpu: "1"
                    memory: "2G"
                  requests:
                    cpu: "500m"
                    memory: "2G"

`;
// 普罗表达式
export const promethusExp = {
  cpu: ['sum(ray_resources{Name="CPU",SessionName=~".+",})'],
  nodeCpu: [
    'ray_node_cpu_utilization{instance=~".+", IsHeadNode="false", SessionName=~".+",} * ray_node_cpu_count{instance=~".+",SessionName=~".+",} / 100',
    'ray_node_cpu_utilization{instance=~".+", IsHeadNode="true", SessionName=~".+",} * ray_node_cpu_count{instance=~".+",SessionName=~".+",} / 100',
    'sum(ray_node_cpu_count{SessionName=~".+",})',
  ],
  manageMemory: ['sum(ray_resources{Name="memory",SessionName=~".+",})'],
  nodeMemory: [
    'ray_node_mem_used{instance=~".+", IsHeadNode="false", SessionName=~".+",}',
    'ray_node_mem_used{instance=~".+", IsHeadNode="true", SessionName=~".+",}',
    'sum(ray_node_mem_total{SessionName=~".+",})',
  ],
  manageGpu: ['sum(ray_resources{Name="CPU",SessionName=~".+",})'],
  nodeGpu: [
    'ray_node_gpus_utilization{instance=~".+", IsHeadNode="false", SessionName=~".+",} / 100',
    'ray_node_gpus_utilization{instance=~".+", IsHeadNode="true", SessionName=~".+",} / 100',
    'sum(ray_node_gpus_available{SessionName=~".+",})',
  ],
  nodeGpuMemory: [
    'ray_node_gram_used{instance=~".+",SessionName=~".+",} * 1024 * 1024',
    '(sum(ray_node_gram_available{SessionName=~".+",}) + sum(ray_node_gram_used{SessionName=~".+",})) * 1024 * 1024',
  ],
  manageNpu: ['sum(ray_resources{Name="NPU",SessionName=~".+",})'],
  nodeNpu: [
    'ray_node_npus_utilization{instance=~".+",SessionName=~".+",} / 100',
    'sum(ray_node_npus_available{SessionName=~".+",})',
    'sum(ray_node_npus_utilization{SessionName=~".+"} / 100)',
  ],
  nodeNpuMemory: [
    'ray_node_hbm_used{instance=~".+",SessionName=~".+",}',
    '(sum(ray_node_hbm_available{SessionName=~".+",}) + sum(ray_node_hbm_used{SessionName=~".+",}))',
    'sum(ray_node_hbm_used{SessionName=~".+",})',
  ],
};

export const getChineseChartType = (intl) => ({
  cpu: intl.formatMessage({ id: 'chart.managedCpu' }),
  nodeCpu: intl.formatMessage({ id: 'chart.cpuUsage' }),
  manageMemory: intl.formatMessage({ id: 'chart.managedMemory' }),
  nodeMemory: intl.formatMessage({ id: 'chart.memoryUsage' }),
  manageGpu: intl.formatMessage({ id: 'chart.managedGpu' }),
  nodeGpu: intl.formatMessage({ id: 'chart.gpuUsage' }),
  nodeGpuMemory: intl.formatMessage({ id: 'chart.gramUsage' }),
  manageNpu: intl.formatMessage({ id: 'chart.managedNpu' }),
  nodeNpu: intl.formatMessage({ id: 'chart.npuUsage' }),
  nodeNpuMemory: intl.formatMessage({ id: 'chart.hbmUsage' }),
});

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

export const lokiExp = {
  // gcs: ['{filename="/tmp/ray/session_latest/logs/gcs_server.out"} | json'],
  gcsStateTotal: ['{filename="/tmp/ray/session_latest/logs/gcs_server.out"} | json'],
  gcsStateActive: ['{filename="/tmp/ray/session_latest/logs/gcs_server.out"} | json'],
  gcsQueueTime: ['{filename="/tmp/ray/session_latest/logs/gcs_server.out"} | json'],
  gcsHealthyCheck: [
    [
      '{filename="/tmp/ray/session_latest/logs/dashboard.log"}',
      '  |~ `GET\\s+(/api/gcs_healthz)\\s` ',
      '  | regexp `.*(GET\\s+(?P<path>/api/gcs_healthz)).*bytes\\s+(?P<response_time>\\d+\.?\\d*)\\s+[a-z]+.*`',
      '  | label_format path="{{.path}}" ',
      '  | line_format "{{.response_time}}"',
    ].join(''),
  ],
  gcsRayLet: [
    [
      '{filename="/tmp/ray/session_latest/logs/dashboard_agent.log"} | json',
      '  |~ `GET\\s+(/api/local_raylet_healthz)\\s`',
      '  | regexp `.*(GET\\s+(?P<path>/api/local_raylet_healthz)).*bytes\\s+(?P<response_time>\\d+\\.?\\d*)\\s+[a-z]+.*`',
      '  | label_format path="{{.path}}"',
      '  | line_format "{{.response_time}}"',
    ].join(''),
  ],
  jobGet: [
    [
      '{filename="/tmp/ray/session_latest/logs/dashboard.log"}',
      '  |~ `GET\\s+/api/jobs/\\S+`',
      '  | regexp `.*bytes\\s+(?P<response_time>\\d+\\.?\\d*)\\s+[a-z]+.*`',
      '  | line_format `{{.response_time}}`',
    ].join(''),
  ],
  jobPost: [
    [
      '{filename="/tmp/ray/session_latest/logs/dashboard.log"}',
      '  |~ `POST\\s+/api/jobs/\\s`',
      '  | regexp `.*bytes\\s+(?P<response_time>\\d+\\.?\\d*)\\s+[a-z]+.*`',
      '  | line_format `{{.response_time}}`',
    ].join(''),
  ],
  jobDelete: [
    [
      '{filename="/tmp/ray/session_latest/logs/dashboard.log"}',
      '  |~ `DELETE\\s+/api/jobs/[^/]+\\s`',
      '  | regexp `.*bytes\\s+(?P<response_time>\\d+\\.?\\d*)\\s+[a-z]+.*`',
      '  | line_format `{{.response_time}}`',
    ].join(''),
  ],
  jobLogGet: [
    [
      '{filename="/tmp/ray/session_latest/logs/dashboard.log"}',
      '  |~ `GET\\s+/api/jobs/[^/]+/logs\\b`',
      '  | regexp `.*bytes\\s+(?P<response_time>\\d+\\.?\\d*)\\s+[a-z]+.*` ',
      '  | line_format `{{.response_time}}`',
    ].join(''),
  ],
  jobLogDelete: [
    [
      '{filename="/tmp/ray/session_latest/logs/dashboard.log"}',
      '  |~ `DELETE\\s+/api/jobs/[^/]+/logs\\b`',
      '  | regexp `.*bytes\\s+(?P<response_time>\\d+\\.?\\d*)\\s+[a-z]+.*` ',
      '  | line_format `{{.response_time}}`',
    ].join(''),
  ],
  dashboardNode: [
    [
      '{filename="/tmp/ray/session_latest/logs/dashboard.log"}',
      '|~ `GET\\s+(/nodes\\?view=summary)\\s`',
      '| regexp `.*(GET\\s+(?P<path>/nodes\\?view=summary)).*bytes\\s+(?P<response_time>\\d+\\.?\\d*)\\s+[a-z]+.*`',
      '| label_format path="{{.path}}"',
      '| line_format "{{.response_time}}"',
    ].join(''),
  ],
  dashboardLog: [
    [
      '{filename="/tmp/ray/session_latest/logs/dashboard.log"}',
      '|~ `GET\\s+(/logical/actors)\\s` ',
      '| regexp `.*(GET\\s+(?P<path>/logical/actors)).*bytes\\s+(?P<response_time>\\d+\\.?\\d*)\\s+[a-z]+.*`',
      '| label_format path="{{.path}}"',
      '| line_format "{{.response_time}}"',
    ].join(''),
  ],
  dashboardJob: [
    [
      '{filename="/tmp/ray/session_latest/logs/dashboard.log"}',
      '|~ `GET\s+(/api/jobs/)\s` ',
      '| regexp `.*(GET\\s+(?P<path>/api/jobs/)).*bytes\\s+(?P<response_time>\\d+\\.?\\d*)\\s+[a-z]+.*` ',
      '| label_format path="{{.path}}"',
      '| line_format "{{.response_time}}"',
    ].join(''),
  ],
};

export const getChineseLokiType = (intl) => ({
  gcsStateTotal: intl.formatMessage({ id: 'loki.gcsStateTotal' }),
  gcsStateActive: intl.formatMessage({ id: 'loki.gcsStateActive' }),
  gcsQueueTime: intl.formatMessage({ id: 'loki.gcsQueueTime' }),
  gcsHealthyCheck: intl.formatMessage({ id: 'loki.gcsHealthyCheck' }),
  gcsRayLet: intl.formatMessage({ id: 'loki.gcsRayLet' }),
  jobGet: intl.formatMessage({ id: 'loki.jobGet' }),
  jobPost: intl.formatMessage({ id: 'loki.jobPost' }),
  jobDelete: intl.formatMessage({ id: 'loki.jobDelete' }),
  jobLogGet: intl.formatMessage({ id: 'loki.jobLogGet' }),
  jobLogDelete: intl.formatMessage({ id: 'loki.jobLogDelete' }),
  dashboardNode: intl.formatMessage({ id: 'loki.dashboardNode' }),
  dashboardLog: intl.formatMessage({ id: 'loki.dashboardLog' }),
  dashboardJob: intl.formatMessage({ id: 'loki.dashboardJob' }),
});

export const lokiResultWord = {
  gcs: {
    gcsStateTotal: 'result',
    gcsStateActive: 'result1',
    gcsQueueTime: 'result2',
  },
};
