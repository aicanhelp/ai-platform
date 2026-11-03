/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
/**
 * 监控实例
 */
const commonYaml = `
metadata:
  namespace: default
  name: 'example'
spec:
  replicas: 1
  selector:
    matchLabels:
      app: example
  template:
    metadata:
      labels:
        app: example
`;

export const serviceMonitorYamlExample = `apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: example-servicemonitor
  namespace: example-namespace
spec:
  endpoints:
    - interval: 30s
      port: web
      path: /metrics
  selector:
    matchLabels:
      app: example-app
  namespaceSelector:
    matchNames:
    - example-namespace
`;

/**
 * 容器组
 */
export const podYamlExample = `apiVersion: v1
kind: Pod 
metadata:
  namespace: default
  name: example-pod
  labels:
    app: example-pod
    namespace: default
spec:
  containers:
    - name: example-container
      image: nginx:1.19.3
      ports:
        - containerPort: 8080
`;

/**
 * 无状态负载
 */
export const deploymentYamlExample = `apiVersion: apps/v1
kind: Deployment ` + `${commonYaml}` + 
`    spec:
       containers:
         - name: example-deployment
           image: 'nginx:1.19.3'
           ports:
             - containerPort: 8091
`;

/**
 * 有状态负载
 */
export const statefulSetYamlExample = `apiVersion: apps/v1
kind: StatefulSet ` + `${commonYaml}` + 
`    spec:
       containers:
         - name: example-statefulset
           image: 'nginx:1.19.3'
           ports:
             - containerPort: 8092
`;

/**
 * 守护进程
 */
export const daemonSetYamlExample = `apiVersion: apps/v1
kind: DaemonSet ` + `${commonYaml}` + 
`    spec:
       containers:
         - name: example-daemonset
           image: 'nginx:1.19.3'
           ports:
             - containerPort: 8093
`;

/**
 * 任务
 */
export const jobYamlExample = `apiVersion: batch/v1
kind: Job
metadata:
  name: example
  namespace: default
spec:
  template:
    spec:
      containers:
        - name: example
          image: busybox
          command: ["sh", "-c", "echo Hello! && sleep 3600"]
      restartPolicy: Never
`;

/**
 * 定时任务
 */
export const cronJobYamlExample = `apiVersion: batch/v1
kind: CronJob
metadata:
  name: example-cronjob
  namespace: default
spec:
  schedule: "0 * * * *"
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: example
            image: busybox
            args:
            - /bin/sh
            - -c
            - date; echo Hello from the Kubernetes cluster
          restartPolicy: OnFailure
`;

/**
 * 配置
 */
export const configYamlExample = `apiVersion: v1
kind: ConfigMap
metadata:
  name: 'my-configmap'
  namespace: 'default'
data: 
  key1: value1
  key2: value2
`;

/**
 * 保密字典
 */
export const secretYamlExample = `apiVersion: v1
kind: Secret
metadata:
  name: example-secret
  namespace: default
data:
  username: xxxxxxxx  # base64 encoded string
  password: xxxxxxxxxxxxxxxx  # base64 encoded string
`;

/**
 * 命名空间
 */
export const namespaceYamlExample = `apiVersion: v1
kind: Namespace
metadata:
  name: example-namespace
`;

/**
 * 资源配额
 */
export const resourceQuotaYamlExample = `apiVersion: v1
kind: ResourceQuota
metadata:
  name: example-quota
  namespace: example-namespace
spec:
  hard:
    pods: "10"
    requests.cpu: "4"
    requests.memory: "16Gi"
    limits.cpu: "8"
    limits.memory: "32Gi"
`;

/**
 * 限制范围
 */
export const limitRangeYamlExample = `apiVersion: v1
kind: LimitRange
metadata:
  name: example-limitrange
  namespace: example-namespace
spec:
  limits:
  - default:
      cpu: "500m"
      memory: "512Mi"
    defaultRequest:
      cpu: "200m"
      memory: "256Mi"
    type: Container
`;

/**
 * 自定义资源定义
 */
export const customResourceDefinitionYamlExample = `apiVersion: apiextensions.k8s.io/v1
kind: CustomResourceDefinition
metadata:
  name: examples.example.com
spec:
  group: example.com
  versions:
    - name: v1
      served: true
      storage: true
      schema:
        openAPIV3Schema:
          type: object
          properties:
            spec:
              type: object
              properties:
                foo:
                  type: string
                bar:
                  type: integer
  scope: Namespaced
  names:
    plural: examples
    singular: example
    kind: Example
    shortNames:
    - ex
`;

/**
 *  CR定义
 */
export const crYamlExample = `apiVersion: example.com/v1
kind: Example
metadata:
  name: example-cr
  namespace: example-namespace
`;

export const podGroupExample = `apiVersion: scheduling.volcano.sh/v1beta1
kind: PodGroup
metadata:
  name: example-podgroup
  namespace: default
spec:
  minMember: 3
  minResources:
    cpu: 3
    memory: 3Gi
`;

export const queueSchedulingExampe = `apiVersion: scheduling.volcano.sh/v1beta1
kind: Queue
metadata:
  name: example-queue
spec:
  weight: 1
  reclaimable: false
  capability:
    cpu: 2
`;

export const jobBatchExample = `apiVersion: batch.volcano.sh/v1alpha1
kind: Job
metadata:
  name: vcjob
  namespace: default
spec:
  schedulerName: volcano
  minAvailable: 4
  tasks:
    - replicas: 4
      name: "test"
      template:
        spec:
          containers:
            - image: alpine
              command: ["/bin/sh", "-c", "sleep 1000"]
              imagePullPolicy: IfNotPresent
              name: running
              resources:
                requests:
                  cpu: "1"
          restartPolicy: OnFailure
`;

/**
 * 网络服务
 */
export const serviceYamlExample = `apiVersion: v1
kind: Service
metadata:
  name: example-service
  namespace: default
spec:
  selector:
    app: example
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8080
`;

/**
 * 应用路由
 */
export const ingressYamlExample = `apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: example-ingress
  namespace: default
spec:
  rules:
    - host: example.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: example-service
                port:
                  number: 80
`;

/**
 * 角色
 */
export const roleYamlExample = `apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  namespace: default
  name: example-role
rules:
  - apiGroups: [""]
    resources: ["pods"]
    verbs: ["get", "list", "watch"]
  - apiGroups: [""]
    resources: ["pods/log"]
    verbs: ["get"]
`;

/**
 * 服务账户
 */
export const serviceAccountYamlExample = `apiVersion: v1
kind: ServiceAccount
metadata:
  name: example-serviceaccount
  namespace: default
`;

/**
 * 角色绑定
 */
export const roleBindYamlExample = `apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: example-rolebinding
  namespace: default
subjects:
  - kind: ServiceAccount
    name: example-serviceaccount
    namespace: default
roleRef:
  kind: Role
  name: example-role
  apiGroup: rbac.authorization.k8s.io
`;

/**
 * 集群角色
 */
export const clusterRoleYamlExample = `apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: example-clusterrole
rules:
- apiGroups: [""]
  resources: ["pods"]
  verbs: ["get", "list", "watch"]
`;

/**
 * 集群角色绑定
 */
export const clusterRoleBindYamlExample = `apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: example-clusterrolebinding
subjects:
- kind: ServiceAccount
  name: example-serviceaccount
  namespace: default
roleRef:
  kind: ClusterRole
  name: example-clusterrole
  apiGroup: rbac.authorization.k8s.io
`;

/**
 * 数据卷 pv
 */
export const pvYamlExample = `apiVersion: v1
kind: PersistentVolume
metadata:
  name: example-pv
spec:
  capacity:
    storage: 1Gi  # 定义 PV 的大小
  accessModes:
    - ReadWriteOnce  # 定义访问模式
  persistentVolumeReclaimPolicy: Retain  # 定义回收策略
  storageClassName: example-sc  # 与 StorageClass 对应
  hostPath:
    path: "/mnt/data"  # 本地路径，表示在主机上的存储位置
`;

/**
 * 数据卷声明 pvc
 */
export const pvcYamlExample = `apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: example-pvc
  namespace: default
spec:
  accessModes:
    - ReadWriteOnce  # 与 PV 的 accessModes 一致
  storageClassName: example-sc  # 对应的 StorageClass
  resources:
    requests:
      storage: 1Gi  # 请求的存储空间大小
`;

/**
 * 存储池 sc
 */
export const scYamlExample = `apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: example-sc
provisioner: kubernetes.io/no-provisioner  # 如果使用静态存储，指定 no-provisioner
volumeBindingMode: Immediate  # 定义绑定模式，可选 Immediate 或 WaitForFirstConsumer
`;

/**
 * 创建优先级队列
 */
export const priorityClassesExample = `apiversion: scheduling.k8s.io/v1
kind: PriorityClass
metadata:
  name: high-priority
value: 100
globalDefault: false
description: 'This priority class should be used for volcano job only'
`;