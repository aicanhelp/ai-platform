/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */

import { getLocale } from '@/i18n';

const newEnvPrepareZhCN = `
## 环境准备
1. 已部署Kubernetes v1.21版本及以上、containerd v1.7.0版本及以上，kube-prometheus v1.19版本及以上。

2. openFuyao的混部调度器选择的是volcano-scheduler在使用时需要预先在Kubernetes中通过helm提前进行安装。目前在1.9.0版本进行了全量测试，功能上晚于1.9.0的预期可以正常使用，用户可以选择部署，但暂不保证功能正确性。

   2.1 通过helm安装volcano-scheduler

   \`\`\`shell
   helm repo add volcano-sh https://volcano-sh.github.io/helm-charts
   helm repo update
   helm install volcano volcano-sh/volcano --version 1.9.0 -n volcano-system --create-namespace
   \`\`\`

   \> ![](/src/assets/md/figures/iconnote.gif)**说明：**
   \> 如果已经在openFuyao上安装了NUMA亲和调度组件，此时volcano组件会被默认安装，此时无需再通过helm进行前置安装。

   2.2 修改volcano-scheduler默认配置

   \`\`\`shell
   kubectl edit cm -n volcano-system volcano-scheduler-configmap
   \`\`\`

   主要修改如下注释部分:

   \`\`\`yaml
   apiVersion: v1
   data:
     volcano-scheduler.conf: |
       actions: "allocate, backfill, preempt"	# 确保actions类别和顺序
       tiers:
       - plugins:
         - name: priority           # 确保tiers[0].plugins[0]中开启优先级调度
         - name: gang
           enablePreemptable: false
           enableJobStarving: false   # 确保关闭 enableJobStarving
        ...
   kind: ConfigMap
   metadata:
       meta.helm.sh/release-name: volcano
       meta.helm.sh/release-namespace: volcano-system
     labels:
       app.kubernetes.io/managed-by: Helm
     name: volcano-scheduler-configmap
     namespace: volcano-system
   \`\`\`

3. openFuyao的混部引擎需要操作系统内核至少在4.19及以上，具体各项混部功能是否可以开启可以参见界面上**混部策略配置**中的**混部能力支持**模块

   \> ![](/src/assets/md/figures/iconnote.gif)**说明：**<br />
   \> 在离线混部的完整功能在openEuler 22.03 LTS-SP3 版本上做了详细的验证，对于其它更新版本，您可以选择部署，但暂不保证功能正确性。

4. 开启kubelet绑核和NUMA亲和策略。

   > ![](/src/assets/md/figures/iconnote.gif)**说明：**<br />此功能是为了配合QoS-level为HLS级别Pod开启绑核。只有当kubelet的static策略开启时，HLS级别的Pod才会具有独占性和NUMA亲和性，提升HLS业务的性能表现。

   使用该模块时，需要修改Kubelet的Config文件，具体配置步骤如下所示:

   4.1 打开kubelet配置文件。

      \`\`\`  shell
       vi /etc/kubernetes/kubelet-config.yaml
      \`\`\`

   > ![](/src/assets/md/figures/iconnote.gif)**说明：**<br />
   > 如果上述位置没有config文件，可以在/var/lib/kubelet/config.yaml位置找到。

   4.2 新增或修改配置项。（修改static策略时要同时配置预留cpu）

     \`\`\`yaml
     cpuManagerPolicy: static
     systemReserved:
       cpu: "0.5"
     # 注：当节点cpu核数较少时，kubeReserved开启可能会导致节点可用cpu不足，kubelet有崩溃风险，请谨慎开启
     kubeReserved:
       cpu: "0.5"
     topologyManagerPolicy: xxx # best-effort / restricted / single-numa-node
     \`\`\`

   4.3 修改应用。

     \`\`\`
     rm -rf /var/lib/kubelet/cpu_manager_state
     systemctl daemon-reload
     systemctl restart kubelet
     \`\`\`

   4.4 查看kubelet运行状态。

     \`\`\`
     systemctl status kubelet
     \`\`\`

     kubelet运行状态为running，则表示成功。
5. 混部节点开启containerd的nri扩展功能。

   5.1 在混部节点上，进入vim /etc/containerd/config.toml，搜索是否有[plugins."io.containerd.nri.v1.nri"]。

   5.2 如果有，则将disable=true改为disable=false。如果没有，则在[plugins]下增加：

      \`\`\`
      [plugins."io.containerd.nri.v1.nri"]

         disable = false

         disable_connections = false

         plugin_config_path = "/etc/nri/conf.d"

         plugin_path = "/opt/nri/plugins"

         plugin_registration_timeout = "5s"

         plugin_request_timeout = "2s"

         socket_path = "/var/run/nri/nri.sock"
      \`\`\`

   5.3 配置完成，执行如下命令重启containerd

      \`\`\`shell
      sudo systemctl restart containerd
      \`\`\`
`;

const newEnvPrepareEnUS = `
## Environment Preparation
1. Kubernetes v1.21 or above, containerd v1.7.0 or above, and kube-prometheus v1.19 or above have been deployed.

2. The colocation scheduler in openFuyao uses volcano-scheduler, which needs to be pre-installed in Kubernetes via helm before use. Full testing has been conducted on version 1.9.0. Versions later than 1.9.0 are expected to work normally, and users can choose to deploy them, but functionality is not guaranteed at this time.

   2.1 Install volcano-scheduler via helm

   \`\`\`shell
   helm repo add volcano-sh https://volcano-sh.github.io/helm-charts
   helm repo update
   helm install volcano volcano-sh/volcano --version 1.9.0 -n volcano-system --create-namespace
   \`\`\`

   \> ![](/src/assets/md/figures/iconnote.gif)**Note:**
   \> If the NUMA affinity scheduling component has already been installed on openFuyao, the volcano component will be installed by default at this time, and no pre-installation via helm is required.

   2.2 Modify the default configuration of volcano-scheduler

   \`\`\`shell
   kubectl edit cm -n volcano-system volcano-scheduler-configmap
   \`\`\`

   Mainly modify the following commented sections:

   \`\`\`yaml
   apiVersion: v1
   data:
     volcano-scheduler.conf: |
       actions: "allocate, backfill, preempt"	# Ensure actions category and order
       tiers:
       - plugins:
         - name: priority           # Ensure priority scheduling is enabled in tiers[0].plugins[0]
         - name: gang
           enablePreemptable: false
           enableJobStarving: false   # Ensure enableJobStarving is disabled
        ...
   kind: ConfigMap
   metadata:
       meta.helm.sh/release-name: volcano
       meta.helm.sh/release-namespace: volcano-system
     labels:
       app.kubernetes.io/managed-by: Helm
     name: volcano-scheduler-configmap
     namespace: volcano-system
   \`\`\`

3. The colocation engine in openFuyao requires the operating system kernel to be at least 4.19 or above. For details on whether various colocation functions can be enabled, please refer to the **Colocation Capability Support** module in **Colocation Strategy Configuration** on the interface.

   \> ![](/src/assets/md/figures/iconnote.gif)**Note:**<br />
   \> The complete functions of online-offline colocation have been thoroughly verified on openEuler 22.03 LTS-SP3 version. For other updated versions, you can choose to deploy them, but functionality is not guaranteed at this time.

4. Enable kubelet core binding and NUMA affinity policies.

   > ![](/src/assets/md/figures/iconnote.gif)**Note:**<br />This function is to enable core binding for HLS-level Pods in conjunction with QoS-level. Only when the kubelet static policy is enabled will HLS-level Pods have exclusivity and NUMA affinity, improving the performance of HLS services.

   When using this module, you need to modify the Kubelet Config file. The specific configuration steps are as follows:

   4.1 Open the kubelet configuration file.

      \`\`\`  shell
       vi /etc/kubernetes/kubelet-config.yaml
      \`\`\`

   > ![](/src/assets/md/figures/iconnote.gif)**Note:**<br />
   > If the config file is not found at the above location, it can be found at /var/lib/kubelet/config.yaml.

   4.2 Add or modify configuration items. (When modifying the static policy, reserved cpu must be configured at the same time)

     \`\`\`yaml
     cpuManagerPolicy: static
     systemReserved:
       cpu: "0.5"
     # Note: When the number of node cpu cores is small, enabling kubeReserved may cause insufficient node available cpu, and kubelet may crash. Please enable it with caution.
     kubeReserved:
       cpu: "0.5"
     topologyManagerPolicy: xxx # best-effort / restricted / single-numa-node
     \`\`\`

   4.3 Apply the changes.

     \`\`\`
     rm -rf /var/lib/kubelet/cpu_manager_state
     systemctl daemon-reload
     systemctl restart kubelet
     \`\`\`

   4.4 Check kubelet running status.

     \`\`\`
     systemctl status kubelet
     \`\`\`

     If kubelet running status is running, it means success.
5. Enable containerd nri extension function on colocation nodes.

   5.1 On the colocation node, enter vim /etc/containerd/config.toml and search for [plugins."io.containerd.nri.v1.nri"].

   5.2 If it exists, change disable=true to disable=false. If it does not exist, add the following under [plugins]:

      \`\`\`
      [plugins."io.containerd.nri.v1.nri"]

         disable = false

         disable_connections = false

         plugin_config_path = "/etc/nri/conf.d"

         plugin_path = "/opt/nri/plugins"

         plugin_registration_timeout = "5s"

         plugin_request_timeout = "2s"

         socket_path = "/var/run/nri/nri.sock"
      \`\`\`

   5.3 After configuration is complete, execute the following command to restart containerd

      \`\`\`shell
      sudo systemctl restart containerd
      \`\`\`
`;

export const getNewEnvPrepare = () => {
  const locale = getLocale();
  return locale === 'en-US' ? newEnvPrepareEnUS : newEnvPrepareZhCN;
};

const workloadZhCN = `
## 工作负载部署

### 部署样例

以deployment部署为例，在使用Qos分级策略时，需要在\`.spec.template.metadata.annotations\`中设置Qos-Level注解。

\`\`\`yaml
# HLS
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nginx
spec:
  replicas: 1
  selector:
    matchLabels:
      app: nginx
  template:
    metadata:
      labels:
        app: nginx
      annotations:
        # 指定Qos-Level注解
        openfuyao.com/qos-level: HLS
    spec:
      containers:
      - name: nginx
        image: nginx
        imagePullPolicy: IfNotPresent
        resources:
          limits:
            cpu: 1
            memory: 100Mi
          requests:
            cpu: 1
            memory: 100Mi
---
# LS
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nginx
spec:
  replicas: 1
  selector:
    matchLabels:
      app: nginx
  template:
    metadata:
      labels:
        app: nginx
      annotations:
        # 指定Qos-Level注解
        openfuyao.com/qos-level: LS
    spec:
      containers:
      - name: nginx
        image: nginx
        imagePullPolicy: IfNotPresent
        resources:
          limits:
            cpu: 2
            memory: 200Mi
          requests:
            cpu: 1
            memory: 100Mi
---
# BE
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nginx
spec:
  replicas: 1
  selector:
    matchLabels:
      app: nginx
  template:
    metadata:
      labels:
        app: nginx
      annotations:
        # 指定Qos-Level注解
        openfuyao.com/qos-level: BE
    spec:
      containers:
      - name: nginx
        image: nginx
        imagePullPolicy: IfNotPresent
        resources:
          limits:
            kubernetes.io/be-cpu: 2000	# 对应毫核
            kubernetes.io/be-memory: 200Mi
          requests:
            kubernetes.io/be-cpu: 1000
            kubernetes.io/be-memory: 100Mi
\`\`\`
`;

const workloadEnUS = `
## Workload Deployment

### Deployment Examples

Taking deployment as an example, when using the QoS tiering strategy, you need to set the Qos-Level annotation in \`.spec.template.metadata.annotations\`.

\`\`\`yaml
# HLS
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nginx
spec:
  replicas: 1
  selector:
    matchLabels:
      app: nginx
  template:
    metadata:
      labels:
        app: nginx
      annotations:
        # Specify Qos-Level annotation
        openfuyao.com/qos-level: HLS
    spec:
      containers:
      - name: nginx
        image: nginx
        imagePullPolicy: IfNotPresent
        resources:
          limits:
            cpu: 1
            memory: 100Mi
          requests:
            cpu: 1
            memory: 100Mi
---
# LS
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nginx
spec:
  replicas: 1
  selector:
    matchLabels:
      app: nginx
  template:
    metadata:
      labels:
        app: nginx
      annotations:
        # Specify Qos-Level annotation
        openfuyao.com/qos-level: LS
    spec:
      containers:
      - name: nginx
        image: nginx
        imagePullPolicy: IfNotPresent
        resources:
          limits:
            cpu: 2
            memory: 200Mi
          requests:
            cpu: 1
            memory: 100Mi
---
# BE
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nginx
spec:
  replicas: 1
  selector:
    matchLabels:
      app: nginx
  template:
    metadata:
      labels:
        app: nginx
      annotations:
        # Specify Qos-Level annotation
        openfuyao.com/qos-level: BE
    spec:
      containers:
      - name: nginx
        image: nginx
        imagePullPolicy: IfNotPresent
        resources:
          limits:
            kubernetes.io/be-cpu: 2000	# Corresponds to milli-cores
            kubernetes.io/be-memory: 200Mi
          requests:
            kubernetes.io/be-cpu: 1000
            kubernetes.io/be-memory: 100Mi
\`\`\`
`;

export const getWorkload = () => {
  const locale = getLocale();
  return locale === 'en-US' ? workloadEnUS : workloadZhCN;
};
