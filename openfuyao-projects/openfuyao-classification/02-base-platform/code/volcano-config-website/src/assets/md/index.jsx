/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */

const getLocale = () => window.__OPENFUYAO_LOCALE__ || localStorage.getItem('locale') || 'zh-CN';

const volcanoZhCN = `
# NUMA亲和策略（numa-aware）

在云原生环境中，对于高性能计算（**HPC**）、实时应用和内存密集型工作负载等需要**CPU**间通信频繁的场景下，跨**NUMA**（**Non-uniform memory access**）节点访问会导致增加延迟和开销，从而降低系统性能。为此，**Volcano**实现了**NUMA**亲和性调度策略，尽可能把**Pod**调度到需要跨**NUMA**节点最少的工作节点上，这种调度策略能够降低数据传输开销，优化资源利用率，从而增强系统的整体性能。

## NUMA亲和策略介绍

当Pod设置了拓扑策略时，Volcano会根据Pod设置的拓扑策略预测匹配的节点列表。

1. 根据Pod设置的Volcano拓扑策略，筛选具有相同策略的节点。
2. 在设置了相同策略的节点中，筛选CPU拓扑满足该策略要求的节点进行调度。

| Volcano拓扑策略        | 1. 筛选具有相同策略的节点                                     | 2. 节点的CPU拓扑满足该策略的要求                                         |
|---------------------|---------------------------------------------------|-----------------------------------------------------------|
| **none**            | 无筛选行为：<br>- none：可调度<br>- best-effort：可调度<br>- restricted：可调度<br>- single-numa-node：可调度 | -                                                         |
| **best-effort**     | 筛选拓扑策略同样为“best-effort”的节点：<br>- none：不可调度<br>- best-effort：可调度<br>- restricted：不可调度<br>- single-numa-node：不可调度 | 尽可能满足策略要求进行调度：优先调度至单NUMA节点，如果单NUMA节点无法满足CPU申请值，允许调度至多个NUMA节点。 |
| **restricted**      | 筛选拓扑策略同样为“restricted”的节点：<br>- none：不可调度<br>- best-effort：不可调度<br>- restricted：可调度<br>- single-numa-node：不可调度 | 严格限制的调度策略：<br>- 单NUMA节点的CPU容量上限大于等于CPU的申请值时，仅允许调度至单NUMA节点。如果单NUMA节点剩余的CPU可使用量不足，则Pod无法调度。<br>- 单NUMA节点的CPU容量上限小于CPU的申请值时，可允许调度至多个NUMA节点。 |
| **single-numa-node**| 筛选拓扑策略同样为“single-numa-node”的节点：<br>- none：不可调度<br>- best-effort：不可调度<br>- restricted：不可调度<br>- single-numa-node：可调度 | 仅允许调度至单NUMA节点。                                   |

### 使用Volcano调度工作负载

**spec.template.spec.schedulerName**：设置为**volcano**，表示使用**Volcano**调度该工作负载。

**spec.template.metadata.annotations**：配置**NUMA**亲和性，示例为**single-numa-node**。
\`\`\`
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
        # 指定numa亲和性
        volcano.sh/numa-topology-policy: single-numa-node
    spec:
      # 指定调度器为Volcano
      schedulerName: volcano
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
\`\`\`

对于需要使用运行时POD亲和优化的工作负载，则不能将pod的QOS类别配置为guranteed类型。
`;

const volcanoEnUS = `
# NUMA Affinity Policy (numa-aware)

In cloud-native environments, for scenarios requiring frequent inter-**CPU** communication such as high-performance computing (**HPC**), real-time applications, and memory-intensive workloads, cross-**NUMA** (**Non-uniform memory access**) node access introduces additional latency and overhead, degrading system performance. To address this, **Volcano** implements a **NUMA** affinity scheduling policy that schedules **Pods** onto worker nodes requiring the fewest cross-**NUMA** node hops. This scheduling strategy reduces data transfer overhead, optimizes resource utilization, and enhances overall system performance.

## NUMA Affinity Policy Overview

When a Pod specifies a topology policy, Volcano predicts the matching node list based on the topology policy set for the Pod.

1. Filter nodes that have the same policy as the Volcano topology policy set for the Pod.
2. Among the nodes with the same policy, filter the nodes whose CPU topology satisfies the policy requirements for scheduling.

| Volcano Topology Policy | 1. Filter nodes with the same policy | 2. Node CPU topology satisfies the policy requirements |
|---------------------|---------------------------------------------------|-----------------------------------------------------------|
| **none**            | No filtering behavior:<br>- none: schedulable<br>- best-effort: schedulable<br>- restricted: schedulable<br>- single-numa-node: schedulable | -                                                         |
| **best-effort**     | Filter nodes whose topology policy is also "best-effort":<br>- none: not schedulable<br>- best-effort: schedulable<br>- restricted: not schedulable<br>- single-numa-node: not schedulable | Schedule while satisfying the policy requirements as much as possible: prefer scheduling to a single NUMA node; if a single NUMA node cannot satisfy the requested CPU, scheduling to multiple NUMA nodes is allowed. |
| **restricted**      | Filter nodes whose topology policy is also "restricted":<br>- none: not schedulable<br>- best-effort: not schedulable<br>- restricted: schedulable<br>- single-numa-node: not schedulable | Strictly restricted scheduling policy:<br>- When the CPU capacity upper limit of a single NUMA node is greater than or equal to the requested CPU, scheduling is only allowed to a single NUMA node. If the remaining usable CPU on a single NUMA node is insufficient, the Pod cannot be scheduled.<br>- When the CPU capacity upper limit of a single NUMA node is less than the requested CPU, scheduling to multiple NUMA nodes is allowed. |
| **single-numa-node**| Filter nodes whose topology policy is also "single-numa-node":<br>- none: not schedulable<br>- best-effort: not schedulable<br>- restricted: not schedulable<br>- single-numa-node: schedulable | Scheduling is only allowed to a single NUMA node.                                   |

### Scheduling Workloads with Volcano

**spec.template.spec.schedulerName**: Set to **volcano**, indicating that **Volcano** is used to schedule the workload.

**spec.template.metadata.annotations**: Configure **NUMA** affinity; the example uses **single-numa-node**.
\`\`\`
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
        # Specify NUMA affinity
        volcano.sh/numa-topology-policy: single-numa-node
    spec:
      # Specify Volcano as the scheduler
      schedulerName: volcano
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
\`\`\`

For workloads that need to use runtime Pod affinity optimization, the Pod's QoS class must not be configured as guaranteed.
`;

const gangZhCN = `
# 组调度（Gang）

组调度（**Gang**）满足了调度过程中“**All or nothing**”的调度需求，避免**Pod**的任意调度导致集群资源的浪费，主要应用于**AI**、大数据等多任务协作场景。启用该能力后，可以解决分布式训练任务之间的资源忙等待和死锁等痛点问题，大幅度提升整体训练性能。

#### 组调度介绍

**Gang**调度策略是**volcano-scheduler**的核心调度算法之一，它满足了调度过程中的“**All or nothing**”的调度需求，避免**Pod**的任意调度导致集群资源的浪费。具体算法是，观察**Job**下的**Pod**已调度数量是否满足了最小运行数量，当**Job**的最小运行数量得到满足时，为**Job**下的所有**Pod**执行调度动作，否则，不执行。

创建工作负载使用**Gang**调度能力

1. 首先创建**PodGroup**，需指定**minMember**和**minResources**：

   **minMember**：归属于当前**PodGroup**的一组**Pod**满足**minMember**数量时，才会被统一调度。

   **minResources**：集群空闲资源满足**minResources**要求时，该组**Pod**才会被统一调度。

2. 创建工作负载时，通过**schedulerName**指定**Volcano**调度器，并通过**annotation**指定其归属的**PodGroup**：

   **schedulerName**：设置为**volcano**，表示使用**Volcano**调度该工作负载。

   **scheduling.k8s.io/group-name**：指定上一步中创建的**PodGroup**，示例为**pg-test1**。
`;

const gangEnUS = `
# Gang Scheduling

Gang scheduling (**Gang**) satisfies the "**All or nothing**" scheduling requirement during the scheduling process, avoiding resource waste caused by arbitrary **Pod** scheduling. It is mainly applied in multi-task collaboration scenarios such as **AI** and big data. Once enabled, it resolves pain points such as resource busy-waiting and deadlocks among distributed training tasks, significantly improving overall training performance.

#### Gang Scheduling Overview

The **Gang** scheduling policy is one of the core scheduling algorithms of **volcano-scheduler**. It satisfies the "**All or nothing**" scheduling requirement during the scheduling process, preventing resource waste caused by arbitrary **Pod** scheduling. The specific algorithm observes whether the number of already-scheduled **Pods** under a **Job** meets the minimum running count; when the minimum running count of the **Job** is met, scheduling actions are performed for all **Pods** under the **Job**, otherwise no action is taken.

Creating a workload that uses the **Gang** scheduling capability

1. First create a **PodGroup**, specifying **minMember** and **minResources**:

   **minMember**: A group of **Pods** belonging to the current **PodGroup** is scheduled together only when the **minMember** count is met.

   **minResources**: The group of **Pods** is scheduled together only when the cluster's free resources meet the **minResources** requirement.

2. When creating the workload, specify the **Volcano** scheduler via **schedulerName**, and specify its **PodGroup** via **annotation**:

   **schedulerName**: Set to **volcano**, indicating that **Volcano** is used to schedule the workload.

   **scheduling.k8s.io/group-name**: Specify the **PodGroup** created in the previous step; the example is **pg-test1**.
`;

const binPackZhCN = `
# 装箱策略（binpack）

装箱调度（**Binpack**）是一种优化算法，以最小化资源使用量为目标，将资源合理地分配给每个任务，使所有资源都可以实现最大化的利用价值。在集群工作负载的调度过程中使用**Binpack**调度策略，调度器会优先将**Pod**调度到资源消耗较多的节点，减少各节点空闲资源碎片，提高集群资源利用率。

​	**Binpack**在对一个节点打分时，会根据**Binpack**插件自身权重和各资源设置的权重值综合打分。首先，对**Pod**请求资源中的每类资源依次打分，以 CPU 为例，CPU 资源在待调度节点的得分信息如下：

​	**CPU.score = CPU.weight \* (request + used) / allocatable**

通过 Binpack 策略的节点总得分如下：

​	**Score = binpack.weight \* (CPU.score + Memory.score) / (CPU.weight+ Memory.weight) \* 100**

点击按钮即可打开/关闭装箱策略。
`;

const binPackEnUS = `
# Binpack Policy

Binpack scheduling (**Binpack**) is an optimization algorithm aimed at minimizing resource usage, reasonably allocating resources to each task so that all resources can achieve maximum utilization value. When the **Binpack** scheduling policy is used during cluster workload scheduling, the scheduler preferentially schedules **Pods** onto nodes with higher resource consumption, reducing idle resource fragmentation across nodes and improving cluster resource utilization.

​	When **Binpack** scores a node, it combines the **Binpack** plugin's own weight and the weight values configured for each resource. First, it scores each resource type in the **Pod**'s requested resources. Taking CPU as an example, the CPU resource score on the node to be scheduled is as follows:

​	**CPU.score = CPU.weight \* (request + used) / allocatable**

The total node score under the Binpack policy is as follows:

​	**Score = binpack.weight \* (CPU.score + Memory.score) / (CPU.weight+ Memory.weight) \* 100**

Click the button to enable/disable the binpack policy.
`;

const priorityZhCN = `
# 优先级调度与抢占（priority）

优先级表示一个作业相对于其他作业的重要性，**Volcano**兼容**Kubernetes**中的**Pod**优先级定义（[PriorityClass](https://kubernetes.io/zh-cn/docs/concepts/scheduling-eviction/pod-priority-preemption/#priorityclass)）。启用该能力后，调度器将优先保障高优先级业务调度。集群资源不足时，调度器主动驱逐低优先级业务，保障调度高优先级业务可以正常调度。

### 优先级调度与抢占策略介绍

用户在集群中运行的业务丰富多样，包括核心业务、非核心业务，在线业务、离线业务等，根据业务的重要程度和SLA要求，可以对不同业务类型设置相应的高优先级。比如对核心业务和在线业务设置高优先级，可以保证该类业务优先获取集群资源。当集群资源被非核心业务占用，整体资源不足时，如果有新的核心业务提交部署请求，可以通过抢占的方式驱逐部分非核心业务，释放集群资源用于核心业务的调度运行。

配置完成后，可以在工作负载或Volcano Job中使用优先级定义（[PriorityClass](https://kubernetes.io/zh-cn/docs/concepts/scheduling-eviction/pod-priority-preemption/#priorityclass)）进行优先级调度。

1. 创建一个或多个优先级定义（**PriorityClass**）
2. 创建工作负载或Volcano Job，并指定priorityClassName。
`;

const priorityEnUS = `
# Priority Scheduling and Preemption (priority)

Priority indicates the importance of a job relative to other jobs. **Volcano** is compatible with the **Pod** priority definition in **Kubernetes** ([PriorityClass](https://kubernetes.io/zh-cn/docs/concepts/scheduling-eviction/pod-priority-preemption/#priorityclass)). Once enabled, the scheduler prioritizes the scheduling of high-priority workloads. When cluster resources are insufficient, the scheduler actively evicts low-priority workloads to ensure high-priority workloads can be scheduled normally.

### Priority Scheduling and Preemption Policy Overview

The workloads running in a user's cluster are diverse, including core and non-core workloads, online and offline workloads, etc. Based on the importance of the workload and SLA requirements, corresponding high priorities can be set for different workload types. For example, setting high priority for core and online workloads ensures that these workloads obtain cluster resources first. When cluster resources are occupied by non-core workloads and overall resources are insufficient, if a new core workload submits a deployment request, some non-core workloads can be evicted through preemption to release cluster resources for scheduling and running core workloads.

After configuration, you can use priority definitions ([PriorityClass](https://kubernetes.io/zh-cn/docs/concepts/scheduling-eviction/pod-priority-preemption/#priorityclass)) for priority scheduling in workloads or Volcano Jobs.

1. Create one or more priority definitions (**PriorityClass**)
2. Create a workload or Volcano Job, and specify priorityClassName.
`;

const drfZhCN = `
# 公平调度（DRF）

**DRF**（**Dominant Resource Fairness**）是主资源公平调度策略，应用于大批量提交AI训练和大数据作业的场景，可增强集群业务的吞吐量，整体缩短业务执行时间，提高训练性能。

#### 公平调度介绍

在实际业务中，经常会遇到将集群稀缺资源分配给多个用户的情况，每个用户获得资源的权利都相同，但是需求数却可能不同，如何公平的将资源分配给每个用户是一项非常有意义的事情。调度层面有一种常用的方法为最大最小化公平分配算法（max-min fairness share），尽量满足用户中的最小的需求，然后将剩余的资源公平分配给剩下的用户。形式化定义如下：

1. 资源分配以需求递增的方式进行分配
2. 每个用户获得的资源不超过其需求
3. 未得到满足的用户等价平分剩下的资源

max-min fairness算法的最大问题是认为资源是单一的，但是现实情况中资源却不是单一的，例如CPU、Memory、GPU等资源在分配时都需要考虑。这个时候DRF应运而生，简单来说DRF就是 max-min fairness 算法的泛化版本，可以支持多种类型资源的公平分配， 即每个用户的主资源满足 max-min fairness 要求。

每个Job资源的Share值计算如下：

**Share = Total Request / Cluster Resources**

当Job具有多个资源时，将Share值最大的资源作为主资源，在进行优先级调度时，仅根据主资源的Share值进行优先级调度。
`;

const drfEnUS = `
# Fair Scheduling (DRF)

**DRF** (**Dominant Resource Fairness**) is a dominant-resource fair scheduling policy. It is applied in scenarios where AI training and big data jobs are submitted in large batches, and can enhance cluster workload throughput, shorten overall workload execution time, and improve training performance.

#### Fair Scheduling Overview

In real-world workloads, it is common to allocate scarce cluster resources to multiple users. Each user has the same right to obtain resources, but the demanded quantities may differ. Fairly allocating resources to each user is a very meaningful task. At the scheduling level, a commonly used method is the max-min fairness share algorithm, which satisfies the smallest demand among users as much as possible, and then fairly allocates the remaining resources to the remaining users. The formal definition is as follows:

1. Resource allocation proceeds in order of increasing demand
2. No user receives more resources than their demand
3. Unsatisfied users equally share the remaining resources

The biggest issue with the max-min fairness algorithm is the assumption that resources are of a single type. In reality, however, resources are not single, e.g., CPU, Memory, and GPU all need to be considered during allocation. DRF emerged in this context — simply put, DRF is a generalized version of the max-min fairness algorithm that supports fair allocation of multiple resource types, i.e., each user's dominant resource satisfies the max-min fairness requirement.

The Share value of each Job's resources is calculated as follows:

**Share = Total Request / Cluster Resources**

When a Job has multiple resources, the resource with the largest Share value is treated as the dominant resource. During priority scheduling, priority scheduling is performed based solely on the Share value of the dominant resource.
`;

const descVolcanoZhCN = `
# 修改kubelet配置文件
1. 在服务器上进入/var/lib/kubelet/config.yaml文件，增加如下内容，以修改拓扑策略为single-numa-node为例:
\`\`\`
topologyManagerPolicy: single-numa-node
cpuManagerPolicy: static
memoryManagerPolicy: None
systemReserved:
  cpu: "0.5"
  memory: "500Mi"
\`\`\`
2. 删除/var/lib/kubelet/cpu_manager_state文件
3. 重启kubelet：
\`\`\`
systemctl daemon-reload
systemctl restart kubelet
systemctl status kubelet
\`\`\`

# 使用Volcano调度工作负载

使用Volcano调度工作负载时，只需要在Pod的spec字段中设置schedulerName参数并指定参数值为volcano，示例如下：

\`\`\`
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nginx
  labels:
    app: nginx
spec:
  replicas: 4
  selector:
    matchLabels:
      app: nginx
  template:
    metadata:
      labels:
        app: nginx
    spec:
      # 指定调度器为Volcano
      schedulerName: volcano
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
        ports:
        - containerPort: 80
\`\`\`

同时，Volcano还支持设置负载所属队列和抢占属性等，可通过Pod的注解实现。目前Volcano支持的Pod注解配置如下：

| Pod注解                                           | 说明                                                         |
| ------------------------------------------------- | ------------------------------------------------------------ |
| scheduling.volcano.sh/queue-name: "<queue-name>"  | 指定负载所在队列，其中<queue-name>为队列名称。               |
| volcano.sh/preemptable: "true"                    | 表示作业是否可抢占。开启后，认为该作业可以被抢占。取值范围：true：开启抢占。（默认为开启状态）false：关闭抢占。 |
| volcano.sh/numa-topology-policy: <topologyPolicy> | 指定负载NUMA亲和的节点的拓扑策略，其中<topologyPolicy>为节点的拓扑策略名称。 |

可通过查询Pod详情查看Pod是否由Volcano调度，以及被分配的队列：

\`\`\`
kubectl describe pod <pod_name>
\`\`\`

回显如下：

\`\`\`
Spec:
  Min Member:  1
  Min Resources:
    Cpu:     100m
    Memory:  100Mi
  Queue:     q1
Status:
  Conditions:
    Last Transition Time:  2024-10-30T01:54:43Z
    Reason:                tasks in gang are ready to be scheduled
    Status:                True
    Transition ID:         70be1d7d-3532-41e0-8324-c7644026b38f
    Type:                  Scheduled
  Phase:                   Running
Events:
  Type    Reason     Age              From     Message
  ----    ------     ----             ----     -------
  Normal  Scheduled  0s (x3 over 2s)  volcano  pod group is ready
\`\`\`
`;

const descVolcanoEnUS = `
# Modifying the kubelet Configuration File
1. On the server, open the /var/lib/kubelet/config.yaml file and add the following content. Taking modifying the topology policy to single-numa-node as an example:
\`\`\`
topologyManagerPolicy: single-numa-node
cpuManagerPolicy: static
memoryManagerPolicy: None
systemReserved:
  cpu: "0.5"
  memory: "500Mi"
\`\`\`
2. Delete the /var/lib/kubelet/cpu_manager_state file
3. Restart kubelet:
\`\`\`
systemctl daemon-reload
systemctl restart kubelet
systemctl status kubelet
\`\`\`

# Scheduling Workloads with Volcano

When using Volcano to schedule workloads, you only need to set the schedulerName parameter in the Pod's spec field and specify the value as volcano. An example is as follows:

\`\`\`
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nginx
  labels:
    app: nginx
spec:
  replicas: 4
  selector:
    matchLabels:
      app: nginx
  template:
    metadata:
      labels:
        app: nginx
    spec:
      # Specify Volcano as the scheduler
      schedulerName: volcano
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
        ports:
        - containerPort: 80
\`\`\`

In addition, Volcano supports setting the workload's queue and preemption attributes via Pod annotations. The Pod annotations currently supported by Volcano are as follows:

| Pod Annotation | Description |
| ------------------------------------------------- | ------------------------------------------------------------ |
| scheduling.volcano.sh/queue-name: "<queue-name>"  | Specifies the queue of the workload, where <queue-name> is the queue name.               |
| volcano.sh/preemptable: "true"                    | Indicates whether the job can be preempted. When enabled, the job is considered preemptable. Values: true: enable preemption (enabled by default). false: disable preemption. |
| volcano.sh/numa-topology-policy: <topologyPolicy> | Specifies the topology policy of the NUMA-affinity nodes for the workload, where <topologyPolicy> is the name of the node topology policy. |

You can query the Pod details to check whether the Pod is scheduled by Volcano and which queue it is assigned to:

\`\`\`
kubectl describe pod <pod_name>
\`\`\`

The output is as follows:

\`\`\`
Spec:
  Min Member:  1
  Min Resources:
    Cpu:     100m
    Memory:  100Mi
  Queue:     q1
Status:
  Conditions:
    Last Transition Time:  2024-10-30T01:54:43Z
    Reason:                tasks in gang are ready to be scheduled
    Status:                True
    Transition ID:         70be1d7d-3532-41e0-8324-c7644026b38f
    Type:                  Scheduled
  Phase:                   Running
Events:
  Type    Reason     Age              From     Message
  ----    ------     ----             ----     -------
  Normal  Scheduled  0s (x3 over 2s)  volcano  pod group is ready
\`\`\`
`;

const environmnetPreparationZhCN = `
# 环境准备

## NUMA亲和性调度+最优NUMA Distance：

修改kubernetes配置文件：

\`\`\`
vi /var/lib/kubelet/config.yaml
\`\`\`

加入以下内容：

\`\`\`
cpuManagerPolicy: static
kubeReserved:
  cpu: 500m
systemReserved:
  cpu: 500m
\`\`\`

并根据需求修改节点的拓扑策略：

\`\`\`
topologyManagerPolicy: single-numa-node
\`\`\`

删除cpu_manager_state文件：

\`\`\`
rm /var/lib/kubelet/cpu_manager_state
\`\`\`

重启kubelet：

\`\`\`
systemctl start kubelet
\`\`\`

查看kubelet状态：

\`\`\`
systemctl status kubelet
\`\`\`

如果kubelet状态为running则表示修改成功。

### 运行时POD亲和优化

修改containerd的配置文件：

\`\`\`
vi /etc/containerd/config.toml
\`\`\`

将其中的内容修改：

\`\`\`
      [plugins."io.containerd.nri.v1.nri"]
      disable = false
\`\`\`

修改操作系统cmdline参数：

\`\`\`
vi /etc/grub2-efi.cfg
\`\`\`

在当前操作系统镜像对应的linux参数后添加：

\`\`\`
mem_sampling_on kpti=off numa_icon=enable
\`\`\`

修改crictl配置：

\`\`\`
vi /etc/crictl.yaml
\`\`\`

修改为：

\`\`\`
runtime-endpoint: unix:///run/containerd/containerd.sock
\`\`\`

在net_rship_no_system文件目录下执行：

\`\`\`
chmod 777 net_rship_no_system
./net_rship_no_system -l -I 6000 -i 3000
\`\`\`

此步骤会打开autonuma插件。
`;

const environmnetPreparationEnUS = `
# Environment Preparation

## NUMA Affinity Scheduling + Optimal NUMA Distance:

Modify the Kubernetes configuration file:

\`\`\`
vi /var/lib/kubelet/config.yaml
\`\`\`

Add the following content:

\`\`\`
cpuManagerPolicy: static
kubeReserved:
  cpu: 500m
systemReserved:
  cpu: 500m
\`\`\`

And modify the node's topology policy as needed:

\`\`\`
topologyManagerPolicy: single-numa-node
\`\`\`

Delete the cpu_manager_state file:

\`\`\`
rm /var/lib/kubelet/cpu_manager_state
\`\`\`

Restart kubelet:

\`\`\`
systemctl start kubelet
\`\`\`

Check the kubelet status:

\`\`\`
systemctl status kubelet
\`\`\`

If the kubelet status is running, the modification is successful.

### Runtime Pod Affinity Optimization

Modify the containerd configuration file:

\`\`\`
vi /etc/containerd/config.toml
\`\`\`

Modify the content as follows:

\`\`\`
      [plugins."io.containerd.nri.v1.nri"]
      disable = false
\`\`\`

Modify the OS cmdline parameters:

\`\`\`
vi /etc/grub2-efi.cfg
\`\`\`

Append the following after the linux parameters of the current OS image:

\`\`\`
mem_sampling_on kpti=off numa_icon=enable
\`\`\`

Modify the crictl configuration:

\`\`\`
vi /etc/crictl.yaml
\`\`\`

Modify to:

\`\`\`
runtime-endpoint: unix:///run/containerd/containerd.sock
\`\`\`

Execute in the directory of the net_rship_no_system file:

\`\`\`
chmod 777 net_rship_no_system
./net_rship_no_system -l -I 6000 -i 3000
\`\`\`

This step enables the autonuma plugin.
`;

const workloadZhCN = `
# 工作负载部署

对于需要使用NUMA亲和性调度的工作负载，需要在spec.template.spec.schedulerName中指定调度器为Volcano。在spec.template.metadata.annotations中配置NUMA亲和性。

\`\`\`
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
        # 指定numa亲和性
        volcano.sh/numa-topology-policy: single-numa-node
    spec:
      # 指定调度器为Volcano
      schedulerName: volcano
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
\`\`\`

对于需要使用运行时POD亲和优化的工作负载，则不能将pod的QOS类别配置为guranteed类型。
`;

const workloadEnUS = `
# Workload Deployment

For workloads that need to use NUMA affinity scheduling, specify Volcano as the scheduler in spec.template.spec.schedulerName. Configure NUMA affinity in spec.template.metadata.annotations.

\`\`\`
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
        # Specify NUMA affinity
        volcano.sh/numa-topology-policy: single-numa-node
    spec:
      # Specify Volcano as the scheduler
      schedulerName: volcano
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
\`\`\`

For workloads that need to use runtime Pod affinity optimization, the Pod's QoS class must not be configured as guaranteed.
`;

const distanceZhCN = `
# 最优NUMA Distance

Kubernetes的kubelet组件存在一个topologyManager模块，通过协调cpuManager、memoryManager和deviceManager模块对节点上的cpu内存和设备进行统筹调度，通过一系列的算法和用户配置，在创建Pod时尽可能地将硬件资源（CPU/内存/设备）分配在同一个NUMA NODE内。在跨NUMA分配的场景，本策略总会选择distance总和最小的hits作为可选分配方案集合。使用时需要完成环境配置。

以一个简单的资源空闲状态为例：

|         | NUMA0 | NUMA1 | NUMA2 |
| ------- | ----- | ----- | ----- |
| MEM(G)  | 1     | 1     | 2     |
| CPU(核) | 1     | 8     | 10    |

上表表示当前节点上的空闲资源，当系统申请一个Pod资源诉求为CPU核数 3，内存 3G时具体的算法如下：

1、获取各个各个节点的资源情况并使用一个对象保存，该对象简称为hit，该例子中存在3个NUMA-NODE，那么hit对象的长度为3位，每一位上值为1时，表示该组合需要使用到此NUMA节点，如满足3核CPU的一种组合是NUMA0+NUMA1，那么此时hit值为[011]，由于本发明不涉及topologyManager的是否跨NUMA策略，单NUMA无法满足pod诉求且存在只跨两个NUMA节点的就能满足诉的选择，所以简化了策略算法删除单节点及三节点的hits，以此类推，获取以下的hits

| MEM  | [101] | [011] |       |
| ---- | ----- | ----- | ----- |
| CPU  | [110] | [101] | [011] |

2、通过以上的hits，将MEM、CPU组合计算，可以得到一共6种hits组合，如下表，当二进制一致时，表示资源匹配，即该这组合是可用的：

| ID       | 1     | 2     | 3     | 4     | 5     | 6     |
| -------- | ----- | ----- | ----- | ----- | ----- | ----- |
| MEM      | [101] | [101] | [101] | [011] | [011] | [011] |
| CPU      | [110] | [101] | [011] | [110] | [101] | [011] |
| 是否可用 | False | True  | False | False | False | True  |

Hits组合

3、计算可得当前可用的hits：[101],[011]，按现有k8s实现代码，当存在多个hits时会选择掩码值最小的hit作为资源分配方案，即hit：[011]。那么此Pod最终跨NUMA0(1U1G)、NUMA1(2U2G)部署。

原生方案默认选择掩码值最小的hit(NUMA0,NUMA1)预留为容器资源，未考虑NUMA间访问的时延问题。

4、本方案在多个hit中选择时，选择distance总和最小的hits作为可选分配方案集合。

| distance | NUMA0 | NUMA1 | NUMA2 |
| -------- | ----- | ----- | ----- |
| NUMA0    | 10    | 30    | 20    |
| NUMA1    | 30    | 10    | 30    |
| NUMA2    | 20    | 30    | 10    |

以本场景为例，NUMA0与NUMA2的距离小于NUMA0与NUMA1的距离，所以选用hit(NUMA1,NUMA2)才是最优解。
提示：本策略仅支持kubelet v1.28.0及以上版本，若版本不支持可通过以下链接下载kubelet并替换：

\`\`\`
https://openfuyao.obs.cn-north-4.myhuaweicloud.com/kubernetes/kubernetes/releases/download/v1.28.8/bin/linux/amd64/kubelet
https://openfuyao.obs.cn-north-4.myhuaweicloud.com/kubernetes/kubernetes/releases/download/v1.28.8/bin/linux/arm64/kubelet
\`\`\`
将以上链接中下载的kubelet二进制替换原有的kubelet，kubelet二进制路径：

\`\`\`
/usr/local/bin
\`\`\`
替换后重启kubelet：
\`\`\`
systemctl restart kubelet
\`\`\`
`;

const distanceEnUS = `
# Optimal NUMA Distance

The kubelet component of Kubernetes contains a topologyManager module that coordinates the cpuManager, memoryManager, and deviceManager modules to jointly schedule the CPU, memory, and devices on a node. Through a series of algorithms and user configuration, it allocates hardware resources (CPU/memory/devices) within the same NUMA NODE as much as possible when creating a Pod. In cross-NUMA allocation scenarios, this policy always selects the hits with the smallest total distance as the set of candidate allocation schemes. Environment configuration must be completed before use.

Taking a simple idle resource state as an example:

|         | NUMA0 | NUMA1 | NUMA2 |
| ------- | ----- | ----- | ----- |
| MEM(G)  | 1     | 1     | 2     |
| CPU(cores) | 1     | 8     | 10    |

The table above shows the idle resources on the current node. When the system requests a Pod with a resource demand of 3 CPU cores and 3 GB memory, the specific algorithm is as follows:

1. Retrieve the resource status of each node and store it in an object, abbreviated as hit. In this example, there are 3 NUMA-NODEs, so the hit object has a length of 3 bits. When a bit value is 1, it indicates that the combination requires using this NUMA node. For example, a combination that satisfies 3 CPU cores is NUMA0+NUMA1, so the hit value is [011]. Since this invention does not involve the topologyManager's cross-NUMA policy, and when a single NUMA cannot meet the Pod's demand and there is an option that only spans two NUMA nodes to satisfy the request, the policy algorithm is simplified by removing single-node and three-node hits. By analogy, the following hits are obtained:

| MEM  | [101] | [011] |       |
| ---- | ----- | ----- | ----- |
| CPU  | [110] | [101] | [011] |

2. Using the hits above, by computing the MEM and CPU combinations, there are a total of 6 hit combinations, as shown in the table below. When the binary values match, it indicates a resource match, i.e., the combination is available:

| ID       | 1     | 2     | 3     | 4     | 5     | 6     |
| -------- | ----- | ----- | ----- | ----- | ----- | ----- |
| MEM      | [101] | [101] | [101] | [011] | [011] | [011] |
| CPU      | [110] | [101] | [011] | [110] | [101] | [011] |
| Available | False | True  | False | False | False | True  |

Hits combinations

3. The calculation shows that the currently available hits are [101] and [011]. According to the existing k8s implementation, when multiple hits exist, the hit with the smallest mask value is selected as the resource allocation scheme, i.e., hit [011]. Therefore, this Pod is ultimately deployed across NUMA0 (1U1G) and NUMA1 (2U2G).

The native scheme by default selects the hit with the smallest mask value (NUMA0, NUMA1) to reserve container resources, without considering the inter-NUMA access latency.

4. When selecting among multiple hits, this scheme selects the hits with the smallest total distance as the set of candidate allocation schemes.

| distance | NUMA0 | NUMA1 | NUMA2 |
| -------- | ----- | ----- | ----- |
| NUMA0    | 10    | 30    | 20    |
| NUMA1    | 30    | 10    | 30    |
| NUMA2    | 20    | 30    | 10    |

Taking this scenario as an example, the distance between NUMA0 and NUMA2 is smaller than the distance between NUMA0 and NUMA1, so selecting hit (NUMA1, NUMA2) is the optimal solution.
Note: This policy only supports kubelet v1.28.0 and later. If the version is not supported, you can download kubelet via the following links and replace it:

\`\`\`
https://openfuyao.obs.cn-north-4.myhuaweicloud.com/kubernetes/kubernetes/releases/download/v1.28.8/bin/linux/amd64/kubelet
https://openfuyao.obs.cn-north-4.myhuaweicloud.com/kubernetes/kubernetes/releases/download/v1.28.8/bin/linux/arm64/kubelet
\`\`\`
Replace the original kubelet binary with the kubelet binary downloaded from the links above. The kubelet binary path is:

\`\`\`
/usr/local/bin
\`\`\`
After replacement, restart kubelet:
\`\`\`
systemctl restart kubelet
\`\`\`
`;

const runningPodZhCN = `
# 运行时POD亲和优化

通过resource-aware模块统计线程亲和性，进而转化为pod亲和性，利用NRI，感知resource-aware并存储pod间亲和关系，修改pod的cpuset/memset，从而将亲和性Pod的资源分配至同一NUMA NODE或距离更近的NUMA NODE中。使用时需要完成环境配置。

1、operator扫描节点上需要纳管的Pod信息，并将这些信息传递到host内核中的亲和性感知模块，判断这些Pod是否存在亲和性关系。

2、若Pod列表中存在亲和性关系，且部署的NUMA distance较远的，则operator调用该Pod的更新接口，触发注册的NRI插件的update。

3、NRI插件的update接口，通过从operator中获取的亲和性关系，调整该Pod的cpuset/memset参数，将其部署距离亲和Pod 的NUMA distance更近的NUMA 位置。

4、调整成功后则动态调优流程结束。
注意：本策略仅支持 arm + openeuler22.03 sp4 环境，且需要完成以下配置：
修改containerd的配置文件：

\`\`\`
vi /etc/containerd/config.toml
\`\`\`
将其中的内容修改：
\`\`\`
    [plugins."io.containerd.nri.v1.nri"]
    disable = false
\`\`\`
`;

const runningPodEnUS = `
# Runtime Pod Affinity Optimization

The resource-aware module collects thread affinity and converts it into Pod affinity. Using NRI, it perceives resource-aware and stores inter-Pod affinity relationships, and modifies the Pod's cpuset/memset, thereby allocating the resources of affinity Pods to the same NUMA NODE or to NUMA NODEs that are closer in distance. Environment configuration must be completed before use.

1. The operator scans the Pod information on the node that needs to be managed, and passes this information to the affinity perception module in the host kernel to determine whether these Pods have affinity relationships.

2. If affinity relationships exist in the Pod list and the deployed NUMA distance is large, the operator calls the Pod's update interface to trigger the update of the registered NRI plugin.

3. The NRI plugin's update interface, using the affinity relationships obtained from the operator, adjusts the Pod's cpuset/memset parameters to deploy it to a NUMA location with a closer NUMA distance to the affinity Pod.

4. After successful adjustment, the dynamic tuning process ends.
Note: This policy only supports the arm + openeuler22.03 sp4 environment, and the following configuration must be completed:
Modify the containerd configuration file:

\`\`\`
vi /etc/containerd/config.toml
\`\`\`
Modify the content as follows:
\`\`\`
    [plugins."io.containerd.nri.v1.nri"]
    disable = false
\`\`\`
`;

const newEnvPrepareZhCN = `
# 运行时Pod亲和优化

修改containerd的配置文件：

\`\`\`
vi /etc/containerd/config.toml
\`\`\`

将其中的内容修改：

\`\`\`
[plugins."io.containerd.nri.v1.nri"]
disable = false
\`\`\`
`;

const newEnvPrepareEnUS = `
# Runtime Pod Affinity Optimization

Modify the containerd configuration file:

\`\`\`
vi /etc/containerd/config.toml
\`\`\`

Modify the content as follows:

\`\`\`
[plugins."io.containerd.nri.v1.nri"]
disable = false
\`\`\`
`;

export const getVolcano = () => {
  const locale = getLocale();
  return locale === 'en-US' ? volcanoEnUS : volcanoZhCN;
};

export const getGang = () => {
  const locale = getLocale();
  return locale === 'en-US' ? gangEnUS : gangZhCN;
};

export const getBinPack = () => {
  const locale = getLocale();
  return locale === 'en-US' ? binPackEnUS : binPackZhCN;
};

export const getPriority = () => {
  const locale = getLocale();
  return locale === 'en-US' ? priorityEnUS : priorityZhCN;
};

export const getDrf = () => {
  const locale = getLocale();
  return locale === 'en-US' ? drfEnUS : drfZhCN;
};

export const getDescVolcano = () => {
  const locale = getLocale();
  return locale === 'en-US' ? descVolcanoEnUS : descVolcanoZhCN;
};

export const getEnvironmnetPreparation = () => {
  const locale = getLocale();
  return locale === 'en-US' ? environmnetPreparationEnUS : environmnetPreparationZhCN;
};

export const getWorkload = () => {
  const locale = getLocale();
  return locale === 'en-US' ? workloadEnUS : workloadZhCN;
};

export const getDistance = () => {
  const locale = getLocale();
  return locale === 'en-US' ? distanceEnUS : distanceZhCN;
};

export const getRunningPod = () => {
  const locale = getLocale();
  return locale === 'en-US' ? runningPodEnUS : runningPodZhCN;
};

export const getNewEnvPrepare = () => {
  const locale = getLocale();
  return locale === 'en-US' ? newEnvPrepareEnUS : newEnvPrepareZhCN;
};

// Keep the original exports for backward compatibility
export const volcano = volcanoZhCN;
export const gang = gangZhCN;
export const binPack = binPackZhCN;
export const priority = priorityZhCN;
export const drf = drfZhCN;
export const descVolcano = descVolcanoZhCN;
export const environmnetPreparation = environmnetPreparationZhCN;
export const workload = workloadZhCN;
export const distance = distanceZhCN;
export const runningPod = runningPodZhCN;
export const newEnvPrepare = newEnvPrepareZhCN;
