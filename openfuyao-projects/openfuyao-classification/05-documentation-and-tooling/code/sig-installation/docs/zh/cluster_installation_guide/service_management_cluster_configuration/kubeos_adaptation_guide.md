# KubeOS自定义配置

## 背景信息

传统节点OS升级依赖包管理器逐包更新，容易导致环境漂移。KubeOS是openEuler针对Kubernetes worker节点场景设计的不可变OS，采用只读rootfs、A/B root分区和持久化PERSIST分区，以整镜像方式原子升级和回滚，确保节点OS状态一致可复现。

openFuyao支持普通openEuler master与KubeOS worker VM共存的混合OS集群形态。当用户需要worker节点具备不可变OS能力、实现OS原子升级和回滚时，可参照本指导完成KubeOS适配部署。集群安装完成后，通过KubeOS addon接管worker节点的OS生命周期管理（升级/回滚/配置）。

**表 1** 普通openEuler与KubeOS对比

| 对比项 | 普通openEuler | KubeOS |
| ---- | ------------- | ------ |
| rootfs | 可读写 | 只读 |
| OS升级方式 | 包管理器逐包更新 | 整镜像原子升级/回滚 |
| 分区模型 | 常规分区 | A/B root + PERSIST持久分区 |
| 二进制安装 | 运行态可安装 | 镜像制作阶段内置 |
| 适用角色 | master/worker | 仅worker |

当前阶段部署形态为普通openEuler master + KubeOS VM worker，不支持master运行KubeOS，不支持物理机KubeOS worker。该部署形态支持小、中、大规模集群，worker数量受虚拟化资源池和集群规模限制。

## 前提条件

- 已完成引导节点初始化，引导集群运行正常。引导节点初始化方法请参见[安装引导集群](../bootstrap_cluster_configuration)。
- 已准备普通openEuler的master节点。
- 已构建KubeOS VM系统盘镜像`system.qcow2`或`system.img`，镜像内置containerd、kubelet、bke-agent、os-agent等运行依赖。构建方法请参见[KubeOS镜像构建手册](https://gitcode.com/openFuyao/kubeos-support/blob/master/docs/KubeOS镜像构建手册.md)。
- 已准备KubeOS addon制品（os-operator镜像、os-proxy镜像、Helm chart），制品来源请参见[kubeos-support仓库](https://gitcode.com/openFuyao/kubeos-support)。
- 虚拟化资源池支持从qcow2/raw系统盘创建VM。

> ![](../figures/icon-note.gif)**说明**：
>
> 下文中bke-cluster.yaml和bke-node.yaml是引导节点上用于配置集群安装信息的文件，默认位于`/bke/cluster/`目录下。

## 使用限制

- master节点不支持运行KubeOS，不支持容器OS升级。
- worker节点必须通过KubeOS VM镜像创建，不接受普通openEuler节点加入KubeOS worker池。
- 当前阶段不支持物理机KubeOS worker，PXE（Preboot eXecution Environment）物理机安装作为后续扩展单独设计。
- 不支持普通openEuler worker在线转换为KubeOS worker。
- KubeOS worker VM必须使用独立系统盘，不能多个VM共享同一个`system.qcow2`系统盘。
- master节点不安装os-agent，也不挂载`/var/run/os-agent`。

## 操作步骤

### 创建KubeOS worker VM

1. 将KubeOS `system.qcow2`或`system.img`导入虚拟化平台。

2. 通过FC（FusionCompute）或QEMU创建多台worker VM，为每台VM分配唯一系统盘、MAC、hostname和网络配置。

   > ![](../figures/icon-notice.gif)**注意**：
   >
   > `system.qcow2`应作为模板镜像，IP、hostname、machine-id需动态配置，不能写为固定值。每个worker必须克隆独立系统盘。

3. 确认worker VM首启后网络和主机名配置完成，且以下挂载已生效：

   ```bash
   # 确认持久化分区已挂载
   mount | grep persist
   # 确认/etc为overlay可写
   mount | grep '/etc'
   # 确认/var为bind mount
   mount | grep '/var'
   # 确认/opt/cni为overlay可写
   mount | grep '/opt/cni'
   ```

### 配置集群安装参数

1. 在bke-cluster.yaml中配置集群信息。

   集群配置与普通集群一致，KubeOS适配无需在bke-cluster.yaml中额外配置参数，通过安装命令的`--immutable-os`参数触发。

2. 在bke-node.yaml中配置节点信息。

   master节点role配置为`master/node`和`etcd`，worker节点role配置为`node`，IP和hostname需与实际KubeOS VM一致。以下仅展示KubeOS worker节点配置，master节点配置与普通集群一致。

   ```yaml
   apiVersion: bke.bocloud.com/v1beta1
   kind: BKENode
   metadata:
     labels:
       cluster.x-k8s.io/cluster-name: bke-cluster
     name: bke-cluster-n1
     namespace: bke-cluster
   spec:
     hostname: n1
     ip: 192.168.201.51
     port: "22"
     role:
     - node
     username: root
     password: 'your-password'
   ```

### 启动集群安装

在引导节点或管理集群节点上执行集群创建命令，通过`--immutable-os=kubeos`参数声明本集群的worker节点使用KubeOS镜像：

```bash
bke cluster create -f /bke/cluster/bke-cluster.yaml -n /bke/cluster/bke-node.yaml --immutable-os=kubeos
```

安装器在BKECluster CR上注入`openfuyao.io/immutable-os=kubeos` annotation，后续安装逻辑根据此标志走KubeOS适配逻辑。集群安装过程中的状态变化和判断方法与普通集群一致。

> ![](../figures/icon-note.gif)**说明**：
>
> `--immutable-os`参数仅对worker节点生效，master节点始终走普通openEuler安装流程。

### 安装适配说明

安装流程检测到KubeOS worker时，对worker节点的安装逻辑进行如下适配：

| 组件 | 普通OS流程 | KubeOS适配流程 |
| ---- | ---------- | -------------- |
| containerd | 下载解压二进制，写入service文件 | 跳过下载（镜像内置），仅渲染`/etc/containerd/config.toml`等配置。 |
| kubelet | 下载二进制，写入service文件 | 跳过下载（镜像内置），仅生成bootstrap配置、kubeconfig等可变配置。 |
| bke-agent | SSH推送二进制到`/usr/local/bin` | 跳过推送（镜像内置），仅下发配置、证书、kubeconfig。 |
| kubectl | 下载并分发到worker | 跳过安装。 |
| CNI Calico | 写入`/opt/cni/bin`和`/etc/cni/net.d` | 保持不变（`/opt/cni`和`/etc`在KubeOS上可写）。 |

worker加入集群后自动打以下标签：

```yaml
node.openfuyao.io/os: kubeos
node.openfuyao.io/role: worker
upgrade.openeuler.org/node-selector: default-worker-pool
```

### 安装KubeOS addon

KubeOS addon需在bke-cluster.yaml中预先声明，集群创建时自动安装。addon包含OS/OSInstance CRD、RBAC、os-operator（Deployment，调度到control-plane）和os-proxy（DaemonSet，仅调度到KubeOS worker），用于接管worker节点的OS生命周期管理。

1. 配置chart repo。

   在bke-cluster.yaml的`spec.clusterConfig.cluster.chartRepo`中声明chart仓库地址：

   ```yaml
   spec:
     clusterConfig:
       cluster:
         chartRepo:
           domain: cr.openfuyao.cn
           port: "443"
           prefix: charts
   ```

   > ![](../figures/icon-note.gif)**说明**：
   >
   > chart仓库认证和TLS配置方法请参见[addon使用指导](./using_add-ons.md#chart形态addon)。

2. 添加KubeOS addon配置。

   在`spec.clusterConfig.addons`中添加：

   ```yaml
   spec:
     clusterConfig:
       addons:
       - name: kubeos-addon
         type: chart
         version: 0.0.0-latest
         releaseName: kubeos-addon
         namespace: upgrade-system
         block: true
   ```

   > ![](../figures/icon-note.gif)**说明**：
   >
   > `block: true`表示阻塞等待该addon安装完成后才继续后续流程。KubeOS addon建议设置为`true`，确保OS管理组件就绪后再进行后续操作。

3. 配置自定义values.yaml（可选）。

   如不配置valuesConfigMapRef，addon将使用chart包默认的values.yaml，镜像默认从`global.imageRegistry`指定的仓库拉取。如需自定义镜像仓库地址或版本，通过valuesConfigMapRef指定。先在管理集群创建ConfigMap：

   ```yaml
   apiVersion: v1
   kind: ConfigMap
   metadata:
     name: kubeos-addon-values
     namespace: bke-cluster
   data:
     values.yaml: |
       global:
         imageRegistry: "cr.openfuyao.cn/openfuyao"
         imageNamespace: "kubeos"
       operator:
         image:
           repository: kubeos-operator
           tag: "v1.0.8"
       proxy:
         image:
           repository: kubeos-proxy
           tag: "v1.0.8"
   ```

   然后在addon配置中引用：

   ```yaml
         valuesConfigMapRef:
           name: kubeos-addon-values
           namespace: bke-cluster
           valuesKey: values.yaml
   ```

   > ![](../figures/icon-note.gif)**说明**：
   >
   > `global.imageRegistry`需填写实际可访问的镜像仓库地址，os-operator和os-proxy镜像需提前推送到该仓库。

4. 验证addon安装。

   集群创建完成后，执行以下命令验证KubeOS addon组件状态：

   ```bash
   # 确认os-operator运行状态，预期1/1 Running
   kubectl get pods -n upgrade-system -l control-plane=upgrade-operator

   # 确认os-proxy覆盖所有KubeOS worker，预期pod数量等于worker节点数量
   kubectl get pods -n upgrade-system -l control-plane=upgrade-proxy -o wide

   # 确认CRD已注册，预期状态为Established
   kubectl get crd os.upgrade.openeuler.org osinstances.upgrade.openeuler.org
   ```

## 配置示例

以下为包含KubeOS addon的完整bke-cluster.yaml配置片段，其中`valuesConfigMapRef`为可选字段：

```yaml
spec:
  clusterConfig:
    cluster:
      chartRepo:
        domain: cr.openfuyao.cn
        port: "443"
        prefix: charts
      containerRuntime:
        cri: containerd
        runtime: runc
        param:
          cgroup-driver: systemd
          data-root: /var/lib/containerd
      containerdVersion: v2.1.1
      kubernetesVersion: v1.34.3-of.1
      imageRepo:
        domain: default
        ip: 192.168.200.71
        port: "40443"
    addons:
    - name: kubeproxy
      version: v1.34.3-of.1
      block: true
      param:
        clusterNetworkMode: calico
    - name: calico
      version: v3.27.3
      param:
        calicoMode: vxlan
    - name: coredns
      version: v1.10.1
    - name: kubeos-addon
      type: chart
      version: 0.0.0-latest
      releaseName: kubeos-addon
      namespace: upgrade-system
      # valuesConfigMapRef为可选字段，不配置时使用chart默认values.yaml
      valuesConfigMapRef:
        name: kubeos-addon-values
        namespace: bke-cluster
        valuesKey: values.yaml
```

## KubeOS不可变性约束

KubeOS采用只读rootfs，以下目录运行态只读，不得写入文件或安装软件包：

| 目录 | 说明 |
| ---- | ---- |
| `/`、`/usr`、`/bin`、`/sbin`、`/lib`、`/lib64` | 系统根文件系统和基础系统目录。 |
| `/usr/local/bin` | 传统bke-agent推送路径，KubeOS下改为镜像内置。 |
| `/usr/lib/systemd/system` | systemd unit目录，运行态不写入service文件。 |
| `/boot` | 内核、initramfs、grub相关内容。 |

以下目录通过overlay和bind mount提供持久可写能力，安装流程的配置渲染和证书写入在这些目录下进行：

| 目录 | 后端存储 | 典型内容 |
| ---- | -------- | -------- |
| `/etc` | overlay到`/persist/etc` | 主机名、网络、containerd配置、kubelet配置、CNI配置。 |
| `/etc/containerd` | `/etc` overlay | `config.toml`、runtime配置、私有仓库证书。 |
| `/etc/kubernetes` | `/etc` overlay | kubelet kubeconfig、CA、bootstrap配置。 |
| `/var` | bind到`/persist/var` | 运行数据、日志、容器和kubelet数据。 |
| `/var/lib/kubelet` | `/var` bind | kubelet状态、Pod volume、config.yaml。 |
| `/var/lib/containerd` | `/var` bind | 镜像缓存、容器快照、运行时状态。 |
| `/opt/cni` | overlay到`/persist/opt` | CNI插件二进制。 |

> ![](../figures/icon-note.gif)**说明**：
>
> `/opt/cni`通过`opt-cni.mount` overlay到PERSIST分区，Calico可按常规方式写入CNI插件和配置，重启和A/B升级后可保留。

## 故障排除

### worker节点无法加入集群

1. 在worker VM上执行`ping <管理节点IP>`确认网络可达，检查DHCP或静态IP配置是否正确。
2. 在worker VM上执行`date`对比管理节点时间。时间偏差会导致TLS证书验证失败，bkeagent反复重启。在worker VM上执行`systemctl status chronyd`或`systemctl status ntpd`确认NTP服务正常运行。
3. 在worker节点执行`systemctl status bkeagent`检查服务状态，如显示`failed`或频繁重启，执行`journalctl -u bkeagent -n 100`查看日志，确认是否能连接管理集群API server。

### containerd配置失败

1. 在worker节点执行`cat /etc/containerd/config.toml`检查配置文件是否存在且内容正确。
2. 执行`ls -la /etc/containerd/`确认目录可写。如目录不存在，检查KubeOS镜像是否正确构建了`/etc` overlay挂载。
3. 执行`journalctl -u bkeagent | grep containerd`查看containerd相关日志。如果报`containerd config version 3 is not supported`，此错误不影响bkeagent运行，可忽略。

### KubeOS addon安装失败

1. 在管理集群节点上执行`curl -k https://cr.openfuyao.cn`确认chart仓库网络可达。
2. 执行`kubectl get configmap -n bke-cluster`确认values.yaml ConfigMap已正确创建。
3. 执行`kubectl describe pod -n upgrade-system`查看pod事件，确认镜像是否可拉取。如报`ImagePullBackOff`，检查values.yaml中镜像仓库地址和镜像是否已推送。
4. 执行`kubectl get nodes --show-labels | grep kubeos`确认worker节点标签`node.openfuyao.io/os=kubeos`存在。
5. 执行`kubectl describe bkecluster -n bke-cluster`查看BKECluster状态，确认addon配置中的chartRepo地址、version和valuesConfigMapRef是否正确。

### os-proxy未调度到KubeOS worker

1. 执行`kubectl get nodes --show-labels | grep kubeos`确认worker节点已打标签`node.openfuyao.io/os=kubeos`。
2. 执行`kubectl get ds -n upgrade-system -o yaml`检查os-proxy DaemonSet的nodeSelector是否为`node.openfuyao.io/os: kubeos`。
3. 执行`kubectl get nodes`确认KubeOS worker节点状态为Ready。如节点NotReady，执行`kubectl describe node <node-name>`查看原因。
