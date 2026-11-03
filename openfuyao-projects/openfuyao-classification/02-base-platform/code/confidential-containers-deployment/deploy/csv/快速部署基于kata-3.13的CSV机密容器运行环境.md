# 快速部署基于 kata-3.13 的CSV机密容器运行环境

## 更新历史

- 2025-11-25：创建文档

## 参考文档

- **Kata Containers 项目**  
  [https://github.com/kata-containers/kata-containers](https://github.com/kata-containers/kata-containers)

- **Confidential Containers (CoCo) 社区**  
  [https://github.com/confidential-containers](https://github.com/confidential-containers)  
  - Operator 项目：[https://github.com/confidential-containers/operator](https://github.com/confidential-containers/operator)
  - 官方文档站：[https://confidentialcontainers.org/](https://confidentialcontainers.org/)

- **Containerd Remote Snapshotter 设计文档**  
  [https://github.com/containerd/containerd/blob/main/docs/remote-snapshotter.md](https://github.com/containerd/containerd/blob/main/docs/remote-snapshotter.md)

- **Kubernetes RuntimeClass 机制**  
  [https://kubernetes.io/docs/concepts/containers/runtime-class/](https://kubernetes.io/docs/concepts/containers/runtime-class/)

## 目录索引

- [CSV 机密容器方案演进](#csv-机密容器方案演进)
  - [起点：Kata-2.5](#起点kata-25)
  - [雏形：Kata-3](#雏形kata-3)
  - [逐步完善：Kata-3.13](#逐步完善kata-313)
- [部署前准备](#部署前准备)
  - [准备支持 CSV 的 K8s 节点](#准备支持-csv-的-k8s-节点)
  - [准备容器镜像](#准备容器镜像)
  - [安装 kustomize 工具](#安装-kustomize-工具)
- [使用 CoCo Operator 部署 CSV 机密容器运行时](#使用-coco-operator-部署-csv-机密容器运行时)
  - [部署 CoCo Operator](#部署-coco-operator)
  - [创建 CcRuntime 实例](#创建-ccruntime-实例)
  - [触发节点安装](#触发节点安装)
  - [验证安装结果](#验证安装结果)
- [创建机密容器](#创建机密容器)
  - [部署测试 Pod](#部署测试-pod)
  - [验证机密虚拟机配置](#验证机密虚拟机配置)

## CSV 机密容器方案演进

CSV 机密容器解决方案，基于 [Kata Containers](https://github.com/kata-containers/kata-containers) 和 [Confidential Containers](https://github.com/confidential-containers)（CoCo）两大开源社区，紧密跟随社区技术路线持续演进。该方案旨在利用硬件级可信执行环境（TEE）技术，在保障容器运行时安全的同时，实现与 Kubernetes 原生生态的无缝集成。以下是其关键发展阶段的演进路径：

### 起点：Kata-2.5

Kata Containers 的核心理念是在轻量级虚拟机（VM）中运行容器，以提供更强的隔离性。在 Kata-2.5 版本中，CSV 方案通过在启动虚拟机（VM）前扩展底层虚拟机监控器（VMM，如 QEMU）的命令行参数，引入并配置机密计算专用 QEMU 对象，用 **CSV 机密虚拟机** 替代普通虚拟机，使容器运行于硬件保护的可信环境中。

### 雏形：Kata-3

CoCo 社区在 Kata 代码仓库中创建 **CCv0 分支**，开始探索更完整的机密容器解决方案。CSV 机密容器方案基于此分支快速迭代，形成了功能化雏形，实现了两大关键特性突破：

- **将容器的镜像下载和容器 rootfs 的创建迁移到虚拟机侧进行**，避免了宿主机侧的数据泄露风险。

- 引入 **simple-kbs** 作为轻量级密钥代理服务（Key Broker Service），提供基本的远程认证能力，允许 Guest VM 向外部 KBS 请求加密资源（如镜像解密密钥）。

### 逐步完善：Kata-3.13

CoCo 社区基于 Kata-3.13 版本发布了 v0.9.0，标志着机密容器技术从功能验证走向成熟。CSV 方案借此实现了多项关键突破，正式具备了生产级的部署与运行能力：

- 首先，方案**不再依赖定制的 containerd 和 kata**。在之前的版本中，方案需要对 containerd 和 Kata Containers 进行定制化修改，这增加了方案的维护成本和兼容性风险。而在 Kata-3.13 中，通过优化架构设计，避免了对 containerd 的修改，并将核心改动合并到 Kata 主分支。这一调整使得方案可直接使用社区官方的 containerd 和 Kata 版本，大幅提升了兼容性与可维护性，显著降低了用户的接入门槛。

- 其次，用 **KBS 微服务替代了 simple-kbs**。simple-kbs 作为早期的密钥管理组件，功能较为单一，难以满足大规模、高可用的生产环境需求。新版本中的 KBS 微服务采用了模块化、分布式的架构设计，不仅支持多种硬件机密计算技术的适配，还提供了丰富的密钥管理策略、细粒度的访问控制以及高可用性保障，能够为大规模机密容器集群提供稳定、安全的密钥和认证服务。

- 最后，方案实现了 **通过 operator 快速部署**。Operator 是云原生领域用于自动化应用部署和管理的核心工具。CSV 方案借助 CoCo 社区提供的 Operator，可在标准 K8s 集群中快速部署机密容器运行时环境，极大地简化了方案的落地难度，为生产环境的规模化应用提供了有力的支撑。

## 部署前准备

本文介绍如何通过 **Operator** 在标准 K8s 集群中快速部署基于 Kata-3.13 方案的 **CSV 机密容器运行时环境**。在正式部署前，需完成以下准备工作：

### 准备支持 CSV 的 K8s 节点

集群中需包含至少一个配置了 CSV 能力的节点。将搭载海光 CPU 的物理机或裸金属服务器作为节点加入集群，并参考文档《[配置海光 CSV 环境](./配置海光CSV环境.md)》完成 CSV 相关配置。

此外，由于方案依赖 containerd 的 [Remote Snapshotter](https://github.com/containerd/containerd/blob/main/docs/remote-snapshotter.md) 功能，要求海光节点上的容器运行时为 **containerd**，且版本**不低于 1.7**。可通过以下命令查看节点的容器运行时版本，确认是否满足要求：
```bash
kubectl get node $NODE_NAME -o jsonpath='{.status.nodeInfo.containerRuntimeVersion}'
```

示例输出：
```
containerd://1.7.27
```

### 准备容器镜像

方案部署需用到以下三个容器镜像，确保集群节点可正常拉取：
- **Operator Controller** 镜像：quay.io/confidential-containers/operator:v0.12.0
- **Pre-install Payload** 镜像：quay.io/confidential-containers/reqs-payload:b495f450fbcd5c199b8635a7dfefe1b5d763c378
- **kata-deploy** 镜像（含 Kata artifacts）：haigon.cn/test/kata-deploy-csv:3.11.0

镜像可直接使用原有仓库，也可下载后推送到集群易访问的私有仓库。其中，海光定制的 `kata-deploy-csv` 镜像需通过 containerd 的 `ctr` 工具拉取，操作命令如下：
```bash
# 使用 ctr 拉取镜像
ctr -n k8s.io image pull "haigon.cn/test/kata-deploy-csv:3.11.0"

# 导出镜像为 tar 包
ctr -n k8s.io image export "kata-deploy-csv.tar" "haigon.cn/test/kata-deploy-csv:3.11.0"

# 导入镜像到 docker
docker load -i "kata-deploy-csv.tar"
```

### 安装 kustomize 工具

为便于对 Operator 部署清单进行定制（如替换镜像地址），需要安装 **kustomize** 工具：

```bash
# 下载 kustomize 压缩包
wget -O "kustomize_v5.4.3_linux_amd64.tar.gz" "https://github.com/kubernetes-sigs/kustomize/releases/download/kustomize%2Fv5.4.3/kustomize_v5.4.3_linux_amd64.tar.gz"

# 解压到系统可执行目录
tar -zxf "kustomize_v5.4.3_linux_amd64.tar.gz" -C /usr/local/bin
```

## 使用 CoCo Operator 部署 CSV 机密容器运行时

本节介绍如何使用 **CoCo Operator** 在支持海光 CSV 的节点上部署机密容器运行时环境。

### 部署 CoCo Operator

1. **获取海光定制的 Operator 源码**  
   克隆指定分支：
   ```bash
   git clone "https://gitee.com/hanliyang-kata-coco/operator.git" -b "0.12.0-hygon-arch-sig" "coco-operator"
   ```

2. **（可选）设置 Operator 镜像地址**  
   若需替换默认镜像仓库，进入源码目录并通过 `kustomize` 修改镜像地址：
   ```bash
   cd coco-operator
   pushd config/release
   kustomize edit set image quay.io/confidential-containers/operator="local.cmecloud.cn/confidential-containers/operator:v0.12.0"
   popd
   ```

3. **部署 Operator**  
  在 `coco-operator` 目录下执行部署命令：
   ```bash
   kubectl apply -k config/release
   ```

4. **验证部署结果**  
   检查 Pod 是否正常运行：
   ```bash
   kubectl get po -n confidential-containers-system
   ```
   示例输出：
   ```
   NAME                                              READY   STATUS    RESTARTS   AGE
   cc-operator-controller-manager-68d589f5f6-gnmrx   1/1     Running   0          5h40m
   ```
   查看 `ccruntimes` CRD 是否存在：
   ```bash
   kubectl get crd | grep ccruntimes
   ```
   示例输出：
   ```
   ccruntimes.confidentialcontainers.org    2025-06-24T03:10:26Z
   ```

### 创建 CcRuntime 实例

1. **（可选）设置镜像地址**  
   若需替换 pre-install payload 和 kata-deploy 镜像，执行以下命令修改镜像地址：
   ```bash
   pushd config/samples/ccruntime/default
   kustomize edit set image quay.io/confidential-containers/reqs-payload="local.cmecloud.cn/confidential-containers/reqs-payload:0.12.0-hygon-arch-sig"
   kustomize edit set image haigon.cn/test/kata-deploy-csv="local.cmecloud.cn/confidential-containers/kata-deploy-csv:3.11.0"
   popd
   ```

2. **应用 CcRuntime 配置**  
   创建机密容器运行时配置实例：
   ```bash
   kubectl apply -k config/samples/ccruntime/default
   ```

### 触发节点安装

为指定的海光节点添加标签，以触发自动安装：
```bash
kubectl label nodes $NODE_NAME node.kubernetes.io/worker=
```

> **说明**：CoCo Operator 会监听带有 `node.kubernetes.io/worker` 标签的节点，并在该节点上部署安装任务。  

安装过程中，将在 `confidential-containers-system` 命名空间创建两个 DaemonSet 管理的 Pod：
```bash
kubectl get po -n confidential-containers-system
```
示例输出：
```
NAME                                     READY   STATUS    RESTARTS   AGE
cc-operator-daemon-install-nfv9p         1/1     Running   0          179m
cc-operator-pre-install-daemon-25hhb     1/1     Running   0          179m
```

### 验证安装结果

1. **检查 Nydus Snapshotter 服务状态**  
   登录目标节点，确认 `nydus-snapshotter` 已成功启动：
   ```bash
   systemctl status nydus-snapshotter.service
   ```
   示例输出：
   ```
   ● nydus-snapshotter.service - Nydus snapshotter
        Loaded: loaded (/etc/systemd/system/nydus-snapshotter.service; enabled; vendor preset: disabled)
        Active: active (running) since Tue 2025-06-24 14:12:58 CST; 3h 2min ago
      Main PID: 2627803 (containerd-nydu)
         Tasks: 39 (limit: 2472149)
        Memory: 32.6M
        CGroup: /system.slice/nydus-snapshotter.service
                └─ 2627803 /opt/confidential-containers/bin/containerd-nydus-grpc --config /opt/confidential-containers/share/nydus-snapshotter/config-coco-guest-pulling.toml --log-to-stdout
   ```

2. **确认 RuntimeClass 已注册**  
   在 K8s 集群中查看是否已创建 CSV 对应的 RuntimeClass：
   ```bash
   kubectl get runtimeclass | grep csv
   ```
   示例输出：
   ```
   kata-qemu-csv    kata-qemu-csv    3h3m
   kata-qemu-csv2   kata-qemu-csv2   3h3m
   kata-qemu-csv3   kata-qemu-csv3   3h3m
   ```

至此，CSV 机密容器运行时已在目标节点成功部署，可使用对应的 `RuntimeClass` 运行机密容器工作负载。

## 创建机密容器

本节通过部署一个测试 Pod，验证 CSV 机密容器运行时是否已正确配置并生效。

### 部署测试 Pod

使用公共 Nginx 镜像创建测试 Pod，并指定 `runtimeClassName: kata-qemu-csv` 以启用 CSV 机密容器运行时：
```bash
kubectl apply -f - <<EOF
apiVersion: v1
kind: Pod
metadata:
  name: test-pod
  labels:
    app: nginx
spec:
  runtimeClassName: kata-qemu-csv
  containers:
  - name: nginx
    image: nginx:latest
    ports:
    - containerPort: 80
EOF
```

验证 Pod 是否成功运行：
```bash
kubectl get pod test-pod
```
预期输出：
```
NAME        READY   STATUS    RESTARTS   AGE
test-pod    1/1     Running   0          27s
```

### 验证机密虚拟机配置

登录目标海光节点，可以查看相应的 QEMU 进程以及启动参数：
```bash
ps aux | grep qemu
```

在输出中应观察到如下关键参数（节选）：
```text
-object sev-guest,id=sev,cbitpos=47,reduced-phys-bits=5,policy=3,kernel-hashes=on
-machine q35,accel=kvm,kernel_irqchip=split,confidential-guest-support=sev
-drive if=pflash,format=raw,readonly=on,file=/opt/confidential-containers/share/ovmf/HYGONCSV.fd
-kernel /opt/confidential-containers/share/kata-containers/vmlinuz-5.10.134-102cc-csv-hygon-gpu
-initrd /opt/confidential-containers/share/kata-containers/kata-containers-initrd-csv-hygon-gpu.img.online_sev_kbc
```

**关键参数说明**：
- `-object sev-guest,id=sev,...`：创建 `sev` 安全对象。。
- `-machine ...,confidential-guest-support=sev`：将 `sev` 安全对象绑定到虚拟机，激活机密计算支持。
- `-drive if=pflash,...,file=.../HYGONCSV.fd`：使用海光定制 OVMF 固件，确保可信启动。
- `-kernel...`：机密虚拟机使用的内核。
- `-initrd...`：机密虚拟机使用的initrd镜像。
