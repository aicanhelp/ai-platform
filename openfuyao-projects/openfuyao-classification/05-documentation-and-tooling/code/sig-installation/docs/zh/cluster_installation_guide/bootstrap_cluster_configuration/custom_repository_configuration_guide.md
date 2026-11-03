# 自定义仓库配置

## 背景信息

* 支持使用自定义仓库配置，支持**镜像仓**、**二进制仓**、**chart仓**的自定义配置。
* 支持镜像仓和chart仓的单向认证访问(containerd、helm对仓库的单向访问，且使用同一CA)，二进制仓不涉及认证访问。
  
 > ![说明](../figures/icon-note.gif) **说明：**<br />chart 仓目前只有 NUMA 组件在 addon 中独立使用。私有 Chart / addon 操作步骤见[使用 Add-ons 安装 NUMA 亲和性扩展指南](https://gitcode.com/openFuyao/sig-installation/blob/master/docs/zh/user_case/numa_affinity_extension_installation_guide_using_add-ons.md)。

## 前提条件

* 需要用户提供自定义的镜像仓、二进制仓、chart仓。

  * 请将所需所有镜像导入到私有镜像仓中（注意命名空间、镜像名称、镜像tag要上传正确）。
  * 将所需的所有二进制文件导入到私有二进制。
  * 将所需的所有chart包导入到私有chart仓中。

- 在引导节点上按约定，将仓库的根CA上传到指定路径`/etc/openFuyao`下。若目录不存在，新建后建议将目录权限收敛为**700**；若目录已存在，则不再修改目录权限。`ca.crt`文件权限建议收敛为**600**。目录结构为`/etc/openFuyao/ca.crt`。

  ```bash
  if [ ! -d /etc/openFuyao ]; then
    mkdir -p /etc/openFuyao
    chmod 700 /etc/openFuyao
  fi
  # 将 ca.crt 放到 /etc/openFuyao/ca.crt 后执行
  chmod 600 /etc/openFuyao/ca.crt
  ```

  > ![注意](../figures/icon-caution.gif) **注意：**<br />`/etc/openFuyao` 已存在时不调整其目录权限，避免影响该路径上已有配置。`ca.crt`虽然通常不包含私钥，但仍建议按最小权限原则限制访问，仅允许文件所有者读取和写入，避免被其他本地账号不必要地读取。

- 需要上传的介质列表详见[附录部分](./custom_repository_configuration_guide.md#附录)。

## 使用限制

无。

## 操作步骤

> ![说明](../figures/icon-note.gif) **说明：**<br />创建管理集群/业务集群时，openFuyao 通过以下自定义资源（CR）承载配置。字段含义与缺省值（openFuyao 平台官方推荐配置）详见[附录：cluster-api-provider-bke 配置参数](../appendix/cluster-api-provider-bke_configuration.md)。
>
> | 缩写 | CR 类型 | kubectl 短名 / 资源名 | 作用 |
> | --- | --- | --- | --- |
> | bc | `BKECluster` | `bc` | 描述待创建集群的版本、仓库、网络、addon 等集群级配置 |
> | kct | `KubeletConfig` | `kct` | 自定义 kubelet 配置，由 bc 的 `spec.KubeletConfigRef` 引用 |
> | cct | `ContainerdConfig` | `cct` | 自定义 containerd 配置，由 bc 的 `spec.clusterConfig.cluster.containerdConfigRef` 引用 |

1. 引导节点初始化。

    将bkeadm二进制工具上传到引导节点
    使用bke init完成引导节点的初始化。下面为初始化命令参考：
    
    ``````
    
      bke init \
      # 1. 镜像仓
      --otherRepo 192.168.100.17:443/openfuyao \   # 自定义镜像仓库（含端口+路径），注意命名空间推荐为openFuyao
      --domain my.harbor.com \                     # 镜像仓库域名（与--otherRepo匹配，确保证书和访问一致性）
      --imageRepoPort 443 \                        # 镜像仓库端口（显式指定端口号）
      --imageRepoCAFile /etc/openFuyao/ca.crt \    # 私有仓库CA证书（指定读取的证书路径）
      # 2. 二进制仓
      --otherSource http://192.168.100.48:8080 \   
      # 3. 功能开关：关闭控制台安装
      --installConsole=false \                     # 不安装BKE自带控制台（默认关闭）
      --enableNTP=false \                          # 不进行NTP同步服务操作
      # 4. 其他
      --confirm                                    # 提前确认，不需要进行后续Y/N确认
    ``````

2. 通过 bc（BKECluster）创建管理集群/业务集群。

    2.1 查询并编辑 bc，在 `spec.clusterConfig.cluster` 下将镜像仓（imageRepo）、二进制/系统包仓（httpRepo）、Chart 仓（chartRepo）改为自定义地址。

    ```bash
    # 查看已有 BKECluster
    kubectl get bc -A
    # 编辑目标集群的 bc（将占位符替换为实际命名空间与名称）
    kubectl edit bc -n <命名空间> <名称>
    ```

    自定义仓库配置样例（写入 `spec.clusterConfig.cluster`）：

    ```yaml
    httpRepo:  # 二进制/系统包仓库
      domain: http.bocloud.k8s   # 可保持默认域名，或改为实际域名
      ip: 192.168.200.173        # 修改为实际的 http 仓库 IP
      port: "40080"              # 修改为实际的 http 仓库端口号
      prefix: ""                 # 修改为实际的路径前缀，默认可为空
    imageRepo:  # 镜像仓库
      domain: deploy.bocloud.k8s # 修改为实际的 image 仓域名
      ip: 192.168.200.173        # 修改为实际的 image 仓 IP
      port: "40443"              # 修改为实际的 image 仓端口号
      prefix: openfuyao          # 修改为实际的项目名
    chartRepo:  # Chart 仓库
      domain: deploy.bocloud.k8s # 修改为实际的 chart 仓域名；无域名时可填 IP
      ip: 192.168.200.173        # 修改为实际的 chart 仓 IP
      port: "38080"              # 修改为实际的 chart 仓端口号
      prefix: ""                 # 修改为实际的路径前缀，默认可为空
    ```

    2.2 创建或编辑 cct（ContainerdConfig），为私有镜像仓补充 containerd registry 配置，并在 bc 中引用。

    ```bash
    # 编辑已有 ContainerdConfig（将占位符替换为实际命名空间与名称）
    kubectl edit cct -n <命名空间> <名称>
    # 或通过 YAML 创建/更新后 apply
    kubectl apply -f my-containerd.yaml
    ```

    registry 配置样例：

    ```yaml
    registry:
      configPath: /etc/containerd/certs.d
      configs:
        "my.harbor.com":            # 镜像仓域名
          capabilities: ["pull", "resolve"]
          host: my.harbor.com
          skipVerify: false
          plainHTTP: false
          tls:
            caFile: /etc/openFuyao/ca.crt
            insecureSkipVerify: false
    ```

    在 bc 中引用该 cct：

    ```yaml
    spec:
      clusterConfig:
        cluster:
          containerdConfigRef:
            name: bke-containerd      # 与上一步 ContainerdConfig 的 metadata.name 一致
            namespace: bke-containerd # 与上一步 ContainerdConfig 的 metadata.namespace 一致
    ```

    2.3（可选）创建或编辑 kct（KubeletConfig），自定义 kubelet 配置，并在 bc 中引用。

    ```bash
    # 编辑已有 KubeletConfig（将占位符替换为实际命名空间与名称）
    kubectl edit kct -n <命名空间> <名称>
    # 或通过 YAML 创建/更新后 apply
    kubectl apply -f my-kubelet.yaml
    ```

    在 bc 中引用该 kct：

    ```yaml
    spec:
      KubeletConfigRef:
        name: bke-kubelet      # 与 KubeletConfig 的 metadata.name 一致
        namespace: bke-kubelet # 与 KubeletConfig 的 metadata.namespace 一致
    ```

    > ![说明](../figures/icon-note.gif) **说明：**<br />kct / cct 的完整字段说明与样例见[二进制组件自定义配置（Containerd和kubelet）](../service_management_cluster_configuration/binary_file_custom_configuration_guide.md)；BKECluster 全量字段见[附录：cluster-api-provider-bke 配置参数](../appendix/cluster-api-provider-bke_configuration.md)。

    > ![注意](../figures/icon-caution.gif) **注意：**<br />生产环境禁止将`insecureSkipVerify`设置为`true`。该配置会跳过仓库服务端证书校验，存在中间人攻击、误连错误仓库以及拉取篡改内容的风险。只有在临时排障或隔离测试环境下，且明确接受风险时，才可短时使用。

## 附录

### 镜像列表

**1. 用于搭建K3s集群的基础组件，架构均为`amd64、arm64`。**

| **镜像地址**                            | **标签**      | **描述**                 |
| ----------------------------------------- | ------------- |------------------------|
| cr.openfuyao.cn/openfuyao/registry        | 2.8.1         | 镜像仓库（与基础镜像重复，统一复用）     |
| cr.openfuyao.cn/openfuyao/nginx_yum_repo  | 1.23.0-alpine | YUM源代理服务（基于Nginx）    |
| cr.openfuyao.cn/openfuyao/chartmuseum     | v0.16.2       | Helm Chart仓库服务        |
| cr.openfuyao.cn/openfuyao/nfs-server-alpine | 0.9.0         | NFS存储服务（轻量Alpine版本） |
| cr.openfuyao.cn/openfuyao/k3s             | v1.25.16-k3s4 | K3s集群核心二进制镜像          |
| cr.openfuyao.cn/openfuyao/mirrored-pause  | 3.6           | 容器暂停镜像（K3s容器生命周期管理）   |

**2. K8s核心组件镜像（needDownload: true）**

覆盖K8s控制平面、网络、负载均衡等核心功能，架构均为`amd64、arm64`。

| **镜像地址**                                  | **标签**   | **描述**                     |
| ----------------------------------------------- | ---------- |----------------------------|
| cr.openfuyao.cn/openfuyao/kubernetes/coredns    | v1.10.1    | K8s DNS服务（CoreDNS）        |
| cr.openfuyao.cn/openfuyao/kubernetes/pause      | 3.9        | K8s容器基础镜像（与K3s区分版本）       |
| cr.openfuyao.cn/openfuyao/kube-apiserver        | of-v1.33.1 | K8s API服务器（openFuyao定制版）  |
| cr.openfuyao.cn/openfuyao/kube-controller-manager | of-v1.33.1 | K8s控制器管理器（openFuyao定制版）   |
| cr.openfuyao.cn/openfuyao/kube-scheduler        | of-v1.33.1 | K8s调度器（openFuyao定制版）      |
| cr.openfuyao.cn/openfuyao/kube-proxy            | of-v1.33.1 | K8s网络代理（openFuyao定制版）     |
| cr.openfuyao.cn/openfuyao/etcd                  | of-v3.5.21 | Etcd数据库（K8s集群存储，openFuyao定制版） |
| cr.openfuyao.cn/openfuyao/keepalived/keepalived | 1.3.5      | 高可用VIP服务（Keepalived）      |
| cr.openfuyao.cn/openfuyao/haproxy/haproxy       | 2.1.4      | 负载均衡服务（HAProxy）            |
| cr.openfuyao.cn/openfuyao/calico/cni            | v3.27.3    | Calico CNI网络插件（容器网络接口）    |
| cr.openfuyao.cn/openfuyao/calico/node           | v3.27.3    | Calico节点代理（网络策略执行）        |
| cr.openfuyao.cn/openfuyao/calico/kube-controllers | v3.27.3    | Calico控制器（网络资源管理）         |
| cr.openfuyao.cn/kubernetes/pause                | 3.9        | K8s容器基础镜像（与K3s区分版本）       |

**3. bke工具本身**

| **镜像地址**                                             | **标签** | **描述**                  |
| ---------------------------------------------------------- | -------- |-------------------------|
| cr.openfuyao.cn/openfuyao/cluster-api-provider-bke         | latest   | BKE集群API提供商（CAPI兼容） |
| cr.openfuyao.cn/openfuyao/bke-manifests                    | latest   | BKE资源清单镜像              |
| cr.openfuyao.cn/openfuyao/cluster-api/cluster-api-controller | v1.4.3   | CAPI集群控制器              |
| cr.openfuyao.cn/openfuyao/cluster-api/kubeadm-bootstrap-controller | v1.4.3   | CAPI Kubeadm引导控制器      |
| cr.openfuyao.cn/openfuyao/cluster-api/kubeadm-control-plane-controller | v1.4.3   | CAPI Kubeadm控制平面控制器    |

**4. 扩展组件**

**4.1 Volcano调度扩展**

架构均为`amd64、arm64`，用于容器调度优化（如NUMA亲和、资源调度）。

| **镜像地址**                                    | **标签** | **描述**                    |
| ------------------------------------------------- | -------- |---------------------------|
| cr.openfuyao.cn/openfuyao/volcano-config-website  | latest   | Volcano配置页面              |
| cr.openfuyao.cn/openfuyao/volcano-config-service  | latest   | Volcano配置服务              |
| cr.openfuyao.cn/openfuyao/numa-affinity-webhook   | latest   | NUMA亲和性Webhook服务        |
| cr.openfuyao.cn/openfuyao/numa-tuning-operator    | v25.03   | NUMA资源调优Operator         |
| cr.openfuyao.cn/openfuyao/volcano/resource-exporter | latest   | Volcano资源指标导出器           |
| cr.openfuyao.cn/openfuyao/oauth-proxy             | latest   | OAuth认证代理（多组件复用）         |
| cr.openfuyao.cn/openfuyao/vc-scheduler            | latest   | Volcano调度器核心             |
| docker.io/volcanosh/vc-webhook-manager            | v1.9.0   | Volcano Webhook管理器（官方镜像） |
| docker.io/volcanosh/vc-controller-manager         | v1.9.0   | Volcano控制器管理器（官方镜像）      |
| docker.io/alpine/openssl                          | 3.5.0    | OpenSSL工具（基础依赖，Alpine版本） |
| cr.openfuyao.cn/openfuyao/kubectl-openfuyao       | latest   | openFuyao定制Kubectl工具    |

### **二进制文件列表**

**1. K8s核心操作工具**

用于K8s集群的命令行管理与节点运行，均提供`amd64`/`arm64`双架构支持。

| **组件名称**               | **架构** | **文件类型** | **描述**                               |
| -------------------------- | ---------- | ------------ |--------------------------------------|
| kubectl-v1.33.1-amd64      | amd64      | 单个二进制   | K8s集群命令行工具（openFuyao定制v1.33.1版本） |
| kubectl-v1.33.1-arm64      | arm64      | 单个二进制   | 同上（适配ARM架构节点）                      |
| kubelet-v1.33.1-amd64      | amd64      | 单个二进制   | K8s节点代理（维持节点与集群通信，v1.33.1版本）       |
| kubelet-v1.33.1-arm64      | arm64      | 单个二进制   | 同上（适配ARM架构节点）                      |
| etcdctl-v3.5.6-linux-amd64 | amd64      | 单个二进制   | Etcd数据库命令行工具（管理集群存储，v3.5.6版本）      |
| etcdctl-v3.5.6-linux-arm64 | arm64      | 单个二进制   | 同上（适配ARM架构节点）                      |

**2. 容器运行时核心组件**

支撑容器启停与资源管理的底层二进制，含运行时主体与核心依赖。

| **组件名称**                         | **架构** | **文件类型** | **描述**                            |
| ------------------------------------ | ---------- | ------------ |-----------------------------------|
| containerd-v2.1.1-linux-amd64.tar.gz | amd64      | 压缩包       | 容器运行时（核心组件，v2.1.1 版本，压缩包含二进制与配置）  |
| containerd-v2.1.1-linux-arm64.tar.gz | arm64      | 压缩包       | 同上（适配ARM架构节点）                   |
| runc-amd64                           | amd64      | 单个二进制   | 轻量级容器运行时（containerd依赖，创建/销毁容器） |
| runc-arm64                           | arm64      | 单个二进制   | 同上（适配ARM架构节点）                   |

**3. 容器网络插件（CNI）**

用于配置容器网络（如网络隔离、跨节点通信），为压缩包格式（内含多个CNI插件二进制）。

| **组件名称**                       | **架构** | **文件类型** | **描述**                                        |
| ---------------------------------- | ---------- | ------------ |-----------------------------------------------|
| cni-plugins-linux-amd64-v1.4.1.tgz | amd64      | 压缩包       | CNI网络插件集合（v1.4.1版本，含bridge、host-local等插件） |
| cni-plugins-linux-arm64-v1.4.1.tgz | arm64      | 压缩包       | 同上（适配ARM架构节点）                               |

**4. 包管理与配置处理工具**

用于Helm Chart管理、YAML/JSON解析等运维操作。

| **组件名称**                    | **架构** | **文件类型** | **描述**                                  |
| ------------------------------- | ---------- | ------------ |-----------------------------------------|
| helm-v3.14.2-linux-amd64.tar.gz | amd64      | 压缩包       | Helm包管理工具（v3.14.2 版本，管理K8s应用Chart）  |
| helm-v3.14.2-linux-arm64.tar.gz | arm64      | 压缩包       | 同上（适配ARM架构节点）                         |
| yq_linux_amd64                  | amd64      | 单个二进制   | YAML数据处理工具（查询/修改YAML文件，v4.43.1版本） |
| yq_linux_arm64                  | arm64      | 单个二进制   | 同上（适配ARM架构节点）                         |
| jq-linux-amd64                  | amd64      | 单个二进制   | JSON数据处理工具（解析/ 过滤JSON，v1.7.1版本）     |
| jq-linux-arm64                  | arm64      | 单个二进制   | 同上（适配ARM架构节点）                         |

**5. 用于RPM包管理、NFS共享配置的辅助压缩包（内含相关二进制或脚本）。**

| **组件名称**    | **架构**             | **文件类型** | **描述**                   |
| --------------- | ---------------------- |----------|--------------------------|
| rpm.tar.gz      | 通用（压缩包内含多架构） | 压缩包      | RPM包管理相关工具集合（适配RPM系操作系统） |
| nfsshare.tar.gz | 通用（压缩包内含多架构） | 压缩包      | NFS共享服务配置工具（实现集群存储共享）    |
| charts.tar.gz   | 必须要有               | 压缩包      | charts包文件                |

**6. BKE集群专属管理工具**

openFuyao BKE集群的初始化与运维工具，双架构支持。

| **组件名称**       | **架构** | **文件类型** | **描述**                  |
| ------------------ | ---------- | ------------ |-------------------------|
| bkeadm_linux_amd64 | amd64      | 单个二进制   | BKE集群管理工具（初始化、升级、运维集群） |
| bkeadm_linux_arm64 | arm64      | 单个二进制   | 同上（适配ARM架构节点）         |

注：cfssl工具在openFuyao核心组件中使用，联创场景未用上。

**二进制文件全量列表**

```yaml
files:
  - address: https://openfuyao.obs.cn-north-4.myhuaweicloud.com/kubernetes/kubernetes/releases/download/of-v1.33.1/bin/linux/arm64/
    files:
      - kubectl-v1.33.1-arm64
  - address: https://openfuyao.obs.cn-north-4.myhuaweicloud.com/kubernetes/kubernetes/releases/download/of-v1.33.1/bin/linux/amd64/
    files:
      - kubectl-v1.33.1-amd64
  - address: https://openfuyao.obs.cn-north-4.myhuaweicloud.com/kubernetes/kubernetes/releases/download/of-v1.33.1/bin/linux/arm64/
    files:
      - kubelet-v1.33.1-arm64
  - address: https://openfuyao.obs.cn-north-4.myhuaweicloud.com/kubernetes/kubernetes/releases/download/of-v1.33.1/bin/linux/amd64/
    files:
      - kubelet-v1.33.1-amd64
  - address: https://openfuyao.obs.cn-north-4.myhuaweicloud.com/containerd/containerd/releases/download/v2.1.1-origin/
    files:
      - containerd-v2.1.1-linux-amd64.tar.gz
      - containerd-v2.1.1-linux-arm64.tar.gz
  - address: https://openfuyao.obs.cn-north-4.myhuaweicloud.com/containernetworking/plugins/releases/download/v1.4.1/
    files:
      - cni-plugins-linux-amd64-v1.4.1.tgz
      - cni-plugins-linux-arm64-v1.4.1.tgz
  - address: https://openfuyao.obs.cn-north-4.myhuaweicloud.com/helm/releases/download/v3.14.2/
    files:
      - helm-v3.14.2-linux-amd64.tar.gz
      - helm-v3.14.2-linux-arm64.tar.gz
  - address: https://openfuyao.obs.cn-north-4.myhuaweicloud.com/mikefarah/yq/releases/download/v4.43.1
    files:
      - yq_linux_arm64
      - yq_linux_amd64
  - address: https://openfuyao.obs.cn-north-4.myhuaweicloud.com/jqlang/jq/releases/download/v1.7.1
    files:
      - jq-linux-arm64
      - jq-linux-amd64
  - address: https://openfuyao.obs.cn-north-4.myhuaweicloud.com/cloudflare/cfssl/releases/download/v1.6.4
    files:
      - cfssl-certinfo_1.6.4_linux_arm64
      - cfssl-certinfo_1.6.4_linux_amd64
      - cfssl_1.6.4_linux_arm64
      - cfssl_1.6.4_linux_amd64
      - cfssljson_1.6.4_linux_arm64
      - cfssljson_1.6.4_linux_amd64
  - address: https://openfuyao.obs.cn-north-4.myhuaweicloud.com/opencontainers/runc/releases/download/v1.1.12/
    files:
      - runc-arm64
      - runc-amd64
  - address: https://openfuyao.obs.cn-north-4.myhuaweicloud.com/etcd-io/etcd/releases/download/v3.5.6/
    files:
      - etcdctl-v3.5.6-linux-amd64
      - etcdctl-v3.5.6-linux-arm64
  - address: https://openfuyao.obs.cn-north-4.myhuaweicloud.com/rpm/releases/download/v0.0.1/
    files:
      - rpm.tar.gz
  - address: https://openfuyao.obs.cn-north-4.myhuaweicloud.com/nfs/
    files:
      - nfsshare.tar.gz
  - address: https://openfuyao.obs.cn-north-4.myhuaweicloud.com/charts/releases/download/latest/
    files:
      - charts.tar.gz
  - address: https://openfuyao.obs.cn-north-4.myhuaweicloud.com/openFuyao/bkeadm/releases/download/latest/
    files:
      - bkeadm_linux_amd64
      - bkeadm_linux_arm64
```

### chart包列表

目前只涉及到NUMA一个组件。

```yaml
  - address: https://openfuyao.obs.cn-north-4.myhuaweicloud.com/charts/releases/download/latest/
    files:
      - numa-affinity-package-0.0.0-latest.tgz
```
