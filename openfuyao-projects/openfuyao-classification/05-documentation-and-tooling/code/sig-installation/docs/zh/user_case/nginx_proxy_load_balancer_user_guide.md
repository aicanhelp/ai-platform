# nginx-proxy负载均衡使用指导

## 背景信息

BKE集群的worker节点访问kube-apiserver需要经过负载均衡。当前提供两种负载均衡模式：

- **ha模式（默认）**：通过keepalived在各master节点间漂移VIP，haproxy侦听VIP并将请求转发到各master节点的kube-apiserver。该模式要求部署环境提供独立的VIP，且master节点之间二层网络可达。

- **nginx模式**：在每个worker节点以静态Pod方式运行一个nginx实例，侦听本机回环地址`127.0.0.1:6443`，通过TCP stream将请求分发到全部master节点的kube-apiserver。worker节点上的kubelet、kube-proxy等组件统一通过`127.0.0.1:6443`访问kube-apiserver。

nginx模式的优势：

1. **降低部署门槛**：消除VIP依赖，适用于IP资源紧张、云环境不支持VIP漂移或二层网络受限的场景。
2. **提升业务面可靠性**：负载均衡下沉到每个worker节点本地，nginx对upstream采用被动探活（开源版nginx无主动健康检查），某master故障时新连接自动转发到其余master，worker本地入口不受影响。

## 前提条件

- 已完成openFuyao管理面安装，且版本支持`loadBalanceType`字段（自本特性发布版本起支持）。
- 工作集群至少包含1个master节点，且worker节点可以访问全部master节点IP的kube-apiserver端口（默认6443）。
- 节点可从镜像仓拉取nginx镜像（离线环境由本地镜像仓提供）。

## 使用限制

- **取值范围**：`loadBalanceType`仅支持`ha`与`nginx`，默认为`ha`。
- **创建后不可变**：`loadBalanceType`字段在集群创建后不可修改（webhook强制校验），不支持运行期在两种模式间切换。
- **与外部负载均衡互斥**：nginx模式不可与`customExtra["extraLoadBalanceIP"]`（外部负载均衡）同时配置。该约束在集群创建时校验，创建后请勿通过更新集群追加`extraLoadBalanceIP`。
- **ControlPlaneEndpoint约束**：nginx模式下`spec.controlPlaneEndpoint.host`如显式指定，必须为某个master节点IP（禁止填写VIP或`127.0.0.1`）；未指定时自动取第一个master节点IP。`port`必须为kube-apiserver端口（默认6443）：仅指定host而省略port时，端口默认取36443（ha模式负载均衡端口），nginx模式下该端口不可用。
- **管理面入口单点**：nginx模式下管理面通过`controlPlaneEndpoint.host`（默认第一个master节点IP）访问工作集群，该节点故障期间管理面将暂时无法管理工作集群，但业务面worker不受影响。对管理面高可用有要求的用户请选择ha模式。

## 操作步骤

按照下列步骤配置nginx-proxy负载均衡。

1. 编辑`BKECluster CR`，在`spec.clusterConfig.cluster`字段配置nginx负载均衡相关参数。

    **表 1** nginx-proxy负载均衡参数与默认值说明
    |参数|说明|默认值|
    |---|---|---|
    |loadBalanceType|worker节点访问kube-apiserver的负载均衡方式，取值ha或nginx。创建后不可修改。|ha|
    |nginxProxyVersion|nginx-proxy使用的nginx镜像tag，仅在loadBalanceType为nginx时生效。通常无需配置，随集群升级流程自动同步。|1.28.0|

    **示例**：以下是一个最小化的nginx模式集群配置（仅列出与本特性相关的字段）。

    ```yaml
    apiVersion: bke.bocloud.com/v1beta1
    kind: BKECluster
    metadata:
      name: bkecluster-sample
    spec:
      # 未指定时自动取第一个master节点IP和端口6443；显式指定host时须同时指定port
      # controlPlaneEndpoint:
      #   host: 192.168.1.10
      #   port: 6443
      clusterConfig:
        cluster:
          loadBalanceType: nginx
    ```

    > ![](./figures/icon-notice.gif)**注意：**<br/> `loadBalanceType`创建后不可修改。创建集群前请根据部署环境确认负载均衡模式：需要VIP或管理面高可用，选择`ha`；无法提供VIP且可接受管理面入口单点，选择`nginx`。

2. 应用`BKECluster CR`创建集群（或在通过管理面创建集群的流程中携带上述配置）。

    调谐器在创建集群过程中检测到`loadBalanceType`为`nginx`后，跳过keepalived/haproxy的安装，并在每个worker节点上渲染以下内容：

    - `/etc/kubernetes/nginx-proxy/nginx.conf`：nginx配置文件，upstream为全部master节点的kube-apiserver地址，侦听`127.0.0.1:6443`。
    - 静态Pod清单`nginx-proxy.yaml`：在每个worker节点拉起nginx实例（Pod名称为`kube-system/nginx-proxy`）。

3. 验证部署。

    3.1 集群状态验证。集群创建完成后，`BKECluster`的`status.clusterHealthState`应为`Healthy`，`status.clusterStatus`应为`Ready`。

    3.2 nginx-proxy静态Pod验证。执行以下命令，验证每个worker节点存在一个`nginx-proxy` Pod且处于`Running`状态：

    ```bash
    kubectl get pod -n kube-system -o wide | grep nginx-proxy
    ```

    3.3 本地入口验证。登录任一worker节点，执行以下命令，预期返回`ok`：

    ```bash
    curl -k https://127.0.0.1:6443/healthz
    ```

    3.4 master故障规避验证（可选）。停止任一master节点的kube-apiserver后，worker节点上的组件仍可通过本地nginx正常访问其余master节点的kube-apiserver，集群业务不受影响。
