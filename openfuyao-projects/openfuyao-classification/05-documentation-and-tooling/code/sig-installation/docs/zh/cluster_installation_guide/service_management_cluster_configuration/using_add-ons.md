# Addon组件自定义配置

## 背景信息

openFuyao社区通过在BKECluster中放置addon数据（分为yaml和chart形态），调谐器在处理BKECluster时可进行扩展件的安装。addon的介绍参见[附录](./using_add-ons.md#附录)章节。

## 前提条件

无。

## 使用限制

- 针对chart形态addon，仅提供chart安装、卸载、升级配置声明框架，chart addon安装前、后对宿主机的配置修改需用户自行处理。  
- 部署的chart addon包含子chart时，不支持在部署时解压chart并修改子chart的values.yaml。  
- 离线场景下，需用户自行确保离线镜像仓和离线chart仓库有待安装addon所需镜像、chart。  

## 操作步骤

按以下步骤安装addon插件，在线场景和离线场景操作一致。

### yaml形态addon

1. 编辑 bc（BKECluster）CR，添加所需的 addon 组件和参数。

   ```bash
   # 查看已有 BKECluster
   kubectl get bc -A
   # 编辑目标集群的 bc（将占位符替换为实际命名空间与名称）
   kubectl edit bc -n <命名空间> <名称>
   ```

   需要在`BKECluster`CRD的`Spec.ClusterConfig.Addons`中声明自定义addon的参数，确保调谐器能识别并解析这些参数。字段含义与缺省值见[附录：cluster-api-provider-bke 配置参数](../appendix/cluster-api-provider-bke_configuration.md)。

   参数格式：使用`Param`字段传递键值对。

   示例：自定义一个扩展件，需要指导镜像版本、副本数、自定义标签：

   ```yaml
   # BKECluster CR中配置自定义addon
   spec:
     clusterConfig:
       addons:
       - name: "my-addon"
         version: "v1.0.0"
         param:
           image: "my-registry/my-addon:v1.0.0"
           replicas: "3"
           labels: "app=my-addon, env=prod"
         block: true # 部署失败时是否阻塞整个流程
   ```

   参数解析：代码`cluster-api-provider-bke/pkg/kube`会通过`prepareAddonParam`函数将这些参数注入到YAML模板中，无需修改CRD结构，直接使用`Param`字段传递即可。

2. 准备addon的YAML模板。

   需要将addon的部署YAML（如Deployment、Service等）放在指定目录，确保调谐器能遍历并渲染这些模板。

   BKE 在引导/管理节点上通过 hostPath 读取 addon 清单，固定目录为 `/etc/openFuyao/addons/manifests/kubernetes`。内置组件已按「组件名/版本号/yaml」放置，可直接参考现有路径，例如：

   ```text
   /etc/openFuyao/addons/manifests/kubernetes/
     coredns/
       v1.10.1/
         coredns.yaml
     calico/
       v3.27.3/
         calico.yaml
     openfuyao-system-controller/
       latest/
         openfuyao-system-controller.yaml
   ```

   自定义 addon 时，在同一根目录下新建 `<addon-name>/<version>/`，放入 yaml 文件，例如：

   ```text
   /etc/openFuyao/addons/manifests/kubernetes/
     my-addon/
       v1.0.0/
         01-deployment.yaml
         02-service.yaml
   ```

   模板变量：YAML中需要替换的动态参数（如镜像地址、副本数）需使用模板占位符，调谐器会通过`parseFabricParam`等函数注入参数。

    > ![说明](../figures/icon-note.gif) **说明：**<br />文件名前缀（如`01-`、`02-`）会影响部署顺序（升序执行），删除时会按降序执行。

   如下展示的是01-deployment.yaml文件，模板占位符会通过参数注入。

   ```yaml
   apiVersion: apps/v1
   kind: Deployment
   metadata:
     name: my-addon
     namespace: {{.namespace}} # 由调谐器注入集群命名空间
   spec:
     replicas: {{.replicas}} # 引用addon参数中的replicas
     selector:
       matchLabels:
         app: my-addon
     template:
       metadata:
         labels: {{.labels}} # 引用addon参数中的labels
       spec:
         containers:
         - name: my-addon
           image: {{.image}} # 引用addon参数中的image
   ```

3. 处理addon的特殊逻辑（可选）。

    如果addon需要前置/后置操作（如创建Secret、配置权限、依赖其他组件），需在调谐器代码中添加自定义逻辑。

    - **前置操作**：在`EnsureAddonDeploy.addonBeforeCreateCustomOperate`中添加分支，处理部署前的准备工作（如创建存储目录、配置RBAC）。
    - **后置操作**：在`EnsureAddonDeploy.addonAfterCreateCustomOperate`中添加分支，处理部署后的收尾工作（如验证状态、注册资源）。

4. 执行部署。
   
   将自定义addon配置到`BKECluster`CR的`Spec.ClusterConfig.Addons`中，调谐器会在`EnsureAddonDeploy`阶段检测到新addon，自动执行部署流程。  
    4.1. 对比Spec（期望状态）和`Status.AddonStatus`（实际状态），发现差异后触发部署。  
    4.2. 解析参数并渲染YAML模板。  
    4.3. 通过`kube.InstallAddon`调用`kubectl apply`部署到目标集群。  
    4.4. 更新`Status.AddonStatus`记录部署结果。  

5. 验证部署。

   - 检查目标集群中是否创建了对应的资源（`kubectl get deployment my-addon -n <namespace>`）。
   - 查看`BKECluster`的状态（`kubectl describe bkecluster <name> -n <namespace>`），确认`ClusterAddonCondition`为True，且`Status.AddonStatus`包含`my-addon`的记录。
  
6. addon更新和卸载（可选）。
   
   - 更新：修改`BKECluster`中addon的`version`或`param`，调谐器会自动触发更新流程（重新渲染YAML并履约）。
   - 卸载：从`Spec.ClusterConfig.Addons`中移除addon，调谐器会按YAML文件名降序执行删除操作（`kubectl delete -f <file>`）。

### chart形态addon

> ![注意](../figures/icon-notice.gif) **注意：**<br/>
> 本功能不会对chart包内容执行任何安全性、完整性或合规性校验，请您在部署前确保chart包的安全性。

1. 编辑 bc（BKECluster）CR，添加 chart repo。

   ```bash
   kubectl get bc -A
   kubectl edit bc -n <命名空间> <名称>
   ```

   需要在`BKECluster`CRD的`spec.clusterConfig.cluster.chartRepo`中声明chart repo，并声明chart repo的认证信息。

   示例：声明一个自签名证书的私有chart repo，示例如下。

   ```yaml
   # BKECluster CR中配置自定义addon
   spec:
     clusterConfig:
       cluster:
         chartRepo: 
           domain: my.chartrepo.cn  # 域名，chart仓库未配置域名时可填写ip地址 
           ip: 192.168.0.2          # IP地址
           port: "443"              # 开放端口号
           prefix: openfuyao        # 拉取chart域名前缀，例如此仓库中拉取chart url为 my.chartrepo.cn/openfuyao
           insecureSkipTLSVerify: false     # 是否跳过TLS验证，默认为false
           authSecretRef:                   # chart仓库认证信息，公开chart仓库时可忽略此字段
             name: chart-repo-auth-secret   # Secret名称
             namespace: bke-cluster         # Secret所在命名空间
             usernameKey: username          # Secret中存放用户名的key，不填写默认为 username
             passwordKey: password          # Secret中存放密码的key，不填写默认为 password
           tlsSecretRef:                    # chart仓库TLS认证信息，无TLS校验时可忽略此字段
             name: chart-repo-tls-secret    # Secret名称
             namespace: bke-cluster         # Secret所在命名空间
             certKey: tls.crt               # Secret中存放证书的key，不填写默认为 tls.crt
             caKey: ca.crt                  # Secret中存放CA的key，不填写默认为 ca.crt
             keyKey: tls.key                # Secret中存放私钥的key，不填写默认为 tls.key
   ```
   
   > ![说明](../figures/icon-note.gif) **说明：**<br /> 
   > 支持配置公开的chart仓库、私有chart仓库、带TLS认证的chart仓库。
   > 支持拉取oci格式chart、传统格式chart。
   > 一个bc CR中仅能配置一个chart repo，不支持一次性从多个chart repo中拉取chart。

   若对接私有chart仓库需要将登录认证信息保存在管理集群Secret中，Secret结构示例如下。

   ```yaml
   # authSecretRef 对应的 Secret YAML 定义
   apiVersion: v1
   kind: Secret
   metadata:
      name: chart-repo-auth-secret
      namespace: bke-cluster
   type: Opaque
   data:
      username: <base64-encoded-username>  # 示例：替换为实际的 base64 编码用户名，例如 echo -n 'your-username' | base64
      password: <base64-encoded-password>  # 示例：替换为实际的 base64 编码密码，例如 echo -n 'your-password' | base64
   ```

   若对接自签名证书的chart仓库需要将TLS证书认证信息保存在管理集群Secret中，Secret结构示例如下。

   ```yaml
   # tlsSecretRef 对应的 Secret YAML 定义
   apiVersion: v1
   kind: Secret
   metadata:
     name: chart-repo-tls-secret
     namespace: bke-cluster
   type: Opaque
   data:
     tls.crt: <base64-encoded-client-certificate>  # 示例：替换为实际的 base64 编码客户端证书内容，例如 cat client.crt | base64 -w 0
     ca.crt: <base64-encoded-ca-certificate>       # 示例：替换为实际的 base64 编码 CA 证书内容，例如 cat ca.crt | base64 -w 0
     tls.key: <base64-encoded-private-key>         # 示例：替换为实际的 base64 编码私钥内容，例如 cat client.key | base64 -w 0
   ```

2. 编辑 bc（BKECluster）CR，添加 chart addon。

   ```bash
   kubectl edit bc -n <命名空间> <名称>
   ```

   需要在`BKECluster`CRD的`spec.clusterConfig.addons`中声明chart addon的配置。

   示例：自定义一个logging-package addon，并配置自定义values.yaml，示例如下。

   ```yaml
   # BKECluster CR中配置自定义addon
   spec:
     clusterConfig:
       addons:
       - name: logging-package     # chart名字，会根据该名称拉取chart，必填
         release: logging          # release名字，会根据该名称创建release，非必填，默认值为addon name
         type: chart               # 插件类型，chart插件需指定为chart
         namespace: logging        # chart安装的命名空间，必填
         timeout: 5                # chart安装、升级超时时间，非必填，默认值为5分钟
         valuesRefConfigMap:       # chart包values.yaml引用配置，非必填，默认使用chart包默认的values.yaml
           name: logging           # configmap名称，必填
           namespace: bke-cluster  # configmap所在命名空间，必填
           valuesKey: values.yaml  # values.yaml文件名，非必填，默认值为values.yaml
         version: 0.0.0-latest     # chart版本，必填
         block: false              # 部署时是否阻塞其他addon的部署，非必填，默认值为false
   ```

   若配置自定义values.yaml，则需将values.yaml保存在管理集群ConfigMap中，ConfigMap结构示例如下。

   > ![说明](../figures/icon-note.gif) **说明：**<br />
   > 示例中的values.yaml需修改为真实配置，可从[logging-operator仓库](https://gitcode.com/openFuyao/logging-operator/blob/master/charts/logging-package/values.yaml)获取真实values.yaml。

   ```yaml
   apiVersion: v1
   kind: ConfigMap
   metadata:
     name: logging
     namespace: bke-cluster
   data:
     values.yaml: |  # 以下values.yaml内容为示例配置，需修改为真实配置
       replicaCount: 2
       image:
         repository: cr.openfuyao.cn/openfuyao
         tag: "2.4"
       service:
         type: ClusterIP
         port: 80
   ```

   > ![注意](../figures/icon-notice.gif) **注意：**<br/> 
   > 部署chart后修改存放values.yaml的configMap不会触发调谐器更新chart addon。 
   > 若为离线场景，需用户自行修改values.yaml中镜像拉取地址。

3. 处理addon的特殊逻辑（可选）。

   如果addon需要前置/后置操作（如创建Secret、配置权限、依赖其他组件），需在调谐器代码中添加自定义逻辑。

    - **前置操作**：在`EnsureAddonDeploy.addonBeforeCreateCustomOperate`中添加分支，处理部署前的准备工作（如创建存储目录、配置RBAC）。
    - **后置操作**：在`EnsureAddonDeploy.addonAfterCreateCustomOperate`中添加分支，处理部署后的收尾工作（如验证状态、注册资源）。

4. 执行部署。

   将自定义addon配置到`BKECluster`CR的`spec.clusterConfig.addons`中，调谐器会在`EnsureAddonDeploy`阶段检测到新addon，自动执行部署流程。  
   4.1. 对比Spec（期望状态）和`Status.AddonStatus`（实际状态），发现差异后触发部署。  
   4.2. 通过`kube.InstallAddon`调用helm sdk拉取chart包到本地临时文件。  
   4.3. 调用helm sdk部署chart到目标集群。  
   4.4. 更新`Status.AddonStatus`记录部署结果。  

5. 验证部署。

    - 检查目标集群中是否创建了对应的资源（`kubectl get deployment my-addon -n <namespace>`）。
    - 查看`BKECluster`的状态（`kubectl describe bkecluster <name> -n <namespace>`），确认`ClusterAddonCondition`为True，且`Status.AddonStatus`包含`my-addon`的记录。

6. addon更新和卸载（可选）。

    - 更新：修改`BKECluster`中addon的`version`，调谐器会自动触发更新流程。
    - 卸载：从`spec.clusterConfig.addons`中移除addon，调谐器会自动删除chart addon。

## 附录

下面介绍下常见扩展件的情况。

- K8s生态组件

   ```yaml
   - name: kubeproxy
     version: v1.33.1-of.1
     block: true # 可选，默认false。设置为true表示阻塞，即等待该组件启动完成后才会部署后续扩展件
     param: # 部署该组件需要的参数
       clusterNetworkMode: calico
   - name: calico
     version: v3.25.0
     param:
       calicoMode: bgp
   - name: coredns
     version: v1.10.1
   - name: nfs-csi
     version: v4.1.0
     block: true
     param:
       nfsServer: 172.28.8.186
   ```

   - 调谐器会按照配置顺序部署扩展件。
   - 如果有扩展件的安装强依赖前续扩展件拉起的服务，可以设置前续扩展件的block参数为true，执行阻塞式安装。
   - 如果需要安装非网络扩展件，将其放置在网络扩展件后面。

- 集群网络组件

   > ![说明](../figures/icon-note.gif) **说明：**<br />1.请将网络组件根据下方示例结合实际情况选用不同组合，并将其写在addon最前方。<br />2.为了保证后续的组件能正常启动，下面所示的有关集群网络的addon均需要添加设置`block: true`参数。<br />3.详细的addon参数和参数说明解释，请看下方样例。
   
   下面介绍两种常见的网络插件，Fabric和Calico。
   > ![说明](../figures/icon-note.gif) **说明：**<br />IPVS仅在使用Calico加kube-proxy时支持，Fabric不支持。

   - Fabric

     Fabric支持Overlay和Underlay两种模式。

     - Overlay模式

       Pod使用独立CIDR，通过VXLAN隧道封装跨节点通信，适用于底层网络受限场景。

       ```yaml
       addons:
       - name: fabric
         block: true
         param:
           fabricMode: overlay
           cidrBlock: "10.250.0.0/16" # 按实际情况填写，填写后同时设置bkecluster.spec.clusterConfig.cluster.networking.podSubnet与其一致
           excludeIps: "" # 没有则填写""，支持ip段和单个ip，使用英文","分割。eg:"10.250.0.1-10.250.0.10,10.250.0.55"
           tunnelType: vxlan # geneve gre erspan vxlan
           subCIDRMask: "24" # 按实际情况填写
         version: 2.6.2
       - name: coredns
         block: true
         version: v1.10.1
       ```
       
     - Underlay模式

       Pod直接使用VPC子网IP地址，依托于底层网络通信，无封装、高性能。

       ```yaml
       addons:
       - name: fabric
         block: true
         param:
           fabricMode: underlay
           cidrBlock: "10.250.0.0/16" # 按实际情况填写，填写后同时设置bkecluster.spec.clusterConfig.cluster.networking.podSubnet与其一致
           excludeIps: "" # 没有则填写""，支持ip段和单个ip，使用英文","分割。eg:"10.250.0.1-10.250.0.10,10.250.0.55"
           vlanID: "" # 按实际情况填写
           gateway: "10.250.0.254" # 按实际情况填写
           mask: "24" # 按实际情况填写
           dataPlaneUnified: "eth1" # 按实际情况填写
         version: 2.6.2
       - name: coredns
         block: true
         version: v1.10.1
       ```

   - Calico
     
     Calico是Kubernetes的网络和网络策略插件，Calico-Typha是其在大规模集群中用于降低数据存储负载的中间缓存代理组件，通过聚合Felix连接、过滤无关更新和水平扩展，显著减轻Kubernetes API Server/etcd压力并优化网络策略分发效率。根据实际业务需求，可通过以下参数调整配置。
    
     - calicoMode：用于设置Calico的网络模式，可选值为`vxlan`、`bgp`、`ipip`。
     - proxyMode：用于设置kube-proxy的转发模式，可选值为`iptables`、`ipvs`，该参数为可选。
     - allowTypha：用于配置是否开启Typha。Typha用于降低数据存储缓存，`calicoMode`与`proxyMode`的模式选择，不会影响Calico-Typha的使用。
     - typhaReplicas：用于设置Typha的副本数。根据[calico官方文档](https://docs.tigera.io/calico/latest/getting-started/kubernetes/self-managed-onprem/onpremises#install-calico)，建议在集群规模大于50个节点时选择开启`Calico-Typha`，每增加200个节点，至少增加1个Typha副本数，同时需要保证Typha副本总数小于集群节点总数。

     ```yaml
     # kube-proxy版本以k8s版本为准
     - name: kubeproxy
       param:
         proxyMode: iptables # iptables、ipvs
       version: v1.33.1-of.1
       block: true
     - name: calico
       param:
         calicoMode: vxlan # vxlan bgp ipip
         allowTypha: true
         typhaReplicas: 2 # 按实际情况填写
       version: v3.31.3 # 仅3.31.3及以上版本支持typha相关配置，否则不生效。
       block: true
     - name: coredns
       version: v1.10.1
       block: true
     ```
- DNS缓存组件
 
  DNS缓存组件是位于DNS客户端和DNS服务器之间的中间层软件，用于缓存DNS查询结果，减少重复查询的延迟和网络开销。
  
  - node-local-dns

    node-local-dns作为Kubernetes集群的内部DNS解析增强组件，能够在节点级别缓存DNS查询结果，有效减少CoreDNS负载，提升服务发现速度并降低网络延迟，同时支持自动重试和故障转移以增强DNS解析可靠性，提高集群在超大规模集群下的业务稳定性和并发处理能力。根据实际业务需求，通过以下参数调整配置：

    - localdns：用于配置侦听地址。在IPTABLES模式下，node-local-dns Pod会同时侦听kube-dns服务的IP地址和`<localdns>`的地址，以便Pod可以使用其中任何一个IP地址来查询DNS记录。在IPVS模式下，因为IPVS负载均衡使用的接口已经占用了该地址，node-local-dns Pod只会侦听`<localdns>`的地址，因此node-local-dns接口不能绑定kube-dns的集群IP地址。
    
    ```yaml
    - name: nodelocaldns            
      param:
        localdns: 10.96.0.10 # 按实际情况填写，注意避免与已使用的端口发生冲突
      version: v1.26.4               
    ```
   
