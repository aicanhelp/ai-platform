# 常见问题解答（FAQ）

本文档旨在提供常见的安装部署问题，并给出可用的解决手段。

## 1. calico组件的install-cni容器一直重启，出现calico-node在init阶段（1/3）且处于CrashLoopbackOff状态

  - 问题原因

    可用在节点上执行`ip route`查看路由信息，该问题通常由以下两种情况引起。

    1. 节点上缺少默认路由。
    2. 节点上存在多条默认路由，且分配给集群部署的不是最高优先级路由（metric数值越小优先级越高）。

  - 解决方法

    针对问题原因1，添加默认路由，执行下面命令。

    ```bash
    ip route add default via 192.168.100.1 dev eth0 proto static metric 100
    ```

    针对问题原因2，删除多余的默认路由，执行下面命令。

    ```bash
    ip route del default via 192.168.90.1 dev eth1
    ```

  - 问题拓展

    Calico要求节点必须有一个优先级最高的默认路由，本质上是为了确保集群内外所有流量都能在节点这个三层路由器上被正确转发。

    默认路由的配置一般在`/etc/sysconfig/network-scripts`目录下，定义和配置系统网络接口（网卡）的持久化配置文件。当系统重启或网络服务重启时，系统会读取这些文件来设置`IP`地址、网关、`DNS`等信息。

## 2. calico-node频繁重启，容器陷入“启动-探测失败-重启”的死循环

  - 问题原因

    该问题通常由设置的`initialDelaySeconds`过短导致的，Pod还未完成初始化、建立连接、加载配置等羁绊操作，`kubelet`就开始探测`calico-node`的健康状态。

  - 解决方法

    设置合适的延迟时间，告诉`kubelet`，在容器启动后，先等待一段时间，然后再执行第一次探测操作，执行如下命令进行设置。

    ```bash
    # 编辑K8s资源yaml，找到readinessProbe就绪探针相关，新增initialDelaySeconds字段
    kubectl edit ds -n kube-system calico-node
    ```

    `initialDelaySeconds`推荐设置如下，实际场景中根据实际情况设置。

    - 小集群（10节点以下），设置为30，足够`calico`完成基本初始化。
    - 大集群（10节点以上），设置为60，节点多、资源竞争激烈，需要更长的启动时间。

## 3. calico-node处于0/1的Running状态

  - 详细现象

    查看详细信息，如下。

    - 节点状态为`NotReady`：这是最直观的表现。通过`kubectl get nodes`命令，你会发现这个节点的状态不是`Ready`，而是`NotReady`。
    - `calico-node`Pod探针失败：查看`calico-node`Pod的详情（`kubectl describe pod ...`）或日志，会反复出现类似以下的错误信息。
      - `Readiness probe failed: calico/node is not ready: BIRD is not ready`
      - `BGP not established with X.X.X.X` （X.X.X.X通常是其他节点的IP）
      - `Error querying BIRD: unable to connect to BIRDv4 socket`
    - Pod跨节点通信中断：因为BGP会话未能建立，路由信息无法同步。最直接的后果是，当前节点上的Pod无法与集群内其他节点上的Pod正常通信。

  - 问题原因

    `calico-node`的`IP_AUTODETECTION_METHOD`这个字段设置错误，导致节点网络不通。

  - 解决方法

    执行如下命令修改环境变量的取值。

    ```bash
    # 编辑K8s资源yaml，设置IP_AUTODETECTION_METHOD环境变量
    kubectl edit ds -n kube-system calico-node
    ```

    常见设置取值如下。

    - `skip-interface=nerdctl*`：`openFuyao`设置的默认策略，跳过`nerdctl`前缀的网卡，选择第一个有效网卡的`IP`。
    - `can-reach=192.168.100.5`：直接指定目标地址的网卡`IP`。
    - `interface=eth4`：使用`eth4`网卡上的`IP`。

  - 问题拓展

    `IP_AUTODETECTION_METHOD`这个字段决定了Calico如何在多网卡的节点上，自动选择用于建立`BGP`邻居和封装流量的正确`IP`地址。

## 4. 引导集群安装业务集群时，业务集群节点的bkeagent无法连接引导集群的APIServer，导致集群安装失败

  - 问题原因

    `bkeagent`启动的时候会指定参数`--kube-config`配置其监控的`APIServer`，`bkeagent`启动后会判断其调谐处理的`CRD`是否在监控的`APIServer`中下发，这就会触发`bkeagent`尝试连接引导集群的`APIServer`，出现了下面的报错日志信息。

    ```
    # v25.12及之前版本日志是/var/log/bkeagenbt.log，之后的版本是/var/log/openFuyao/bkeagent.log
    The CRD cannot be installed in the target cluster, xxx
    ```

    查看配置文件（v25.12及之前版本日志是`/etc/bkeagent/config`，之后的版本是`/etc/openFuyao/bkeagent/config`），发现`server`对应的`IP`地址并非给定的引导节点的地址。

  - 解决方法

    执行下面的命令重置引导节点，然后指定`IP`地址进行初始化。

    ```bash
    # 重置引导节点
    bke reset --all --mount
    # 指定IP地址进行初始化
    bke init --hostIP=1.2.3.4
    ```

## 5. 引导集群部署openFuyao管理面时，coredns处于CrashLoopbackOff状态

  - 详细信息

    查看`coredns`的详细日志（`kubectl logs -n kube-system coredns-xxx`），出现下面的日志（实际日志`IP`地址和`Port`端口不同）。

    ```
    [ERROR] plugin/errors: 2 . NS: read udp 100.20.0.15:59690->100.10.0.10:53: i/o timeout
    [FATAL] plugin/loop: Loop(127.0.0.1:34812 -> :53) detected for zone ".", see https://coredns.io/plugins/loop#troubleshooting. Query: "HINFO ***"
    ```

  - 问题原因

    出现上面的日志，说明`coredns`进行域名解析时陷入了`loop`死循环。

  - 解决方法

    `kubectl get cm -n kube-system coredns -o yaml`可以看到`forward`设置为`/etc/resolv.conf`，这导致了无其他可用的`DNS`服务器时就会使用`coredns`的`server ip`作为服务器进行域名解析，最后导致死循环。

    执行下面命令，设置服务器。

    ```bash
    # 设置forward为forward . 8.8.8.8，如果有可用的DNS服务器，可以设置为对应的IP地址
    kubectl edit cm -n kube-system coredns
    ```

  - 问题拓展

    `forward`插件是将集群内无法解析的`DNS`请求，转发给指定的上游`DNS`服务器。

## 6. 使用虚拟IP部署高可用集群失败，使用bke reset进行节点重置后，再次使用已占用的虚拟IP部署高可用集群，部署失败

  - 问题原因

    高可用集群的`keepalived`组件会将虚拟`IP`绑定在高可用集群的某个节点上，使用`bke reset`在每个节点上重置环境后，并不会将虚拟`IP`从已绑定的节点上解绑，导致二次使用时，出现报错，最终导致集群无法拉起。

  - 解决方法

    登录到首次安装的高可用集群的 Master 节点上，执行下面的命令解绑虚拟`IP`。

    ```bash
    # 查看节点的网卡绑定的IP地址
    ip addr
    # 如果查询到绑定了虚拟IP，执行下面的命令进行解绑，vip替换为实际使用的寻IP，eth0替换为实际绑定的网卡
    ip addr del <vip> dev <eth0>
    ```

## 7. openFuyao安装的集群，除了管理面删除集群外，后端如何删除集群

  正式步骤、删除顺序、清理范围与成功判据见[集群卸载](../../cluster_uninstallation/overview.md)及[通过后台（命令行）删除业务集群](../../cluster_uninstallation/delete_service_cluster_command.md)。

  引导集群或者管理集群的`installer-service`会从`APIServer`中读取`BKECluster`数据。除管理面删除外，也可在**引导节点**或**管理集群**终端执行：

  ```bash
  # 查询集群信息
  kubectl get bc -A
  # 将占位符替换为实际命名空间与名称后编辑
  kubectl edit bc -n <命名空间> <名称>
  ```

  在打开的 YAML 编辑器中设置以下两项（彻底清理目标节点时）：

  ```yaml
  metadata:
    annotations:
      bke.bocloud.com/ignore-target-cluster-delete: "false"
  spec:
    reset: true
  ```

## 8. 后端如何实现openFuyao集群的扩缩容操作

  openFuyao管理面提供集群的生命周期管理能力，包括集群的扩容、缩容、升级、安装和卸载等能力。这里提供后端进行集群的扩缩容处理，需要在集群处于健康的情况下进行操作，集群处于非健康状态时，扩缩容操作可能会出现错误。

  - 缩容操作，将节点从已有集群中删除。

    查看已有`BKENode`资源。

    ```bash
    # bke-cluster替换为实际集群的信息
    kubectl get bn -n bke-cluster
    ```

    删除对应的BKENode资源。

    ```bash
    # bke-cluster-n1替换为实际的节点名称
    kubectl delete bn -n bke-cluster bke-cluster-n1
    ```

    再次执行查看已有`BKENode`资源命令发现无对应节点即删除成功。

  - 扩容操作，将新节点加入到已有集群中。

    编写新的节点的配置文件(newNode.yaml)。
    ```yaml
    apiVersion: bke.bocloud.com/v1beta1
    kind: BKENode
    metadata:
      name: bke-cluster-n1
      namespace: bke-cluster
      labels:
        cluster.x-k8s.io/cluster-name: bke-cluster
    spec:
      hostname: n1
      ip: <node-ip>
      password: '<encrypted>'
      port: "22"     
      role:
      - node
      username: root
    ```
    执行下面命令实现扩容操作。
    ```bash
    kubectl apply -f newNode.yaml
    ```

    执行查看已有`BKENode`资源命令发现对应节点Ready即扩容成功。

 注：v25.12及之前版本参考以下指南操作。

  - 缩容操作，将节点从已有集群中删除。

    编辑`BKECluster`资源。

    ```bash
    # bke-cluster替换为实际集群的信息
    kubectl edit bc -n bke-cluster bke-cluster
    ```

    设置预约删除的节点。

    ```yaml
    metadata:
      annotations:
        # 节点预约删除，节点删除是个危险的动作故增加此注释做二次确认，默认无
        # 删除节点时除从spec中将节点删除，还需将要删除的节点ip填入，多个ip之间使用‘,’分割
        # 两步操作少之其一都不会触发节点删除
        bke.bocloud.com/appointment-deleted-nodes: "172.100.200.10"
    ```

    将节点信息从`Spec`中删除。

    ```yaml
    spec:
      clusterConfig:
        nodes:                
        - hostname: master-1  
          ip: 172.100.200.10
          username: root      
          password: password0
          port: "22"
          role:             
          - master
          - etcd
    ```

  - 扩容操作，将新节点加入到已有集群中。

    编辑`BKECluster`资源。

    ```bash
    # bke-cluster替换为实际集群的信息
    kubectl edit bc -n bke-cluster bke-cluster
    ```

    将节点信息加入到`Spec`中。

    ```yaml
    spec:
      clusterConfig:
        nodes:                
        - hostname: master-1  
          ip: 172.100.200.10
          username: root      
          password: password0
          port: "22"
          role:             
          - master
          - etcd
    ```

## 9. 节点 iptables FORWARD 链默认策略为 DROP 时，导致容器网络通信异常、集群初始化或创建失败

  - 详细现象

    节点上存在 iptables 且 FORWARD 链默认策略为 DROP 时，可能出现以下一种或多种现象。

    1. `bke init` 初始化引导节点后，引导节点上的容器（镜像仓库、YUM 仓库、Chart 仓库）服务正常启动，但引导集群（k3s）中的 Pod 间通信异常或 Pod 访问外部服务超时。
    2. 引导集群部署 openFuyao 管理面时，部分 Pod（如 cluster-api-provider-bke 控制器）出现网络不通，日志中存在连接超时或拒绝等报错。
    3. openFuyao 前端管理面访问超时，在引导节点上通过 `curl` 访问后端服务可正常输出，但在其余节点上执行 `curl` 访问则超时或不通。
    4. 业务集群节点初始化或节点加入集群过程中，镜像拉取超时，节点无法加入集群。
    5. 集群创建后，跨节点 Pod 通信失败，CoreDNS 解析超时，Calico BGP 会话无法建立。

  - 解决方法

    在出现上述现象的节点上，手动将 iptables FORWARD 链默认策略修改为 ACCEPT。

    ```bash
    # 查看当前 FORWARD 链默认策略
    iptables -t filter -nvL FORWARD

    # 将 FORWARD 链默认策略设置为 ACCEPT
    iptables -t filter -P FORWARD ACCEPT
    ```

    如需在引导节点初始化之前预先设置，可以在执行 `bke init` 之前执行上述命令。

  - 问题拓展

    iptables FORWARD 链默认策略为 DROP 的常见来源包括：

    - 系统安装时启用了 firewalld 防火墙，firewalld 默认将 FORWARD 链策略设为 DROP。即使后续停止了 firewalld 服务，已设置的 iptables 策略不会自动恢复为 ACCEPT。
    - 安全加固场景中，管理员手动将 FORWARD 链策略设为 DROP 以限制未授权的流量转发。
    - 部分 Linux 发行版（如某些最小化安装的 CentOS/RHEL）默认将 FORWARD 策略设为 DROP。

## 10. 安装过程中 /etc/resolv.conf 被 NetworkManager 刷新为默认内容，导致节点网络异常

  - 问题原因

    安装过程中，`NetworkManager`（NM）服务会管理节点的 `DNS` 配置，并将 `/etc/resolv.conf` 刷新为默认内容，覆盖已配置的有效 `DNS` 服务器，导致节点域名解析异常，进而引发网络不通、镜像拉取失败、组件通信超时等问题。

  - 解决方法

    修改 `NetworkManager` 的配置，设置 `dns=none`，禁止 `NetworkManager` 管理 `DNS` 配置。

    ```bash
    # 编辑 NetworkManager 主配置文件
    vi /etc/NetworkManager/NetworkManager.conf
    ```

    在 `[main]` 段中新增或修改如下配置。

    ```ini
    [main]
    dns=none
    ```

    保存后重启 `NetworkManager` 服务使配置生效，并确认 `/etc/resolv.conf` 中的 `DNS` 配置正确。

    ```bash
    # 重启 NetworkManager 服务
    systemctl restart NetworkManager
    # 确认 resolv.conf 内容未被覆盖为默认配置
    cat /etc/resolv.conf
    ```

  - 问题拓展

    `dns=none` 表示 `NetworkManager` 不再写入或更新 `/etc/resolv.conf`，由管理员自行维护 `DNS` 配置。安装前建议提前配置，避免安装过程中被意外刷新。
## 11. 引导节点 k3s 容器一直间断重启
  - 详细现象

    在引导节点执行 `ps -ef | grep k3s` 和 `nerdctl ps -a`，发现k3s server会间断重启。
    使用`nerdctl logs kubernetes`发现日志中有 **failed to run iptables command to create KUBE-ROUTER-OUTPUT chain due to running [/bin/aux/iptables -t filter -S KUBE-ROUTER-OUTPUT 1 --wait]: exit status 3**。

  - 问题原因
  
    执行`lsmod | grep -E 'ip_tables|iptable_filter'`无输出。
    出现上述现象的原因是k3s在启动网络组件时，因为宿主机内核的`iptables`相关模块缺失，导致网路策略控制器初始化失败，触发panic并退出。

  - 解决方法

    可以执行以下命令加载对应模块：
    ```bash
    modprode ip_tables
    modprode iptables_filter
    modprode iptables_nat
    ```
    
## 12. 单节点集群两个coredns pod中一个会偶发Pending
- 详细现象
    
  创建的单节点集群执行`kubectl get pod -n kube-system | grep coredns`发现一个状态是`Running`，一个状态是`Pending`。

- 解决方法
  
  执行以下命令将 CoreDNS 副本降为 1 即可。对于单节点环境，1 个 CoreDNS 副本已经完全足够提供 DNS 解析服务。
  ```bash
  kubectl scale deployment coredns --replicas=1 -n kube-system
  ```

## 13. 共部署场景下，业务集群 Pod 无日志输出，宿主机 /var/log/pods 下的日志目录每分钟被清理

  - 详细现象

    引导节点同时作为业务集群节点（共部署）时，部署完成后出现以下一种或多种现象。

    1. 业务集群部署在该节点上的 Pod 执行 `kubectl logs` 报错（如 `no such file or directory`）或无任何日志输出。
    2. 登录该节点查看，`/var/log/pods` 下业务集群 Pod 的日志目录（命名格式为 `<命名空间>_<Pod名>_<UID>`）约每分钟消失一次。
    3. 引导集群在该节点上的 Pod 日志存在同样的被清理现象。

  - 问题原因

    自 openFuyao v26.09 版本起（引入 k3s 容器日志挂载宿主机行为的版本），引导节点初始化时会将宿主机的 `/var/log/pods` 与 `/var/log/containers` 以相同路径挂载（bind mount）进 k3s 容器，便于在宿主机直接查看引导集群日志。

    共部署场景下，节点上同时运行两个 kubelet：k3s 内嵌的 kubelet（引导集群）与业务集群的 kubelet，两者共用宿主机的 `/var/log/pods` 目录。

    kubelet 的容器 GC 每 1 分钟执行一次，其中包含清理 Pod 日志目录的逻辑：`/var/log/pods` 为 kubelet 硬编码路径，无任何参数可修改或关闭该清理行为，凡是当前 kubelet 不认识的 Pod 日志目录都会被当作孤儿目录直接删除。因此两个 kubelet 会互相将对方 Pod 的日志目录当作孤儿清理，表现为业务集群 Pod 持续丢失日志，引导集群 Pod 日志同样受损。

    说明：常规部署（引导节点不作为业务集群节点）不受此问题影响，无需执行以下操作。

  - 解决方法

    核心操作为更换 k3s 容器的日志挂载路径，使两个集群的日志目录物理隔离，按部署阶段分为以下两种场景。

    场景一：引导节点部署成功后、共部署业务集群之前（前置规避），整体顺序为：保存已有日志 → 重建 k3s 容器（更换日志挂载路径）→ 验证引导集群恢复 → 再共部署业务集群。

    1. 保存已有日志，共两部分。

       - Pod 日志：重建容器并共部署业务集群后，宿主机 `/var/log/pods` 中遗留的引导集群日志目录会被业务集群的 kubelet 当作孤儿目录清理。
       - k3s 容器自身日志：k3s 进程（含内嵌 kubelet、apiserver 等控制面组件）的日志输出至容器 stdout，由 nerdctl 落盘保存，执行 `nerdctl rm` 删除容器时会一并清除。k3s 内嵌 containerd 的日志位于 `/var/lib/rancher/k3s/agent/containerd/containerd.log`，该目录已挂载至宿主机持久保存，重建容器不受影响，无需备份。

       ```bash
       mkdir -p /var/log/bke/backup
       cp -a /var/log/pods /var/log/bke/backup/pods-$(date +%Y%m%d%H%M%S)
       cp -a /var/log/containers /var/log/bke/backup/containers-$(date +%Y%m%d%H%M%S)
       # 导出 k3s 容器自身日志
       nerdctl logs kubernetes > /var/log/bke/backup/k3s-container-$(date +%Y%m%d%H%M%S).log
       ```

    2. 创建引导集群专用日志目录。

       ```bash
       mkdir -p /var/log/bke/bootstrap/pods /var/log/bke/bootstrap/containers
       ```

    3. 重建 k3s 容器。引导集群状态保存在宿主机 `/etc/rancher/k3s` 与 `/var/lib/rancher/k3s`，重建容器不影响集群数据。

       ```bash
       # 获取当前容器完整启动参数，记录端口映射、--add-host、镜像地址等实际取值
       nerdctl inspect kubernetes

       # 停止并删除容器
       nerdctl stop kubernetes
       nerdctl rm kubernetes
       ```

       使用原启动参数重建容器，仅将以下两个挂载：

       ```bash
       -v /var/log/pods:/var/log/pods
       -v /var/log/containers:/var/log/containers
       ```

       替换为：

       ```bash
       -v /var/log/bke/bootstrap/pods:/var/log/pods
       -v /var/log/bke/bootstrap/containers:/var/log/containers
       ```

       其余参数保持与 `nerdctl inspect kubernetes` 的结果一致。注意：实际环境的启动参数可能比示例更多，例如管理面安装后会追加 `--dns` 参数，部分环境还存在 `--kube-apiserver-arg`（Webhook 认证配置）等参数，重建时必须以 inspect 输出为准一并带上，遗漏会改变集群行为。示例如下（尖括号内容替换为实际取值）。

       ```bash
       nerdctl run -d --name=kubernetes \
         -p <kubernetesPort>:36443 -p <consolePort>:30010 \
         --privileged --restart=always \
         --add-host <imageRepo>:<imageRepoIP> \
         -v /etc/rancher/k3s:/etc/rancher/k3s \
         -v /etc/timezone:/etc/timezone \
         -v /etc/docker:/etc/docker \
         -v /etc/localtime:/etc/localtime \
         -v /var/lib/rancher/k3s:/var/lib/rancher/k3s \
         -v /bke:/bke \
         -v /etc/openFuyao:/etc/openFuyao \
         -v /etc/openFuyao/addons/manifests:/etc/openFuyao/addons/manifests \
         -v /var/log/bke/bootstrap/pods:/var/log/pods \
         -v /var/log/bke/bootstrap/containers:/var/log/containers \
         <k3s镜像> server --snapshotter=native --https-listen-port=<kubernetesPort> \
         --service-cidr=100.10.0.0/16 --cluster-cidr=100.20.0.0/16 \
         --advertise-address=<hostIP> --tls-san=<hostIP> --node-name=kubernetes \
         --pause-image=<pause镜像> \
         --disable=coredns,servicelb,traefik,local-storage,metrics-server
       ```

       示例中尖括号占位符的取值可通过 `nerdctl inspect kubernetes` 的输出获取，对应关系如下。

       | 占位符 | 获取方式 |
       | ---- | ---- |
       | `<kubernetesPort>` | `HostConfig.PortBindings` 中容器端口 `36443/tcp` 对应的宿主机端口，与顶层 `Args` 中 `--https-listen-port` 的取值一致 |
       | `<consolePort>` | `HostConfig.PortBindings` 中容器端口 `30010/tcp` 对应的宿主机端口 |
       | `<imageRepo>`、`<imageRepoIP>` | `HostConfig.ExtraHosts` 中条目的域名与 IP |
       | `<k3s镜像>` | 顶层 `Image` 字段 |
       | `<hostIP>` | 顶层 `Args` 中 `--advertise-address`、`--tls-san` 的取值（即 `bke init --hostIP` 指定的引导节点 IP） |
       | `<pause镜像>` | 顶层 `Args` 中 `--pause-image` 的取值 |

       另外，inspect 输出 `Mounts` 中的匿名卷（`/var/lib/cni`、`/var/lib/kubelet`、`/var/log`）由镜像 VOLUME 声明自动创建，重建容器时会重新生成，无需在启动参数中手动指定。

    4. 验证引导集群恢复正常。

       ```bash
       # 确认节点就绪
       kubectl get nodes
       # 确认容器内日志挂载已指向新路径
       nerdctl exec kubernetes sh -c "mount | grep /var/log"
       # 确认引导集群日志写入新路径
       ls /var/log/bke/bootstrap/pods
       ```

    5. 再执行业务集群共部署。部署完成后确认 `/var/log/pods` 中业务集群 Pod 的日志目录持续存在（超过 2 分钟未消失），且 `kubectl logs` 输出正常。

    场景二：业务集群已共部署完成（事后修复）

    1. 保存已有日志。共部署期间两个 kubelet 已在互相清理对方的日志目录，已被清理的日志无法恢复，先备份当前尚存内容，命令同场景一第 1 步。
    2. 依次执行场景一第 2~4 步：创建引导集群专用日志目录、重建 k3s 容器（更换日志挂载路径）、验证引导集群恢复。重建期间引导集群（含管理面）短暂中断；业务集群的 kubelet 与容器运行时独立于 k3s 容器，不受影响，无需重新部署业务集群。
    3. 重启该节点上受影响的业务集群 Pod。共部署期间正在运行的 Pod，其日志文件已被删除，需待容器重启后才会重新生成日志文件，建议对受影响的工作负载执行滚动重启以避免业务中断。

       ```bash
       # 以 Deployment 为例执行滚动重启，其他工作负载类型同理
       kubectl rollout restart deployment/<name> -n <namespace>
       ```

    4. 验证业务集群日志恢复。

       ```bash
       # 日志目录持续存在（超过 2 分钟未消失）
       ls /var/log/pods
       # kubectl logs 输出正常
       kubectl logs <pod-name> -n <namespace>
       ```

  - 问题拓展

    - 本操作需在引导节点部署（含管理面）全部完成之后执行。管理面安装、升级或重新初始化等会重建 k3s 容器的操作会把原始日志挂载重新带回，此类操作完成后需重新执行本操作，直至包含修复的版本发布。
    - 处理完成后，引导集群 Pod 日志位于宿主机 `/var/log/bke/bootstrap/pods`，业务集群 Pod 日志仍位于 `/var/log/pods`，两个集群日志目录物理隔离、互不清理，且均可在宿主机直接查看与审计。
  