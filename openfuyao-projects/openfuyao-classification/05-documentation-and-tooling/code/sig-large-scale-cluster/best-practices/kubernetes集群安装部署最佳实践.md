# 最佳实践

## kubernetes集群安装部署最佳实践

描述了使用部署工具部署超大规模kubernetes集群流程。部署工具支持在、离线部署kubernetes集群。

### 目标

执行几条命令即可安装kubernetes集群。

### 前提条件

- 要求待安装集群机器已经制备OS  
- 所有机器可使用root用户登录  
- 所有机器网络互通  
- 要求所有节点根目录至少有100GB剩余磁盘空间  
- etcd与etcdData节点需要有单独一块未挂载的高速SSD硬盘（只能有一块），该硬盘空间 ≥ 128GB，8KB顺序IOPS ≥ 500，读写速度 ≥ 400MiB/s，最好是NVMe接口  
- 建议节点资源16U32G  
- 所有节点要提前安装好tar  

### 注意事项

- 所有节点环境干净，未安装runc、containerd、docker、docker-ce、kubeadm、kubectl、kubelet、crictl等组件  
- 离线部署制品包中默认携带openEuler 22.03 lts sp3的conntrack、ipset、iptables、ipvsadm、networkManager、openssl、socat、systemd-pam组件的rpm包，若部署环境不是openEuler 22.03 lts sp3，则需要能直接使用包管理工具安装上述基础组件，否则将会部署失败  
- 创建高可用集群时如果haproxy和kube-apiserver在同一节点，需要修改haproxy的loadBalancerPort参数或kube-apiserver的apiserverBindPort参数，防止出现端口冲突  
- 此文档未描述功能均未测试，集群扩容功能待测试，目前仅支持扩容worker节点  

### 安装须知

- fuyao（部署工具可执行文件）二进制执行日志在`/var/log/fuyao-installer/fuyao/installer.log`内
- shell脚本（部署工具执行的shell脚本）执行日志在集群内各个节点的`/var/log/fuyao-installer/shell.log`内，因为需要在每个节点都执行shell脚本，所以shell脚本日志记录在各个节点
- 集群配置文件中advancedParams项参数未经过测试验证,建议使用默认值
- 如果集群配置文件中节点信息即填写了密码又填写了密钥路径，则使用密钥进行登录认证，如果只使用密码进行登录认证则需要删除privateKeyPath，若配置使用密钥进行登录认证，则需要提前配置从执行命令节点到集群各节点免密登录
- 集群配置文件中，node role有一下几种：

| 角色        | 描述                            | 备注                                   |
|-----------|-------------------------------|--------------------------------------|
| loadBalance | 负载均衡节点，keepalived和haproxy组件部署节点 | 当前采用keepalived+haproxy实现apiserver高可用 |
| kubeApiserver    | apiserver pod部署节点                       |                                      |
|kubeScheduler|scheduler pod部署节点|                                      |
|kubeControllerManager|controller-manager pod部署节点|                                      |
|etcd|etcd部署节点，此节点存放data与events| 数量必须为单数个                             |
|etcdData|etcd data部署节点，存放deployment、pod yaml等集群资源文件| 数量必须为单数个                              |
|etcdEvents|etcd events部署节点，存放集群事件| 数量必须为单数个                              |
|coreDns|coredns pod部署节点|                                      |
|worker|worker节点|                                      |

- 在集群配置文件中etcd角色不能与etcdData、etcdEvents共存，配置文件内必须包含kubeApiserver、kubeScheduler、kubeControllerManager三种角色，必须包含etcd或者 etcdData、etcdEvents角色  
- 一个节点可以配置多个角色，但是etcdData与etcdEvents角色不能在一个节点共存，worker与其他角色在一个节点不能共存  

### 使用限制

#### 支持操作系统

| 操作系统      | 版本             | 架构       |
|-----------|----------------| ---------- |
| openEuler | 22.03(lts sp3) | arm64、x86 |
| Ubuntu    | 22.04 | arm64、x86 |

> **说明：**<br />其他版本操作系统暂未测试，可能会出现不在预期内的问题。

#### 支持Kubernetes版本

- v1.28：1.28.15

### 部署形态

- 单节点部署
- 多节点部署
- apiserver高可用部署
- 离线部署

### 操作步骤-在线安装

#### 在线安装节点规划示例

> 此规划示例部署形态可支持4000节点kubernetes集群，若安装更大规模集群请上调master、ha、etcd data与etcd events节点CPU、内存资源

| ip              | hostname | 组件           | 角色          | CPU(vCPU) | 内存(BG) |
|-----------------|----------|--------------|-------------|-----------|--------|
| 192.168.200.240 | node1    | keepalived、haproxy | ha节点        | 32        | 64     |
| 192.168.200.239 | node2    | keepalived、haproxy | ha节点      | 32        | 64     |
| 192.168.200.238 | node3    | apiserver    | master      | 16        | 32     |
| 192.168.200.237 | node4    | apiserver    | master      | 16        | 32     |
| 192.168.200.236 | node5    | apiserver    | master      | 16        | 32     |
| 192.168.200.235 | node6    | apiserver    | master      | 16        | 32     |
| 192.168.200.234 | node7    | apiserver    | master      | 16        | 32     |
| 192.168.200.233 | node8    | apiserver    | master      | 16        | 32     |
| 192.168.200.232 | node9    | control-manager | master      | 16        | 32     |
| 192.168.200.231 | node10   | control-manager | master      | 16        | 32     |
| 192.168.200.230 | node11   | kube-scheduler | master      | 16        | 32     |
| 192.168.200.229 | node12   | kube-scheduler | master      | 16        | 32     |
| 192.168.200.228 | node13   | etcd         | etcd data   | 16        | 32     |
| 192.168.200.227 | node14   | etcd         | etcd data   | 16        | 32     |
| 192.168.200.226 | node15   | etcd         | etcd data   | 16        | 32     |
| 192.168.200.225 | node16   | etcd         | etcd events | 16        | 32     |
| 192.168.200.224 | node17   | etcd         | etcd events | 16        | 32     |
| 192.168.200.223 | node18   | etcd         | etcd events | 16        | 32     |
| 192.168.200.222 | node19   | coredns、kubeproxy等组件 | worker      | 16        | 32     |
| 192.168.200.221 | node20   | coredns、kubeproxy等组件 | worker      | 16        | 32     |
| 192.168.200.220 | node21   | coredns、kubeproxy等组件 | worker      | 16        | 32     |
| 192.168.200.219 | node22   | calico、kubeproxy等组件 | worker      | 16        | 32     |
| 192.168.200.218 | node23   | calico、kubeproxy等组件 | worker      | 16        | 32     |
| 192.168.200.217 | node24   | 业务组件         | worker      | 16        | 32     |
| 192.168.200.216 | node25   | 业务组件         | worker      | 16        | 32     |
| 192.168.200.241 | 无        | 无            | 高可用部署虚拟IP   | /         | /      |

1. 以root用户登录node1节点。

2. 下载安装工具与安装脚本。

   - 自动下载

    ```
    curl -sfL https://openfuyao.obs.cn-north-4.myhuaweicloud.com/openFuyao/installer/releases/download/large-cluster/download.sh | bash -
    ```

   执行上述命令默认下载最新版本的安装包，若想下载其他版本，请使用如下命令指定版本后进行下载。

    ```
    export VERSION=x.y.z

    curl -sfL https://openfuyao.obs.cn-north-4.myhuaweicloud.com/openFuyao/installer/releases/download/large-cluster/download.sh | bash -
    ```

   - 手动构建  
     - 手动构建要求执行命令机器系统为linux且机器已联网  
     - 手动构建前需要配置gitcode ssh 公钥，[参考链接](https://docs.gitcode.com/docs/help/home/user_center/security_management/ssh/)  

   ```shell
   set -ex
   VERSION="1.21.8"
   ARCH=$(uname -m)
   case $ARCH in
       x86_64) ARCH="amd64";;
       aarch64) ARCH="arm64";;
   esac
   
   mkdir -p $HOME/tools && cd $HOME/tools
   which wget || yum install wget -y
   wget https://golang.google.cn/dl/go${VERSION}.linux-${ARCH}.tar.gz
   which tar || yum install tar -y
   rm -rf /usr/local/go && tar -C /usr/local -xzf go${VERSION}.linux-${ARCH}.tar.gz
   echo 'export PATH=$PATH:/usr/local/go/bin' >> ~/.bashrc
   source ~/.bashrc
   
   which git || yum install git -y
   
   git clone git@gitcode.com:openFuyao/installer.git
   cd installer
   git checkout large-cluster
   
   go build -o fuyao -tags netgo,osusergo
   sudo ./fuyao create component-config  # 此处可修改fuyao-component-config.yaml文件，用来构建不同操作系统的安装包
   sudo ./fuyao create component-package -f fuyao-component-config.yaml
   # 构建出的安装包存放在./targets文件夹内，安装包包括fuyao二进制与asset文件夹，asset文件夹存放配置文件与对应操作系统的安装脚本压缩包
   cd ./targets/ && ls
   ```

3. 生成集群规划配置文件。

   - 进入安装包目录。

   > **说明：**<br />
   > 此处以openEuler22.03 arm64为例，具体路径应视实际情况而定。

   ```shell
   export INSTALLER_PACKAGE_PATH=fuyao-v0.1.0-openEuler-22.03-arm64

   cd $INSTALLER_PACKAGE_PATH
   ```

   - 执行命令，生成默认带有全部参数的集群配置文件。

   ```shell
   ./fuyao create cluster-config --all-params
   ```

   - 修改后的配置文件如下：
      - 需要填写节点配置信息，包括hostname、password、ip、port、privateKeyPath、roles等
      - 若安装高可用集群需要配置keepalived和haproxy组件参数，非高可用集群无需配置
      - 其他组件参数可按需修改

   > 若在线部署想使用外部镜像仓库，请修改组件参数中imageRepo参数的值，同时确保镜像仓库内有对应的组件镜像。
   > 所有组件请使用配置文件内默认提供的版本，暂未提供可供选择的其他版本。

    ```yaml
    apiVersion: openfuyao.io/v1beta1
    kind: cluster
    metadata:
       name: example-ha-cluster
    spec:
       nodes:
          - hostname: loadbalance-0  # 同一集群内hostname不能重复
            user: root
            password: passwd  # 修改密码
            ip: 192.168.200.240
            port: 22
            roles:
               - loadBalance
          - hostname: loadbalance-1
            user: root
            password: passwd
            ip: 192.168.200.239
            port: 22
            roles:
               - loadBalance
          - hostname: apiserver-01
            user: root
            password: passwd
            ip: 192.168.200.238
            port: 22
            roles:
               - kubeApiserver
          - hostname: apiserver-02
            user: root
            password: passwd
            ip: 192.168.200.237
            port: 22
            roles:
               - kubeApiserver
          - hostname: apiserver-03
            user: root
            password: passwd
            ip: 192.168.200.236
            port: 22
            roles:
               - kubeApiserver
          - hostname: apiserver-04
            user: root
            password: passwd
            ip: 192.168.200.235
            port: 22
            roles:
               - kubeApiserver
          - hostname: apiserver-05
            user: root
            password: passwd
            ip: 192.168.200.234
            port: 22
            roles:
               - kubeApiserver
          - hostname: apiserver-06
            user: root
            password: passwd
            ip: 192.168.200.233
            port: 22
            roles:
               - kubeApiserver
          - hostname: controller-manager-01
            user: root
            password: passwd
            ip: 192.168.200.232
            port: 22
            roles:
               - kubeControllerManager
          - hostname: controller-manager-02
            user: root
            password: passwd
            ip: 192.168.200.231
            port: 22
            roles:
               - kubeControllerManager
          - hostname: kube-scheduler-01
            user: root
            password: passwd
            ip: 192.168.200.230
            port: 22
            roles:
               - kubeScheduler
          - hostname: kube-scheduler-02
            user: root
            password: passwd
            ip: 192.168.200.229
            port: 22
            roles:
               - kubeScheduler
          - hostname: etcd-data-01
            user: root
            password: passwd
            ip: 192.168.200.228
            port: 22
            roles:
               - etcdData
          - hostname: etcd-data-02
            user: root
            password: passwd
            ip: 192.168.200.227
            port: 22
            roles:
               - etcdData
          - hostname: etcd-data-03
            user: root
            password: passwd
            ip: 192.168.200.226
            port: 22
            roles:
               - etcdData
          - hostname: etcd-events-01
            user: root
            password: passwd
            ip: 192.168.200.225
            port: 22
            roles:
               - etcdEvents
          - hostname: etcd-events-02
            user: root
            password: passwd
            ip: 192.168.200.224
            port: 22
            roles:
               - etcdEvents
          - hostname: etcd-events-03
            user: root
            password: passwd
            ip: 192.168.200.223
            port: 22
            roles:
               - etcdEvents
          - hostname: worker-coredns-01
            user: root
            password: passwd
            ip: 192.168.200.222
            port: 22
            roles:
               - coreDns
          - hostname: worker-coredns-02
            user: root
            password: passwd
            ip: 192.168.200.221
            port: 22
            roles:
               - coreDns
          - hostname: worker-coredns-03
            user: root
            password: passwd
            ip: 192.168.200.220
            port: 22
            roles:
               - coreDns
          - hostname: worker-01
            user: root
            password: passwd
            ip: 192.168.200.219
            port: 22
          - hostname: worker-02
            user: root
            password: passwd
            ip: 192.168.200.218
            port: 22
          - hostname: worker-03
            user: root
            password: passwd
            ip: 192.168.200.217
            port: 22
          - hostname: worker-04
            user: root
            password: passwd
            ip: 192.168.200.216
            port: 22
       components:
          - name: runc
            param:
               - name: version
                 value: 1.1.12
                 type: string
                 describe: runc版本
          - name: cni-plugins
            param:
               - name: version
                 value: 1.4.1
                 type: string
                 describe: CNI plugins版本
          - name: calico
            param:
               - name: imageRepo
                 value: harbor.openfuyao.com/openfuyao
                 type: string
                 describe: 镜像仓地址
               - name: routeReflectorClusterID
                 value: 244.0.0.1
                 type: string
                 describe: bgp route reflector clusterID
               - name: nodeNames
                 value: ""
                 type: string
                 describe: rr节点名称，填写集群中的节点名称，多个节点用空格分隔，建议选择2-3个节点
               - name: calicoVersion
                 value: 3.27.3
                 type: string
                 describe: calico版本
               - name: tigeraOperatorVersion
                 value: 1.32.7
                 type: string
                 describe: tigera operator版本
          - name: containerd
            param:
               - name: imageRepo
                 value: harbor.openfuyao.com/openfuyao
                 type: string
                 describe: containerd config镜像地址
               - name: rootPath
                 value: /var/lib/containerd
                 type: string
                 describe: containerd数据存放路径
               - name: statePath
                 value: /run/containerd
                 type: string
                 describe: containerd运行状态存放路径
               - name: version
                 value: 1.7.14
                 type: string
                 describe: containerd版本
          - name: helm
            param:
               - name: version
                 value: 3.14.2
                 type: string
                 describe: helm版本
          - name: ipset
            param: []
          - name: tar
            param: []
          - name: iptables
            param: []
          - name: openssl
            param: []
          - name: network-manager
            param: []
          - name: socat
            param: []
          - name: conntrack
            param: []
          - name: ipvsadm
            param: []
          - name: calicoctl
            param:
               - name: version
                 value: 3.27.3
                 type: string
                 describe: calicoctl版本
          - name: yq
            param:
               - name: version
                 value: 4.43.1
                 type: string
                 describe: yq版本
          - name: step
            param:
               - name: version
                 value: 0.28.2
                 type: string
                 describe: step版本
          - name: kubectl
            param:
               - name: version
                 value: 1.28.15-large-scale-cluster
                 type: string
                 describe: kubectl版本
          - name: coredns
            param:
               - name: version
                 value: 1.10.1
                 type: string
                 describe: coredns版本。
               - name: SchedulerNodeNames
                 value: ""
                 type: string
                 describe: coredns pod调度到哪些节点。填写配置文件内节点名，多个节点名用英文逗号分隔。默认随机调度到控制平面节点
               - name: replicas
                 value: "3"
                 type: int
                 describe: coredns pod副本数
          - name: kubelet
            param:
               - name: version
                 value: 1.28.15-large-scale-cluster
                 type: string
                 describe: kubelet版本
          - name: kube-proxy
            param:
               - name: kubeProxyMode
                 value: ipvs
                 type: string
                 describe: kube-proxy模式
               - name: ipvsScheduler
                 value: lc
                 type: string
                 describe: kube-proxy ipvs算法
               - name: version
                 value: 1.28.15
                 type: string
                 describe: kube-proxy版本
          - name: etcd
            param:
               - name: version
                 value: 3.5.18
                 type: string
                 describe: etcd版本
          - name: kube-scheduler
            param:
               - name: version
                 value: 1.28.15-large-scale-cluster
                 type: string
                 describe: kube-scheduler版本
               - name: leaderElectionLeaseDuration
                 value: 65s
                 type: string
                 describe: 选主租约时长，超过此时长开始选主
               - name: leaderElectionRenewDeadline
                 value: 60s
                 type: string
                 describe: 选主租约续约时间，超过此时长开始续约
               - name: kubeAPIBurst
                 value: "10000"
                 type: int
                 describe: 与apiserver服务器通信时突发峰值请求个数上限(次/s)
               - name: kubeAPIQPS
                 value: "5000"
                 type: int
                 describe: 与apiserver服务器通信时每秒请求个数上限(次/s)
               - name: featureGates
                 value: ComponentSLIs=false
                 type: string
                 describe: 是否允许抓取健康指标
               - name: goMemLimit
                 value: 67500MiB
                 type: string
                 describe: go运行时可用的总体内存限制。建议配置总内存的85%
               - name: goMaxProcs
                 value: "32"
                 type: int
                 describe: 设置运行时核心数量。建议配置为环境cpu核心数
          - name: local-dns
            param:
               - name: imageRepo
                 value: harbor.openfuyao.com/openfuyao
                 type: string
                 describe: local-dns拉取镜像时镜像仓库地址
               - name: version
                 value: 1.25.0
                 type: string
                 describe: local-dns版本
          - name: kube-apiserver
            param:
               - name: imageRepo
                 value: harbor.openfuyao.com/openfuyao
                 type: string
                 describe: kube-apiserver拉取镜像时镜像仓库地址
               - name: apiserverBindPort
                 value: "6443"
                 type: string
                 describe: kube-apiserver端口号
               - name: version
                 value: 1.28.15-large-scale-cluster
                 type: string
                 describe: kube-apiserver version
               - name: podSubnet
                 value: 109.0.0.0/8
                 type: string
                 describe: pod ip地址段.
               - name: clusterDNS
                 value: 169.254.20.10
                 type: string
                 describe: clusterDNS ip 地址.
               - name: maxRequestsInflight
                 value: "10000"
                 type: string
                 describe: kube-apiserver 最大并发请求数(次/s)
               - name: maxMutatingRequestsInflight
                 value: "5000"
                 type: string
                 describe: kube-apiserver 最大并发变更请求数(次/s)
               - name: etcdMaxCallRecvMsgSize
                 value: "2147483647"
                 type: int
                 describe: 客户端请求etcd可接收的最大字节数
               - name: etcdMaxCallSendMsgSize
                 value: "110100480"
                 type: int
                 describe: 客户端请求etcd可发送的最大字节数
               - name: hwAccessLogCheckLogDeletedPeriod
                 value: "5"
                 type: int
                 describe: 访问日志重建扫描周期
               - name: hwAccessLogPermissions
                 value: "0600"
                 type: string
                 describe: 访问日志文件权限
               - name: auditLogMaxAge
                 value: "30"
                 type: int
                 describe: 审计日志最大保留天数
               - name: auditLogMaxBackup
                 value: "50"
                 type: int
                 describe: 审计日志最大保留个数
               - name: auditLogMaxSize
                 value: "10"
                 type: int
                 describe: 审计日志文件大小(MB)
               - name: auditLogMode
                 value: batch
                 type: string
                 describe: 审计日志保存模式
               - name: goMaxProcs
                 value: "32"
                 type: int
                 describe: 设置运行时核心数量，多节点场景下建议配置为cpu核心数
          - name: kube-controller-manager
            param:
               - name: version
                 value: 1.28.15-large-scale-cluster
                 type: string
                 describe: kube-controller-manager版本
               - name: leaderElectionLeaseDuration
                 value: 65s
                 type: string
                 describe: 选主租约时长，超过此时长开始选主
               - name: leaderElectionRenewDeadline
                 value: 60s
                 type: string
                 describe: 选主租约续约时间，超过此时长开始续约
               - name: kubeAPIBurst
                 value: "15000"
                 type: int
                 describe: 与apiserver服务器通信时突发峰值请求个数上限(次/s)
               - name: kubeAPIQPS
                 value: "10000"
                 type: int
                 describe: 与apiserver服务器通信时每秒请求个数上限(次/s)
               - name: featureGates
                 value: ComponentSLIs=false
                 type: string
                 describe: 是否允许抓取健康指标
               - name: nodeMonitorPeriod
                 value: 5s
                 type: string
                 describe: cloud-node-lifecycle-controller对节点状态进行同步的周期
               - name: nodeMonitorGracePeriod
                 value: 40s
                 type: string
                 describe: 在将一个Node标记为不健康之前允许其无限响应的时长上限
               - name: goMemLimit
                 value: 67500MiB
                 type: string
                 describe: go运行时可用的总体内存限制。建议配置总内存的85%
               - name: goMaxProcs
                 value: "32"
                 type: int
                 describe: 设置运行时核心数量。建议配置为环境cpu核心数
          - name: kubeadm
            param:
               - name: version
                 value: 1.28.15-large-scale-cluster
                 type: string
                 describe: kubeadm版本
          - name: haproxy
            param:
               - name: loadBalancerPort
                 value: "6443"
                 type: string
                 describe: haproxy监听端口
          - name: keepalived
            param:
               - name: vip
                 value: 192.168.200.241/24  # 虚拟ip
                 type: string
                 describe: 虚拟ip, 如192.168.200.241/24, 注意不要与其他集群重复且要指定网段
               - name: interface  # vip绑定的网卡，要求loadbalance-0和loadbalance-1都绑定同一网卡
                 value: eth0
                 type: string
                 describe: 为vip指定的网卡
          - name: apull
            param: []
    ```

   - 若集群节点数量较多，可执行如下命令从execl文件内读取节点信息
     - 若不填写--hostnameCol，在安装集群时默认使用宿主机的hostname做为集群内节点名字  
     - 若不填写--portCol，默认连接机器的端口号为22  
     - 若不填写--userCol，默认登录机器使用的用户名为root  
     - 若不填写--rowStart，默认从第一行开始  
     - 若不填写--rowEnd，默认读取到最后一行  
     - --ipCol和--passwdCol为必填参数  
     - 若表格内roleCol列值为空(不填写任何字符)，默认节点权限为worker。表格内roleCol列每一行只能填写一个节点权限名或不填写节点权限名  
     - 表格sheetName必须为Sheet1  

   ```shell
   ./fuyao create cluster-config -f /root/nodes.xlsx --all-params --hostnameCol xx --ipCol xx --passwdCol xx --portCol xx --roleCol xx --userCol xx --rowStart xx --rowEnd xx
   ```

4. 在安装包目录内执行安装。

   ```
   ./fuyao create cluster -f example-cluster.yaml
   ```

5. 确认安装结果。

   - 查看机器上所有pod信息，当所有pod状态都为running后表示环境准备完毕。

   ```
   kubectl get pod -A
   ```

   >  **说明：**<br />
   > - 所有pod状态都为running状态则安装完毕。具体时间与实际网络情况有关，网速较慢时等待时间会比较长。

#### 在线卸载集群

1. 以root身份登录安装执行机。

2. 执行如下命令，进入安装包目录。

   ```shell
   cd $INSTALLER_PACKAGE_PATH
   ```

   编写配置文件，配置文件修改内容与安装时一致，填写集群中master与worker节点的hostname、user、ip、port、privateKeyPath

   ```shell
   vi example-cluster.yaml
   ```

   > **说明：**<br />若集群中节点无改变，可直接使用安装集群时的配置文件进行集群卸载。

3. 在安装包目录内执行卸载。

    - 镜像仓库在执行命令节点

   ```shell
   ./fuyao destroy cluster -f example-cluster.yaml
   ```

   > 若出现长时间未完成卸载请重新执行卸载命令

4. 出现`Successfully completed cluster destroy`字样后即完成集群卸载。

   卸载操作未删除安装与卸载过程中打印的日志，日志存放路径为`/var/log/fuyao-installer/`，若想删除日志可执行如下命令：

   ```shell
   rm -rf /var/log/fuyao-installer/
   ```

### 操作步骤-离线安装

#### 离线安装节点规划示例

> 此规划示例部署形态可支持4000节点kubernetes集群，若安装更大规模集群请上调master、ha、etcd data与etcd events节点CPU、内存资源

| ip              | hostname       | 组件                   | 角色          | CPU(vCPU) | 内存(BG) |
|-----------------|----------------|----------------------|-------------|-----------|--------|
| 192.168.100.240 | offline-node1  | keepalived、haproxy   | ha节点        | 32        | 64     |
| 192.168.100.239 | offline-node2  | keepalived、haproxy   | ha节点        | 32        | 64     |
| 192.168.100.238 | offline-node3  | apiserver            | master      | 16        | 32     |
| 192.168.100.237 | offline-node4  | apiserver            | master      | 16        | 32     |
| 192.168.100.236 | offline-node5  | apiserver            | master      | 16        | 32     |
| 192.168.100.235 | offline-node6  | apiserver            | master      | 16        | 32     |
| 192.168.100.234 | offline-node7  | apiserver            | master      | 16        | 32     |
| 192.168.100.233 | offline-node8  | apiserver            | master      | 16        | 32     |
| 192.168.100.232 | offline-node9  | control-manager      | master      | 16        | 32     |
| 192.168.100.231 | offline-node10 | control-manager      | master      | 16        | 32     |
| 192.168.100.230 | offline-node11 | kube-scheduler       | master      | 16        | 32     |
| 192.168.100.229 | offline-node12 | kube-scheduler       | master      | 16        | 32     |
| 192.168.100.228 | offline-node13 | etcd                 | etcd data   | 16        | 32     |
| 192.168.100.227 | offline-node14 | etcd                 | etcd data   | 16        | 32     |
| 192.168.100.226 | offline-node15 | etcd                 | etcd data   | 16        | 32     |
| 192.168.100.225 | offline-node16 | etcd                 | etcd events | 16        | 32     |
| 192.168.100.224 | offline-node17 | etcd                 | etcd events | 16        | 32     |
| 192.168.100.223 | offline-node18 | etcd                 | etcd events | 16        | 32     |
| 192.168.100.222 | offline-node19 | coredns、kubeproxy等组件 | worker      | 16        | 32     |
| 192.168.100.221 | offline-node20 | coredns、kubeproxy等组件 | worker      | 16        | 32     |
| 192.168.100.220 | offline-node21 | coredns、kubeproxy等组件 | worker      | 16        | 32     |
| 192.168.100.219 | offline-node22 | calico、kubeproxy等组件  | worker      | 16        | 32     |
| 192.168.100.218 | offline-node23 | calico、kubeproxy等组件  | worker      | 16        | 32     |
| 192.168.100.217 | offline-node24 | 业务组件                 | worker      | 16        | 32     |
| 192.168.100.216 | offline-node25 | 业务组件                 | worker      | 16        | 32     |
| 192.168.100.215 | offline-node26 | harbor               | 离线镜像仓库      | 16        | 32     |
| 192.168.100.214 | online-node1   | containerd、nerdctl等  | 在线导制品节点     | 8         | 16     |
| 192.168.100.242 | 无              | 无                    | 高可用部署虚拟IP   | /         | /      |

1. 以root用户登录online-node1节点。

2. 下载安装工具与安装脚本。

   - 自动下载

    ```
    curl -sfL https://openfuyao.obs.cn-north-4.myhuaweicloud.com/openFuyao/installer/releases/download/large-cluster/download.sh | bash -
    ```

   执行上述命令默认下载最新版本的安装包，若想下载其他版本，请使用如下命令指定版本后进行下载。

    ```
    export VERSION=x.y.z

    curl -sfL https://openfuyao.obs.cn-north-4.myhuaweicloud.com/openFuyao/installer/releases/download/large-cluster/download.sh | bash -
    ```

    - 手动构建  
        - 手动构建要求执行命令机器系统为linux且机器已联网  
        - 手动构建前需要配置gitcode ssh 公钥，[参考链接](https://docs.gitcode.com/docs/help/home/user_center/security_management/ssh/)

   ```shell
   VERSION="1.21.8"
   ARCH=$(uname -m)
   case $ARCH in
       x86_64) ARCH="amd64";;
       aarch64) ARCH="arm64";;
   esac
   
   mkdir -p $HOME/tools && cd $HOME/tools
   which wget || yum install wget -y
   wget https://golang.google.cn/dl/go${VERSION}.linux-${ARCH}.tar.gz
   which tar || yum install tar -y
   rm -rf /usr/local/go && tar -C /usr/local -xzf go${VERSION}.linux-${ARCH}.tar.gz
   echo 'export PATH=$PATH:/usr/local/go/bin' >> ~/.bashrc
   source ~/.bashrc
   
   which git || yum install git -y
   
   git clone git@gitcode.com:openFuyao/installer.git
   cd installer
   git checkout large-cluster
   
   go build -o fuyao -tags netgo,osusergo
   sudo ./fuyao create component-config  # 此处可修改fuyao-component-config.yaml文件，用来构建不同操作系统的安装包
   sudo ./fuyao create component-package -f fuyao-component-config.yaml
   # 构建出的安装包存放在./targets文件夹内，安装包包括fuyao二进制与asset文件夹，asset文件夹存放配置文件与对应操作系统的安装脚本压缩包
   cd ./targets/ && ls
   ```

3. 导出离线安装所需制品。

   - 进入安装包目录。

   > **说明：**<br />
   > 此处以openEuler22.03 arm64为例，具体路径应视实际情况而定。

    ```shell
    export INSTALLER_PACKAGE_PATH=fuyao-v0.1.0-openEuler-22.03-arm64

    cd $INSTALLER_PACKAGE_PATH
    ```

   - 在安装包目录执行命令导出制品。

    ```shell
       ./fuyao export artifact
    ```

   - 查看制品。
      - 制品默认存放路径为二进制存放路径。
      - fuyao-offline-artifact.tar.gz即为制品文件。

     ```yaml
     ls
     ```

4. 将下载的制品和安装包通过U盘等介质复制至离线环境安装节点。

5. 以root用户登录离线offline-node26节点。

6. 进入到安装包目录。

   > **说明：**<br />
   > 此处以openEuler22.03 arm64为例，具体路径应视实际情况而定。

    ```shell
    export INSTALLER_PACKAGE_PATH=fuyao-v0.1.0-openEuler-22.03-arm64

    cd $INSTALLER_PACKAGE_PATH
    ```

7. 创建离线镜像仓并上传制品。

   - 将离线镜像仓库创建在执行命令节点
    ```shell
   # 离线制品路径根据实际情况修改
    export OPENFUYAO_ARTIFACT_PATH=/root/fuyao-offline-artifact.tar.gz

    ./fuyao create registry --artifact-path $OPENFUYAO_ARTIFACT_PATH
    ```

   - 将离线镜像仓库创建在其他节点
    ```shell
   # 离线制品路径根据实际情况修改
    export OPENFUYAO_ARTIFACT_PATH=/root/fuyao-offline-artifact.tar.gz

    ./fuyao create registry --artifact-path $OPENFUYAO_ARTIFACT_PATH --registry-node-host 192.168.0.12 --registry-node-user root --registry-node-passwd xxxxx
    ```

   - `--artifact-path`后需填写制品存放路径。
   - 待出现`successfully`字样则表明已创建离线镜像仓并上传制品。

   >**说明：**<br />
   >离线harbor采用https方式访问，故而采用自签名方式生成证书，证书存放路径为`/etc/ssl/certs/openfuyao-harbor`与`/etc/ssl/certs/openfuyao-harbor/openfuyao-harbor-ca-bk.crt`，
   >在安装集群过程中会复制harbor证书到各节点，复制到各节点中的证书存放路径为`/etc/ssl/certs/openfuyao-harbor/openfuyao-harbor-ca.crt`
   > 完成安装后会在asset文件夹内创建registry-config.yaml文件，用于存放harbor ip、域名与证书存放路径，在执行离线安装时会从该文件夹内读取离线镜像仓库的配置，请勿删除该文件。
   > 若离线部署使用外部镜像仓库，可在registry-config.yaml文件内配置外部镜像仓库ip、domain、caPath、originDomain，其中domain带有镜像仓库的project，originDomain不带有镜像仓库project。目前仅支持使用https方式访问镜像仓库。

8. 生成集群规划配置文件。

   - 执行命令，生成配置文件。

    ```shell
    ./fuyao create cluster-config
    ```

   - 按需修改集群规划配置文件(可参考在线部署配置文件样例修改，在、离线部署配置文件相同)

9. 在安装包目录内执行安装。

   - 离线安装需要额外添加`--offline`标志，并指定制品路径。
   - 需在执行离线创建镜像仓库的安装包内执行，因为此安装包asset目录中registry-config.yaml文件记录了离线镜像仓库信息，在离线安装时会读取该文件。

   ```
   ./fuyao create cluster -f example-cluster.yaml --offline --artifact-path $OPENFUYAO_ARTIFACT_PATH
   ```

10. 安装结果确认。

    - 查看机器上所有pod信息，当所有pod状态都为running后表示环境准备完毕。

     ```
     kubectl get pod -A
     ```

    >  **说明：**
    >  - 所有pod状态都为running状态则安装完毕。具体时间与实际网络情况有关，网速较慢时等待时间会比较长。

#### 离线卸载集群

1. 以root身份登录安装执行机。

2. 执行如下命令，进入安装包目录。

   ```shell
   cd $INSTALLER_PACKAGE_PATH
   ```

   编写配置文件，配置文件修改内容与安装时一致，填写集群中master与worker节点的hostname、user、ip、port、privateKeyPath

   ```shell
   vi example-cluster.yaml
   ```

   > **说明：**<br />若集群中节点无改变，可直接使用安装集群时的配置文件进行集群卸载。

3. 在安装包目录内执行卸载。

    - 镜像仓库在执行命令节点

   ```shell
   ./fuyao destroy cluster -f example-cluster.yaml --offline
   ```
   
    > 若出现长时间未完成卸载请重新执行卸载命令
    >  此命令会默认卸载镜像仓库，因为加了--offline标志，默认认为离线镜像仓库在执行卸载命令节点，如不想卸载离线镜像仓库请去掉--offline参数

    - 镜像仓库在其他节点

   ```shell
   ./fuyao destroy cluster -f example-cluster.yaml --offline --registry-node-host xxx --registry-node-passwd xxx --registry-node-user xxx 
   ```

   > 若出现长时间未完成卸载请重新执行卸载命令

4. 出现`Successfully completed cluster destroy`字样后即完成集群卸载。

   卸载操作未删除安装与卸载过程中打印的日志，日志存放路径为`/var/log/fuyao-installer/`，若想删除日志可执行如下命令：

   ```shell
   rm -rf /var/log/fuyao-installer/
   ```
