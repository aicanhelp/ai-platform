## openFuyao 全量组件清单

| 所属sig                    | 功能分类           | 组件名（二进制名称）                  | 功能简介                                                 | 语言           | 代码仓 | 是否核心组件 | 能否独立发布使用 |
|--------------------------|----------------|-----------------------------|------------------------------------------------------|--------------|------- |--------|----------|
| sig-Installation         | ClusterAPI安装部署 | bkeadm                      | K8s容器编排工具，为用户提供便捷高效的容器编排和应用部署能力                      | Golang       | https://gitcode.com/openFuyao/bkeadm | 是      ||
| sig-Installation         | ClusterAPI安装部署 | bkeagent                    | 集群节点部署的系统级服务进程，执行bke-controller下发的内置指令、ke8s指令和shell指令 | Golang       | https://gitcode.com/openFuyao/bkeagent | 是      ||
| sig-Installation         | ClusterAPI安装部署 | bke-controller-manager      | 自定义集群部署的CRD资源的调谐处理                                   | Golang       | https://gitcode.com/openFuyao/cluster-api-provider-bke | 是      ||
| sig-Installation         | ClusterAPI安装部署 | bke-manifests | 存放产品组件的各个部署版本的yaml文件 | Shell/Makefile | https://gitcode.com/openFuyao/bke-manifests | 是 ||
| sig-Installation | ClusterAPI安装部署 | bke-console-website | 引导集群前端管理面，为集群生命周期管理提供前端用户界面 | JavaScript | https://gitcode.com/openFuyao/bke-console-website | 是 ||
| sig-Installation | ClusterAPI安装部署 | bke-console-service | 引导集群管理的后端服务 | Golang | https://gitcode.com/openFuyao/bke-console-service | 是 ||
| sig-Installation         | ClusterAPI安装部署 | installer-service           | 集群生命周期管理后端处理服务                                       | Golang       | https://gitcode.com/openFuyao/installer-service | 是      ||
| sig-Installation         | ClusterAPI安装部署 | installer-website           | 集群生命周期管理前端处理服务                                       | JavaScript   | https://gitcode.com/openFuyao/installer-website | 是      ||
| sig-Installation         | ClusterAPI安装部署 | openfuyao-system-controller | 用于在k8s集群上快速部署、卸载openFuyao管理面                         | Shell        | https://gitcode.com/openFuyao/openfuyao-system-controller | 是      ||
| sig-Installation         | ClusterAPI安装部署 | k3s                  | CNCF 沙箱项目，一个经过认证的、轻量级的 Kubernetes 发行版 | Golang | https://github.com/k3s-io/k3s/tree/v1.25.16%2Bk3s4 | 是      |三方件|
| sig-Installation | ClusterAPI安装部署 | registry | 存储、管理和分发容器镜像 | Golang | https://github.com/distribution/distribution/tree/v2.8.1 | 是 |三方件|
| sig-Installation | ClusterAPI安装部署 | nginx_yum_repo | 托管yum软件仓库的nginx web服务器 | Shell | https://github.com/nginx/docker-nginx/tree/1.23.0 | 是 |三方件|
| sig-Installation | ClusterAPI安装部署 | chartmuseum | 私有的Helm Chart存储、管理和分发服务 | Golang | https://github.com/helm/chartmuseum/tree/v0.16.2 | 是 |三方件|
| sig-Installation | ClusterAPI安装部署 | nfs-server-alpine | 部署网络文件系统服务器 | Shell | https://github.com/ehough/docker-nfs-server | 是 |三方件|
| sig-Installation | ClusterAPI安装部署 | capi-controller-manager | 集群的Kubernetes控制器 | Golang | https://github.com/kubernetes-sigs/cluster-api/tree/v1.4.3 | 是 |三方件|
| sig-Installation | ClusterAPI安装部署 | kubeadm-bootstrap-controller | Kubeadm引导程序提供商的控制器 | Golang | https://github.com/kubernetes-sigs/cluster-api/tree/v1.4.3 | 是 |三方件|
| sig-Installation | ClusterAPI安装部署 | kubeadm-control-plane-controller | 负责管理工作负载集群控制平面的控制器 | Golang | https://github.com/kubernetes-sigs/cluster-api/tree/v1.4.3 | 是 |三方件|
| sig-Installation | ClusterAPI安装部署 | pause | pod基础设施容器 | C | https://github.com/kubernetes/kubernetes/tree/master/build/pause | 是 |三方件|
| sig-Installation | ClusterAPI安装部署 | haproxy | TCP/HTTP负载均衡 | C | https://github.com/haproxy/haproxy/tree/v2.1-dev4 | 是 |三方库|
| sig-Installation | ClusterAPI安装部署 | keepalived | 高可用和负载均衡 | C | https://github.com/acassen/keepalived/tree/v1.3.5 | 是 |三方件|
| sig-Installation | ClusterAPI安装部署 | console-website | 前端管理面，为应用管理、资源管理、告警监控等openFuyao核心功能提供前端用户界面 | JavaScript | https://gitcode.com/openFuyao/console-website | 是 ||
| sig-Installation | ClusterAPI安装部署 | console-service | 集群管理的后端服务 | Golang | https://gitcode.com/openFuyao/console-service | 是 ||
| sig-Distributed          | 分布式框架          | ray                         | 扩展AI和Python应用程序的统一框架，包含一个核心分布式运行时环境和一组用于简化机器学习计算的AI库 | Python/C++   |https://gitcode.com/openFuyao/ray | 否      ||
| sig-Distributed          | 扩展组件           | ray-service                 | 以扩展组件形式为openFuyao提供ray相关能力，同时支持其他兼容平台独立部署            | Golang       |https://gitcode.com/openFuyao/ray-service | 否      ||
| sig-Distributed          | 扩展组件           | ray-website                 | ray-service在openFuyao管理面的前端界面                        | JavaScript   | https://gitcode.com/openFuyao/ray-website | 否      ||
| sig-AI-Inference         | 集成方案           | ai-inference-integration               | LLM推理加速集成套件                   | Golang | https://gitcode.com/openFuyao/ai-inference-integration | 否      ||
| sig-AI-Inference         | 集成方案           | AI-all-in-one               | AI一体机集成解决方案，支持基础LLM推理全栈与DeepSeek模型                   | Python/Shell | https://gitcode.com/openFuyao/ai-all-in-one | 否      ||
| sig-AI-Inference         | 扩展组件           | mooncake               | 提供分布式KVCache管理能力，在原生moon cake基础上进行增强，优先回合上游 | C++       |https://gitcode.com/openFuyao/mooncake| 三方件      ||
| sig-AI-Inference         | 扩展组件           | cache-indexer               | 提供分布式KVCache全局索引能力 | Python       |https://gitcode.com/openFuyao/cache-indexer| 否      ||
| sig-AI-Inference         | 扩展组件           | vllm-ascend               | LLM推理引擎、含昇腾适配，推理引擎增强 | Python       |https://gitcode.com/openFuyao/vllm-ascend| 三方件      ||
| sig-AI-Inference         | 扩展组件           | hermes-router               | 提供AI推理场景下的路由方案，包含基础路由模块，用户配置以及基于kv-aware的带资源与负载感知的路由策略 | Golang       |https://gitcode.com/openFuyao/hermes-router| 否      ||
| sig-AI-Inference         | 扩展组件           | mis-management              | 提供与昇腾推理微服务的交互能力，包含大模型推理实例、模型文件管理等功能                  | Golang       |https://gitcode.com/openFuyao/mis-management| 否      ||
| sig-AI-Inference         | 扩展组件           | mis-management-website      | mis-management在openFuyao管理面的前端界面                     | JavaScript   |https://gitcode.com/openFuyao/mis-management-website| 否      ||
| sig-AI-Inference         | 扩展组件           | eagle-eye      | 推理场景近实时可观测子系统                     | Golang   |https://gitcode.com/openFuyao/eagle-eye| 否      ||
| sig-orchestration-engine | 容器编排           | kube-apiserver              | kubernetes集群的服务入口，负责公开了 Kubernetes API，处理请求，将数据写入存储(ETCD) | Golang       |https://gitcode.com/openFuyao/kubernetes| 是      ||
| sig-orchestration-engine | 容器编排           | kube-controller-manager     | 集群默认资源如：node、pod、PV等资源的控制器，负责使资源对象达到预期的状态            | Golang       |https://gitcode.com/openFuyao/kubernetes| 是      ||
| sig-orchestration-engine | 容器编排           | kube-scheduler              | 集群中的Pod调度器，监听新Pod，根据资源需求、策略等为其选择最合适的节点运行             | Golang       |https://gitcode.com/openFuyao/kubernetes| 是      ||
| sig-orchestration-engine | 容器编排           | kubelet                     | 集群中运行在每个节点上的代理组件，负责维护节点上的Pod生命周期，与容器运行时交互            | Golang       |https://gitcode.com/openFuyao/kubernetes| 是      ||
| sig-orchestration-engine | 容器编排           | kube-proxy                  | 集群中每个节点上所运行的网络代理，维护节点上的一些网络规则，做流量转发与通信               | Golang       |https://gitcode.com/openFuyao/kubernetes| 是      ||
| sig-orchestration-engine | 容器编排           | kubectl                     | Kubernetes提供的使用API与集群的控制面进行通信的命令行工具                  | Golang       |https://gitcode.com/openFuyao/kubernetes| 是      ||
| sig-orchestration-engine | 容器编排           | etcd                        | 一个高可用的键值数据库，存储了集群所有的状态和配置数据                          | Golang       |https://gitcode.com/openFuyao/etcd| 是      ||
| sig-orchestration-engine | 容器编排           | coredns                     | 集群的内部DNS服务器，为Service和Pod提供域名解析，是实现服务发现的核心组件          | Golang       |https://gitcode.com/openFuyao/coredns| 是      ||
| sig-orchestration-engine | 容器编排           | kubeone                     | 将kubernetes、etcd、coredns合并在一个进程中的极致轻量化的kubernetes容器编排底座 | Golang       |https://gitcode.com/openFuyao/kubeone| 否      ||
| sig-orchestration-engine | 容器运行时          | containerd                  | 标准的容器运行时，负责在主机上管理容器的完整生命周期                           | Golang       |https://gitcode.com/openFuyao/containerd| 是      ||
| sig-orchestration-engine | 网络插件           | villus                      | 加载数据面并为数据面提供Pod和Service路由信息，为Pod分配IP等                | Golang       |https://gitcode.com/openFuyao/villus| 是      ||
| sig-orchestration-engine | 网络插件           | villus-cni                  | CNI插件，与villus交互获取分配IP并为Pod创建网卡                       | Golang       |https://gitcode.com/openFuyao/villus| 是      ||
| sig-orchestration-engine | 存储插件           | localcsi-controller         | 负责管理本地存储资源的控制器组件                                     | Golang       |待创建| 否      ||
| sig-orchestration-engine | 存储插件           | localcsi-node               | 运行在节点上的负责节点存储资源分配与挂载的agent组件                         | Golang       |待创建| 否      ||
| sig-orchestration-engine | 扩展组件           | pod-lm-controller           | Pod热迁移的控制器组件，负责管理Pod热迁移完整流程                          | Golang       |待创建| 否      ||
| sig-orchestration-engine | 扩展组件           | pod-lm-agent                | Pod热迁移的节点级组件，负责Pod热迁移的具体操作，如内存数据dump、传输、restore等     | Golang       |待创建| 否      ||
| sig-orchestration-engine | 扩展组件           | node-checker                | 负责OS一致性检查的工具                                         | Golang       |https://gitcode.com/openFuyao/node-checker| 否      ||
| sig-Colocation           | 扩展组件           | colocation-website          | 在离线混部组件在openFuyao管理面的前端界面                            | JavaScript   |https://gitcode.com/openFuyao/colocation-website| 否      ||
| sig-Colocation           | 扩展组件           | colocation-service          | 在离线混部的后端，负责配置管理、混部监控以及节点混部功能支持情况的状态聚合                | Golang       |https://gitcode.com/openFuyao/colocation-service| 否      ||
| sig-Colocation           | 扩展组件           | colocation-management       | 在离线混部的资源超卖管理、负载准入管理                                  | Golang       |https://gitcode.com/openFuyao/colocation-management| 否      ||
| sig-Large-Scale-Cluster  | 调度组件           | volcano                     | 用于批量调度AI作业                                           | Golang       |https://gitcode.com/openFuyao/volcano-ext| 否      | 是        |
| sig-Large-Scale-Cluster  | 自动化部署          | clusters-computing-operator | 自动化部署超大规模集群多集群                                       | Golang       |https://gitcode.com/openFuyao/clusters-computing-operator|否      | 是      |
| sig-Large-Scale-Cluster  | 自动化部署          | ascend-deployer             | 昇腾软件栈自动化部署工具                                         | Python       |https://gitcode.com/openFuyao/ascend-deployer|否      | 是      |
| sig-Large-Scale-Cluster  | 模拟组件           | kwok                        |    模拟Kubernetes节点和Pod                                                  | Golang       |https://gitcode.com/openFuyao/kwok| 否      |是      |
| sig-Large-Scale-Cluster  | AI集群系统软件           | mind-cluster                | 支持NPU训练和推理硬件的深度学习组件，提供NPU集群作业调度、运维监测、故障恢复等功能                                         | Golang       |https://gitcode.com/openFuyao/mind-cluster| 否      |是      |
| sig-Container-Platform   | 内置认证服务器      | oauth-server                | openFuyao容器平台中内置的认证服务器，负责校验平台用户登陆密码，生成授权码与token。支持用户登录、退出登录、首次登录强制修改用户密码，修改用户密码等功能                                         | Golang       | https://gitcode.com/openFuyao/oauth-server                | 是      | 否        |
| sig-Container-Platform   | 内置认证webhook服务 | oauth-webhook               | 作为openFuyao容器平台中kube-apiserver的认证webhook服务，校验openFuyao平台中oauth-server生成的access_token                                         | Golang       | https://gitcode.com/openFuyao/oauth-webhook               | 是      | 否        |
| sig-Container-Platform   | 服务认证边车       | oauth-proxy                 | 用于openFuyao的扩展组件认证鉴权，以边车的形式随扩展组件一同部署                                         | Golang       | https://gitcode.com/openFuyao/oauth-proxy                 | 否      | 否        |
| sig-Container-Platform   | 用户管理         | user-management-operator    | openFuyao容器平台中用户管理组件，为容器平台提供了一整套灵活的用户和角色管理功能。该系统支持用户的增删改查以及角色的分配，查看与解除绑定等               | Golang       | https://gitcode.com/openFuyao/user-management-operator    | 是      | 否        |
| sig-Container-Platform   | 监控组件         | monitoring-service          | openFuyao监控服务与Prometheus深度集成，内置常用指标查询能力，并提供监控目标（targets）和告警规则（alerting rules）的查看功能。               | Golang       | https://gitcode.com/openFuyao/monitoring-service          | 是      | 否        |
| sig-Container-Platform   | NUMA亲和扩展组件前端 | multi-cluster-website       | openFuyao多集群管理扩展组件前端，支持集群纳管、解纳管、集群标签管理以及服务跨集群访问               | JavaScript       | https://gitcode.com/openFuyao/multi-cluster-website       | 否      | 是        |
| sig-Container-Platform   | NUMA亲和扩展组件后端 | multi-cluster-service       | openFuyao多集群管理扩展组件后端，支持集群纳管、解纳管、集群标签管理以及服务跨集群访问            | Golang       | https://gitcode.com/openFuyao/multi-cluster-service       | 否      | 是        |
| sig-numa-affinity        | NUMA亲和扩展组件前端 | volcano-config-website     | NUMA亲和调度扩展组件前端               | JavaScript   | https://gitcode.com/openFuyao/volcano-config-website     | 否      | 是        |
| sig-numa-affinity        | NUMA亲和扩展组件后端 | volcano-config-service      | 面向性能敏感型业务，提供基于硬件拓扑感知的集群级和节点级资源调度和资源分配能力增强，显著提升集群作业性能            | Golang       | https://gitcode.com/openFuyao/volcano-config-service      | 否      | 是        |