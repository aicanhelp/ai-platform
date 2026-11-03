# 术语

| 中文名称 | 英文全称 | 缩略语               | 说明 |
|-----|---------|-------------------|------|
| 扩展件 | addon | - | 使用openFuyao社区BKE安装工具创建集群时需要安装的组件，比如calico。 |
| - | All-In-One | AIO               | 将Kubernetes与fuyao-system等组件均部署在同一节点。 |
| 准入Webhook | Admission Webhook | -                 | Kubernetes准入控制机制，在API对象持久化前拦截并修改（Mutating）或校验（Validating）请求。 |
| API网关 | API Gateway | APIG              | 位于客户端和API之间的一个单点入口，它作为反向代理将客户端的请求路由到后面的一组API。 |
| 应用程序编程接口 | Application Programming Interface | API               | 一些预先定义的函数，目的是提供应用程序与开发人员基于某软件或硬件得以访问一组例程的能力，而又无需访问源码，或理解内部工作机制的细节。 |
| 昇腾编程语言 | Ascend Computing Language | ACL | 提供运行时管理、单算子调用、模型推理、媒体数据处理等API，能够实现利用底层硬件计算资源，在CANN平台上进行深度学习推理计算、图形图像预处理、单算子加速计算等。 |
| 昇腾镜像仓库 | AscendHub | -                 | 昇腾开放Docker镜像仓库。 |
| 属性 | Attributes | -                 | OpenTelemetry中用于描述遥测数据上下文信息的键值对元数据，可附加在Span、Metric、Log等对象上，用于记录业务属性和运行环境信息。 |
| API服务器 | API Server | apiserver         | Kubernetes API服务器，提供集群的REST API接口。 |
| - | blackbox_exporter | -                 | Prometheus官方提供的exporter之一，可以提供HTTP、HTTPS、DNS、TCP以及ICMP的方式对网络进行探测。 |
| 引导节点 | Bootstrap Node | -                 | 集群初始化时创建的第一个节点，用于引导整个集群的创建过程。 |
| - | BatchTransfer | - | BatchTransfer封装了操作请求，具体负责将一个Segment中非连续的一组数据空间的数据和另外一组Segment的对应空间进行Read/Write数据同步。 |
| - | Mooncake CacheTier | CacheTier | Mooncake中Tier Backend架构下的不同层级缓存层，用于分层存放KVCache数据。 |
| - | cAdvisor | -                 | 一个谷歌开发的容器监控工具，它被内嵌到Kubernetes中作为监控组件使用。 |
| 云原生混部 | Cloud Native Colocation | -                 | 使用云原生的方式将在线业务和离线业务部署在同一个集群，通过调配在线业务在业务波谷和波峰时集群资源的使用，提升集群的整体资源利用率的一种部署方式。 |
| 云原生计算基金会 | Cloud Native Computing Foundation | CNCF              | 云原生计算基金会，一个开源软件基金会。 |
| 控制台 | Console | -                 | 前台Web页面控制台。 |
| 容器 | Container | -                 | 基于镜像创建的运行实例，它可以被启动、开始、停止、删除。每个容器都是相互隔离的、保证安全的平台。 |
| 容器内存共享                    | Container memory sharing               | -             | 基于UB内存池化机制，在裸机容器场景节点或者NUMA的内存使用率达到阈值时，触发内存借用，将一部分内存压力无感地分担到远端内存池上的技术。                                 |
| 容器内存借用                    | Container memory borrowing             | -             | UBS-Core的内存池化组件，支持通过内存池化能力将内存块在UBS Server集群内导入导出，以达到在裸机上跨节点以及多进程共享使用内存的目标。                                 |
|容器设备接口|Container Device Interface|CDI|容器运行时的设备注入标准。DRA插件完成设备分配后生成CDI规范文件，由containerd等容器运行时将设备节点、环境变量等挂载进业务容器。|
|通用表达式语言|Common Expression Language|CEL|Kubernetes中用于校验和筛选的表达式语言。npu-dra-plugin支持在ResourceClaim中使用CEL表达式，按NUMA亲和性、芯片型号等设备属性筛选NPU设备。| 
| 协调世界时 | Coordinated Universal Time | UTC               | UTC是一种时间标准，用于在全球范围内统一时间。 |
| - | CronJob | -                 | 创建基于时隔重复调度的Job。 |
| - | Custom Resource | CR                | Kubernetes中自定义资源。 |
| 自定义资源定义 | Custom Resource Definition | CRD               | Kubernetes一个资源扩展机制，允许定义定制资源。 |
| 证书颁发机构 | Certificate Authority | CA                | 负责签发和管理数字证书的权威机构。 |
| 证书签发请求 | Certificate Signing Request | CSR               | 包含证书申请者信息的文件，用于向CA申请证书。 |
| 通用名称 | Common Name | CN                | X.509证书中的字段，通常用于标识证书持有者的名称。 |
| 证书文件扩展名 | Certificate | CRT               | 通常用于存储X.509证书。 |
| 配置映射 | Configuration Map | ConfigMap         | Kubernetes中的配置对象，用于存储非敏感的配置数据。 |
| 证书撤销列表 | Certificate Revocation List | CRL               | 包含已被撤销证书的列表。 |
| 控制器管理器 | Controller Manager | controller-manager | Kubernetes控制器管理器，运行各种控制器来维护集群状态。 |
| - | Customstatlogger  | -  | vLLM对外开放了StatLoggerBase抽象类，能够自定义指标及指标上报方式。|
|-|DaVinci Card Management Interface|DCMI|昇腾NPU的设备管理API接口，提供设备发现、状态查询、固件升级等能力。npu-dra-plugin通过DCMI接口发现并上报NPU设备信息，DCMI不可用时回退使用npu-smi。|
| - | DaemonSet | -                 | 确保全部（或者某些）节点上运行一个Pod的副本。 |
| 监控看板 | Dashboard | -                 | 监控看板由多个用户自定义的监控组件组成，支持用户根据自身需求监控各种指标。 |
| 数据并行 | Data Parallelism | DP                | 每个设备上都会有一个模型的完整副本，每个设备独立地处理一部分数据集，然后将各自的梯度汇总起来。 |
| 解码阶段 | Decode | -                 | 生成首个token到推理停止的过程。 |
| - | Deployment | -                 | 为Pod和ReplicaSet提供声明式的更新能力。 |
| 设备类别 | DeviceClass | -                 | Kubernetes DRA框架中的一个集群级API对象，用于定义某类设备资源的抽象类别及其选择语义，以供ResourceClaim在请求资源时引用。 |
| 设备到设备传输 | Device-to-Device Transmission | D2D Transmission | 数据不经过CPU或主机内存，直接在同一加速卡内部或不同加速卡之间搬运的传输方式。 |
| 域名系统 | Domain Name System | DNS               | 将域名和IP地址相互映射的服务，可更好地进行网络访问。 |
| - | Dubbo | -                 | 阿里巴巴公司开源的一个高性能的服务框架，拥有高性能和透明化的RPC远程服务调用及服务治理方案。 |
| 动态资源分配 | Dynamic Resource Allocation | DRA               | Kubernetes提供的一种设备资源管理机制，用于通过结构化API对象实现设备资源的请求、调度与分配，并允许设备驱动参与资源选择与分配决策过程。 |
| 分布式追踪 | Distributed Tracing | -                 | 追踪请求在分布式系统中的完整执行路径，记录各环节耗时和关键属性，用于性能分析和故障诊断。 |
| - | DCGM Exporter                        | -              | 采集GPU运行与健康指标，包括GPU利用率、PCIe传输速率、温度、功率使用等信息。 |
| 嵌入 | Embedding | EMB               | 数据向量化嵌入操作。 |
| - | Elasticsearch | ES                | 分布式搜索和分析引擎，常用于存储和查询日志、指标和追踪数据。 |
| 扩展密钥用途 | Extended Key Usage | extKeyUsage       | 证书扩展字段，指定证书的特定用途（如服务器认证、客户端认证等）。 |
| - | etcd | -            | 分布式键值存储系统，Kubernetes用于存储集群状态和配置数据。 |
| 端点选择器 | EndpointPicker | EPP | 在Kubernetes Gateway API Inference Extension中，负责选择合适的后端实例的组件，支持基于不同路由策略的端点挑选。 |
|-|Felix|-|Calico的节点代理组件，运行在每个节点上，负责配置网络、路由和策略规则。|
| 完全限定域名 | Fully qualified domain name | FQDN              | 同时带有主机名和域名的名称，FQDN=Hostname+DomainName。 |
| 吉字节 | Gigabyte | GB                | 一种十进制的信息计量单位，常用于标识电脑硬盘、存储器等具有较大容量的储存媒介之储存容量。 |
| 网关推理扩展 | Gateway API Inference Extension | GIE | 在Kubernetes Gateway API基础上扩展推理相关能力，用于定义和管理推理服务的路由与流量策略。 |
| - | Helm | -                 | Kubernetes中的一种包管理器，用于简化在Kubernetes集群中部署和管理应用程序。 |
| - | Helm Chart | -                 | Helm的一个核心概念，它是一个预先配置好的应用程序资源包。 |
| 高可用 | High Availability | HA                | 系统或服务能够以高可靠性和持续可用性运行，即使在面临硬件故障或其他异常情况下也能保持正常运行。 |
| 高性能RSA加速引擎 | High Performance RSA Engine | HPRE              | KAE高性能RSA加速引擎模块。 |
| 高性能zlib/Gzip压缩引擎ZIP | High Performance ZIP Engine | ZIP               | KAE高性能zlib/Gzip压缩引擎模块。 |
|高带宽内存|High Bandwidth Memory|HBM|NPU专用的高带宽显存。npu-dra-plugin软切分场景下，通过memCapacity按1Mi步长为vNPU分配HBM内存配额。|
|Pod水平自动扩缩 | Horizontal Pod Autoscaler | HPA               | 自动更新工作负载资源（例如Deployment或者StatefulSet），目的是自动扩缩工作负载以满足需求。 |
| 主机到设备传输 | Host_to_Device Transmission | H2D Transmission | 从CPU/主机内存向NPU/GPU等加速设备显存复制数据的传输过程。 |
|华为缓存一致性系统|Huawei Cache Coherence System|HCCS|昇腾NPU之间的硬件直连互联链路，分为HCCS（直连）和HCCS_SW（经HCCS交换设备）两类连接，二者共同构成HCCS环，用于NPU拓扑亲和调度。|
| 超文本安全传输协议 | Hypertext Transfer Protocol Secure | HTTPS             | 以安全为目标的HTTP通道，在HTTP的基础上通过传输加密和身份认证保证了传输过程的安全性。 |
| 监控指标 | Indicator | -                 | 监控指标是数据采集系统（如普罗米修斯）支持用户进行监控的指标，一个监控指标可以包含多个监控实例。 |
| - | Ingress | -                 | 对集群中服务的外部访问进行管理的API对象，可以提供负载均衡、SSL终结和基于名称的虚拟托管。 |
| 监控实例 | Instance | -                 | 监控实例是Kubernetes上可被监控的最小粒度对象。每个监控实例被某些键值对标签的集合唯一标识。 |
| - | Job | -                 | 在集群中运行一次性任务，关注的是执行一次性任务而不是保持指定数量的实例运行。Job控制器会创建一个或多个Pod，以运行指定的任务。当任务完成后，Job控制器将删除Pod。 |
| - | Jaeger | -                 | 开源的分布式追踪系统，用于监控和故障排查微服务架构，由Uber开源并捐赠给CNCF。 |
| 键值缓存 | Key-Value Cache | KVCache          | 大模型推理加速的常用策略，通过缓存大模型推理过程中自注意力机制生成的Key（K）和Values（V）矩阵来工作，避免重复计算，提高推理速度。 |
| - | kube-apiserver | -                 | 验证并配置API对象的数据，这些对象包括Pods、Services、ReplicationControllers等。API服务器为REST操作提供服务，并为集群的共享状态提供前端，所有其他组件都通过该前端进行交互。 |
| - | kubectl | -                 | Kubernetes API与Kubernetes集群的控制面进行通信的命令行工具。 |
| - | kubelet | -                 | 是Kubernetes集群中的一个重要组件，运行在每个节点上，负责管理该节点上的容器。它是Kubernetes系统中的节点代理，与主控制平面中的控制器进行通信，以确保容器按照预期状态在节点上运行。 |
| - | Kube-rbac-proxy | -                 | 一个轻量级的HTTP代理服务，专为Kubernetes设计，它利用Kubernetes的SubjectAccessReview功能执行RBAC（基于角色的访问控制）授权。这个项目的目标是限制Pod之间的通信，只允许持有有效的RBAC授权令牌的Pod访问其他Pod。 |
| - | Kubernetes | K8s               | Kubernetes是一个可移植、可扩展的开源平台，用于管理容器化的工作负载和服务，方便进行声明式配置和自动化。 |
| - | kube-state-metrics | -                 | 采集API Server生成有关资源对象的状态指标，比如Deployment、Node、Pod。 |
| 鲲鹏加速引擎 | Kunpeng Accelerator Engine | KAE               | 鲲鹏加速引擎KAE（Kunpeng Accelerator Engine）是基于鲲鹏920处理器提供的硬件加速解决方案。 |
| 私钥文件扩展名 | Key | KEY               | 用于存储私钥。 |
| Kubernetes配置文件 | Kubernetes Configuration | kubeconfig        | 包含集群访问信息、认证信息和上下文配置。 |
| 密钥用途 | Key Usage | keyUsage          | 证书扩展字段，指定证书密钥的用途（如数字签名、密钥加密等）。 |
| - | kube-proxy | -       | Kubernetes网络代理，负责维护节点上的网络规则和负载均衡。 |
| 大语言模型 | Large Language Model | LLM               | 基于大量文本数据训练的深度学习模型。 |
| 负载均衡器 | Load Balancer | -                 | 用于在多个服务器之间分配网络流量的设备或服务。 |
| - | metrics-server | -                 | Kubernetes监控体系中的核心组件之一，负责从Kubelet收集资源指标，然后对这些指标监控数据进行聚合（依赖kube-aggregator），并在Kubernetes API Server中通过Metrics API（/apis/metrics.k8s.io/）公开暴露它们，但是metrics-server只存储最新的指标数据（CPU/Memory）。 |
| 昇腾推理微服务 | Mind Inference Service | MIS               | 昇腾提供的基于容器化部署的大模型推理API服务。 |
| 昇腾推理微服务调度组件 | Mind Inference Service Operator | MIS-Operator      | 实现推理微服务实例生命周期管理的组件。 |
| 众核 | Multi-core | -                 | 指在单一芯片上集成大量处理核心。众核场景特指集群中CPU数大于256的节点。 |
| 双向TLS验证 | Mutual TLS | mTLS              | 在服务端和客户端之间使用双向加密通道。 |
| - | Mooncake | -        | 开源社区，并且提出一种以KVCache为中心的LLM服务解耦架构。 |
| - | Mooncake Store | -        | 专为LLM推理场景设计的高性能分布式键值KVCache存储引擎。 |
| - | Mooncake Store Master Service | Master Service        |  Mooncake Store中负责管理整个集群的逻辑存储空间池，并处理节点的加入与退出事件。 |
| - | Mooncake Store Client | Mooncake Client        | Mooncake Store的客户端，负责由上层应用调用发起get/put请求以及提供实际KVCache存储。 |
| - | Mooncake Transfer Engine | TE        | 是一个围绕Segment和BatchTransfer两个核心抽象设计的高性能，零拷贝数据传输库。 |
| 命名空间 | Namespace | -                 | Kubernetes命名空间，在平台上是项目中相互隔离更小的资源空间，也是用户实现生产的工作区间。一个项目可以创建多个命名空间，可占用的资源配额综合不能超过项目配额。命名空间更细粒度的划分了资源配额的同时，还限制了命名空间下容器的大小（CPU、内存），有效的提升了资源利用率。 |
| - | Nginx | -                 | 一个高性能的HTTP和反向代理web服务器，同时也提供了IMAP/POP3/SMTP服务。 |
| 节点 | Node | -                 | 根据集群配置，节点可以是一个虚拟机或者物理机器。 |
| 节点特性感知服务 | Node Feature Discovery | NFD               | Kubernetes的节点特性感知功能。它检测Kubernetes集群中每个节点上可用的硬件功能，并使用节点标签、注释和节点污点来标记这些功能。 |
| - | node_exporter | -  | 用于收集和公开主机系统的指标，例如CPU、磁盘、内存、网络等。它可以与Prometheus或其他监控工具一起使用，并支持各种收集器和自定义指标。 |
| 非一致性内存访问 | Non-Uniform Memory Access | NUMA              | NUMA是现代多核和多处理器系统中的一种内存架构，它通过将处理器和内存划分为多个节点来优化系统的内存访问速度。 |
| - | NATS  | -   | NATS是一款开源、轻量、高性能的分布式消息系统，提供发布/订阅、请求/回复和队列订阅等通信模型。                                                                                                                                 |
| - | NATS Prometheus Exporter  | -  | 采集NATS服务器的监控端点（如varz、connz、subsz、routez）指标，包括连接数、订阅数、消息吞吐量、传输速率、客户端延迟等信息，用于监控消息系统的性能与健康状态。|
| - | OAuth2-Server | -                 | openFuyao中提供OAuth2.0协议实现的服务端。 |
| - | OAuth-proxy | -                 | 提供基于OAuth2的身份验证和授权功能。它可以帮助保护Web应用程序或API，确保只有经过验证的用户才能访问受保护的内容。 |
| 混部离线业务 | Offline Workload | -                 | 对服务质量要求相对比较低、响应延迟不敏感的业务，如大数据分析、转码、AI训练等。 |
| 混部在线业务 | Online Workload | -                 | 对服务质量要求比较高、响应延迟敏感的业务，如Web服务、电子商务等。 |
| - | Open Authorization 2.0 | OAuth2.0          | OAuth 2.0是业界标准的授权协议。OAuth 2.0注重简化客户端开发人员的工作，同时为Web应用程序、桌面应用程序、移动电话和客厅设备提供特定的授权流程。 |
| 开放遥测标准 | OpenTelemetry | OTel              | 云原生可观测性的开源标准，提供统一的API、SDK和工具来采集、处理和导出遥测数据（指标、日志、追踪）。 |
| 开放遥测协议 | OpenTelemetry Protocol | OTLP              | OpenTelemetry定义的传输协议，用于在客户端和服务端之间传输遥测数据。 |
| 操作系统 | Operating System | OS                | 一种内置的程序，用来协调计算机的各种硬件，与用户进行交互。常见有Windows，macOS和开源的Linux。 |
| 组织 | Organization | O                 | X.509证书中的字段，用于标识证书持有者所属的组织。 |
| 持久卷 | Persistent Volume | PV                | 集群中的一块存储，由管理员负责制备和管理。 |
| 持久化卷声明 | Persistent Volume Claim | PVC               | 用于文件存储的Kubernetes存储资源。 |
| 流水线并行 | pipeline parallel | PP                | 一种将模型按层分割为多个部分，在不同设备上进行计算的技术。 |
| - | Pod | -                 | 在Kubernetes中创建和管理的、最小的可部署的计算单元。 |
| 预填充阶段 | Prefill | -                 | 用户输入完prompt到生成首个token的过程。 |
| 预填充-解码分离架构 | Prefill-Decode Disaggregation | PD                | 一种将大模型推理过程中的Prefill和Decode两个阶段调度到不同硬件集群中执行，以优化资源配置和提升系统性能的架构。 |
| 普罗米修斯 | Prometheus | -                 | 一款开源的系统监控和警报工具集，用于收集和处理实时指标。通过HTTP协议周期性地从目标服务或代理中拉取监控数据，并将其存储在高度可用的时间序列数据库中。用户可以使用PromQL查询语言对这些数据进行查询、聚合和可视化，并根据预设规则触发警报。 |
| 提示词 | Prompt | -                 | 用户输入给模型的信息，模型生成符合预期的输出。 |
| 公钥基础设施 | Public Key Infrastructure | PKI               | 用于管理数字证书和公钥-私钥对的系统。 |
| PKCS#1标准 | Public-Key Cryptography Standards #1 | PKCS#1            | RSA加密标准，定义了RSA私钥的存储格式。 |
| 隐私增强邮件格式 | Privacy-Enhanced Mail | PEM               | 一种Base64编码的证书和密钥存储格式。 |
| 配置项 | Profile | -          | 签发策略配置中的配置项，定义了特定类型证书的签发参数。 |
| - | PodGroup | -               | 作业开发者声明一个PodGroup，调度器以“组”为单位进行资源判断与预留，确保作业的所有Pod能同时启动（Gang Scheduling） |
| 服务质量 | Quality of Service | QoS               | Kubernetes根据Pod的资源请求和限制，将Pod分为Guaranteed、Burstable和BestEffort三种服务质量等级，用于决定资源分配和驱逐优先级。业务也可以自定义QoS策略，区分不同服务或任务的优先级和资源保障。QoS有助于提升关键业务的稳定性和资源利用率。 |
| - | RayCluster | -                 | 基础的Ray集群，由1个head node及0到若干个worker node组成应用集群。 |
| - | RayJob | -                 | 用于提交并执行单个作业。每次提交的作业会独立创建一个Ray集群，在集群准备好后执行任务，并在任务完成后自动销毁，实现集群级别的隔离性。 |
| - | RayService | -                 | 对Ray Serve进行部署，部署时会创建独立的Ray集群，并支持服务热更新、高可用等能力。 |
| RAII资源管理 | Resource Acquisition Is Initialization | RAII | 一种C++资源管理范式，将内存、文件句柄、锁等资源的生命周期绑定到局部对象，通过构造获取资源、析构自动释放，减少资源泄漏并提升异常安全性。 |
| 远程直接存储器访问 | Remote Direct Memory Access | RDMA | 从一台计算机的存储器直接访问另一台计算机的存储器的技术。它使得网卡能够直接访问应用存储器，支持零拷贝网络通信。 |
| 远程过程调用 | Remote Procedure Call | RPC | 一个计算机通信协议。该协议允许运行于一台计算机的程序调用另一个地址空间（通常为一个开放网络的一台计算机）的子程序，而程序员就像调用本地程序一样，无需额外地为这个交互进行编程（即无需关注细节）。 |
| 资源 | Resource | -                 | Kubernetes中的内置资源与自定义资源。 |
| 资源超卖 | Resource Overselling | -                 | 混部在线业务在业务波谷时，申请资源的剩余量往往较高，资源超卖即是将这部分资源动态分配给混部离线作业使用的行为。 |
| 资源声明 | ResourceClaim | -                 | Kubernetes DRA框架中的一个命名空间级API对象，用于表示对某一类设备资源的具体请求，并在调度和分配过程中绑定到特定节点上的实际资源实例。 |
| 资源声明模板 | ResourceClaimTemplate | -                 | Kubernetes DRA框架中的一个命名空间级API对象，用于定义生成ResourceClaim的模板规范。 |
| 资源切片 | ResourceSlice | -                 | Kubernetes DRA框架中的一个集群级API对象，用于表示某一DRA Driver在特定节点上可供分配的一组结构化资源实例。 | 
| 基于角色的访问控制 | Role-based access control | RBAC              | 是一种访问控制方法。通过用户角色来管理对系统资源的访问权限。在RBAC中，权限与角色相关联，用户通过其所属的角色来获得相应的权限。 |
| - | Secret | -                 | 一种包含少量敏感信息例如密码、令牌或密钥的对象。 |
| 硬件安全加速引擎 | Security Engine | SEC               | KAE硬件安全加速引擎模块 |
| 服务 | Service | -                 | Kubernetes中将运行在一个或一组Pod上的网络应用程序公开为网络服务的方法。 |
| 服务级别目标 | Service Level Object | SLO               | 确保所提供的服务符合客户的期望的目标。 |
| - | ServiceMonitor | -                 | 是PrometheusOperator对监控系统的核心抽象之一，通过ServiceMonitor可方便地进行指标监控。 |
| 告警静默 | Silence | -                 | 告警组件提供的基础能力，根据设置好的静默规则进行告警匹配，一旦有匹配成功的告警，则该告警被静默，即不被推送。 |
| - | Spring Cloud | -                 | 基于Spring Boot框架的一套完整的微服务解决方案。 |
| 跨度 | Span | -                 | 分布式追踪中的基本工作单元，代表请求链路中的一个操作或步骤，包含操作名称、开始/结束时间、属性等信息。 |
| - | StatefulSet | -                 | 用来管理有状态应用的工作负载API对象。 |
| 主题备用名称 | Subject Alternative Name | SAN               | 证书中用于指定多个域名或IP地址的扩展字段。 |
| 调度器 | scheduler | - | Kubernetes调度器，负责将Pod调度到合适的节点上运行。 |
| - | Segment | - | 代表一段可被远程读写的连续地址空间。 |
| 张量并行 | Tensor Parallel | TP                | 一种将模型权重矩阵分割为多个部分，在不同设备上进行计算的技术。 |
| - | Tier Backend | - | Mooncake中支持分层存储的后端系统，负责统一管理不同存储层级上的KV缓存数据。 |
| 传输层安全协议 | Transport Layer Security | TLS               | 用于在网络上提供加密通信。 |
| 首Token延迟 | Time To First Token | TTFT        | 大模型推理从输入到输出第一个token的延迟。 |
| 通用唯一识别码 | Universally Unique Identifier | UUID              | UUID是一种软件建构的标准，由时间戳、时钟序列和全局唯一的节点标识符（如主机名的哈希值）组成。 |
| 显存 | Video Random Access Memory | VRAM | 显卡专用的高速内存，用于暂存渲染所需的纹理、帧缓冲等图形数据。 |
| Virtual Central Processing Unit | vCPU | -                 | 虚拟中央处理器，是在虚拟环境中使用的处理器资源。它是物理CPU的一部分，可以被虚拟机独立使用。与实际物理CPU不同，vCPU通过超线程技术将一个物理处理器划分为多个虚拟处理器核心，实现资源的共享和动态分配。 |
|维多利亚|VictoriaMetrics|VM|VictoriaMetrics是一款高性能、低成本、可水平扩展的开源时序数据库和监控解决方案，专为大规模指标数据存储和查询优化设计，兼容Prometheus生态。|
| 视觉语言大模型 | Visual Language Model | VLM               | 基于大量视觉-文本数据训练的深度学习模型。 |
| - | vLLM | -                 | 一种面向大语言模型设计的高效推理引擎和框架，可优化大模型推理性能。 |
| 存储卷 | Volume | -                 | Kubernetes中的一种抽象概念，用于提供Pod中容器的持久化存储。 |
| 监控组件 | Widget | -                 | 监控组件是一个包含名称和数据图表的组件，以卡片形式展示。 |
| - | - | xPyD              | PD架构中含x个P节点与y个D节点的架构形态。 |
| X.509标准 | - | X.509             | 国际电信联盟（ITU）制定的公钥证书标准，定义了数字证书的格式和结构。 |
| 等价类调度 | Equivalent Class Scheduling  |  - |  将具有相同资源请求、亲和性等条件的Pod视为“等价类”，一次调度决策可应用于整个类，极大降低大规模作业的调度计算开销。 |
|  拓扑感知 | Topology-aware Scheduling  |  - |  结合节点网络拓扑（NVLink，RDMA）与硬件信息，优先将需要高速通信的Pod调度到同一节点或邻近节点。 |
| 无头服务 | Headless Service | -                 | Kubernetes中一种特殊的Service，通过将`clusterIP`设置为`None`，不为服务分配集群IP，也不提供负载均衡，而是通过DNS直接返回后端Pod的IP地址。常与StatefulSet配合使用，为每个Pod提供稳定且独立的网络标识。 |
