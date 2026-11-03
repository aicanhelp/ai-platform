# Terminology

| English Full Name | Abbreviation | Description |
|---------|-------------------|------|
| addon | - | Components that need to be installed when creating a cluster using the openFuyao community BKE installation tool, such as calico. |
| All-In-One | AIO | Deploying Kubernetes and fuyao-system components on the same node. |
| Admission Webhook | - | Kubernetes admission control mechanism that intercepts and modifies (Mutating) or validates (Validating) requests before API objects are persisted. |
| API Gateway | APIG | A single entry point located between the client and APIs, acting as a reverse proxy to route client requests to a set of backend APIs. |
| Application Programming Interface | API | Predefined functions that provide applications and developers with the ability to access a set of routines based on software or hardware, without needing to access source code or understand internal working mechanism details. |
| Ascend Computing Language | ACL | Provides runtime management, single-operator invocation, model inference, media data processing, and other APIs, enabling the use of underlying hardware computing resources for deep learning inference computation, graphics/image preprocessing, single-operator accelerated computing, etc. on the CANN platform. |
| AscendHub | - | Ascend open Docker image repository. |
| Attributes | - | Key-value pair metadata in OpenTelemetry used to describe telemetry data context information, attachable to Span, Metric, Log, and other objects for recording business attributes and runtime environment information. |
| API Server | apiserver | Kubernetes API server, providing the cluster's REST API interface. |
| blackbox_exporter | - | One of the official exporters provided by Prometheus, which can probe networks via HTTP, HTTPS, DNS, TCP, and ICMP. |
| Bootstrap Node | - | The first node created during cluster initialization, used to bootstrap the entire cluster creation process. |
| BatchTransfer | - | BatchTransfer encapsulates operation requests, specifically responsible for Read/Write data synchronization of a group of non-contiguous data spaces in one Segment with corresponding spaces in another group of Segments. |
| Mooncake CacheTier | CacheTier | Different hierarchical cache layers in the Mooncake Tier Backend architecture, used for layered storage of KVCache data. |
| cAdvisor | - | A container monitoring tool developed by Google, embedded into Kubernetes as a monitoring component. |
| Cloud Native Colocation | - | A deployment approach that uses cloud-native methods to deploy online and offline workloads in the same cluster, adjusting resource usage during online workload troughs and peaks to improve overall cluster resource utilization. |
| Cloud Native Computing Foundation | CNCF | Cloud Native Computing Foundation, an open source software foundation. |
| Console | - | Frontend web page console. |
| Container | - | A runtime instance created from an image that can be started, stopped, and deleted. Each container is isolated and secure. |
| Container memory sharing | - | A technology based on the UB memory pooling mechanism that triggers memory borrowing when the memory usage rate of a bare-metal container node or NUMA reaches a threshold, imperceptibly offloading some memory pressure to a remote memory pool. |
| Container memory borrowing | - | The memory pooling component of UBS-Core, supporting importing and exporting memory blocks within a UBS Server cluster through memory pooling capabilities, to achieve the goal of cross-node and multi-process shared memory usage on bare metal. |
| Container Device Interface | CDI | A device injection standard for container runtimes. After the DRA plugin completes device allocation, it generates CDI specification files, and container runtimes such as containerd mount device nodes, environment variables, etc. into business containers. |
| Common Expression Language | CEL | An expression language in Kubernetes used for validation and filtering. npu-dra-plugin supports using CEL expressions in ResourceClaim to filter NPU devices by device attributes such as NUMA affinity and chip model. |
| Coordinated Universal Time | UTC | UTC is a time standard used to unify time globally. |
| CronJob | - | Creates Jobs with time-interval-based repeated scheduling. |
| Custom Resource | CR | Custom resource in Kubernetes. |
| Custom Resource Definition | CRD | A resource extension mechanism in Kubernetes that allows defining custom resources. |
| Certificate Authority | CA | An authoritative organization responsible for issuing and managing digital certificates. |
| Certificate Signing Request | CSR | A file containing certificate applicant information, used to apply for a certificate from a CA. |
| Common Name | CN | A field in X.509 certificates, typically used to identify the certificate holder's name. |
| Certificate | CRT | Typically used to store X.509 certificates. |
| Configuration Map | ConfigMap | A configuration object in Kubernetes for storing non-sensitive configuration data. |
| Certificate Revocation List | CRL | A list containing revoked certificates. |
| Controller Manager | controller-manager | Kubernetes controller manager that runs various controllers to maintain cluster state. |
| Customstatlogger | - | vLLM exposes the StatLoggerBase abstract class, allowing customization of metrics and metric reporting methods. |
| DaVinci Card Management Interface | DCMI | The device management API interface for Ascend NPUs, providing device discovery, status query, firmware upgrade, and other capabilities. npu-dra-plugin discovers and reports NPU device information through the DCMI interface, falling back to npu-smi when DCMI is unavailable. |
| DaemonSet | - | Ensures that a copy of a Pod runs on all (or certain) nodes. |
| Dashboard | - | A monitoring dashboard composed of multiple user-customized monitoring widgets, supporting users to monitor various metrics according to their needs. |
| Data Parallelism | DP | Each device has a complete copy of the model; each device independently processes a portion of the dataset, and then aggregates their respective gradients. |
| Decode | - | The process from generating the first token to the end of inference. |
| Deployment | - | Provides declarative update capabilities for Pods and ReplicaSet. |
| DeviceClass | - | A cluster-scoped API object in the Kubernetes DRA framework, used to define an abstract category and selection semantics for a type of device resource, referenced by ResourceClaim when requesting resources. |
| Device-to-Device Transmission | D2D Transmission | A transmission method where data is transferred directly within the same accelerator card or between different accelerator cards without going through CPU or host memory. |
| Domain Name System | DNS | A service that maps domain names to IP addresses and vice versa for better network access. |
| Dubbo | - | An open-source high-performance service framework from Alibaba, featuring high-performance and transparent RPC remote service invocation and service governance solutions. |
| Dynamic Resource Allocation | DRA | A device resource management mechanism provided by Kubernetes, used to implement device resource request, scheduling, and allocation through structured API objects, and allows device drivers to participate in resource selection and allocation decision processes. |
| Distributed Tracing | - | Tracking the complete execution path of a request in a distributed system, recording the duration and key attributes of each stage for performance analysis and fault diagnosis. |
| DCGM Exporter | - | Collects GPU runtime and health metrics, including GPU utilization, PCIe transfer rate, temperature, power usage, and other information. |
| Embedding | EMB | Data vectorization embedding operation. |
| Elasticsearch | ES | Distributed search and analytics engine, commonly used for storing and querying logs, metrics, and tracing data. |
| Extended Key Usage | extKeyUsage | A certificate extension field specifying the specific purposes of the certificate (e.g., server authentication, client authentication, etc.). |
| etcd | - | Distributed key-value storage system used by Kubernetes to store cluster state and configuration data. |
| EndpointPicker | EPP | In the Kubernetes Gateway API Inference Extension, the component responsible for selecting appropriate backend instances, supporting endpoint selection based on different routing strategies. |
| Felix | - | Calico's node agent component, running on each node, responsible for configuring network, routing, and policy rules. |
| Fully qualified domain name | FQDN | A name with both hostname and domain name, FQDN=Hostname+DomainName. |
| Gigabyte | GB | A decimal information measurement unit, commonly used to identify the storage capacity of computer hard disks, memory, and other storage media with larger capacities. |
| Gateway API Inference Extension | GIE | Extends inference-related capabilities based on the Kubernetes Gateway API, used to define and manage routing and traffic policies for inference services. |
| Helm | - | A package manager in Kubernetes used to simplify deploying and managing applications in Kubernetes clusters. |
| Helm Chart | - | A core concept of Helm; a pre-configured application resource package. |
| High Availability | HA | The ability of a system or service to run with high reliability and continuous availability, maintaining normal operation even in the face of hardware failures or other exceptions. |
| High Performance RSA Engine | HPRE | KAE high-performance RSA acceleration engine module. |
| High Performance ZIP Engine | ZIP | KAE high-performance zlib/Gzip compression engine module. |
| High Bandwidth Memory | HBM | Dedicated high-bandwidth memory for NPUs. In npu-dra-plugin soft partitioning scenarios, HBM memory quota is allocated to vNPUs through memCapacity with a 1Mi step size. |
| Horizontal Pod Autoscaler | HPA | Automatically updates workload resources (e.g., Deployment or StatefulSet) to automatically scale workloads to meet demand. |
| Host_to_Device Transmission | H2D Transmission | The process of copying data from CPU/host memory to the memory of acceleration devices such as NPU/GPU. |
| Huawei Cache Coherence System | HCCS | A hardware direct-connect interconnect link between Ascend NPUs, divided into HCCS (direct connect) and HCCS_SW (via HCCS switch device) types, which together form an HCCS ring for NPU topology-aware scheduling. |
| Hypertext Transfer Protocol Secure | HTTPS | An HTTP channel with security as the goal, ensuring transmission security through transport encryption and identity authentication on top of HTTP. |
| Indicator | - | Monitoring indicators are metrics supported by data collection systems (such as Prometheus) for user monitoring; one monitoring indicator can contain multiple monitoring instances. |
| Ingress | - | An API object that manages external access to services in the cluster, providing load balancing, SSL termination, and name-based virtual hosting. |
| Instance | - | A monitoring instance is the smallest granularity object that can be monitored on Kubernetes. Each monitoring instance is uniquely identified by a set of key-value pair labels. |
| Job | - | Runs one-time tasks in the cluster, focusing on executing one-time tasks rather than maintaining a specified number of running instances. The Job controller creates one or more Pods to run the specified task. When the task completes, the Job controller deletes the Pods. |
| Jaeger | - | An open-source distributed tracing system for monitoring and troubleshooting microservice architectures, open-sourced by Uber and donated to CNCF. |
| Key-Value Cache | KVCache | A common strategy for accelerating large model inference, working by caching the Key (K) and Values (V) matrices generated during the self-attention mechanism in the large model inference process, avoiding redundant computation and improving inference speed. |
| kube-apiserver | - | Validates and configures data for API objects, including Pods, Services, ReplicationControllers, etc. The API server serves REST operations and provides a frontend for the cluster's shared state, through which all other components interact. |
| kubectl | - | A command-line tool for the Kubernetes API to communicate with the Kubernetes cluster's control plane. |
| kubelet | - | An important component in the Kubernetes cluster, running on each node, responsible for managing containers on that node. It is the node agent in the Kubernetes system, communicating with controllers in the master control plane to ensure containers run as expected on the node. |
| Kube-rbac-proxy | - | A lightweight HTTP proxy service designed for Kubernetes, using Kubernetes' SubjectAccessReview functionality to perform RBAC (Role-Based Access Control) authorization. This project aims to restrict communication between Pods, only allowing Pods with valid RBAC authorization tokens to access other Pods. |
| Kubernetes | K8s | Kubernetes is a portable, extensible open-source platform for managing containerized workloads and services, facilitating both declarative configuration and automation. |
| kube-state-metrics | - | Collects state metrics about resource objects generated by the API Server, such as Deployment, Node, and Pod. |
| Kunpeng Accelerator Engine | KAE | Kunpeng Accelerator Engine (KAE) is a hardware acceleration solution based on the Kunpeng 920 processor. |
| Key | KEY | Used to store private keys. |
| Kubernetes Configuration | kubeconfig | Contains cluster access information, authentication information, and context configuration. |
| Key Usage | keyUsage | A certificate extension field specifying the purpose of the certificate key (e.g., digital signature, key encryption, etc.). |
| kube-proxy | - | Kubernetes network proxy, responsible for maintaining network rules and load balancing on nodes. |
| Large Language Model | LLM | A deep learning model trained on large amounts of text data. |
| Load Balancer | - | A device or service used to distribute network traffic across multiple servers. |
| metrics-server | - | One of the core components in the Kubernetes monitoring system, responsible for collecting resource metrics from Kubelet, aggregating them (depending on kube-aggregator), and exposing them through the Metrics API (/apis/metrics.k8s.io/) in the Kubernetes API Server. However, metrics-server only stores the latest metric data (CPU/Memory). |
| Mind Inference Service | MIS | A containerized large model inference API service provided by Ascend. |
| Mind Inference Service Operator | MIS-Operator | A component that implements lifecycle management for inference microservice instances. |
| Multi-core | - | Refers to integrating a large number of processing cores on a single chip. Multi-core scenarios specifically refer to nodes in a cluster with more than 256 CPUs. |
| Mutual TLS | mTLS | Using bidirectional encrypted channels between server and client. |
| Mooncake | - | An open-source community that proposed a KVCache-centric LLM service decoupled architecture. |
| Mooncake Store | - | A high-performance distributed key-value KVCache storage engine designed specifically for LLM inference scenarios. |
| Mooncake Store Master Service | Master Service | In Mooncake Store, responsible for managing the logical storage space pool of the entire cluster and handling node join and leave events. |
| Mooncake Store Client | Mooncake Client | The client of Mooncake Store, responsible for initiating get/put requests called by upper-layer applications and providing actual KVCache storage. |
| Mooncake Transfer Engine | TE | A high-performance, zero-copy data transfer library designed around two core abstractions: Segment and BatchTransfer. |
| Namespace | - | Kubernetes namespace; on the platform, it is a smaller resource space isolated within a project and the working space for users to perform production. A project can create multiple namespaces, and the total resource quota occupied cannot exceed the project quota. Namespaces provide finer-grained resource quota division while also limiting container sizes (CPU, memory) within the namespace, effectively improving resource utilization. |
| Nginx | - | A high-performance HTTP and reverse proxy web server, also providing IMAP/POP3/SMTP services. |
| Node | - | Depending on cluster configuration, a node can be a virtual machine or a physical machine. |
| Node Feature Discovery | NFD | Kubernetes node feature discovery functionality. It detects hardware features available on each node in the Kubernetes cluster and labels them using node labels, annotations, and node taints. |
| node_exporter | - | Used to collect and expose host system metrics such as CPU, disk, memory, network, etc. It can be used with Prometheus or other monitoring tools and supports various collectors and custom metrics. |
| Non-Uniform Memory Access | NUMA | NUMA is a memory architecture in modern multi-core and multi-processor systems that optimizes memory access speed by dividing processors and memory into multiple nodes. |
| NATS | - | NATS is an open-source, lightweight, high-performance distributed messaging system that provides publish/subscribe, request/reply, and queue subscription communication models. |
| NATS Prometheus Exporter | - | Collects metrics from NATS server monitoring endpoints (such as varz, connz, subsz, routez), including connection count, subscription count, message throughput, transfer rate, client latency, and other information, for monitoring the performance and health status of the messaging system. |
| OAuth2-Server | - | The server-side implementation of the OAuth2.0 protocol in openFuyao. |
| OAuth-proxy | - | Provides OAuth2-based authentication and authorization capabilities. It helps protect web applications or APIs, ensuring that only authenticated users can access protected content. |
| Offline Workload | - | Workloads with relatively low service quality requirements and insensitive to response latency, such as big data analytics, transcoding, AI training, etc. |
| Online Workload | - | Workloads with high service quality requirements and sensitive to response latency, such as web services, e-commerce, etc. |
| Open Authorization 2.0 | OAuth2.0 | OAuth 2.0 is an industry-standard authorization protocol. OAuth 2.0 focuses on simplifying the work of client developers while providing specific authorization flows for web applications, desktop applications, mobile phones, and living room devices. |
| OpenTelemetry | OTel | An open-source standard for cloud-native observability, providing unified APIs, SDKs, and tools to collect, process, and export telemetry data (metrics, logs, traces). |
| OpenTelemetry Protocol | OTLP | A transport protocol defined by OpenTelemetry for transmitting telemetry data between clients and servers. |
| Operating System | OS | A built-in program used to coordinate various hardware of the computer and interact with users. Common ones include Windows, macOS, and the open-source Linux. |
| Organization | O | A field in X.509 certificates used to identify the organization to which the certificate holder belongs. |
| Persistent Volume | PV | A piece of storage in the cluster, provisioned and managed by an administrator. |
| Persistent Volume Claim | PVC | Kubernetes storage resource for file storage. |
| Pipeline Parallel | PP | A technique that splits a model by layers into multiple parts and computes on different devices. |
| Pod | - | The smallest deployable computing unit created and managed in Kubernetes. |
| Prefill | - | The process from the user completing the prompt input to generating the first token. |
| Prefill-Decode Disaggregation | PD | An architecture that schedules the Prefill and Decode phases of the large model inference process to different hardware clusters to optimize resource allocation and improve system performance. |
| Prometheus | - | An open-source system monitoring and alerting toolkit used for collecting and processing real-time metrics. It periodically pulls monitoring data from target services or agents via the HTTP protocol and stores it in a highly available time-series database. Users can query, aggregate, and visualize this data using the PromQL query language, and trigger alerts based on preset rules. |
| Prompt | - | Information input by the user to the model; the model generates output that meets expectations. |
| Public Key Infrastructure | PKI | A system for managing digital certificates and public-private key pairs. |
| Public-Key Cryptography Standards #1 | PKCS#1 | RSA encryption standard that defines the storage format of RSA private keys. |
| Privacy-Enhanced Mail | PEM | A Base64-encoded certificate and key storage format. |
| Profile | - | A configuration item in the certificate issuance strategy that defines the issuance parameters for a specific type of certificate. |
| PodGroup | - | A job developer declares a PodGroup, and the scheduler performs resource judgment and reservation in "group" units, ensuring that all Pods of a job can start simultaneously (Gang Scheduling). |
| Quality of Service | QoS | Kubernetes classifies Pods into three QoS levels — Guaranteed, Burstable, and BestEffort — based on Pod resource requests and limits, used to determine resource allocation and eviction priority. Businesses can also customize QoS policies to differentiate the priority and resource guarantees of different services or tasks. QoS helps improve the stability and resource utilization of critical workloads. |
| RayCluster | - | A basic Ray cluster, consisting of 1 head node and 0 to several worker nodes forming an application cluster. |
| RayJob | - | Used for submitting and executing a single job. Each submitted job independently creates a Ray cluster, executes the task after the cluster is ready, and is automatically destroyed after the task is completed, achieving cluster-level isolation. |
| RayService | - | Deploys Ray Serve, creating an independent Ray cluster during deployment, and supports service hot updates, high availability, and other capabilities. |
| Resource Acquisition Is Initialization | RAII | A C++ resource management paradigm that binds the lifecycle of resources such as memory, file handles, and locks to local objects, acquiring resources through construction and automatically releasing them through destruction, reducing resource leaks and improving exception safety. |
| Remote Direct Memory Access | RDMA | A technology for directly accessing the memory of one computer from another. It enables network cards to directly access application memory, supporting zero-copy network communication. |
| Remote Procedure Call | RPC | A computer communication protocol that allows a program running on one computer to call a subroutine in another address space (typically a computer on an open network), as if the programmer were calling a local program, without additional programming for this interaction (i.e., without needing to worry about details). |
| Resource | - | Built-in and custom resources in Kubernetes. |
| Resource Overselling | - | The behavior of dynamically allocating the remaining resources from online workload resource requests during workload troughs to offline colocation jobs. |
| ResourceClaim | - | A namespace-scoped API object in the Kubernetes DRA framework, used to represent a specific request for a type of device resource and bound to an actual resource instance on a specific node during scheduling and allocation. |
| ResourceClaimTemplate | - | A namespace-scoped API object in the Kubernetes DRA framework, used to define a template specification for generating ResourceClaims. |
| ResourceSlice | - | A cluster-scoped API object in the Kubernetes DRA framework, used to represent a set of structured resource instances available for allocation by a specific DRA Driver on a specific node. |
| Role-based access control | RBAC | An access control method that manages access permissions to system resources through user roles. In RBAC, permissions are associated with roles, and users obtain corresponding permissions through their assigned roles. |
| Secret | - | An object containing a small amount of sensitive information such as passwords, tokens, or keys. |
| Security Engine | SEC | KAE hardware security acceleration engine module. |
| Service | - | A method in Kubernetes for exposing network applications running on one or a group of Pods as network services. |
| Service Level Objective | SLO | A goal that ensures the provided service meets customer expectations. |
| ServiceMonitor | - | One of the core abstractions of Prometheus Operator for the monitoring system, facilitating metric monitoring through ServiceMonitor. |
| Silence | - | A basic capability provided by the alerting component that matches alerts based on preset silence rules; once an alert matches successfully, it is silenced (i.e., not pushed). |
| Spring Cloud | - | A complete microservices solution based on the Spring Boot framework. |
| Span | - | The basic unit of work in distributed tracing, representing an operation or step in the request chain, containing operation name, start/end time, attributes, and other information. |
| StatefulSet | - | A workload API object used to manage stateful applications. |
| Subject Alternative Name | SAN | An extension field in certificates used to specify multiple domain names or IP addresses. |
| scheduler | - | Kubernetes scheduler, responsible for scheduling Pods to appropriate nodes for execution. |
| Segment | - | Represents a contiguous address space that can be remotely read and written. |
| Tensor Parallel | TP | A technique that splits model weight matrices into multiple parts and computes on different devices. |
| Tier Backend | - | A backend system in Mooncake that supports tiered storage, responsible for uniformly managing KV cache data across different storage tiers. |
| Transport Layer Security | TLS | Used to provide encrypted communication over networks. |
| Time To First Token | TTFT | The latency from input to the first token output in large model inference. |
| Universally Unique Identifier | UUID | UUID is a software construction standard consisting of a timestamp, clock sequence, and a globally unique node identifier (such as a hostname hash). |
| Video Random Access Memory | VRAM | Dedicated high-speed memory for graphics cards, used to store textures, frame buffers, and other graphics data needed for rendering. |
| vCPU | - | Virtual central processing unit, a processor resource used in virtual environments. It is a portion of a physical CPU that can be independently used by a virtual machine. Unlike actual physical CPUs, vCPUs use hyper-threading technology to divide one physical processor into multiple virtual processor cores, enabling resource sharing and dynamic allocation. |
| VictoriaMetrics | VM | VictoriaMetrics is a high-performance, low-cost, horizontally scalable open-source time-series database and monitoring solution, optimized for large-scale metric data storage and querying, and compatible with the Prometheus ecosystem. |
| Visual Language Model | VLM | A deep learning model trained on large amounts of visual-text data. |
| vLLM | - | An efficient inference engine and framework designed for large language models, optimizing large model inference performance. |
| Volume | - | An abstract concept in Kubernetes used to provide persistent storage for containers in a Pod. |
| Widget | - | A monitoring widget is a component containing a name and data chart, displayed in card form. |
| - | xPyD | An architecture form in PD architecture with x P nodes and y D nodes. |
| - | X.509 | A public key certificate standard developed by the International Telecommunication Union (ITU), defining the format and structure of digital certificates. |
| Equivalent Class Scheduling | - | Treating Pods with the same resource requests, affinities, and other conditions as an "equivalent class," where one scheduling decision can be applied to the entire class, greatly reducing the scheduling computational overhead for large-scale jobs. |
| Topology-aware Scheduling | - | Combining node network topology (NVLink, RDMA) and hardware information, prioritizing scheduling Pods that require high-speed communication to the same node or nearby nodes. |
| Headless Service | - | A special type of Service in Kubernetes that, by setting `clusterIP` to `None`, does not allocate a cluster IP for the service and does not provide load balancing, but instead returns backend Pod IP addresses directly through DNS. Often used with StatefulSet to provide stable and independent network identities for each Pod. |
