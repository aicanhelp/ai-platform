# 1 kubelet、CRI与容器运行时发展

**从Docker到Containerd 1.0**

![cri-containerd architecture](https://kubernetes.io/images/blog/2018-05-24-kubernetes-containerd-integration-goes-ga/cri-containerd.png)

Docker在OCI规范定义之后，拆分出来containerd和runc，接着K8s推出了CRI接口，统一不同容器引擎之间交互的接口，引入了Pod的概念及Pod生命周期的管理；K8s通过调用docker-shim来调用docker api；为了更好的将Containerd接入到CRI标准中，K8s又搞出了`cri-containerd`项目，`cri-containerd`是一个守护进程用来实现Kubelet和Containerd之间的交互。



**Containerd 从1.0到1.1**

![containerd architecture](https://kubernetes.io/images/blog/2018-05-24-kubernetes-containerd-integration-goes-ga/containerd.png)

在containerd 1.1中，cri-containerd守护进程现在被重构为一个容器化的CRI插件。CRI插件内置在containerd 1.1中，默认情况下启用。与cri-containerd不同，CRI插件通过直接函数调用与containerd交互，更加稳定和高效。



# 2 切换关注点

## 2.1 命令行工具

切换到 Containerd 容器运行时可以消除掉中间环节，所有的容器都可由 Containerd 这类容器运行时来运行，对 Docker 是不可见的。 因此，无法使用相关的Docker 工具或UI，不能再使用 `docker ps` 或 `docker inspect` 命令来获取容器信息。 也就不能获取日志、停止容器，以及使用 `docker exec` 在容器中执行命令。需要排查业务是否有使用docker相关的命令进行操作，需要进行替换。

Containerd提供了相应的命令行工具进行容器的维护：

- Nerdctl：Containerd提供的与Docker兼容的命令行工具；
- Ctr：Containerd自带的一个低阶管理工具，主要关注容器、镜像、任务等；
- Crictl：K8s生态提供的基于CRI的命令行工具，调用containerd的端点进行通信，主要关注镜像、Pods、容器。



## 2.2 Kubelet（K8s v1.22）

### 2.2.1 启动参数

#### 2.2.1.1 kubelet支持 --container-runtime 配置为 remote（默认为docker）(后续移除)

 用于设置对接容器运行时的方式，支持docker、remote两个值。docker表示使用kubelet内部的dockershim代码对接dockerd，**remote**表示基于CRI接口标准对接containerd等其他容器运行时。

```go
// pkg/kubelet/config/flags.go
fs.StringVar(&s.ContainerRuntime, "container-runtime", s.ContainerRuntime, "The container runtime to use. Possible values: 'docker', 'remote'.")
```

#### 2.2.1.2 kubelet新增 --container-runtime-endpoint 指定containerd的通信地址

- 默认值：`unix:///var/run/dockershim.sock`
- 目标值：`unix:///var/run/containerd/containerd.sock`

#### 2.2.1.3 --pod-infra-container-image和--docker-endpoint参数（后续移除）

当前kubelet使用dockershim对接dockerd时（`--container-runtime`=docker）：

通过`--pod-infra-container-image`参数指定sandbox镜像

通过`--docker-endpoint`指定docker的通信地址

```go
// pkg/kubelet/config/flags.go
// Docker-specific settings.
fs.StringVar(&s.PodSandboxImage, "pod-infra-container-image", s.PodSandboxImage, fmt.Sprintf("The image whose network/ipc namespaces containers in each pod will use. %s", dockerOnlyWarning))
fs.StringVar(&s.DockerEndpoint, "docker-endpoint", s.DockerEndpoint, fmt.Sprintf("Use this for the docker endpoint to communicate with. %s", dockerOnlyWarning))
```

#### 2.2.1.4 --network-plugin、--cni-conf-dir、--cni-bin-dir参数（后续移除）

仅dockershim中的CNI有用

```go
// Network plugin settings for Docker.
fs.StringVar(&s.NetworkPluginName, "network-plugin", s.NetworkPluginName, fmt.Sprintf("<Warning: Alpha feature> The name of the network plugin to be invoked for various events in kubelet/pod lifecycle. %s", dockerOnlyWarning))
fs.StringVar(&s.CNIConfDir, "cni-conf-dir", s.CNIConfDir, fmt.Sprintf("<Warning: Alpha feature> The full path of the directory in which to search for CNI config files. %s", dockerOnlyWarning))
fs.StringVar(&s.CNIBinDir, "cni-bin-dir", s.CNIBinDir, fmt.Sprintf("A comma-separated list of full paths of directories in which to search for CNI plugin binaries. %s", dockerOnlyWarning))
```

#### 2.2.1.5 cAdvisor指标格式差异与指标缺失

Kubelet通过 `/metrics/cadvisor` 端点提供 Prometheus 指标

- Docker 节点上的指标格式为 `k8s_<container-name>_<pod-name>_<namespace>_<pod-uid>_<restart-count>`， 但其他运行时的格式不同。例如，在 containerd 节点上它是 `<container-id>`。

- 一些文件系统指标缺失，如下所示：

  ```
  container_fs_inodes_free
  container_fs_inodes_total
  container_fs_io_current
  container_fs_io_time_seconds_total
  container_fs_io_time_weighted_seconds_total
  container_fs_limit_bytes
  container_fs_read_seconds_total
  container_fs_reads_merged_total
  container_fs_sector_reads_total
  container_fs_sector_writes_total
  container_fs_usage_bytes
  container_fs_write_seconds_total
  container_fs_writes_merged_total
  ```

 建议：通过使用独立的cAdvisor守护进程来处理，或通过 Kubelet 的 `/stats/summary` 端点来获取指标。



### 2.2.2 系统进程配置

#### 2.2.2.1 kubelet系统进程配置依赖containerd

当前由systemd拉起的kubelet配置文件中配置了对docker的依赖：`After=docker.service`，需要修改为对containerd的依赖：`After=containerd.service`

 

## 2.3 containerd

### 2.3.1 containerd主要配置

kubelet切换CRI对接containerd后，不再支持通过--pod-infra-container-image参数指定sandbox镜像（仅dockershim对接dockerd时有效），sandbox镜像需要在containerd配置文件中配置，另外需要配置相关的CNI和镜像仓的访问；以下是几个关键的配置示例：

- /etc/containerd/config.toml

  ```toml
  disabled_plugins = []  # 默认为空，启用CRI插件
  ...
  [plugins]
    [plugins."io.containerd.grpc.v1.cri"]
  ...
      sandbox_image = "registry.caas.local/pause:3.6"  # sandbox镜像
  ...
      [plugins."io.containerd.grpc.v1.cri".cni]  # cni插件配置
        bin_dir = "/opt/cni/bin"
        conf_dir = "/etc/cni/net.d"
  ...
      [plugins."io.containerd.grpc.v1.cri".registry]  # 镜像仓访问证书配置
        config_path = "/etc/containerd/certs.d"
  ...
  ```

- /etc/containerd/certs.d/{registry_host:port}/hosts.toml

  ```toml
  server = "https://registry.caas.local"         # 镜像仓库服务器，值与父目录的名称一致（包括端口），示例目录为registry.caas.local
  [host."https://192.168.0.1:5000"]              # 一个镜像仓库的端点（Endpoint）
    ca = "/opt/caas/ssl/containerd/ca.pem"       # CA证书文件路径
    client = [["./client.cert", "./client.key"]] # 用于双向 TLS 认证的客户端证书和私钥文件路径
    capabilities = ["pull", "resolve", "push"]   # 该端点提供的能力（拉取镜像、镜像名解析、推送镜像）
  [host."https://192.167.0.2:5000"]
    ca = "/opt/caas/ssl/containerd/ca.pem"
    client = [["./client.cert", "./client.key"]]
    capabilities = ["pull", "resolve", "push"]
  ```

# 3 切换流程
通常的切换流程为清除节点容器，停止docker，安装containerd，配置kubelet使用containerd作为容器运行时，重启kubelet，验证节点状态是否正常，最后移除docker，恢复节点为可调度。

为了保证无感切换，需要业务支持高可用，且有配置反亲和策略，保证正在进行替换的节点的Pod清除之后，仍有可用实例提供服务，保证业务不中断。

K8s社区提供的切换指导参考文档：https://kubernetes.io/zh-cn/docs/tasks/administer-cluster/migrating-from-dockershim/

