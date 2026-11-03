# 通信矩阵

## 宿主机端口占用清单

### 宿主机对外暴露端口

| 源设备 | 源IP | 源端口 | 目的设备        | 目的IP | 目的端口（侦听） | 协议 | 端口说明 | 侦听端口是否可更改 | 认证方式 | 加密方式 | 所属服务/微服务 | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ALL | ALL | ALL | openFuyao服务器 | 宿主机IP | 36443 | TCP | 宿主机侧暴露端口（HostPort） |是 | 口令认证和单向证书认证 | TLS1.3(缺省)/TLS1.2 | kubernetes | 由 CNI-HOSTPORT 规则转发到 PodIP:36443 |
| ALL | ALL | ALL | openFuyao服务器 | 宿主机IP | 38080 | TCP | 宿主机侧暴露端口（HostPort） | 是 | 无，采用TCP标准协议 | 无 | bocloud_chart_registry | 由 CNI-HOSTPORT 规则转发到 PodIP:8080 |
| ALL | ALL | ALL | openFuyao服务器 | 宿主机IP | 40080 | TCP | 宿主机侧暴露端口（HostPort） | 是 | 无，采用TCP标准协议 | 无 | bocloud_yum_registry | 由 CNI-HOSTPORT 规则转发到 PodIP:80 |
| ALL | ALL | ALL | openFuyao服务器 | 宿主机IP | 40443 | TCP | 宿主机侧暴露端口（HostPort） | 是 | 单向证书认证 | TLS1.3(缺省)/TLS1.2 | bocloud_image_registry | 由 CNI-HOSTPORT 规则转发到 PodIP:443 |
| ALL | ALL | ALL | openFuyao服务器 | 宿主机IP | 30010 | TCP | 宿主机侧暴露端口（NodePort） | 否 | 用户名和密码认证 | TLS1.3(缺省)/TLS1.2  | ingress-nginx-controller | 由 Kubernetes NodePort 暴露 |
| ALL | ALL | ALL | openFuyao服务器 | 宿主机IP | 31616 | TCP | 宿主机侧暴露端口（NodePort） | 否 | 用户名和密码认证 | TLS1.3(缺省)/TLS1.2  | ingress-nginx-controller | 由 Kubernetes NodePort 暴露 |
| ALL | ALL | ALL | openFuyao服务器 | 宿主机IP | 22 | TCP | 宿主机进程监听端口 | 否 | 口令/密钥认证 | SSH加密 | sshd |-  |
| ALL | ALL | ALL | openFuyao服务器 | 宿主机IP | 123 | UDP | 宿主机进程监听端口 | 否 | 无，采用UDP标准协议 | 无 | bke |- |
| ALL | ALL | ALL | openFuyao服务器 | 宿主机IP | 6443 | TCP | 宿主机进程监听端口 | 是 | 口令认证和单向证书认证 | TLS1.3(缺省)/TLS1.2 | kube-apiserver |- |
| ALL | ALL | ALL | openFuyao服务器 | 宿主机IP | 10250 | TCP | 宿主机进程监听端口 | 是 | Token认证和证书认证 | TLS1.3(缺省)/TLS1.2 | kubelet | - |
| ALL | ALL | ALL | openFuyao服务器 | 宿主机IP | 10249 | TCP | 宿主机进程监听端口 | 是 | 无，采用TCP标准协议 | 无 | kube-proxy |-  |
| ALL | ALL | ALL | openFuyao服务器 | 宿主机IP | 10256 | TCP | 宿主机进程监听端口 | 是 | 无，采用TCP标准协议 | 无 | kube-proxy |-  |
| ALL | ALL | ALL | openFuyao服务器 | 宿主机IP | 10257 | TCP | 宿主机进程监听端口 | 是 | 双向证书认证 | TLS1.3(缺省)/TLS1.2  | kube-controller-manager |-  |
| ALL | ALL | ALL | openFuyao服务器 | 宿主机IP | 10259 | TCP | 宿主机进程监听端口 | 是 | 双向证书认证 | TLS1.3(缺省)/TLS1.2 | kube-scheduler |-  |
| ALL | ALL | ALL | openFuyao服务器 | 宿主机IP | 2379 | TCP | 宿主机进程监听端口 | 是 | 双向证书认证 | TLS1.3(缺省)/TLS1.2| etcd |-  |
| ALL | ALL | ALL | openFuyao服务器 | 宿主机IP | 2380 | TCP | 宿主机进程监听端口 | 是 | 双向证书认证 | TLS1.3(缺省)/TLS1.2 | etcd | - |
| ALL | ALL | ALL | openFuyao服务器 | 宿主机IP | 2381 | TCP | 宿主机进程监听端口 | 是 | 无，采用TCP标准协议 | 无 | etcd | - |
| ALL | ALL | ALL | openFuyao服务器 | 宿主机IP | 8443 | TCP | 宿主机进程监听端口 | 是 | 单向证书认证 | TLS1.3(缺省)/TLS1.2 | socat | - |
| ALL | ALL | ALL | openFuyao服务器 | 宿主机IP | 9100 | TCP | 宿主机进程监听端口 | 是 | 无，采用TCP标准协议 | 无| kube-rbac-proxy | - |
| ALL | ALL | ALL | openFuyao服务器 | 宿主机IP | 4789 | UDP | 宿主机进程监听端口 | 是 | 无，采用UDP标准协议 | 无 | calico/vxlan |-  |

### 宿主机本地依赖端口

| 源设备 | 源IP | 源端口 | 目的设备 | 目的IP | 目的端口（侦听） | 协议 | 端口说明 | 侦听端口是否可更改 | 认证方式 | 加密方式 | 所属服务/微服务 | 备注 |
| --- | --- | --- | --- | --- |----------| --- | --- | --- | --- | --- | --- |  --- |
| ALL | ALL | ALL | openFuyao服务器 | 环回地址 | 1338     | TCP | 宿主机本地进程侦听端口（部署/运行必需） | 是 | 无，采用TCP标准协议 | 无| containerd | 仅本机访问 |
| ALL | ALL | ALL | openFuyao服务器 | 环回地址 | 10248    | TCP | 宿主机本地进程侦听端口（部署/运行必需） | 是 | 无，采用TCP标准协议 | 无  | kubelet | 仅本机访问 |
| ALL | ALL | ALL | openFuyao服务器 | 环回地址 | 9099     | HTTP | 宿主机本地进程侦听端口（部署/运行必需） | 是 | 无，采用TCP标准协议 | 无 | calico-node | 仅本机访问 |
| ALL | ALL | ALL | openFuyao服务器 | 环回地址 | 58080    | HTTP | 宿主机本地进程侦听端口（部署/运行必需） | 是 | 无，采用TCP标准协议 | 无 | bkeagent | 仅本机访问 |
| ALL | ALL | ALL | openFuyao服务器 | 环回地址 | 323      | UDP | 宿主机本地进程侦听端口（部署/运行必需） | 是 | 无，采用UDP标准协议 | 无 | chronyd | 若该节点启用chronyd则存在 |

> ![](./figures/icon-note.gif)**说明：**<br/>
> 本节按“宿主机对外暴露端口”和“宿主机本地依赖端口”分类，汇总与集群部署/运行相关的端口。前者既包含可由`ss`/`netstat`直接观察到的主机进程监听端口，也包含NodePort、HostPort/CNI DNAT等转发形成的可达端口；后者通常仅绑定`127.0.0.1`/`::1`，不对外暴露，但被组件部署、健康检查或运行时通信依赖，若被占用同样可能导致组件启动失败、健康检查异常或集群拉起失败。

## Kubernetes相关通信矩阵

| 源设备 | 源IP | 源端口 | 目的设备        | 目的IP | 目的端口（侦听） | 协议 | 端口说明 | 侦听端口是否可更改 | 认证方式 | 加密方式 | 所属服务/微服务 | 备注 | 
| ----- | ---- | ------ | ---------------| ------ | -------------- | ---- | ------- | ----------------- | ------- | ------- | -------| -------------- |
| ALL   | ALL  | ALL    | openFuyao服务器 | 集群IP | 6443           | TCP  | 用于第三方系统通过此端口获取openFuyao提供的聚合服务能力，是kube-apiserver的侦听端口，该端口是SSL加密端口。 | 是 | 口令认证和单向证书认证 | TLS1.3(缺省)/TLS1.2 | kube-apiserver | - | 
| ALL   | ALL  | ALL    | openFuyao服务器 | 集群IP |  443           | TCP  | 用于第三方系统通过此端口获取openFuyao提供的聚合服务能力，是kube-apiserver的侦听端口，该端口是SSL加密端口。 | 否 | token认证或单向证书认证 | TLS1.3(缺省)/TLS1.2 | kube-apiserver | - | 
| ALL   | ALL  | ALL    | openFuyao服务器 | 环回地址 |  10257       | TCP  | 用于访问kube-controller-manager的HTTPS服务，是kube-controller-manager的侦听端口，该端口是SSL加密端口。 | 是 | 双向证书认证 | TLS1.3(缺省)/TLS1.2  | kube-controller-manager | - | 
| ALL   | ALL  | ALL    | openFuyao服务器 | 环回地址 |  10259       | TCP  | 用于访问kube-scheduler的HTTPS服务，是kube-scheduler的侦听端口，该端口是SSL加密端口。 | 是 | 双向证书认证 | TLS1.3(缺省)/TLS1.2 | kube-scheduler | - | 
| ALL   | ALL  | ALL    | openFuyao服务器 | 集群IP |  53         | TCP/UDP  | 用于服务域名解析，是coredns的侦听端口，仅域名解析使用。 | 否 | 无，采用TCP标准协议 | 无 |coredns | - | 
| ALL   | ALL  | ALL    | openFuyao服务器 | 集群IP |  9153       | TCP  | 用于采集coredns的监控指标。 | 是 | 无，采用TCP标准协议 | TLS1.3(缺省)/TLS1.2| coredns | - | 
| ALL   | ALL  | ALL    | openFuyao服务器 | 集群IP |  8080      | TCP  | 用于检查coredns的健康状态。 | 是 | 无，采用TCP标准协议 | TLS1.3(缺省)/TLS1.2 | coredns | - | 
| ALL   | ALL  | ALL    | openFuyao服务器 | 集群IP |  2379       | TCP  | etcd通过此端口提供服务，该端口是SSL加密端口。 | 是 | 双向证书认证 | TLS1.3(缺省)/TLS1.2| etcd | - | 
| ALL   | ALL  | ALL    | openFuyao服务器 | 集群IP |  2380       | TCP  | 用于etcd集群成员间的通信，该端口是SSL加密端口。 | 是 | 双向证书认证 | TLS1.3(缺省)/TLS1.2 | etcd | - | 
| ALL   | ALL  | ALL    | openFuyao服务器 | 集群IP |  2381       | TCP  | 用于采集etcd集群对外提供的监控指标。 | 是 | 无，采用TCP标准协议 | TLS1.3(缺省)/TLS1.2| etcd | - | 
| ALL   | ALL  | ALL    | openFuyao服务器 | 环回地址 |  9099     | HTTP  | 用于提供服务存活状态检查功能。 | 是 | 无，采用TCP标准协议 | 无 | calico-node | - | 
| ALL   | ALL  | ALL    | openFuyao服务器 | 集群IP |  179     | TCP  | calico组件bird BGP协议侦听端口，用于同步节点间的路由。 | 是 | 口令认证 | 无| calico-node | - | 
| ALL   | ALL  | ALL    | openFuyao服务器 | 环回地址 |  10248    | TCP  | 用于检查kubelet的健康状态。 | 是 | 无，采用TCP标准协议 | 无 | kubelet | - | 
| ALL   | ALL  | ALL    | openFuyao服务器 | 集群IP |  10250    | TCP  | 用于与kube-apiserver进行通信，该端口是SSL加密端口。 | 是 | Token认证和证书认证 | TLS1.3(缺省)/TLS1.2 | kubelet | - | 
| ALL   | ALL  | ALL    | openFuyao服务器 | 环回地址 |  10249    | TCP  | 用于采集监控指标。 | 是 | 无，采用TCP标准协议 | 无 | kube-proxy | - | 
| ALL   | ALL  | ALL    | openFuyao服务器 | 环回地址 |  10256    | TCP  | 用于健康状态检查。 | 是 | 无，采用TCP标准协议 | 无 | kube-proxy | - | 

> ![](./figures/icon-note.gif)**说明：**<br/>
> Kubernetes官方文档请参见[端口和协议](https://kubernetes.io/zh-cn/docs/reference/networking/ports-and-protocols/)。

## openFuyao应用组件相关通信矩阵

| 源设备 | 源IP | 源端口 | 目的设备        | 目的IP | 目的端口（侦听） | 协议 | 端口说明 | 侦听端口是否可更改 | 认证方式        | 加密方式              | 所属服务/微服务                       | 备注 |
| ----- | ---- | ------ | ---------------| ------ |----------| ---- | ------- | ----------------- |-------------|-------------------|--------------------------------| --- |
| ALL   | ALL  | ALL    | openFuyao服务器 | PodIP | 443      | TCP  | ingress-nginx-controller服务端口 | 否 | 用户名和密码认证    | TLS1.3(缺省)/TLS1.2 | ingress-nginx-controller       | - |
| ALL   | ALL  | ALL    | openFuyao服务器 | 集群IP | 443      | TCP  | ingress-nginx-controller服务端口 | 否 | 用户名和密码认证    | TLS1.3(缺省)/TLS1.2 | ingress-nginx-controller       | - |
| ALL   | ALL  | ALL    | openFuyao服务器 | IngressIP | 30010    | TCP  | 引导节点管理面对外提供的访问端口 | 否 | 用户名和密码认证    | TLS1.3(缺省)/TLS1.2 | ingress-nginx-controller       | - |
| ALL   | ALL  | ALL    | openFuyao服务器 | IngressIP | 31616    | TCP  | 业务集群&管理集群管理面对外提供的访问端口 | 否 | 用户名和密码认证    | TLS1.3(缺省)/TLS1.2 | ingress-nginx-controller       | - |
| ALL   | ALL  | ALL    | openFuyao服务器 | 集群IP | 9072     | TCP  | web-terminal-service后端Pod访问端口 | 是 | token认证     | TLS1.3(缺省)/TLS1.2 | web-terminal-service           | - |
| ALL   | ALL  | ALL    | openFuyao服务器 | PodIP  | 9072     | TCP  | web-terminal-service后端Service访问端口  | 是 | token认证     | TLS1.3(缺省)/TLS1.2 | web-terminal-service           | - |
| ALL   | ALL  | ALL    | openFuyao服务器 | PodIP  | 80       | TCP  | monitoring-service后端Pod访问端口   | 是 | 无，采用TCP标准协议 | 无                 | monitoring-service             | - |
| ALL   | ALL  | ALL    | openFuyao服务器 | PodIP  | 9096     | TCP  | 内置认证服务器Pod访问端口     | 是 | 双向证书认证      | TLS1.3            | oauth-server                   | - |
| ALL   | ALL  | ALL    | openFuyao服务器 | 集群IP  | 9096     | TCP  | 内置认证服务器Service访问端口     | 是 | 双向证书认证      | TLS1.3            | oauth-server                   | - |
| ALL   | ALL  | ALL    | openFuyao服务器 | PodIP  | 9095     | TCP  | 认证webhookPod访问端口     | 是 | 双向证书认证      | TLS1.3            | oauth-webhook                  | - |
| ALL   | ALL  | ALL    | openFuyao服务器 | 集群IP  | 9095     | TCP  | 认证webhook Service访问端口     | 是 | 双向证书认证      | TLS1.3            | oauth-webhook                  | - |
| ALL   | ALL  | ALL    | openFuyao服务器 | PodIP  | 9175     | TCP  | 用户管理服务Pod访问端口     | 是 | 无，采用TCP标准协议 | 无                 | user-management-operator       | - |
| ALL   | ALL  | ALL    | openFuyao服务器 | 集群IP  | 80       | TCP  | 用户管理服务Service访问端口     | 是 | 无，采用TCP标准协议 | 无                 | user-management-operator       | - |
| ALL   | ALL  | ALL    | openFuyao服务器 | PodIP  | 9093     | TCP  | 应用管理服务后端Pod访问端口     | 是 | 无，采用TCP标准协议 | 无                 | application-management-service | - |
| ALL   | ALL  | ALL    | openFuyao服务器 | 集群IP  | 80       | TCP  | 应用管理服务后端Service访问端口     | 是 | 无，采用TCP标准协议 | 无                 | application-management-service | - |
| ALL   | ALL  | ALL    | openFuyao服务器 | PodIP  | 9093     | TCP  | 应用市场服务后端Pod访问端口     | 是 | 无，采用TCP标准协议 | 无                 | marketplace-service            | - |
| ALL   | ALL  | ALL    | openFuyao服务器 | 集群IP  | 80       | TCP  | 应用市场服务后端Service访问端口     | 是 | 无，采用TCP标准协议 | 无                 | marketplace-service            | - |
| ALL   | ALL  | ALL    | 引导节点 | PodIP  | 8080     | TCP  | 集群生命周期管理前端Pod访问端口     | 是 | 无，采用TCP标准协议 | 无                 |                                | installer-website | - |
| ALL   | ALL  | ALL    | 引导节点 | 集群IP  | 80       | TCP  | 集群生命周期管理前端Service访问端口     | 是 | 无，采用TCP标准协议 | 无                 | installer-website              | - |
| ALL   | ALL  | ALL    | 引导节点 | PodIP  | 9210     | TCP  | 集群生命周期管理后端Pod访问端口     | 是 | 无，采用TCP标准协议 | 无                 | installer-service              | - |
| ALL   | ALL  | ALL    | 引导节点 | 集群IP  | 80       | TCP  | 集群生命周期管理后端Service访问端口     | 是 | 无，采用TCP标准协议 | 无                 | installer-service              | - |
| ALL   | ALL  | ALL    | 引导节点 | PodIP  | 8080     | TCP  | 引导节点对外提供的控制台前端Pod访问端口     | 是 | 无，采用TCP标准协议 | 无                 | bke-console-website            | - |
| ALL   | ALL  | ALL    | 引导节点 | 集群IP  | 80       | TCP  | 引导节点对外提供的控制台前端Service访问端口     | 是 | 无，采用TCP标准协议 | 无                 | bke-console-website            | - |
| ALL   | ALL  | ALL    | 引导节点 | PodIP  | 9037     | TCP  | 引导节点对外提供的控制台后端Pod访问端口     | 是 | 无，采用TCP标准协议 | 无                 | bke-console-website            | - |
| ALL   | ALL  | ALL    | 引导节点 | 集群IP  | 80       | TCP  | 引导节点对外提供的控制台后端Pod访问端口     | 是 | 无，采用TCP标准协议 | 无                 | bke-console-website            | - |
| ALL   | ALL  | ALL    | openFuyao服务器 | PodIP  | 9093     | TCP  | 扩展组件管理后端Pod访问端口     | 是 | 无，采用TCP标准协议 | 无                 | plugin-management-service      | - |
| ALL   | ALL  | ALL    | openFuyao服务器 | 集群IP  | 80       | TCP  | 扩展组件管理后端Service访问端口     | 是 | 无，采用TCP标准协议 | 无                 | plugin-management-service      | - |
| ALL   | ALL  | ALL    | openFuyao服务器 | 集群IP  | 9100     | TCP  | Node Exporter Service访问端口     | 是 | 无，采用TCP标准协议 | 无                 | node-exporter                  | - |
| ALL   | ALL  | ALL    | 引导节点 | 容器IP  | 36443    | HTTPS | 引导节点对外K8s API端口   | 是 | 口令认证和单向证书认证 | TLS1.3(缺省)/TLS1.2 | kubernetes                     | - |
| ALL   | ALL  | ALL    | 引导节点 | 容器IP  | 38080    | HTTP | 引导节点对外提供charts服务端口   | 是 | 无，采用TCP标准协议 | 无                 | bocloud_chart_registry         | - |
| ALL   | ALL  | ALL    | 引导节点 | 容器IP  | 40080    | HTTP | 引导节点对外提供yum源服务端口   | 是 | 无，采用TCP标准协议 | 无                 | bocloud_yum_registry           | - |
| ALL   | ALL  | ALL    | 引导节点 | 容器IP  | 40443    | HTTPS | 引导节点对外提供镜像源服务端口   | 是 | 单向证书认证      | TLS1.3(缺省)/TLS1.2 | bocloud_image_registry         | - |
| ALL | ALL | ALL | openFuyao服务器 | 集群IP | 8443     | HTTPS | 健康探测端口 | 是 | 单向证书认证      | TLS1.3(缺省)/TLS1.2 | bkeagent-deployer              | - |
| ALL | ALL | ALL | openFuyao服务器 | 集群IP | 3377     | HTTP | 健康探测端口 | 是 | 无           | 无                 | bkeagent-launcher              | - |
| ALL | ALL | ALL | openFuyao服务器 | 环回地址 | 58080    | HTTP | 系统服务侦听端口 | 是 | 无，采用TCP标准协议 | 无                 | bkeagent                       | - |

> ![](./figures/icon-note.gif)**说明：**<br/>
> 通过应用市场安装的组件的通信端口信息请从相关组件提供方进行了解。