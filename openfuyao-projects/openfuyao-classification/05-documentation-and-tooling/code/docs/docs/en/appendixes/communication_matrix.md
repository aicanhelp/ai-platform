# Communication Matrix

## Host Port Usage List

### Host Exposed Ports

| Source Device | Source IP | Source Port | Destination Device | Destination IP | Destination Port (Listening) | Protocol | Port Description | Listening Port Configurable | Authentication Mode | Encryption Mode | Service/Microservice | Remarks |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ALL | ALL | ALL | openFuyao Server | Host IP | 36443 | TCP | Host exposed port (HostPort) | Yes | Password authentication and one-way certificate authentication | TLS1.3 (default)/TLS1.2 | kubernetes | Forwarded to PodIP:36443 via CNI-HOSTPORT rules |
| ALL | ALL | ALL | openFuyao Server | Host IP | 38080 | TCP | Host exposed port (HostPort) | Yes | None, standard TCP protocol | None | bocloud_chart_registry | Forwarded to PodIP:8080 via CNI-HOSTPORT rules |
| ALL | ALL | ALL | openFuyao Server | Host IP | 40080 | TCP | Host exposed port (HostPort) | Yes | None, standard TCP protocol | None | bocloud_yum_registry | Forwarded to PodIP:80 via CNI-HOSTPORT rules |
| ALL | ALL | ALL | openFuyao Server | Host IP | 40443 | TCP | Host exposed port (HostPort) | Yes | One-way certificate authentication | TLS1.3 (default)/TLS1.2 | bocloud_image_registry | Forwarded to PodIP:443 via CNI-HOSTPORT rules |
| ALL | ALL | ALL | openFuyao Server | Host IP | 30010 | TCP | Host exposed port (NodePort) | No | Username and password authentication | TLS1.3 (default)/TLS1.2 | ingress-nginx-controller | Exposed via Kubernetes NodePort |
| ALL | ALL | ALL | openFuyao Server | Host IP | 31616 | TCP | Host exposed port (NodePort) | No | Username and password authentication | TLS1.3 (default)/TLS1.2 | ingress-nginx-controller | Exposed via Kubernetes NodePort |
| ALL | ALL | ALL | openFuyao Server | Host IP | 22 | TCP | Host process listening port | No | Password/key authentication | SSH encryption | sshd | - |
| ALL | ALL | ALL | openFuyao Server | Host IP | 123 | UDP | Host process listening port | No | None, standard UDP protocol | None | bke | - |
| ALL | ALL | ALL | openFuyao Server | Host IP | 6443 | TCP | Host process listening port | Yes | Password authentication and one-way certificate authentication | TLS1.3 (default)/TLS1.2 | kube-apiserver | - |
| ALL | ALL | ALL | openFuyao Server | Host IP | 10250 | TCP | Host process listening port | Yes | Token authentication and certificate authentication | TLS1.3 (default)/TLS1.2 | kubelet | - |
| ALL | ALL | ALL | openFuyao Server | Host IP | 10249 | TCP | Host process listening port | Yes | None, standard TCP protocol | None | kube-proxy | - |
| ALL | ALL | ALL | openFuyao Server | Host IP | 10256 | TCP | Host process listening port | Yes | None, standard TCP protocol | None | kube-proxy | - |
| ALL | ALL | ALL | openFuyao Server | Host IP | 10257 | TCP | Host process listening port | Yes | Two-way certificate authentication | TLS1.3 (default)/TLS1.2 | kube-controller-manager | - |
| ALL | ALL | ALL | openFuyao Server | Host IP | 10259 | TCP | Host process listening port | Yes | Two-way certificate authentication | TLS1.3 (default)/TLS1.2 | kube-scheduler | - |
| ALL | ALL | ALL | openFuyao Server | Host IP | 2379 | TCP | Host process listening port | Yes | Two-way certificate authentication | TLS1.3 (default)/TLS1.2 | etcd | - |
| ALL | ALL | ALL | openFuyao Server | Host IP | 2380 | TCP | Host process listening port | Yes | Two-way certificate authentication | TLS1.3 (default)/TLS1.2 | etcd | - |
| ALL | ALL | ALL | openFuyao Server | Host IP | 2381 | TCP | Host process listening port | Yes | None, standard TCP protocol | None | etcd | - |
| ALL | ALL | ALL | openFuyao Server | Host IP | 8443 | TCP | Host process listening port | Yes | One-way certificate authentication | TLS1.3 (default)/TLS1.2 | socat | - |
| ALL | ALL | ALL | openFuyao Server | Host IP | 9100 | TCP | Host process listening port | Yes | None, standard TCP protocol | None | kube-rbac-proxy | - |
| ALL | ALL | ALL | openFuyao Server | Host IP | 4789 | UDP | Host process listening port | Yes | None, standard UDP protocol | None | calico/vxlan | - |

### Host Local Dependency Ports

| Source Device | Source IP | Source Port | Destination Device | Destination IP | Destination Port (Listening) | Protocol | Port Description | Listening Port Configurable | Authentication Mode | Encryption Mode | Service/Microservice | Remarks |
| --- | --- | --- | --- | --- |------------------------------| --- | --- | --- | --- | --- | --- | --- |
| ALL | ALL | ALL | openFuyao Server | Loopback address | 1338                         | TCP | Host local process listening port (required for deployment/runtime) | Yes | None, standard TCP protocol | None | containerd | Local access only |
| ALL | ALL | ALL | openFuyao Server | Loopback address | 10248                        | TCP | Host local process listening port (required for deployment/runtime) | Yes | None, standard TCP protocol | None | kubelet | Local access only |
| ALL | ALL | ALL | openFuyao Server | Loopback address | 9099                         | HTTP | Host local process listening port (required for deployment/runtime) | Yes | None, standard TCP protocol | None | calico-node | Local access only |
| ALL | ALL | ALL | openFuyao Server | Loopback address | 58080                        | HTTP | Host local process listening port (required for deployment/runtime) | Yes | None, standard TCP protocol | None | bkeagent | Local access only |
| ALL | ALL | ALL | openFuyao Server | Loopback address | 323                          | UDP | Host local process listening port (required for deployment/runtime) | Yes | None, standard UDP protocol | None | chronyd | Only exists if chronyd is enabled on the node |

> ![](./figures/icon-note.gif)**Note:**<br/>
> This section categorizes ports related to cluster deployment/runtime into "Host Exposed Ports" and "Host Local Dependency Ports." The former includes both host process listening ports that can be directly observed via `ss`/`netstat`, as well as reachable ports formed by NodePort and HostPort/CNI DNAT forwarding. The latter typically only binds `127.0.0.1`/`::1` and is not externally exposed, but is depended upon by component deployment, health checks, or runtime communication. If occupied, it may also cause component startup failures, health check anomalies, or cluster startup failures.

## Kubernetes Communication Matrix

| Source Device | Source IP | Source Port | Destination Device | Destination IP | Destination Port (Listening) | Protocol | Port Description | Listening Port Configurable | Authentication Mode | Encryption Mode | Service/Microservice | Remarks |
| ----- | ---- | ------ | ---------------| ------ | -------------- | ---- | ------- | ----------------- | ------- | ------- | -------| -------------- |
| ALL | ALL | ALL | openFuyao Server | Cluster IP | 6443 | TCP | Used by third-party systems to obtain the aggregated service capabilities provided by openFuyao through this port. It is the listening port of kube-apiserver, which is an SSL-encrypted port. | Yes | Password authentication and one-way certificate authentication | TLS1.3 (default)/TLS1.2 | kube-apiserver | - |
| ALL | ALL | ALL | openFuyao Server | Cluster IP | 443 | TCP | Used by third-party systems to obtain the aggregated service capabilities provided by openFuyao through this port. It is the listening port of kube-apiserver, which is an SSL-encrypted port. | No | Token authentication or one-way certificate authentication | TLS1.3 (default)/TLS1.2 | kube-apiserver | - |
| ALL | ALL | ALL | openFuyao Server | Loopback address | 10257 | TCP | Used to access the HTTPS service of kube-controller-manager. It is the listening port of kube-controller-manager, which is an SSL-encrypted port. | Yes | Two-way certificate authentication | TLS1.3 (default)/TLS1.2 | kube-controller-manager | - |
| ALL | ALL | ALL | openFuyao Server | Loopback address | 10259 | TCP | Used to access the HTTPS service of kube-scheduler. It is the listening port of kube-scheduler, which is an SSL-encrypted port. | Yes | Two-way certificate authentication | TLS1.3 (default)/TLS1.2 | kube-scheduler | - |
| ALL | ALL | ALL | openFuyao Server | Cluster IP | 53 | TCP/UDP | Used for service domain name resolution. It is the listening port of coredns, used only for domain name resolution. | No | None, standard TCP protocol | None | coredns | - |
| ALL | ALL | ALL | openFuyao Server | Cluster IP | 9153 | TCP | Used to collect monitoring metrics of coredns. | Yes | None, standard TCP protocol | TLS1.3 (default)/TLS1.2 | coredns | - |
| ALL | ALL | ALL | openFuyao Server | Cluster IP | 8080 | TCP | Used to check the health status of coredns. | Yes | None, standard TCP protocol | TLS1.3 (default)/TLS1.2 | coredns | - |
| ALL | ALL | ALL | openFuyao Server | Cluster IP | 2379 | TCP | etcd provides services through this port, which is an SSL-encrypted port. | Yes | Two-way certificate authentication | TLS1.3 (default)/TLS1.2 | etcd | - |
| ALL | ALL | ALL | openFuyao Server | Cluster IP | 2380 | TCP | Used for communication between etcd cluster members, which is an SSL-encrypted port. | Yes | Two-way certificate authentication | TLS1.3 (default)/TLS1.2 | etcd | - |
| ALL | ALL | ALL | openFuyao Server | Cluster IP | 2381 | TCP | Used to collect monitoring metrics externally provided by the etcd cluster. | Yes | None, standard TCP protocol | TLS1.3 (default)/TLS1.2 | etcd | - |
| ALL | ALL | ALL | openFuyao Server | Loopback address | 9099 | HTTP | Used to provide service liveness status check functionality. | Yes | None, standard TCP protocol | None | calico-node | - |
| ALL | ALL | ALL | openFuyao Server | Cluster IP | 179 | TCP | calico component bird BGP protocol listening port, used for synchronizing routes between nodes. | Yes | Password authentication | None | calico-node | - |
| ALL | ALL | ALL | openFuyao Server | Loopback address | 10248 | TCP | Used to check the health status of kubelet. | Yes | None, standard TCP protocol | None | kubelet | - |
| ALL | ALL | ALL | openFuyao Server | Cluster IP | 10250 | TCP | Used to communicate with kube-apiserver, which is an SSL-encrypted port. | Yes | Token authentication and certificate authentication | TLS1.3 (default)/TLS1.2 | kubelet | - |
| ALL | ALL | ALL | openFuyao Server | Loopback address | 10249 | TCP | Used to collect monitoring metrics. | Yes | None, standard TCP protocol | None | kube-proxy | - |
| ALL | ALL | ALL | openFuyao Server | Loopback address | 10256 | TCP | Used for health status check. | Yes | None, standard TCP protocol | None | kube-proxy | - |

> ![](./figures/icon-note.gif)**Note:**<br/>
> For the Kubernetes official documentation, see [Ports and Protocols](https://kubernetes.io/docs/reference/networking/ports-and-protocols/).

## openFuyao Application Component Communication Matrix

| Source Device | Source IP | Source Port | Destination Device | Destination IP | Destination Port (Listening) | Protocol | Port Description | Listening Port Configurable | Authentication Mode                                            | Encryption Mode         | Service/Microservice | Remarks |
| ----- | ---- | ------ | ---------------| ------ | -------------- | ---- | ------- |-----------------------------|----------------------------------------------------------------|-------------------------| -------------- | --- |
| ALL | ALL | ALL | openFuyao Server | PodIP | 443 | TCP | ingress-nginx-controller service port | No                          | Username and password authentication                           | TLS1.3 (default)/TLS1.2 | ingress-nginx-controller | - |
| ALL | ALL | ALL | openFuyao Server | Cluster IP | 443 | TCP | ingress-nginx-controller service port | No                          | Username and password authentication                           | TLS1.3 (default)/TLS1.2 | ingress-nginx-controller | - |
| ALL | ALL | ALL | openFuyao Server | IngressIP | 30010 | TCP | Bootstrap node management plane externally provided access port | No                          | Username and password authentication                           | TLS1.3 (default)/TLS1.2 | ingress-nginx-controller | - |
| ALL | ALL | ALL | openFuyao Server | IngressIP | 31616 | TCP | Business cluster & management cluster management plane externally provided access port | No                          | Username and password authentication                           | TLS1.3 (default)/TLS1.2 | ingress-nginx-controller | - |
| ALL | ALL | ALL | openFuyao Server | Cluster IP | 9072 | TCP | web-terminal-service backend Pod access port | Yes                         | Token authentication                                           | TLS1.3 (default)/TLS1.2 | web-terminal-service | - |
| ALL | ALL | ALL | openFuyao Server | PodIP | 9072 | TCP | web-terminal-service backend Service access port | Yes                         | Token authentication                                           | TLS1.3 (default)/TLS1.2 | web-terminal-service | - |
| ALL | ALL | ALL | openFuyao Server | PodIP | 80 | TCP | monitoring-service backend Pod access port | Yes                         | None, standard TCP protocol                                    | None                    | monitoring-service | - |
| ALL | ALL | ALL | openFuyao Server | PodIP | 9096 | TCP | Built-in authentication server Pod access port | Yes                         | Two-way certificate authentication                             | TLS1.3                  | oauth-server | - |
| ALL | ALL | ALL | openFuyao Server | Cluster IP | 9096 | TCP | Built-in authentication server Service access port | Yes                         | Two-way certificate authentication                             | TLS1.3                  | oauth-server | - |
| ALL | ALL | ALL | openFuyao Server | PodIP | 9095 | TCP | Authentication webhook Pod access port | Yes                         | Two-way certificate authentication                             | TLS1.3                  | oauth-webhook | - |
| ALL | ALL | ALL | openFuyao Server | Cluster IP | 9095 | TCP | Authentication webhook Service access port | Yes                         | Two-way certificate authentication                             | TLS1.3                  | oauth-webhook | - |
| ALL | ALL | ALL | openFuyao Server | PodIP | 9175 | TCP | User management service Pod access port | Yes                         | None, standard TCP protocol                                    | None                    | user-management-operator | - |
| ALL | ALL | ALL | openFuyao Server | Cluster IP | 80 | TCP | User management service Service access port | Yes                         | None, standard TCP protocol                                    | None                    | user-management-operator | - |
| ALL | ALL | ALL | openFuyao Server | PodIP | 9093 | TCP | Application management service backend Pod access port | Yes                         | None, standard TCP protocol                                    | None                    | application-management-service | - |
| ALL | ALL | ALL | openFuyao Server | Cluster IP | 80 | TCP | Application management service backend Service access port | Yes                         | None, standard TCP protocol                                    | None                    | application-management-service | - |
| ALL | ALL | ALL | openFuyao Server | PodIP | 9093 | TCP | Application market service backend Pod access port | Yes                         | None, standard TCP protocol                                    | None                    | marketplace-service | - |
| ALL | ALL | ALL | openFuyao Server | Cluster IP | 80 | TCP | Application market service backend Service access port | Yes                         | None, standard TCP protocol                                    | None                    | marketplace-service | - |
| ALL | ALL | ALL | Bootstrap Node | PodIP | 8080 | TCP | Cluster lifecycle management frontend Pod access port | Yes                         | None, standard TCP protocol                                    | None                    | installer-website | - |
| ALL | ALL | ALL | Bootstrap Node | Cluster IP | 80 | TCP | Cluster lifecycle management frontend Service access port | Yes                         | None, standard TCP protocol                                    | None                    | installer-website | - |
| ALL | ALL | ALL | Bootstrap Node | PodIP | 9210 | TCP | Cluster lifecycle management backend Pod access port | Yes                         | None, standard TCP protocol                                    | None                    | installer-service | - |
| ALL | ALL | ALL | Bootstrap Node | Cluster IP | 80 | TCP | Cluster lifecycle management backend Service access port | Yes                         | None, standard TCP protocol                                    | None                    | installer-service | - |
| ALL | ALL | ALL | Bootstrap Node | PodIP | 8080 | TCP | Bootstrap node externally provided console frontend Pod access port | Yes                         | None, standard TCP protocol                                    | None                    | bke-console-website | - |
| ALL | ALL | ALL | Bootstrap Node | Cluster IP | 80 | TCP | Bootstrap node externally provided console frontend Service access port | Yes                         | None, standard TCP protocol                                    | None                    | bke-console-website | - |
| ALL | ALL | ALL | Bootstrap Node | PodIP | 9037 | TCP | Bootstrap node externally provided console backend Pod access port | Yes                         | None, standard TCP protocol                                    | None                    | bke-console-website | - |
| ALL | ALL | ALL | Bootstrap Node | Cluster IP | 80 | TCP | Bootstrap node externally provided console backend Pod access port | Yes                         | None, standard TCP protocol                                    | None                    | bke-console-website | - |
| ALL | ALL | ALL | openFuyao Server | PodIP | 9093 | TCP | Extension component management backend Pod access port | Yes                         | None, standard TCP protocol                                    | None                    | plugin-management-service | - |
| ALL | ALL | ALL | openFuyao Server | Cluster IP | 80 | TCP | Extension component management backend Service access port | Yes                         | None, standard TCP protocol                                    | None                    | plugin-management-service | - |
| ALL | ALL | ALL | openFuyao Server | Cluster IP | 9100 | TCP | Node Exporter Service access port | Yes                         | None, standard TCP protocol                                    | None                    | node-exporter | - |
| ALL | ALL | ALL | Bootstrap Node | Container IP | 36443 | HTTPS | Bootstrap node externally provided K8s API port | Yes                         | Password authentication and one-way certificate authentication | TLS1.3 (default)/TLS1.2 | kubernetes | - |
| ALL | ALL | ALL | Bootstrap Node | Container IP | 38080 | HTTP | Bootstrap node externally provided charts service port | Yes                         | None, standard TCP protocol                                    | None                    | bocloud_chart_registry | - |
| ALL | ALL | ALL | Bootstrap Node | Container IP | 40080 | HTTP | Bootstrap node externally provided yum repository service port | Yes                         | None, standard TCP protocol                                    | None                    | bocloud_yum_registry | - |
| ALL | ALL | ALL | Bootstrap Node | Container IP | 40443 | HTTPS | Bootstrap node externally provided image repository service port | Yes                         | One-way certificate authentication                             | TLS1.3 (default)/TLS1.2 | bocloud_image_registry | - |
| ALL | ALL | ALL | openFuyao Server | Cluster IP | 8443 | HTTPS | Health probe port | Yes                         | One-way certificate authentication                             | TLS1.3 (default)/TLS1.2 | bkeagent-deployer | - |
| ALL | ALL | ALL | openFuyao Server | Cluster IP | 3377     | HTTP | Health probe port | Yes                         | None                                                           | None                    | bkeagent-launcher              | - |
| ALL | ALL | ALL | openFuyao Server | Loopback address | 58080 | HTTP | System service listening port | Yes                         | None, standard TCP protocol                                    | None                    | bkeagent | - |

> ![](./figures/icon-note.gif)**Note:**<br/>
> For communication port information of components installed through the application market, please consult the relevant component providers.
