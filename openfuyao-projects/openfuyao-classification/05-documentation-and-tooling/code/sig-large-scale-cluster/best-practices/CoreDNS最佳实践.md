# CoreDNS最佳实践

在 Kubernetes 中，CoreDNS 是默认的 DNS 服务器，负责为集群内的服务提供名称解析。其作用包括：

1. 服务发现

   - 内部服务发现：CoreDNS 解析 Kubernetes 服务的内部 DNS 名称，使得 Pod 可以通过服务名称互相通信。
   - 外部服务发现：允许 Kubernetes 集群内的 Pod 解析外部 DNS 名称。

2. 负载均衡

   - 负载均衡：当多个 Pod 提供相同的服务时，CoreDNS 能够返回所有 Pod 的 IP 地址，客户端可以轮询这些 IP 实现负载均衡。

本最佳实践聚集在 coredns 本身，以支持大规模场景下coredns的性能与稳定性。

## 目标

在大规模Kubernetes集群中，CoreDNS可能会遇到以下几个问题：

- 内存使用：CoreDNS的内存使用量主要受集群中Pod和Service数量的影响。在大规模集群中，随着Pod和Service数量的增加，CoreDNS的内存需求也会显著增加。
- 查询性能：CoreDNS的查询性能在高并发情况下可能会受到影响。默认配置下，CoreDNS的查询性能可能无法满足大规模集群的需求，需要进行优化和调整。
- 插件负载：某些插件（如autopath插件）虽然可以优化查询性能，但也会增加CoreDNS的内存和CPU负载。

针对这些问题，本最佳实践聚集如下的优化措施：

- 调整CoreDNS的配置参数
- 使用更高性能的节点
- 合理分配资源。

## 前提条件

CoreDNS 为集群内的默认 DNS 服务器。

## 使用限制

本最佳实践基于 kubernetes 1.28 版本,coredns v1.10.1 版本进行分析。

## 背景信息

在大规模 Kubernetes 集群中，CoreDNS 的性能优化是非常重要的，因为它直接影响到整个集群的 DNS 解析效率和可靠性。

1. 高查询量

    在大规模集群中，CoreDNS 需要处理大量的 DNS 查询（如服务和 Pod 的解析）。高查询量会增加 CoreDNS 的负载，从而影响其性能。

2. 内存使用

    CoreDNS 的内存使用主要受到集群中 Pod 和服务的数量影响。需要合理评估内存资源使用量。

3. 查询延迟

    高查询量和内存使用会导致 CoreDNS 的查询延迟增加。

4. 资源限制

    在大规模集群中，CoreDNS 的 CPU 和内存资源限制需要根据集群规模进行调整。如果资源限制不足，CoreDNS 可能无法处理高负载，导致性能下降。

5. 负载均衡

    为了提高可用性和容错能力，可以在多个 CoreDNS 实例之间分配负载。这样可以确保即使某个实例出现故障，其他实例仍然可以继续处理 DNS 查询。

6. 监控和日志

    定期监控 CoreDNS 的性能和健康状况，确保其正常运行。日志记录功能的开启也有助于故障排除和性能分析。

## 操作步骤

### CoreDNS配置优化

在 Kubernetes 中修改CoreDNS配置，可以通过编辑 CoreDNS 配置文件来实现这一点。以下是具体步骤：

1. 编辑 CoreDNS 配置文件

    首先，找到 CoreDNS 的 ConfigMap 配置文件。通常它位于 kube-system 命名空间中，名字为 coredns。

    ```sh
    kubectl -n kube-system edit configmap coredns
    ```

2. 修改配置

    具体优化配置点见下面章节内容。

    优化后的建议配置如下：
    
    ```yaml
    apiVersion: v1
    data:
      Corefile: |
        .:53 {
            errors
            health {
               lameduck 5s
            }
            ready
            kubernetes cluster.local in-addr.arpa ip6.arpa {
               fallthrough in-addr.arpa ip6.arpa
               ttl 600
            }
            prometheus :9153
            forward . /etc/resolv.conf {
               max_concurrent 1000
            }
            cache 600        
            loop
            reload
            loadbalance
        }
    kind: ConfigMap
    metadata:
      name: coredns
      namespace: kube-system
    ```

3. 应用配置

    保存并退出编辑器，CoreDNS 将自动重新加载配置。

4. 检查配置

    确认配置已生效，可以查看 CoreDNS 的日志或监控其性能指标：
    
    ```sh
    kubectl -n kube-system logs -l k8s-app=kube-dns
    ```

5. 监控缓存效果

    使用 Prometheus 或其他监控工具来监控 CoreDNS 的缓存效果，包括缓存命中率和缓存的大小。这有助于优化缓存配置，确保其在实际使用中发挥最佳效果。

通过这些步骤，可以在 Kubernetes 集群中启用 CoreDNS 的缓存功能，提高 DNS 查询的性能。

#### 启用缓存

在 Kubernetes 中启用 CoreDNS 的缓存功能可以显著提高 DNS 查询的性能。

在 CoreDNS 配置文件中，找到 Corefile 部分，以在 Corefile 中添加 cache 插件，并设置合适的缓存时间

```plaintext
.:53 {
    cache 30
    ...
}
```

在上面的示例中，cache 30 表示启用缓存功能，并设置缓存有效期为 30 秒。
> 根据集群的实际需求调整缓存时间。缓存时间过长可能导致缓存中的数据过时，而缓存时间过短则无法充分利用缓存的优势。通常，30秒到60秒是一个合理的设置。

#### 定期清理缓存（可选）

当配置了缓存时间，CoreDNS 会在缓存失效之前响应相同的查询请求，并在缓存时间过期后自动从上游 DNS 服务器重新获取数据。这种机制已经能有效管理缓存中的数据。

在如下场景下，仍然需要定期清理缓存来确保最佳性能：

- 高查询量场景：在高查询量的环境中，定期清理缓存可以防止缓存中的无用记录累积，避免过多的内存占用。
- 数据变化频繁：如果 DNS 数据变化非常频繁，定期清理缓存可以确保尽快刷新过时的数据。
- 性能优化：在进行性能调优和监控时，定期清理缓存有助于保持高效的缓存管理，减少因缓存命中率降低导致的查询延迟。


CoreDNS 的 reload 插件不直接用于清理缓存，但它可以通过定期重新加载配置文件间接清理缓存。reload 插件会检测 Corefile 文件的更改并重新加载配置，从而刷新缓存：
```plaintext
.:53 {
    cache 30
    reload 30s 15s
    ...
}
```
> - 在这个配置中，reload 30s 表示每 30 秒重新加载一次配置文件，从而间接清理缓存。
> - reload 30s 15s 表示CoreDNS会在 30 秒到 45 秒之间随机检查 Corefile 是否有变化。

#### 限制外部 DNS 查询

CoreDNS 的 stub domains 能够显著优化外部域名的解析。以下是配置 stub domains 的作用和优点：

- 减少查询延迟：stub domains 可以直接将指定域名的解析请求转发到特定的上游 DNS 服务器，从而减少查询延迟。
- 分流查询流量：通过将不同的域名解析请求分流到不同的上游 DNS 服务器，可以减少对默认 DNS 服务器的压力，优化查询性能。
- 提高解析效率：对于特定域名的查询，stub domains 可以确保请求直接到达最合适的 DNS 服务器，提高解析效率。

以下是使用 forward 插件限制外部查询的配置示例，展示了如何配置 stub domains：

```plaintext
.:53 {
    forward . 127.0.0.1 { 
      force_tcp
    }
    ...
}
```

> 在这个配置中，所有未匹配的查询都会被转发到 127.0.0.1，一个无效的 DNS 服务器地址，从而有效地限制外部 DNS 查询。

#### 启用Autopath插件

高查询量和内存使用会导致 CoreDNS 的查询延迟增加。为了减少延迟，可以启用 autopath 插件，它通过减少因 ndots:5 问题导致的性能开销来优化查询性能。

> 启用 autopath 插件后，内存使用会增加，但查询性能会有所提升。

1. 编辑 CoreDNS 配置文件

    首先，找到 CoreDNS 的 ConfigMap 配置文件。通常它位于 kube-system 命名空间中，名字为 coredns。
    
    ```sh
    kubectl -n kube-system edit configmap coredns
    ```

2. 启用 autopath 插件

   ```plaintext
   .:53 {
       autopath @kubernetes
       ...
   }
   ```

> 注意事项:
> - 启用autopath后，CoreDNS会需要更多的内存来存储有关Pod的信息，因此需要调整CoreDNS的内存请求和限制。如果 Pod 的数量较多，CoreDNS 的 Readiness 探针可能会超时，导致 CoreDNS 无法服务。
> - 如果 autopath 无法启用，则在应用程序中尽量使用完全限定域名（FQDN），以避免搜索域的附加查询。例如，将 my-service 改为 my-service.default.svc.cluster.local。

#### 启用loadbalance插件

在 CoreDNS 中，loadbalance 插件的作用是对 DNS 响应中的 IP 地址进行负载均衡。它通过重新排列返回的 IP 地址列表，确保每个客户端都可以获得不同的 IP 地址，从而实现负载分散。这对于高可用性和分布式系统来说尤为重要。

- 实现负载均衡：在多个服务实例间分配负载，避免单个实例过载。
- 提高可用性：通过分散负载，减少单点故障的影响，提高系统的整体可用性。
- 均匀分发流量：确保不同的客户端可以获得不同的 IP 地址，从而均匀分发流量。

以下是一个使用 loadbalance 插件的 CoreDNS 配置示例：

```plaintext
.:53 {
    ...
    loadbalance
}
```

在这个配置中，loadbalance 插件被添加到 Corefile 中。每次查询都会重新排列返回的 IP 地址列表，从而实现负载均衡。

### 配置合理的资源

根据集群中的 Pod 和 Service 数量，合理配置 CoreDNS 的内存资源。例如，可以使用以下公式来估算所需的内存：

1. 未启用 autopath 插件
    
    ```
    MB required (default settings) = (Number of Pods + Services) / 1000 + 54
    ```
   
2. 启用 autopath 插件

    ```
    MB required (with autopath) = (Number of Pods + Services) / 250 + 56
    ```

    > - 内存建议 2Gi
    > - CPU 建议 10vcpu

### 使用高性能的节点类型

> 选择适合CoreDNS的高性能节点类型，以确保能够处理高查询量。

### 使用 NodeLocal DNSCache

NodeLocal DNSCache 通过在集群节点上作为 DaemonSet 运行 DNS 缓存代理来提高集群 DNS 性能。

1. 下载nodelocaldns配置清单

    ```
    wget https://raw.githubusercontent.com/kubernetes/kubernetes/refs/heads/master/cluster/addons/dns/nodelocaldns/nodelocaldns.yaml
    ```

2. 为清单里参数设置变量及值

    ```shell
    kubedns=`kubectl get svc kube-dns -n kube-system -o jsonpath={.spec.clusterIP}`
    domain=<cluster-domain>
    localdns=<node-local-address>
    ```

    > - `<cluster-domain>` 的默认值是 "cluster.local"。
    > - `<node-local-address>` 是 NodeLocal DNSCache 选择的本地侦听 IP 地址（建议取值 169.254.20.10）。

3. 使用变量替换清单里的值（kube-proxy 运行在 IPVS 模式）

   如果 kube-proxy 运行在 IPVS 模式,则执行如下命令进行替换：
    ```shell
    sed -i "s/__PILLAR__LOCAL__DNS__/$localdns/g; s/__PILLAR__DNS__DOMAIN__/$domain/g; s/,__PILLAR__DNS__SERVER__//g; s/__PILLAR__CLUSTER__DNS__/$kubedns/g" nodelocaldns.yaml
    ```

    在 IPVS 模式下，node-local-dns Pod 只会侦听 `<node-local-address>` 的地址。node-local-dns 接口不能绑定 kube-dns 的集群 IP 地址，因为 IPVS 负载均衡使用的接口已经占用了该地址。node-local-dns Pod 会设置 `__PILLAR__UPSTREAM__SERVERS__`。

    参数说明：

   - `__PILLAR__DNS__SERVER__` ：表示 kube-dns Service 的 ClusterIP。
   - `__PILLAR__LOCAL__DNS__`：表示 DNSCache 本地监听的 IP 地址，该地址可以是任何地址，只要该地址不和你的集群里现有的 IP 地址发生冲突。推荐使用本地范围内的地址，例如 IPv4 链路本地区段 169.254.0.0/16 内的地址（默认一般取 169.254.20.10 即可），或者 IPv6 唯一本地地址区段 fd00::/8内的地址。
   - `__PILLAR__DNS__DOMAIN__`：表示集群域，默认就是 cluster.local。

4. 运行 `kubectl create -f nodelocaldns.yaml` 应用清单

    > 应用后，node-local-dns Pod 将在每个集群节点上的 kube-system 名字空间中运行。

5. 修改 kubelet 的 `--cluster-dns` 参数(kube-proxy 运行在 IPVS 模式)

   - 如果 kube-proxy 运行在 IPVS 模式，则需要修改 kubelet 的 `--cluster-dns` 参数为 NodeLocal DNSCache 正在侦听的 `<node-local-address>` 地址。
       - 修改 kube-system 命名空间里的 kubelet-config configmap
       - 修改 kubelet 启动参数里的`--config` 的文件
       - 重启 kubelet 应用新的配置`systemctl restart kubelet`。

    > 配置会影响新创建Pod的DNS设置，而现有Pod的DNS设置不会自动更新。

- 否则，不需要修改 `--cluster-dns` 参数，因为 NodeLocal DNSCache 会同时侦听 kube-dns 服务的 IP 地址和 `<node-local-address>` 的地址。

## 后续步骤

### NodeLocal DNSCache验证

使用nslookup或dig命令测试DNS查询是否正常工作。

```
kubectl run -it --rm --restart=Never --image=infoblox/dnstools:latest dnstools
dnstools# dig kubernetes.default A +search

; <<>> DiG 9.11.3 <<>> kubernetes.default A +search
;; global options: +cmd
;; Got answer:
;; WARNING: .local is reserved for Multicast DNS
;; You are currently testing what happens when an mDNS query is leaked to DNS
;; ->>HEADER<<- opcode: QUERY, status: NOERROR, id: 17578
;; flags: qr aa rd; QUERY: 1, ANSWER: 1, AUTHORITY: 0, ADDITIONAL: 1
;; WARNING: recursion requested but not available

;; OPT PSEUDOSECTION:
; EDNS: version: 0, flags:; udp: 4096
; COOKIE: 0c9fd27773778031 (echoed)
;; QUESTION SECTION:
;kubernetes.default.svc.cluster.local. IN A

;; ANSWER SECTION:
kubernetes.default.svc.cluster.local. 11 IN A   10.10.0.1

;; Query time: 0 msec
;; SERVER: 169.254.20.10#53(169.254.20.10)
;; WHEN: Thu Dec 12 03:42:32 UTC 2024
;; MSG SIZE  rcvd: 129
```

> 上面可以看到 DNSServer 走了本地查询且返回了 10.10.0.1 ANSWER 结果。

## 结论

通过修改 coredDNS 的配置启用缓存及开启相关的插件，提升 coreDNS 自身的处理性能，另外通过在每台节点部署 LocalDNS 缓存代理来提高集群 DNS 性能。
通过这些最佳实践能够使coreDNS在大规模集群下高效稳定的运行。

## 参考资料

> 列出参考或引用的任何相关资料、链接或资源。
- [Scaling CoreDNS in Kubernetes Clusters](https://coredns.io/2018/11/15/scaling-coredns-in-kubernetes-clusters/)
- [在 Kubernetes 集群中使用 NodeLocal DNSCache](https://v1-28.docs.kubernetes.io/zh-cn/docs/tasks/administer-cluster/nodelocaldns/)
- [How do I test the DNS service?](https://github.com/coredns/deployment/blob/master/kubernetes/FAQs.md)