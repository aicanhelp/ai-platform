# CoreDNS最佳实践

在Kubernetes中，CoreDNS是默认的DNS服务器，负责为集群内的服务提供名称解析。其作用包括。

1. 服务发现

   - 内部服务发现：CoreDNS解析Kubernetes服务的内部DNS名称，使得Pod可以通过服务名称互相通信。
   - 外部服务发现：允许Kubernetes集群内的Pod解析外部DNS名称。

2. 负载均衡

   - 负载均衡：当多个Pod提供相同的服务时，CoreDNS能够返回所有Pod的IP地址，客户端可以轮询这些IP地址实现负载均衡。

本最佳实践聚集在CoreDNS本身，以支持大规模场景下CoreDNS的性能与稳定性。

## 目标

在大规模Kubernetes集群中，CoreDNS可能会遇到以下几个问题。

- 内存使用：CoreDNS的内存使用量主要受集群中Pod和Service数量的影响。在大规模集群中，随着Pod和Service数量的增加，CoreDNS的内存需求也会显著增加。
- 查询性能：CoreDNS的查询性能在高并发情况下可能会受到影响。默认配置下，CoreDNS的查询性能可能无法满足大规模集群的需求，需要进行优化和调整。
- 插件负载：某些插件（如autopath插件）虽然可以优化查询性能，但也会增加CoreDNS的内存和CPU负载。

针对这些问题，本最佳实践聚集如下的优化措施。

- 调整CoreDNS的配置参数。
- 使用更高性能的节点。
- 合理分配资源。

## 前提条件

CoreDNS为集群内的默认DNS服务器。

## 使用限制

本最佳实践基于Kubernetes v1.28、v1.33、v1.34，CoreDNS v1.10.1进行分析。

## 背景信息

在大规模Kubernetes集群中，CoreDNS的性能优化是非常重要的，因为它直接影响到整个集群的DNSS解析效率和可靠性。

- 高查询量：在大规模集群中，CoreDNS需要处理大量的DNS查询（如服务和Pod的解析）。高查询量会增加CoreDNS的负载，从而影响其性能。

- 内存使用：CoreDNS的内存使用主要受到集群中Pod和服务的数量影响。需要合理评估内存资源使用量。

- 查询延迟：高查询量和内存使用会导致CoreDNS的查询延迟增加。

- 资源限制：在大规模集群中，CoreDNS的CPU和内存资源限制需要根据集群规模进行调整。如果资源限制不足，CoreDNS可能无法处理高负载，导致性能下降。

- 负载均衡：为了提高可用性和容错能力，可以在多个CoreDNS实例之间分配负载。这样可以确保即使某个实例出现故障，其他实例仍然可以继续处理DNS查询。

- 监控和日志：定期监控CoreDNS的性能和健康状况，确保其正常运行。日志记录功能的开启也有助于故障排除和性能分析。

## 操作步骤

### CoreDNS配置优化

在Kubernetes中修改CoreDNS配置，可以通过编辑CoreDNS配置文件来实现这一点。以下是具体步骤。

1. 编辑CoreDNS配置文件。

    1.1 定位CoreDNS的ConfigMap配置文件。该配置通常部署于kube-system命名空间下，其资源名称为coredns。执行以下命令编辑该配置。

    ```sh
    kubectl -n kube-system edit configmap coredns
    ```

    1.2 修改配置。

        具体优化配置点见下面章节内容，优化后的建议配置如下。
        
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

    1.3 应用配置。

        按`Esc`后输入`:wq`，保存并退出编辑器，CoreDNS将自动重新加载配置。

    1.4 检查配置。

        确认配置已生效，可以查看CoreDNS的日志或监控其性能指标。
        
        ```sh
        kubectl -n kube-system logs -l k8s-app=kube-dns  # 查看CoreDNS的日志
        ```


2. 启用缓存。
 
    监控缓存效果：使用Prometheus或其他监控工具来监控CoreDNS的缓存效果，包括缓存命中率和缓存的大小。这有助于优化缓存配置，确保其在实际使用中发挥最佳效果。

    在Kubernetes中启用CoreDNS的缓存功能可以显著提高DNS查询的性能。通过这些步骤，可以在Kubernetes集群中启用CoreDNS的缓存功能，提高DNS查询的性能。

    在CoreDNS配置文件中，找到Corefile部分，以在Corefile中添加cache插件，并设置合适的缓存时间。

    ```plaintext
    .:53 {
        cache 30
        ...
    }
    ```

    > ![](./figures/icon-note.gif)**说明：**
    > 
    >在上面的示例中，cache 30表示启用缓存功能，并设置缓存有效期为30秒。
    >
    > 根据集群的实际需求调整缓存时间。缓存时间过长可能导致缓存中的数据过时，而缓存时间过短则无法充分利用缓存的优势。通常，30秒到60秒是一个合理的设置。

3. 定期清理缓存（可选）。

    配置缓存时间后，CoreDNS在缓存有效期内将直接返回缓存结果，过期后自动重新查询上游DNS并更新缓存。该机制可有效管理缓存数据。

    在如下场景下，仍然需要定期清理缓存来确保最佳性能。

    - 高查询量场景：在高查询量的环境中，定期清理缓存可以防止缓存中的无用记录累积，避免过多的内存占用。
    - 数据变化频繁：如果DNS数据变化非常频繁，定期清理缓存可以确保尽快刷新过时的数据。
    - 性能优化：在进行性能调优和监控时，定期清理缓存有助于保持高效的缓存管理，减少因缓存命中率降低导致的查询延迟。


    CoreDNS的reload插件不直接用于清理缓存，但它可以通过定期重新加载配置文件间接清理缓存。reload插件会检测Corefile文件的更改并重新加载配置，从而刷新缓存。
    ```plaintext
    .:53 {
        cache 30
        reload 30s 15s
        ...
    }
    ```
    > ![](./figures/icon-note.gif)**说明：**
    > 
    > - 在这个配置中，reload 30s表示每30秒重新加载一次配置文件，从而间接清理缓存。
    > - reload 30s 15s表示CoreDNS会在30秒到45秒之间随机检查Corefile是否有变化。

4. 限制外部DNS查询。

    CoreDNS的stub domains能够显著优化外部域名的解析，以下是配置stub domains的作用和优点。

    - 减少查询延迟：stub domains可以直接将指定域名的解析请求转发到特定的上游DNS服务器，从而减少查询延迟。
    - 分流查询流量：通过将不同的域名解析请求分流到不同的上游DNS服务器，可以减少对默认DNS服务器的压力，优化查询性能。
    - 提高解析效率：对于特定域名的查询，stub domains可以确保请求直接到达最合适的DNS服务器，提高解析效率。

    以下是使用forward插件限制外部查询的配置示例，展示了如何配置stub domains。

    ```plaintext
    .:53 {
        forward . 127.0.0.1 { 
        force_tcp
        }
        ...
    }
    ```

    > ![](./figures/icon-note.gif)**说明：**
    > 
    > 在这个配置中，所有未匹配的查询都会被转发到127.0.0.1，一个无效的DNS服务器地址，从而有效地限制外部DNS查询。

5. 启用Autopath插件。

    高查询量和内存使用会导致CoreDNS的查询延迟增加。为了减少延迟，可以启用autopath插件，它通过减少因ndots:5问题导致的性能开销来优化查询性能。
    > ![](./figures/icon-note.gif)**说明：**
    >
    > 启用autopath插件后，内存使用会增加，但查询性能会有所提升。

    5.1 编辑CoreDNS配置文件。

        首先，找到CoreDNS的ConfigMap配置文件。通常它位于kube-system命名空间中，名字为coredns。
        
        ```sh
        kubectl -n kube-system edit configmap coredns
        ```

    5.2 启用autopath插件。

    ```plaintext
    .:53 {
        autopath @kubernetes
        ...
    }
    ```

    > 注意事项:
    > - 启用autopath后，CoreDNS会需要更多的内存来存储有关Pod的信息，因此需要调整CoreDNS的内存请求和限制。如果Pod的数量较多，CoreDNS的Readiness探针可能会超时，导致CoreDNS无法服务。
    > - 如果autopath无法启用，则在应用程序中尽量使用完全限定域名（FQDN），以避免搜索域的附加查询。例如，将my-service改为my-service.default.svc.cluster.local。

6. 启用loadbalance插件。

    在CoreDNS中，loadbalance插件的作用是对DNS响应中的IP地址进行负载均衡。它通过重新排列返回的IP地址列表，确保每个客户端都可以获得不同的IP地址，从而实现负载分散。这对于高可用性和分布式系统来说尤为重要。

    - 实现负载均衡：在多个服务实例间分配负载，避免单个实例过载。
    - 提高可用性：通过分散负载，减少单点故障的影响，提高系统的整体可用性。
    - 均匀分发流量：确保不同的客户端可以获得不同的IP地址，从而均匀分发流量。

    以下是一个使用loadbalance插件的CoreDNS配置示例。

    ```plaintext
    .:53 {
        ...
        loadbalance
    }
    ```

    > ![](./figures/icon-note.gif)**说明：**
    >
    > 在这个配置中，loadbalance插件被添加到Corefile中。每次查询都会重新排列返回的IP地址列表，从而实现负载均衡。

### 配置合理的资源

根据集群中的Pod和Service数量，合理配置CoreDNS的内存资源。例如，可以使用以下公式来估算所需的内存。

- 未启用autopath插件。
    
    ```
    MB required (default settings) = (Number of Pods + Services) / 1000 + 54
    ```
   
- 启用autopath插件。

    ```
    MB required (with autopath) = (Number of Pods + Services) / 250 + 56
    ```

> ![](./figures/icon-note.gif)**说明：**
>
> - 内存建议2Gi。
> - CPU建议10vcpu。

### 使用高性能的节点类型

选择适合CoreDNS的高性能节点类型，以确保能够处理高查询量。

### 使用NodeLocal DNSCache

NodeLocal DNSCache通过在集群节点上作为DaemonSet运行DNS缓存代理来提高集群DNS性能。

1. 执行以下命令，下载nodelocaldns配置清单。

    ```
    wget https://raw.githubusercontent.com/kubernetes/kubernetes/refs/heads/master/cluster/addons/dns/nodelocaldns/nodelocaldns.yaml
    ```

2. 执行以下命令，为清单里参数设置变量及值。

    ```shell
    kubedns=`kubectl get svc kube-dns -n kube-system -o jsonpath={.spec.clusterIP}`
    domain=<cluster-domain>
    localdns=<node-local-address>
    ```

    > ![](./figures/icon-note.gif)**说明：**
    >
    > - `<cluster-domain>` 的默认值是"cluster.local"。
    > - `<node-local-address>` 是NodeLocal DNSCache选择的本地侦听IP地址（建议取值 169.254.20.10）。

3. 执行以下命令，使用变量替换清单里的值（kube-proxy运行在IPVS模式）。

   如果kube-proxy运行在IPVS模式，则执行如下命令进行替换。
    ```shell
    sed -i "s/__PILLAR__LOCAL__DNS__/$localdns/g; s/__PILLAR__DNS__DOMAIN__/$domain/g; s/,__PILLAR__DNS__SERVER__//g; s/__PILLAR__CLUSTER__DNS__/$kubedns/g" nodelocaldns.yaml
    ```

    在IPVS模式下，node-local-dns Pod只会侦听 `<node-local-address>`的地址。node-local-dns接口不能绑定kube-dns的集群IP地址，因为IPVS负载均衡使用的接口已经占用了该地址。node-local-dns Pod会设置 `__PILLAR__UPSTREAM__SERVERS__`。

    > ![](./figures/icon-note.gif)**说明：**
    >
    > - `__PILLAR__DNS__SERVER__`：对应kube-dns服务的ClusterIP地址。
    > - `__PILLAR__LOCAL__DNS__`：指定NodeLocal DNSCache的本地侦听IP地址。需确保该地址与集群现有地址无冲突，建议使用保留地址段，如IPv4链路本地地址169.254.0.0/16（默认采用 169.254.20.10），或IPv6唯一本地地址fd00::/8。
    > - `__PILLAR__DNS__DOMAIN__`：定义集群DNS域名，默认值为cluster.local。

4. 执行以下命令，部署NodeLocal DNSCache。
    ```
    kubectl create -f nodelocaldns.yaml
    ```

    执行后，node-local-dns Pod将以DaemonSet形式在每个集群节点的kube-system命名空间中启动。

5. 修改kubelet的`--cluster-dns`参数（kube-proxy运行在IPVS模式）。

   - 如果kube-proxy运行在IPVS模式，则需要修改kubelet的`--cluster-dns`参数为NodeLocal DNSCache正在侦听的`<node-local-address>`地址。
       - 修改kube-system命名空间里的kubelet-config configmap。
       - 修改kubelet启动参数里的`--config`的文件。
       - 重启kubelet应用新的配置`systemctl restart kubelet`。

        > ![](./figures/icon-note.gif)**说明：**
        >
        > 配置会影响新创建Pod的DNS设置，而现有Pod的DNS设置不会自动更新。

    - 否则，不需要修改`--cluster-dns`参数，因为NodeLocal DNSCache会同时侦听kube-dns服务的IP地址和 `<node-local-address>` 的地址。

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

> ![](./figures/icon-note.gif)**说明：**
>
> 通过以上信息可见，DNS查询请求已由本地缓存处理，并成功返回ANSWER字段结果10.10.0.1。

## 结论

通过修改CoreDNS的配置启用缓存及开启相关的插件，提升CoreDNS自身的处理性能，另外通过在每台节点部署LocalDNS缓存代理来提高集群DNS性能。
通过这些最佳实践能够使CoreDNS在大规模集群下高效稳定的运行。

## 参考资料

- [Scaling CoreDNS in Kubernetes Clusters](https://coredns.io/2018/11/15/scaling-coredns-in-kubernetes-clusters/)
- [在Kubernetes集群中使用NodeLocal DNSCache](https://kubernetes.io/zh-cn/docs/tasks/administer-cluster/nodelocaldns/)
- [How do I test the DNS service?](https://github.com/coredns/deployment/blob/master/kubernetes/FAQs.md)