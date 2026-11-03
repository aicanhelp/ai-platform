# CoreDNS Best Practices

In Kubernetes, CoreDNS is the default DNS server responsible for providing name resolution for services within the cluster. Its main functions include:

1. Service Discovery

   - Internal service discovery: CoreDNS resolves internal DNS names for Kubernetes services, allowing Pods to communicate with each other through service names.
   - External service discovery: Allows Pods in the Kubernetes cluster to resolve external DNS names.

2. Load Balancing

   - Load balancing: When multiple Pods provide the same service, CoreDNS can return all Pod IP addresses, and clients can poll these IP addresses to achieve load balancing.

This best practice focuses on CoreDNS itself to support CoreDNS performance and stability in large-scale scenarios.

## Objectives

In large-scale Kubernetes clusters, CoreDNS may encounter the following issues:

- Memory usage: CoreDNS's memory usage is mainly affected by the number of Pods and Services in the cluster. In large-scale clusters, as the number of Pods and Services increases, CoreDNS's memory requirements will also increase significantly.
- Query performance: CoreDNS's query performance may be affected under high concurrency. Under default configuration, CoreDNS's query performance may not meet the needs of large-scale clusters and requires optimization and adjustment.
- Plugin load: Some plugins (such as the autopath plugin), although they can optimize query performance, will also increase CoreDNS's memory and CPU load.

For these issues, this best practice focuses on the following optimization measures:

- Adjusting CoreDNS configuration parameters.
- Using higher performance node types.
- Reasonable resource allocation.

## Prerequisites

CoreDNS is the default DNS server within the cluster.

## Usage Limitations

This best practice is based on analysis of Kubernetes v1.28, v1.33, v1.34, CoreDNS v1.10.1.

## Background Information

In large-scale Kubernetes clusters, CoreDNS performance optimization is very important because it directly affects the DNS resolution efficiency and reliability of the entire cluster.

- High query volume: In large-scale clusters, CoreDNS needs to handle a large number of DNS queries (such as service and Pod resolution). High query volume increases CoreDNS's load, thereby affecting its performance.

- Memory usage: CoreDNS's memory usage is mainly affected by the number of Pods and services in the cluster. Reasonable assessment of memory resource usage is needed.

- Query latency: High query volume and memory usage will cause CoreDNS's query latency to increase.

- Resource limits: In large-scale clusters, CoreDNS's CPU and memory resource limits need to be adjusted according to cluster scale. If resource limits are insufficient, CoreDNS may not be able to handle high loads, leading to performance degradation.

- Load balancing: To improve availability and fault tolerance, load can be distributed among multiple CoreDNS instances. This ensures that even if an instance fails, other instances can continue to handle DNS queries.

- Monitoring and logging: Regularly monitor CoreDNS's performance and health status to ensure its normal operation. Enabling logging also helps troubleshoot and analyze performance.

## Procedure

### CoreDNS Configuration Optimization

To modify CoreDNS configuration in Kubernetes, you can edit the CoreDNS configuration file to achieve this. Here are the specific steps:

1. Edit the CoreDNS configuration file.

    1.1 Locate the ConfigMap configuration file for CoreDNS. This configuration is usually deployed in the kube-system namespace with the resource name coredns. Execute the following command to edit the configuration.

    ```sh
    kubectl -n kube-system edit configmap coredns
    ```

    1.2 Modify configuration.

    For specific optimization configuration points, see the following sections. The recommended configuration after optimization is as follows:
        
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

    1.3 Apply configuration.

        Press `Esc` and enter `:wq` to save and exit the editor. CoreDNS will automatically reload the configuration.

    1.4 Check configuration.

    Confirm the configuration has taken effect by checking CoreDNS's logs or monitoring its performance metrics.
        
    ```sh
    kubectl -n kube-system logs -l k8s-app=kube-dns  # Check CoreDNS logs
    ```


2. Enable caching.
 
    Monitor cache effectiveness: Use Prometheus or other monitoring tools to monitor CoreDNS's cache effectiveness, including cache hit rate and cache size. This helps optimize cache configuration, ensuring it works best in actual use.

    Enabling CoreDNS's caching feature in Kubernetes can significantly improve DNS query performance. Through these steps, you can enable CoreDNS's caching feature in a Kubernetes cluster, improving DNS query performance.

    In the CoreDNS configuration file, find the Corefile section to add the cache plugin in the Corefile and set appropriate cache time.

    ```plaintext
    .:53 {
        cache 30
        ...
    }
    ```

    > ![](./figures/icon-note.gif)**Note:**
    > 
    >In the above example, cache 30 means enable caching feature with a cache validity period of 30 seconds.
    >
    > Adjust cache time according to the cluster's actual needs. Too long cache time may cause cached data to become stale, while too short cache time cannot fully utilize the caching advantage. Usually, 30 seconds to 60 seconds is a reasonable setting.

3. Regularly clean cache (optional).

    After configuring cache time, CoreDNS will directly return cached results within the cache validity period, and automatically requery upstream DNS and update cache after expiration. This mechanism can effectively manage cached data.

    In the following scenarios, regular cache cleaning is still needed to ensure optimal performance:

    - High query volume scenarios: In high query volume environments, regularly cleaning cache can prevent accumulation of useless records in the cache, avoiding excessive memory usage.
    - Frequent data changes: If DNS data changes very frequently, regularly cleaning cache can ensure timely refresh of stale data.
    - Performance optimization: When performing performance tuning and monitoring, regularly cleaning cache helps maintain efficient cache management, reducing query latency caused by decreased cache hit rate.


    CoreDNS's reload plugin is not directly used to clean cache, but it can indirectly clean cache by periodically reloading configuration files. The reload plugin detects changes in the Corefile file and reloads configuration, thereby refreshing cache.
    ```plaintext
    .:53 {
        cache 30
        reload 30s 15s
        ...
    }
    ```
    > ![](./figures/icon-note.gif)**Note:**
    > 
    > - In this configuration, reload 30s means reload the configuration file every 30 seconds, thereby indirectly cleaning cache.
    > - reload 30s 15s means CoreDNS will randomly check if Corefile has changed between 30 seconds to 45 seconds.

4. Limit external DNS queries.

    CoreDNS's stub domains can significantly optimize external domain name resolution. Here are the functions and advantages of configuring stub domains:

    - Reduce query latency: stub domains can directly forward resolution requests for specified domain names to specific upstream DNS servers, thereby reducing query latency.
    - Distribute query traffic: By distributing different domain name resolution requests to different upstream DNS servers, pressure on the default DNS server can be reduced, optimizing query performance.
    - Improve resolution efficiency: For queries of specific domain names, stub domains can ensure requests directly reach the most appropriate DNS server, improving resolution efficiency.

    The following is a configuration example using the forward plugin to limit external queries, showing how to configure stub domains:

    ```plaintext
    .:53 {
        forward . 127.0.0.1 { 
        force_tcp
        }
        ...
    }
    ```

    > ![](./figures/icon-note.gif)**Note:**
    > 
    > In this configuration, all unmatched queries will be forwarded to 127.0.0.1, an invalid DNS server address, thereby effectively limiting external DNS queries.

5. Enable Autopath plugin.

    High query volume and memory usage will cause CoreDNS's query latency to increase. To reduce latency, you can enable the autopath plugin, which optimizes query performance by reducing performance overhead caused by ndots:5 issues.
    > ![](./figures/icon-note.gif)**Note:**
    >
    > After enabling the autopath plugin, memory usage will increase, but query performance will improve.

    5.1 Edit the CoreDNS configuration file.

    First, find CoreDNS's ConfigMap configuration file. It is usually located in the kube-system namespace, named coredns.
        
    ```sh
    kubectl -n kube-system edit configmap coredns
    ```

    5.2 Enable autopath plugin.

    ```plaintext
    .:53 {
        autopath @kubernetes
        ...
    }
    ```

    > Notice:
    > - After enabling autopath, CoreDNS will need more memory to store information about Pods, so CoreDNS's memory requests and limits need to be adjusted. If there are many Pods, CoreDNS's Readiness probe may timeout, causing CoreDNS to be unable to serve.
    > - If autopath cannot be enabled, try to use fully qualified domain names (FQDN) in applications to avoid additional queries from search domains. For example, change my-service to my-service.default.svc.cluster.local.

6. Enable loadbalance plugin.

    In CoreDNS, the loadbalance plugin's function is to load balance IP addresses in DNS responses. It rearranges the returned IP address list, ensuring each client can get different IP addresses, thereby achieving load distribution. This is especially important for high availability and distributed systems.

    - Achieve load balancing: Distribute load among multiple service instances to avoid single instance overload.
    - Improve availability: By distributing load, reduce the impact of single point failures and improve overall system availability.
    - Evenly distribute traffic: Ensure different clients can get different IP addresses, thereby evenly distributing traffic.

    The following is a CoreDNS configuration example using the loadbalance plugin:

    ```plaintext
    .:53 {
        ...
        loadbalance
    }
    ```

    > ![](./figures/icon-note.gif)**Note:**
    >
    > In this configuration, the loadbalance plugin is added to Corefile. Each query will rearrange the returned IP address list, thereby achieving load balancing.

### Configure Reasonable Resources

According to the number of Pods and Services in the cluster, reasonably configure CoreDNS's memory resources. For example, you can use the following formula to estimate the required memory:

- Without autopath plugin enabled.
    
    ```
    MB required (default settings) = (Number of Pods + Services) / 1000 + 54
    ```
   
- With autopath plugin enabled.

    ```
    MB required (with autopath) = (Number of Pods + Services) / 250 + 56
    ```

> ![](./figures/icon-note.gif)**Note:**
>
> - Memory recommendation: 2Gi.
> - CPU recommendation: 10vcpu.

### Use High Performance Node Types

Select high-performance node types suitable for CoreDNS to ensure it can handle high query volumes.

### Use NodeLocal DNSCache

NodeLocal DNSCache improves cluster DNS performance by running DNS caching agents on cluster nodes as a DaemonSet.

1. Execute the following command to download the nodelocaldns configuration manifest.

    ```
    wget https://raw.githubusercontent.com/kubernetes/kubernetes/refs/heads/master/cluster/addons/dns/nodelocaldns/nodelocaldns.yaml
    ```

2. Execute the following command to set variables and values for parameters in the manifest.

    ```shell
    kubedns=`kubectl get svc kube-dns -n kube-system -o jsonpath={.spec.clusterIP}`
    domain=<cluster-domain>
    localdns=<node-local-address>
    ```

    > ![](./figures/icon-note.gif)**Note:**
    >
    > - The default value for `<cluster-domain>` is "cluster.local".
    > - `<node-local-address>` is the local listening IP address selected by NodeLocal DNSCache (recommended value: 169.254.20.10).

3. Execute the following command to use variables to replace values in the manifest (kube-proxy running in IPVS mode).

   If kube-proxy is running in IPVS mode, execute the following command for replacement:
    ```shell
    sed -i "s/__PILLAR__LOCAL__DNS__/$localdns/g; s/__PILLAR__DNS__DOMAIN__/$domain/g; s/,__PILLAR__DNS__SERVER__//g; s/__PILLAR__CLUSTER__DNS__/$kubedns/g" nodelocaldns.yaml
    ```

    In IPVS mode, the node-local-dns Pod only listens on the `<node-local-address>` address. The node-local-dns interface cannot bind kube-dns's cluster IP address because the interface used by IPVS load balancing has already occupied that address. The node-local-dns Pod will set `__PILLAR__UPSTREAM__SERVERS__`.

    > ![](./figures/icon-note.gif)**Note:**
    >
    > - `__PILLAR__DNS__SERVER__`: Corresponds to kube-dns service's ClusterIP address.
    > - `__PILLAR__LOCAL__DNS__`: Specifies NodeLocal DNSCache's local listening IP address. Ensure this address doesn't conflict with existing cluster addresses. Recommended to use reserved address ranges, such as IPv4 link-local addresses 169.254.0.0/16 (default: 169.254.20.10), or IPv6 unique local addresses fd00::/8.
    > - `__PILLAR__DNS__DOMAIN__`: Defines cluster DNS domain name, default value is cluster.local.

4. Execute the following command to deploy NodeLocal DNSCache.
    ```
    kubectl create -f nodelocaldns.yaml
    ```

    After execution, node-local-dns Pods will start as a DaemonSet in the kube-system namespace on each cluster node.

5. Modify kubelet's `--cluster-dns` parameter (kube-proxy running in IPVS mode).

   - If kube-proxy is running in IPVS mode, you need to modify kubelet's `--cluster-dns` parameter to the `<node-local-address>` address that NodeLocal DNSCache is listening on.
       - Modify the kubelet-config configmap in the kube-system namespace.
       - Modify the `--config` file in kubelet startup parameters.
       - Restart kubelet to apply the new configuration `systemctl restart kubelet`.

        > ![](./figures/icon-note.gif)**Note:**
        >
        > Configuration will affect DNS settings for newly created Pods, while DNS settings for existing Pods won't automatically update.

    - Otherwise, you don't need to modify the `--cluster-dns` parameter, because NodeLocal DNSCache will listen on both the kube-dns service IP address and the `<node-local-address>` address.

## Post-procedure

### NodeLocal DNSCache Verification

Use nslookup or dig command to test whether DNS queries work normally.

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

> ![](./figures/icon-note.gif)**Note:**
>
> From the above information, you can see that DNS query requests have been processed by local cache and successfully returned ANSWER field result 10.10.0.1.

## Conclusion

By modifying CoreDNS configuration to enable caching and relevant plugins, improving CoreDNS's own processing performance, and deploying LocalDNS caching agents on each node to improve cluster DNS performance.
Through these best practices, CoreDNS can run efficiently and stably in large-scale clusters.

## References

- [Scaling CoreDNS in Kubernetes Clusters](https://coredns.io/2018/11/15/scaling-coredns-in-kubernetes-clusters/)
- [Using NodeLocal DNS Cache in Kubernetes Clusters](https://kubernetes.io/zh-cn/docs/tasks/administer-cluster/nodelocaldns/)
- [How do I test the DNS service?](https://github.com/coredns/deployment/blob/master/kubernetes/FAQs.md)