# Best Practices

## Best Practices for kube-apiserver Load Balancing in the Ultra-Large-Scale Cluster Scenario

- [Keepalived](https://www.keepalived.org/) provides VRRP implementation and allows configuring Linux machines for service load balancing to prevent single points of failure.

- [HAProxy](http://www.haproxy.org/) provides reliable, high-performance load balancing and works perfectly with Keepalived.

### Objectives

This best practice focuses on guiding how to implement kube-apiserver load balancing functionality using a combination of keepalived and haproxy.

### Prerequisites

- The cluster uses kube-apiserver as the API server for the Kubernetes cluster and is deployed on multiple nodes.

- There are two or more additional load balancing (LB) nodes responsible for configuring load balancing, with keepalived and haproxy services installed on the nodes, and the virtual IP can be reached within the cluster.

### Restrictions
None

### Background

High-availability Kubernetes clusters need to ensure no service interruptions during runtime and reasonably distribute requests across multiple kube-apiserver instances to avoid individual instances receiving too many requests, leading to excessive resource usage and inability to provide normal services.

### Procedure

Install keepalived and haproxy on LB nodes. When one of these LB nodes fails, the virtual IP address can automatically transfer to other LB nodes, allowing the cluster to continue operating normally, achieving high availability.

1. Log in to the LB node and complete the following haproxy configuration.

    1.1 All LB nodes need to configure the same `/etc/haproxy/haproxy.cfg` file. Example content is as follows:

    ```shell
    global
    maxconn 180000                                    # Maximum connections, recommended to configure as 150000 to ensure virtual nodes can be created normally.
    uid 0                                            # User ownership, configure according to actual situation.
    gid 0                                            # Group ownership, configure according to actual situation.
    
    defaults
    log global                                       # Use global logging.
    retries 3                                        # Maximum number of connection attempts, exceeding this value considers the backend server unavailable. Recommended to configure as 3, combined with connection timeout, ensure detection of backend server abnormalities within 10s.
    timeout connect 3s                               # Set timeout for waiting to forward client requests to backend server. Recommended to configure as 3s, combined with connection timeout, ensure detection of backend server abnormalities within 10s.
    timeout client 60s                               # Set the longest inactivity timeout for clients, set to 60s to reduce repeated connection establishment.
    timeout server 60s                               # Set the longest inactivity timeout for servers, set to 60s to reduce repeated connection establishment.
    
    frontend k8s-master                               # Frontend
    bind xxxx.xxxx.xxxx.xxxx:xxxx                    # Bind frontend vip ip as the virtual ip address to be used in kubeconfig, port is the port to be used in kubeconfig, configure according to actual situation.
    mode tcp                                         # Use tcp protocol to interact with kube-apiserver.
    default_backend k8s-master                       # Default backend host to call, defaults to k8s-master
    use_backend k8s-master                           # Call specified backend host k8s-master
    
    backend k8s-master                                # Backend
    option tcplog
    balance roundrobin                               # Load balancing algorithm: recommended to configure as roundrobin to evenly distribute requests across multiple kube-apiserver instances.
    default-server inter 3s rise 2 fall 3            # Specify default settings for backend servers, can also configure individually for servers below.
                                                    # inter 3s: monitoring status check interval, in milliseconds, default is 2000, recommended to set to 3 seconds, check every 3 seconds if kube-apiserver endpoint is normal, ensure detection of backend server abnormalities within 10s.
                                                    # rise 2: If two consecutive checks succeed, this kube-apiserver endpoint is considered available.
                                                    # fall 3: If three consecutive checks fail, this kube-apiserver endpoint is considered unavailable.
    server  master1  xxx.xxx.xxx.xxx:xxxx check      # kube-apiserver instance 1 ip:port information, configure according to actual situation.
    server  master2  xxx.xxx.xxx.xxx:xxxx check      # kube-apiserver instance 2 ip:port information, configure according to actual situation.
    server  master3  xxx.xxx.xxx.xxx:xxxx check      # kube-apiserver instance 3 ip:port information, configure according to actual situation.
    ```

    1.2 Add a `haproxy.conf` file in the `/etc/rsyslog.d` directory and write the content.

    ```
    # This line configures haproxy log output file, can be customized freely
    local1.* /etc/haproxy/haproxy.log
    # Output logs using socket method
    $ModLoad imuxsock
    $AddUnixListenSocket /etc/haproxy/dev/log
    ```

    1.3 After saving the configuration file, execute the following command to restart the haproxy service to make the new configuration effective.

    ```shell
    systemctl restart haproxy
    ```

    1.4 Set haproxy to start automatically after boot.

    ```shell
    systemctl enable haproxy
    ```
2. Log in to the LB node and complete the following keepalived configuration.

    2.1 Configure the `/etc/keepalived/keepalived.conf` file on each node separately, with different configuration content for different nodes.

    ```shell
    global_defs {
        router_id 1                           # Used to identify the keepalived server, default value is hostname. This parameter has no substantial impact.
        vrrp_garp_master_refresh 10           # ARP aging parameter, refreshes route ARP at specified intervals to prevent vip connectivity issues due to untimely ARP refresh. Recommended to configure as 10.
    }

    vrrp_script chk_haproxy {                 # Configure detection script, check haproxy status every 3s, consider haproxy unavailable on this node after 3 consecutive failures, consider available after 2 consecutive successes.
    script "killall -0 haproxy"
    interval 3
    weight 100
    fall 3
    rise 2
    }

    vrrp_script chk_apiserver {                 # Configure detection script, check cluster accessibility status every 3s, consider haproxy unavailable on this node after 3 consecutive failures, consider available after 2 consecutive successes.
    script "/etc/keepalived/check_apiserver.sh"
    interval 3
    weight -100
    fall 3
    rise 2
    }

    vrrp_instance VI1 {                       # VI1 is the instance name, configure the same for multiple keepalived instances, value can be customized.
        state BACKUP                          # Both LB nodes are configured as BACKUP to ensure the two LB nodes are equal.
        priority 100                          # Priority of this keepalived instance, configure with the same value for multiple instances.
        interface eth0                        # Network interface where the IP to be routed is located, i.e., the network interface where the kube-apiserver bound IP is located, election will use the configured interface.
        virtual_router_id 24                  # Virtual router ID, range 0-255, multiple keepalived in the same cluster must use the same ID, value must meet range requirements.
        nopreempt                             # Working mode, nopreempt means working in non-preemptive mode, default is preemptive mode preempt, use non-preemptive mode to reduce frequent vip switching.
        advert_int 1                          # Heartbeat notification frequency, multiple keepalived must configure the same heartbeat notification frequency. Recommended to configure as 1 to ensure high availability.
        virtual_ipaddress {                   # Virtual VIP address, configure as the ip to be used by kubeconfig, kubeconfig will use this ip for load balancing.
            xxx.xxx.xxx.xxx/xx dev eth0       # IP/mask of virtual VIP address and on which network interface it's configured, configure according to actual situation.
        }
        notify_master ""                      # Scripts to execute after host state changes to Master|Backup|Fault
        notify_backup ""
        notify_fault ""
        unicast_src_ip xxx.xxx.xxx.xxx        # IP address of this LB host, configure according to actual situation.
        unicast_peer {
            xxx.xxx.xxx.xxx                   # IP addresses of other LB hosts, configure according to actual situation.          
        }
        
        track_script {
            chk_haproxy
            chk_apiserver
        }
    }
    ```

    2.2 Configure cluster status detection script `/etc/keepalived/check_apiserver.sh`.

    ```shell
    #!/bin/sh
    VIP_ADDRESS=            # Fill in the frontend virtual IP bound by haproxy here
    VIP_BIND_PORT=          # Fill in the frontend port bound by haproxy here
    if ip addr | grep -q ${VIP_ADDRESS}; then
        curl --silent --max-time 2 --insecure https://${VIP_ADDRESS}:${VIP_BIND_PORT}/healthz 
        exit $?
    fi
    exit 0
    ```

    2.3 Execute the following command to set permissions for the script.

    ```shell
    chmod +x /etc/keepalived/check_apiserver.sh
    ```

    2.4 After saving the configuration file, execute the following command to restart the keepalived service to make the new configuration effective.

    ```shell
    systemctl restart keepalived
    ```

    2.5 Execute the following command to set haproxy to start automatically after boot.

    ```shell
    systemctl enable keepalived
    ```

3. Verify high availability

    3.1 Execute the following command on all LB nodes.

    ```shell
    ip a
    ```

    3.2 It can be observed that only one LB node has the virtual IP address on the network interface configured with the virtual IP address. Execute the following command to restart haproxy on the node where the current virtual IP address is located.

    ```shell
    systemctl stop haproxy
    ```

    3.3 After waiting 10s, execute the command again on all LB nodes.

    ```shell
    ip a
    ```

    At this time, it can be observed that only one LB node has the virtual IP address on the network interface configured with the virtual IP address, and it is different from the previous node where the virtual IP address was located. This proves that the virtual IP address can switch between multiple LB nodes, achieving high availability.

    3.4 Execute the following command to restore the stopped haproxy.
    ```shell
    systemctl restart haproxy
    ```

4. Configure kubeconfig

    In Kubernetes, each component can use the kubeconfig file to access the kube-apiserver endpoint specified in the kubeconfig file. By setting a virtual IP address, requests can be routed to multiple kube-apiserver instances, achieving load balancing.

    > ![](./figures/icon-note.gif)**Note:**
    >
    > Note that when using a virtual IP address to configure kubeconfig, the pre-planned virtual IP address also needs to be added when issuing certificates.

5. Restart all Kubernetes components to use the new kubeconfig for services.

### References

- [Keepalived](https://www.keepalived.org/) 
- [HAProxy](http://www.haproxy.org/)
