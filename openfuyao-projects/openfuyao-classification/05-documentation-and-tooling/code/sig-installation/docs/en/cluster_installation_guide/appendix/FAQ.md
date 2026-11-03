# Frequently Asked Questions (FAQ)

This document aims to provide common installation and deployment issues along with available solutions.

## 1. The install-cni container of the calico component keeps restarting, with calico-node stuck in the init phase (1/3) and in a CrashLoopbackOff state

  - Cause

    You can run `ip route` on the node to view routing information. This issue is usually caused by one of the following two situations.

    1. The node is missing a default route.
    2. The node has multiple default routes, and the route assigned for cluster deployment is not the one with the highest priority (a smaller metric value indicates a higher priority).

  - Solution

    For cause 1, add a default route by running the following command.

    ```bash
    ip route add default via 192.168.100.1 dev eth0 proto static metric 100
    ```

    For cause 2, delete the redundant default route by running the following command.

    ```bash
    ip route del default via 192.168.90.1 dev eth1
    ```

  - Extended Discussion

    Calico requires that each node has a default route with the highest priority, essentially to ensure that all traffic to and from the cluster can be correctly forwarded at the node, which acts as a Layer 3 router.

    The default route is typically configured under the `/etc/sysconfig/network-scripts` directory, which defines and configures persistent configuration files for system network interfaces (network cards). When the system restarts or the network service restarts, the system reads these files to set the `IP` address, gateway, `DNS`, and other information.

## 2. calico-node restarts frequently, and the container falls into a "start - probe failure - restart" infinite loop

  - Cause

    This issue is usually caused by setting `initialDelaySeconds` too short. The Pod has not yet completed initialization, connection establishment, configuration loading, and other startup operations when `kubelet` starts probing the health status of `calico-node`.

  - Solution

    Set an appropriate delay time to tell `kubelet` to wait for a period of time after the container starts before performing the first probe. Run the following command to configure it.

    ```bash
    # Edit the K8s resource YAML, find the readinessProbe section, and add the initialDelaySeconds field
    kubectl edit ds -n kube-system calico-node
    ```

    The recommended `initialDelaySeconds` settings are as follows; adjust according to actual scenarios.

    - Small clusters (fewer than 10 nodes): set to 30, which is sufficient for `calico` to complete basic initialization.
    - Large clusters (more than 10 nodes): set to 60, as more nodes and intense resource competition require a longer startup time.

## 3. calico-node is in a 0/1 Running state

  - Detailed Symptoms

    View the detailed information as follows.

    - Node status is `NotReady`: This is the most obvious symptom. Through the `kubectl get nodes` command, you will find that the node status is not `Ready`, but `NotReady`.
    - `calico-node` Pod probe failure: When viewing the details of the `calico-node` Pod (`kubectl describe pod ...`) or logs, error messages similar to the following will appear repeatedly.
      - `Readiness probe failed: calico/node is not ready: BIRD is not ready`
      - `BGP not established with X.X.X.X` (X.X.X.X is usually the IP of another node)
      - `Error querying BIRD: unable to connect to BIRDv4 socket`
    - Cross-node Pod communication is interrupted: Because the BGP session cannot be established, routing information cannot be synchronized. The most direct consequence is that Pods on the current node cannot communicate normally with Pods on other nodes in the cluster.

  - Cause

    The `IP_AUTODETECTION_METHOD` field of `calico-node` is set incorrectly, causing node network inaccessibility.

  - Solution

    Run the following command to modify the value of the environment variable.

    ```bash
    # Edit the K8s resource YAML and set the IP_AUTODETECTION_METHOD environment variable
    kubectl edit ds -n kube-system calico-node
    ```

    Common settings are as follows.

    - `skip-interface=nerdctl*`: The default policy set by `openFuyao`, which skips network cards with the `nerdctl` prefix and selects the IP of the first valid network card.
    - `can-reach=192.168.100.5`: Directly specifies the network card IP of the target address.
    - `interface=eth4`: Uses the IP on the `eth4` network card.

  - Extended Discussion

    The `IP_AUTODETECTION_METHOD` field determines how Calico automatically selects the correct IP address for establishing `BGP` neighbors and encapsulating traffic on nodes with multiple network cards.

## 4. When the bootstrap cluster installs a service cluster, the bkeagent of the service cluster cannot connect to the APIServer of the bootstrap cluster, causing cluster installation to fail

  - Cause

    When `bkeagent` starts, it specifies the `--kube-config` parameter to configure the `APIServer` it monitors. After `bkeagent` starts, it checks whether the `CRD` it reconciles has been deployed in the monitored `APIServer`, which triggers `bkeagent` to attempt to connect to the bootstrap cluster's `APIServer`, resulting in the following error log.

    ```
    # For v25.12 and earlier versions, the log is /var/log/bkeagenbt.log; for later versions, it is /var/log/openFuyao/bkeagent.log
    The CRD cannot be installed in the target cluster, xxx
    ```

    When viewing the configuration file (for v25.12 and earlier versions, the log is at `/etc/bkeagent/config`; for later versions, it is at `/etc/openFuyao/bkeagent/config`), you will find that the `IP` address corresponding to `server` is not the address of the given bootstrap node.

  - Solution

    Run the following command to reset the bootstrap node, then specify the `IP` address for initialization.

    ```bash
    # Reset the bootstrap node
    bke reset --all --mount
    # Specify the IP address for initialization
    bke init --hostIP=1.2.3.4
    ```

## 5. When the bootstrap cluster deploys the openFuyao management plane, coredns is in a CrashLoopbackOff state

  - Details

    View the detailed logs of `coredns` (`kubectl logs -n kube-system coredns-xxx`), and the following logs appear (the actual log `IP` addresses and `Port` values differ).

    ```
    [ERROR] plugin/errors: 2 . NS: read udp 100.20.0.15:59690->100.10.0.10:53: i/o timeout
    [FATAL] plugin/loop: Loop(127.0.0.1:34812 -> :53) detected for zone ".", see https://coredns.io/plugins/loop#troubleshooting. Query: "HINFO ***"
    ```

  - Cause

    The above logs indicate that `coredns` falls into a `loop` infinite cycle during domain name resolution.

  - Solution

    `kubectl get cm -n kube-system coredns -o yaml` shows that `forward` is set to `/etc/resolv.conf`, which causes `coredns`'s `server ip` to be used as the server for domain name resolution when no other `DNS` server is available, ultimately resulting in an infinite loop.

    Run the following command to set the server.

    ```bash
    # Set forward to "forward . 8.8.8.8"; if a DNS server is available, set it to the corresponding IP address
    kubectl edit cm -n kube-system coredns
    ```

  - Extended Discussion

    The `forward` plugin forwards `DNS` requests that cannot be resolved within the cluster to the specified upstream `DNS` server.

## 6. Deploying a high-availability cluster using a virtual IP fails; after resetting nodes with bke reset, redeploying the high-availability cluster using the occupied virtual IP fails again

  - Cause

    The `keepalived` component of the high-availability cluster binds the virtual `IP` to a node in the high-availability cluster. After resetting the environment on each node using `bke reset`, the virtual `IP` is not unbound from the node it was bound to, causing an error when reused, ultimately preventing the cluster from being brought up.

  - Solution

    Log in to the Master node of the high-availability cluster from the first installation, and run the following command to unbind the virtual `IP`.

    ```bash
    # View the IP addresses bound to the node's network cards
    ip addr
    # If a virtual IP is found to be bound, run the following command to unbind it; replace vip with the actual virtual IP used, and eth0 with the actual bound network card
    ip addr del <vip> dev <eth0>
    ```

## 7. For clusters installed by openFuyao, besides deleting a cluster through the management plane, how can the backend delete a cluster

  For the formal steps, deletion order, cleanup scope, and success criteria, see [Cluster Uninstallation](../../cluster_uninstallation/overview.md) and [Delete a Service Cluster via Backend (Command Line)](../../cluster_uninstallation/delete_service_cluster_command.md).

  The `installer-service` of the bootstrap cluster or management cluster reads `BKECluster` data from the `APIServer`. In addition to deletion through the management plane, you can also execute the following in the **bootstrap node** or **management cluster** terminal:

  ```bash
  # Query cluster information
  kubectl get bc -A
  # Replace the placeholders with the actual namespace and name, then edit
  kubectl edit bc -n <namespace> <name>
  ```

  In the opened YAML editor, set the following two items (when performing a thorough cleanup of the target node):

  ```yaml
  metadata:
    annotations:
      bke.bocloud.com/ignore-target-cluster-delete: "false"
  spec:
    reset: true
  ```

## 8. How does the backend perform scaling operations for an openFuyao cluster

  The openFuyao management plane provides cluster lifecycle management capabilities, including cluster scaling in, scaling out, upgrade, installation, and uninstallation. Here we provide backend procedures for cluster scaling operations, which must be performed when the cluster is in a healthy state. When the cluster is in an unhealthy state, scaling operations may result in errors.

  - Scale in operation: remove nodes from an existing cluster.

    View existing `BKENode` resources.

    ```bash
    # Replace bke-cluster with the actual cluster information
    kubectl get bn -n bke-cluster
    ```

    Delete the corresponding BKENode resource.

    ```bash
    # Replace bke-cluster-n1 with the actual node name
    kubectl delete bn -n bke-cluster bke-cluster-n1
    ```

    Run the command to view existing `BKENode` resources again; if the corresponding node is no longer present, the deletion was successful.

  - Scale out operation: add new nodes to an existing cluster.

    Write the configuration file (newNode.yaml) for the new node.
    ```yaml
    apiVersion: bke.bocloud.com/v1beta1
    kind: BKENode
    metadata:
      name: bke-cluster-n1
      namespace: bke-cluster
      labels:
        cluster.x-k8s.io/cluster-name: bke-cluster
    spec:
      hostname: n1
      ip: <node-ip>
      password: '<encrypted>'
      port: "22"     
      role:
      - node
      username: root
    ```
    Run the following command to perform the scale out operation.
    ```bash
    kubectl apply -f newNode.yaml
    ```

    Run the command to view existing `BKENode` resources; if the corresponding node is Ready, the scale out was successful.

 Note: For v25.12 and earlier versions, follow the guide below.

  - Scale in operation: remove nodes from an existing cluster.

    Edit the `BKECluster` resource.

    ```bash
    # Replace bke-cluster with the actual cluster information
    kubectl edit bc -n bke-cluster bke-cluster
    ```

    Set the nodes scheduled for deletion.

    ```yaml
    metadata:
      annotations:
        # Node scheduled deletion; deleting a node is a dangerous action, so this annotation is added for secondary confirmation; none by default
        # When deleting a node, in addition to removing it from spec, you also need to fill in the IP of the node to be deleted; separate multiple IPs with ','
        # Both steps are required; missing either will not trigger node deletion
        bke.bocloud.com/appointment-deleted-nodes: "172.100.200.10"
    ```

    Remove the node information from `Spec`.

    ```yaml
    spec:
      clusterConfig:
        nodes:                
        - hostname: master-1  
          ip: 172.100.200.10
          username: root      
          password: password0
          port: "22"
          role:             
          - master
          - etcd
    ```

  - Scale out operation: add new nodes to an existing cluster.

    Edit the `BKECluster` resource.

    ```bash
    # Replace bke-cluster with the actual cluster information
    kubectl edit bc -n bke-cluster bke-cluster
    ```

    Add the node information to `Spec`.

    ```yaml
    spec:
      clusterConfig:
        nodes:                
        - hostname: master-1  
          ip: 172.100.200.10
          username: root      
          password: password0
          port: "22"
          role:             
          - master
          - etcd
    ```

## 9. When the default policy of the node iptables FORWARD chain is DROP, it causes abnormal container network communication and cluster initialization or creation failures

  - Detailed Symptoms

    When iptables exists on a node and the default policy of the FORWARD chain is DROP, one or more of the following symptoms may occur.

    1. After `bke init` initializes the bootstrap node, the container services (image repository, YUM repository, Chart repository) on the bootstrap node start normally, but inter-Pod communication in the bootstrap cluster (k3s) is abnormal, or Pod access to external services times out.
    2. When the bootstrap cluster deploys the openFuyao management plane, some Pods (such as the cluster-api-provider-bke controller) experience network inaccessibility, with errors such as connection timeouts or refusals in the logs.
    3. The openFuyao frontend management plane access times out; accessing the backend service via `curl` on the bootstrap node works normally, but running `curl` to access it on other nodes times out or is unreachable.
    4. During service cluster node initialization or the node joining process, image pulling times out, and the node cannot join the cluster.
    5. After the cluster is created, cross-node Pod communication fails, CoreDNS resolution times out, and Calico BGP sessions cannot be established.

  - Solution

    On the node experiencing the above symptoms, manually change the default policy of the iptables FORWARD chain to ACCEPT.

    ```bash
    # View the current default policy of the FORWARD chain
    iptables -t filter -nvL FORWARD

    # Set the default policy of the FORWARD chain to ACCEPT
    iptables -t filter -P FORWARD ACCEPT
    ```

    If you need to set this before initializing the bootstrap node, you can run the above command before executing `bke init`.

  - Extended Discussion

    Common sources of the iptables FORWARD chain default policy being DROP include:

    - firewalld firewall was enabled during system installation, and firewalld sets the FORWARD chain policy to DROP by default. Even if the firewalld service is subsequently stopped, the iptables policies already set will not automatically revert to ACCEPT.
    - In security hardening scenarios, administrators manually set the FORWARD chain policy to DROP to restrict unauthorized traffic forwarding.
    - Some Linux distributions (such as certain minimal installations of CentOS/RHEL) set the FORWARD policy to DROP by default.

## 10. During installation, /etc/resolv.conf is refreshed to default content by NetworkManager, causing node network abnormalities

  - Cause

    During installation, the `NetworkManager` (NM) service manages the node's `DNS` configuration and refreshes `/etc/resolv.conf` to default content, overwriting the configured valid `DNS` servers. This causes abnormal domain name resolution on the node, which in turn leads to network inaccessibility, image pulling failures, component communication timeouts, and other issues.

  - Solution

    Modify the configuration of `NetworkManager` and set `dns=none` to prevent `NetworkManager` from managing the `DNS` configuration.

    ```bash
    # Edit the NetworkManager main configuration file
    vi /etc/NetworkManager/NetworkManager.conf
    ```

    Add or modify the following configuration in the `[main]` section.

    ```ini
    [main]
    dns=none
    ```

    After saving, restart the `NetworkManager` service for the configuration to take effect, and confirm that the `DNS` configuration in `/etc/resolv.conf` is correct.

    ```bash
    # Restart the NetworkManager service
    systemctl restart NetworkManager
    # Confirm that the resolv.conf content has not been overwritten with default configuration
    cat /etc/resolv.conf
    ```

  - Extended Discussion

    `dns=none` means that `NetworkManager` will no longer write to or update `/etc/resolv.conf`, and the `DNS` configuration is maintained by the administrator. It is recommended to configure this before installation to avoid unexpected refresh during the installation process.

## 11. Bootstrap node k3s container keeps restarting intermittently

  - Detailed Phenomenon

    On the bootstrap node, execute `ps -ef | grep k3s` and `nerdctl ps -a`, and find that the k3s server restarts intermittently.
    Using `nerdctl logs kubernetes`, the log shows **failed to run iptables command to create KUBE-ROUTER-OUTPUT chain due to running [/bin/aux/iptables -t filter -S KUBE-ROUTER-OUTPUT 1 --wait]: exit status 3**.

  - Cause

    Execute `lsmod | grep -E 'ip_tables|iptable_filter'` with no output.
    The cause of this phenomenon is that when k3s starts the network component, the `iptables` related modules of the host kernel are missing, causing the network policy controller initialization to fail, triggering a panic and exit.

  - Solution

    The corresponding modules can be loaded with the following commands:
    ```bash
    modprobe ip_tables
    modprobe iptables_filter
    modprobe iptables_nat
    ```

## 12. On a single-node cluster, one of the two CoreDNS Pods occasionally stays in the Pending state

  - Detailed Phenomenon

    On the created single-node cluster, run `kubectl get pod -n kube-system | grep coredns`, and you will find that one CoreDNS Pod is in the `Running` state while the other is in the `Pending` state.

  - Solution

    Run the following command to scale down the CoreDNS replicas to 1. For a single-node environment, one CoreDNS replica is fully sufficient to provide DNS resolution services.
    ```bash
    kubectl scale deployment coredns --replicas=1 -n kube-system
    ```

## 13. In co-deployment scenarios, service cluster Pods have no log output, and log directories under /var/log/pods on the host are cleaned up every minute

  - Detailed Symptoms

    When the bootstrap node also serves as a service cluster node (co-deployment), one or more of the following symptoms occur after deployment.

    1. Running `kubectl logs` on service cluster Pods deployed on this node returns an error (such as `no such file or directory`) or produces no log output.
    2. When checking on this node, the log directories of service cluster Pods under `/var/log/pods` (named in the format `<namespace>_<PodName>_<UID>`) disappear approximately once per minute.
    3. The logs of bootstrap cluster Pods on this node are cleaned up in the same way.

  - Cause

    Since openFuyao v26.09 (the version that introduced mounting k3s container logs to the host), bootstrap node initialization bind-mounts the host's `/var/log/pods` and `/var/log/containers` into the k3s container at the same paths, so that bootstrap cluster logs can be viewed directly on the host.

    In a co-deployment scenario, two kubelets run on the node simultaneously: the kubelet embedded in k3s (bootstrap cluster) and the service cluster's kubelet, and both share the host's `/var/log/pods` directory.

    The kubelet's container GC runs once per minute and includes logic that cleans up Pod log directories: `/var/log/pods` is a hardcoded kubelet path, with no parameter available to modify or disable this cleanup behavior, and any Pod log directory unknown to the current kubelet is deleted as an orphan directory. As a result, the two kubelets treat each other's Pod log directories as orphans and delete them, manifesting as service cluster Pods continuously losing logs, and bootstrap cluster Pod logs being damaged as well.

    Note: Standard deployments (where the bootstrap node is not a service cluster node) are not affected by this issue and do not require the operations below.

  - Solution

    The core operation is to change the log mount paths of the k3s container so that the log directories of the two clusters are physically isolated. Depending on the deployment stage, there are two scenarios.

    Scenario 1: After the bootstrap node is successfully deployed and before co-deploying the service cluster (proactive prevention). The overall sequence is: save existing logs → recreate the k3s container (changing the log mount paths) → verify that the bootstrap cluster recovers → then co-deploy the service cluster.

    1. Save existing logs, in two parts.

       - Pod logs: After the container is recreated and the service cluster is co-deployed, the bootstrap cluster log directories left under the host's `/var/log/pods` will be cleaned up as orphan directories by the service cluster's kubelet.
       - k3s container's own logs: The logs of the k3s process (including the embedded kubelet, apiserver, and other control plane components) are output to the container's stdout and persisted by nerdctl; they are deleted together with the container when `nerdctl rm` is executed. The log of the containerd embedded in k3s is located at `/var/lib/rancher/k3s/agent/containerd/containerd.log`; this directory is bind-mounted to the host for persistence, is not affected by container recreation, and does not need to be backed up.

       ```bash
       mkdir -p /var/log/bke/backup
       cp -a /var/log/pods /var/log/bke/backup/pods-$(date +%Y%m%d%H%M%S)
       cp -a /var/log/containers /var/log/bke/backup/containers-$(date +%Y%m%d%H%M%S)
       # Export the k3s container's own logs
       nerdctl logs kubernetes > /var/log/bke/backup/k3s-container-$(date +%Y%m%d%H%M%S).log
       ```

    2. Create dedicated log directories for the bootstrap cluster.

       ```bash
       mkdir -p /var/log/bke/bootstrap/pods /var/log/bke/bootstrap/containers
       ```

    3. Recreate the k3s container. The bootstrap cluster state is saved in the host's `/etc/rancher/k3s` and `/var/lib/rancher/k3s`, so recreating the container does not affect the cluster data.

       ```bash
       # Get the complete startup parameters of the current container, and record the actual values of port mappings, --add-host, image address, etc.
       nerdctl inspect kubernetes

       # Stop and remove the container
       nerdctl stop kubernetes
       nerdctl rm kubernetes
       ```

       Recreate the container with the original startup parameters, replacing only the following two mounts:

       ```bash
       -v /var/log/pods:/var/log/pods
       -v /var/log/containers:/var/log/containers
       ```

       with:

       ```bash
       -v /var/log/bke/bootstrap/pods:/var/log/pods
       -v /var/log/bke/bootstrap/containers:/var/log/containers
       ```

       Keep all other parameters consistent with the output of `nerdctl inspect kubernetes`. Note that the actual startup parameters may include more than the example: for example, the `--dns` parameter is appended after the management plane is installed, and some environments also have parameters such as `--kube-apiserver-arg` (Webhook authentication configuration). When recreating the container, always follow the inspect output and include these parameters; omitting them changes the cluster behavior. An example is as follows (replace the content in angle brackets with actual values).

       ```bash
       nerdctl run -d --name=kubernetes \
         -p <kubernetesPort>:36443 -p <consolePort>:30010 \
         --privileged --restart=always \
         --add-host <imageRepo>:<imageRepoIP> \
         -v /etc/rancher/k3s:/etc/rancher/k3s \
         -v /etc/timezone:/etc/timezone \
         -v /etc/docker:/etc/docker \
         -v /etc/localtime:/etc/localtime \
         -v /var/lib/rancher/k3s:/var/lib/rancher/k3s \
         -v /bke:/bke \
         -v /etc/openFuyao:/etc/openFuyao \
         -v /etc/openFuyao/addons/manifests:/etc/openFuyao/addons/manifests \
         -v /var/log/bke/bootstrap/pods:/var/log/pods \
         -v /var/log/bke/bootstrap/containers:/var/log/containers \
         <k3sImage> server --snapshotter=native --https-listen-port=<kubernetesPort> \
         --service-cidr=100.10.0.0/16 --cluster-cidr=100.20.0.0/16 \
         --advertise-address=<hostIP> --tls-san=<hostIP> --node-name=kubernetes \
         --pause-image=<pauseImage> \
         --disable=coredns,servicelb,traefik,local-storage,metrics-server
       ```

       The values of the angle-bracket placeholders in the example can be obtained from the output of `nerdctl inspect kubernetes`. The mapping is as follows.

       | Placeholder | How to obtain |
       | ---- | ---- |
       | `<kubernetesPort>` | The host port corresponding to container port `36443/tcp` in `HostConfig.PortBindings`, consistent with the value of `--https-listen-port` in the top-level `Args` |
       | `<consolePort>` | The host port corresponding to container port `30010/tcp` in `HostConfig.PortBindings` |
       | `<imageRepo>` / `<imageRepoIP>` | The domain name and IP of the entry in `HostConfig.ExtraHosts` |
       | `<k3sImage>` | The top-level `Image` field |
       | `<hostIP>` | The value of `--advertise-address` or `--tls-san` in the top-level `Args` (i.e., the bootstrap node IP specified by `bke init --hostIP`) |
       | `<pauseImage>` | The value of `--pause-image` in the top-level `Args` |

       In addition, the anonymous volumes in the `Mounts` section of the inspect output (`/var/lib/cni`, `/var/lib/kubelet`, and `/var/log`) are automatically created by the VOLUME declarations of the image and are regenerated when the container is recreated; there is no need to specify them manually in the startup parameters.

    4. Verify that the bootstrap cluster has recovered.

       ```bash
       # Confirm the node is ready
       kubectl get nodes
       # Confirm the log mounts inside the container point to the new paths
       nerdctl exec kubernetes sh -c "mount | grep /var/log"
       # Confirm that bootstrap cluster logs are written to the new path
       ls /var/log/bke/bootstrap/pods
       ```

    5. Then co-deploy the service cluster. After deployment, confirm that the log directories of service cluster Pods under `/var/log/pods` persist (without disappearing for more than 2 minutes) and that `kubectl logs` outputs normally.

    Scenario 2: The service cluster has already been co-deployed (post-incident fix)

    1. Save existing logs. During the co-deployment period, the two kubelets have been cleaning up each other's Pod log directories; logs that have already been cleaned up cannot be recovered. Back up the currently remaining content first, using the same commands as in Scenario 1, step 1.
    2. Perform steps 2 to 4 of Scenario 1 in sequence: create dedicated log directories for the bootstrap cluster, recreate the k3s container (changing the log mount paths), and verify that the bootstrap cluster recovers. During the recreation, the bootstrap cluster (including the management plane) is briefly interrupted; the service cluster's kubelet and container runtime are independent of the k3s container and are not affected, so there is no need to redeploy the service cluster.
    3. Restart the affected service cluster Pods on this node. For Pods that were running during the co-deployment period, their log files have been deleted, and log files are regenerated only after the containers restart. A rolling restart of the affected workloads is recommended to avoid service interruption.

       ```bash
       # Perform a rolling restart with a Deployment as an example; the same applies to other workload types
       kubectl rollout restart deployment/<name> -n <namespace>
       ```

    4. Verify that service cluster logs have recovered.

       ```bash
       # Log directories persist (without disappearing for more than 2 minutes)
       ls /var/log/pods
       # kubectl logs outputs normally
       kubectl logs <pod-name> -n <namespace>
       ```

  - Extended Discussion

    - This operation must be performed after the bootstrap node deployment (including the management plane) is fully completed. Operations that recreate the k3s container, such as management plane installation, upgrade, or re-initialization, will bring back the original log mounts, and this operation must be performed again after such operations until a release containing the fix is published.
    - After the operation, bootstrap cluster Pod logs are located at `/var/log/bke/bootstrap/pods` on the host, and service cluster Pod logs remain at `/var/log/pods`. The log directories of the two clusters are physically isolated and do not clean up each other, and both can be viewed and audited directly on the host.
