# nginx-proxy Load Balancer User Guide

## Background Information

Worker nodes in a BKE cluster access kube-apiserver through a load balancer. Two load balancing modes are available:

- **ha mode (default)**: keepalived maintains a floating VIP across master nodes, and haproxy listens on the VIP and forwards requests to kube-apiserver on each master. This mode requires an independent VIP and Layer-2 connectivity among master nodes.

- **nginx mode**: an nginx instance runs as a static Pod on each worker node, listening on the loopback address `127.0.0.1:6443` and distributing requests via TCP stream to kube-apiserver on all master nodes. Components on worker nodes such as kubelet and kube-proxy access kube-apiserver through `127.0.0.1:6443`.

Benefits of nginx mode:

1. **Lower Deployment Barrier**: no VIP is required, which suits environments with tight IP resources, cloud platforms without VIP floating support, or restricted Layer-2 networks.
2. **Higher Data-Plane Reliability**: load balancing runs locally on each worker node. nginx uses passive failure detection (open-source nginx has no active health check): when a master fails, new connections are automatically routed to the remaining masters, leaving the local entry point on the worker unaffected.

## Prerequisites

- The openFuyao management plane has been installed, and its version supports the `loadBalanceType` field (supported since the release that introduces this feature).
- The worker cluster contains at least 1 master node, and worker nodes can reach the kube-apiserver port (default 6443) on all master node IPs.
- Nodes can pull the nginx image from the image repository (provided by the local image repository in offline environments).

## Usage Limitations

- **Allowed Values**: `loadBalanceType` supports only `ha` and `nginx`, and defaults to `ha`.
- **Immutable After Creation**: the `loadBalanceType` field cannot be modified after the cluster is created (enforced by webhook). Switching between the two modes at runtime is not supported.
- **Mutually Exclusive With External Load Balancing**: nginx mode cannot be used together with `customExtra["extraLoadBalanceIP"]` (external load balancing). The constraint is enforced at cluster creation; do not add `extraLoadBalanceIP` by updating the cluster afterwards.
- **ControlPlaneEndpoint Constraint**: in nginx mode, if `spec.controlPlaneEndpoint.host` is explicitly specified, it must be a master node IP (a VIP or `127.0.0.1` is rejected); if unspecified, it defaults to the first master node IP. `port` must be the kube-apiserver port (default 6443): if only `host` is specified, the port defaults to 36443 (the ha-mode load balancer port), which is unavailable in nginx mode.
- **Single Management-Plane Entry Point**: in nginx mode, the management plane accesses the worker cluster through `controlPlaneEndpoint.host` (the first master node IP by default). The management plane cannot manage the worker cluster while that node is down (worker nodes on the data plane are unaffected). Users who require management-plane high availability should choose ha mode.

## Procedure

Follow the steps below to configure nginx-proxy load balancing.

1. Edit the `BKECluster CR` and configure the nginx load balancing parameters in the `spec.clusterConfig.cluster` field.

    **Table 1** nginx-proxy Load Balancing Parameters and Default Values
    |Parameter|Description|Default Value|
    |---|---|---|
    |loadBalanceType|Load balancing mode for worker nodes to access kube-apiserver. Valid values are ha and nginx. Immutable after creation.|ha|
    |nginxProxyVersion|nginx image tag used by nginx-proxy. Effective only when loadBalanceType is nginx. Usually no configuration is needed; it is synchronized automatically during cluster upgrade.|1.28.0|

    **Example**: a minimal nginx-mode cluster configuration (only fields related to this feature are listed).

    ```yaml
    apiVersion: bke.bocloud.com/v1beta1
    kind: BKECluster
    metadata:
      name: bkecluster-sample
    spec:
      # Defaults to the first master node IP and port 6443; if host is specified, port must be specified as well
      # controlPlaneEndpoint:
      #   host: 192.168.1.10
      #   port: 6443
      clusterConfig:
        cluster:
          loadBalanceType: nginx
    ```

    > ![](./figures/icon-notice.gif)**Notice:**<br> `loadBalanceType` cannot be modified after creation. Confirm the load balancing mode based on the deployment environment before creating the cluster: choose `ha` if a VIP or management-plane high availability is required; choose `nginx` if no VIP is available and a single management-plane entry point is acceptable.

2. Apply the `BKECluster CR` to create the cluster (or include the preceding configuration when creating the cluster through the management plane).

    During cluster creation, the reconciler detects that `loadBalanceType` is `nginx`, skips the installation of keepalived/haproxy, and renders the following on each worker node:

    - `/etc/kubernetes/nginx-proxy/nginx.conf`: the nginx configuration file. Its upstream lists the kube-apiserver addresses of all master nodes; nginx listens on `127.0.0.1:6443`.
    - the static Pod manifest `nginx-proxy.yaml`: starts an nginx instance on each worker node (Pod name `kube-system/nginx-proxy`).

3. Verify the deployment.

    3.1 Verify the cluster status. After the cluster is created, `status.clusterHealthState` of the `BKECluster` should be `Healthy` and `status.clusterStatus` should be `Ready`.

    3.2 Verify the nginx-proxy static Pods. Execute the following command to verify that one `nginx-proxy` Pod in `Running` state exists on each worker node:

    ```bash
    kubectl get pod -n kube-system -o wide | grep nginx-proxy
    ```

    3.3 Verify the local entry point. Log in to any worker node and execute the following command. The expected output is `ok`:

    ```bash
    curl -k https://127.0.0.1:6443/healthz
    ```

    3.4 Verify service continuity during a master failure (optional). After kube-apiserver on any master node is stopped, components on worker nodes can still access kube-apiserver on the remaining masters through the local nginx, and cluster services are unaffected.
