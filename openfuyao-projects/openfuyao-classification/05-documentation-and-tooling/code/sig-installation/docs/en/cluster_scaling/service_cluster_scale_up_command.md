# Scaling Up a Service Cluster via Backend (Command Line)

Cluster scale-up refers to adding new worker nodes to an existing service cluster to increase its compute resources and scheduling capacity. By operating Kubernetes Custom Resources (CRs) via the command line, users can perform service cluster scale-up operations from the backend.

## Prerequisites

- The service cluster to be scaled up is in a "Healthy" state.
- The scale-up operation must be performed on the bootstrap node or management cluster used when creating the cluster.
- New nodes to be added to the cluster have been prepared and meet the following requirements:
  - The node can connect to the management cluster network.
  - The node can be accessed via SSH using the root user.
  - The node is a bare-metal OS with no docker or Kubernetes components installed.
  - The node's IP address is not used by any existing cluster.
  - The time difference between the node and the management cluster is no more than 10 seconds.

> ![icon](figures/icon-notice.gif) **Notice:**
> - Performing a scale-up operation when the cluster is in an unhealthy state may cause errors. Perform the operation only when the cluster status is "Healthy".
> - The cluster scale-up operation must be performed on the bootstrap node or management cluster used when creating the cluster; otherwise, the target cluster cannot be managed.

## Usage Restrictions

- Only worker nodes can be added during scale-up. Through Webhook validation, adding Master nodes to an existing cluster is not allowed (creating a `BKENode` resource with a role containing master will be rejected).
- The IP address of the node to be added must be unique across all clusters; otherwise, `BKENode` creation will be intercepted by Webhook validation.
- The role of a node cannot be changed after creation. Changing a worker node to a Master node is not supported.
- Only nodes with IPv4 addresses are currently supported.

## Operation Steps

### 1. View Cluster Information

Confirm the name and namespace of the cluster to be scaled up, and view the current cluster node information.

```bash
# View all BKECluster resources to confirm the cluster name and namespace (namespace is usually the same as the cluster name)
kubectl get bkecluster -A

# View the BKENode resources of the current cluster
# Replace <bke-cluster> with the actual cluster namespace
kubectl get bn -n bke-cluster
```

### 2. Write the Node Configuration File

Create the `BKENode` resource configuration file for the new node (e.g., `newNode.yaml`).

```yaml
apiVersion: bke.bocloud.com/v1beta1
kind: BKENode
metadata:
  name: bke-cluster-n1          # Node name, must be unique within the cluster
  namespace: bke-cluster         # Cluster namespace, consistent with the BKECluster namespace
  labels:
    cluster.x-k8s.io/cluster-name: bke-cluster   # Associated cluster name, consistent with the BKECluster name
spec:
  hostname: n1                   # Node hostname
  ip: 192.168.200.101            # Replace with the actual node IP
  password: '<password>'          # Replace with the actual node password; Webhook will automatically AES-encrypt it
  port: "22"                     # SSH port, default is 22
  role:
  - node                         # role must be node (worker node) when scaling up
  username: root                 # SSH username, default is root
```

**Table 1** BKENode Configuration Parameter Description

| Parameter | Description |
|:---------- |:------- |
| metadata.name | BKENode resource name, must be unique within the cluster namespace. The "cluster name-node sequence number" format is recommended. |
| metadata.namespace | Namespace, must be consistent with the namespace of the BKECluster to be scaled up. |
| metadata.labels | Must contain the `cluster.x-k8s.io/cluster-name` label, with the value being the name of the target BKECluster, used to associate the node with the cluster. |
| spec.hostname | The hostname of the node, which will be used as the Kubernetes node name. |
| spec.ip | The IP address of the node, must be a valid IPv4 address and unique across all clusters. |
| spec.port | SSH login port number, default is "22". |
| spec.username | SSH login username, default is root. |
| spec.password | SSH login password. Fill in the password during creation; Webhook will automatically perform AES encryption for storage. |
| spec.role | Node role. Must be set to `node` (worker node) during scale-up. Adding master role nodes to an existing cluster is not supported. |

> ![icon](figures/icon-note.gif) **Note:**
> - If you need to scale up multiple nodes at once, you can use `---` in the same YAML file to separate multiple `BKENode` resource definitions, or create multiple configuration files separately.
> - The `cluster.x-k8s.io/cluster-name` value in `metadata.labels` must match the `metadata.name` of `BKECluster`, and `metadata.namespace` must match the `metadata.namespace` of `BKECluster`; otherwise, the node cannot be correctly associated with the target cluster.

### 3. Create the BKENode Resource

Execute the following command to create the `BKENode` resource and trigger the scale-up process.

```bash
kubectl apply -f newNode.yaml
```

If scaling up multiple nodes at once, write multiple node configurations into the same file:

```bash
kubectl apply -f newNodes.yaml
```

### 4. View Scale-up Progress

After creating the `BKENode` resource, the cluster-api-provider-bke controller will automatically start the scale-up process. You can view the scale-up progress with the following commands.

```bash
# View the BKENode resource status; the State field shows the node status
kubectl get bn -n bke-cluster

# View the BKECluster status; the ClusterStatus field shows the cluster status
kubectl get bkecluster -n bke-cluster

# View node details, focusing on the State and Message fields
kubectl describe bn bke-cluster-n1 -n bke-cluster
```

During the scale-up process, the node status (State) changes as follows:

| Node Status | Description |
|:----- |:----------- |
| Initializing | bkeagent is being pushed to the node; environment initialization in progress. |
| BootStrapping | The node is executing kubeadm join to join the cluster. |
| NotReady | The node has joined the cluster but has not yet passed health checks. |
| Ready | The node is ready; scale-up succeeded. |

The cluster status (ClusterStatus) changes as follows:

| Cluster Status | Frontend Display | Description |
|:----- |:----- |:----------- |
| Initializing | Installing | Pushing the bkeagent proxy to the new node, node environment initialization (installing containerd and other components), and post-processing script phase. |
| ScalingWorkerNodesUp | Scaling Up | Worker nodes are executing kubeadm join to join the service cluster. |
| Ready | Healthy | Scale-up completed; the cluster is in a healthy state. |
| WorkerScalingUpFailed | Scale Failed | An error occurred during scale-up; the node failed to join the cluster. |

> ![icon](figures/icon-note.gif) **Note:**
> - The scale-up process is asynchronous. After creating the `BKENode` resource, the command returns immediately, and the scale-up operation runs asynchronously in the background.
> - During scale-up, the cluster status will switch between `Initializing` (Installing) and `ScalingWorkerNodesUp` (Scaling Up), which is normal.
> - During scale-up, you can view events (Events) via `kubectl describe bkecluster <cluster-name> -n <namespace>` to learn detailed progress.
> - Scale-up time depends on the node network conditions and environment configuration; typically, each node takes 3 to 10 minutes.

### 5. Verify Scale-up Results

After scale-up is complete, verify in the target service cluster whether the new node has successfully joined.

```bash
# View the BKENode resources in the management cluster (bootstrap node) to confirm the node is in the list
kubectl get bn -n bke-cluster

# View the service cluster node list
kubectl get nodes
```

The status of the newly added node should be `Ready`, indicating that the scale-up succeeded.

> ![icon](figures/icon-notice.gif) **Notice:**
> If the node status remains at `Initializing` or `BootStrapping` for an extended period, or the cluster status shows `WorkerScalingUpFailed`, refer to the [Cluster Deployment Issue Diagnosis Guide](../cluster_installation_guide/appendix/cluster_deployment_issue_diagnosis_guide.md) to troubleshoot. Common issues include node network unreachable, incorrect SSH password, and time desynchronization.

## Follow-up Operations

After completing the cluster scale-up, if you need to scale down the cluster, see [Scaling Down a Service Cluster via Backend (Command Line)](./service_cluster_scale_down_command.md).
