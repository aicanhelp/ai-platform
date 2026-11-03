# Scaling Down a Service Cluster via Backend (Command Line)

Cluster scale-down refers to removing existing worker nodes from a service cluster to release computing resources. By operating on Kubernetes custom resources (CRs) via the command line, users can perform service cluster scale-down operations in the backend.

## Prerequisites

- The service cluster to be scaled down is in a "Healthy" state.
- The scale-down operation must be performed on the bootstrap node or management cluster used when creating the cluster.
- The nodes to be deleted are worker nodes, and no critical services are running on the nodes, or Pod migration has been prepared.

> ![icon](figures/icon-notice.gif) **Notice:**
> - Performing a scale-down operation while the cluster is in an unhealthy state may cause errors. Perform the operation only when the cluster is in a "Healthy" state.
> - Pods on the nodes to be deleted will be forcibly evicted or deleted. Ensure that your services can tolerate the nodes going offline.
> - The cluster scale-down operation must be performed on the bootstrap node or management cluster used when creating the cluster. Otherwise, the target cluster cannot be managed.

## Usage Restrictions

- During scale-down, only worker nodes can be deleted. Manual deletion of Master nodes is not supported. Through Webhook validation, manually deleting a `BKENode` resource with the Master role will be rejected. Deletion is only allowed when initiated by the controller or when the node is in a Failed state.
- Deleted nodes cannot be restored. Proceed with caution.
- Nodes in the `Pending` (deploying) or `Upgrading` (upgrading) state cannot be deleted.
- Only nodes with IPv4 addresses are currently supported.

## Operation Steps

### 1. View Cluster Node Information

Confirm the name and namespace of the cluster to be scaled down, and view the current cluster node information to determine the nodes to be deleted.

```bash
# View all BKECluster resources to confirm the cluster name and namespace (the namespace is usually the same as the cluster name)
kubectl get bkecluster -A

# View the current BKENode resources of the cluster, focusing on the Name and State fields
# Replace <bke-cluster> with the actual cluster namespace
kubectl get bn -n bke-cluster
```

Output example:

```text
NAME             STATE    IP              ROLE     HOSTNAME
bke-cluster-m1   Ready    192.168.200.1   master   m1
bke-cluster-n1   Ready    192.168.200.2   node     n1
bke-cluster-n2   Ready    192.168.200.3   node     n2
```

> ![icon](figures/icon-note.gif) **Note:**
> - Nodes with `ROLE` set to `master` are Master nodes and cannot be deleted.
> - Nodes with `ROLE` set to `node` are worker nodes and can be deleted during scale-down.
> - Record the `BKENode` resource name (NAME column) of the nodes to be deleted. It will be used in subsequent steps.

### 2. Delete the BKENode Resource

Execute the following command to delete the `BKENode` resource corresponding to the node to be scaled down, triggering the scale-down process.

**Single node deletion:**

```bash
# Replace bke-cluster with the actual cluster namespace, and bke-cluster-n1 with the actual BKENode resource name
kubectl delete bn -n bke-cluster bke-cluster-n1
```

**Batch deletion:**

```bash
# Delete multiple nodes at the same time
kubectl delete bn -n bke-cluster bke-cluster-n1 bke-cluster-n2
```

> ![icon](figures/icon-note.gif) **Note:**
> - After the `BKENode` resource is deleted, the command returns immediately. The scale-down operation is executed asynchronously in the backend.
> - The controller automatically handles node Drain, Pod eviction, kubeadm reset, and other operations. No manual intervention is required.

### 3. View Scale-Down Progress

After the `BKENode` resource is deleted, the cluster-api-provider-bke controller automatically starts the scale-down process. You can view the scale-down progress with the following commands.

```bash
# View the BKECluster status. The ClusterStatus field shows the cluster status
kubectl get bkecluster -n bke-cluster

# View the remaining BKENode resources to confirm whether the nodes to be deleted have been removed
kubectl get bn -n bke-cluster

# View cluster events to learn the detailed scale-down progress
kubectl describe bkecluster bke-cluster -n bke-cluster
```

During the scale-down process, the cluster status (ClusterStatus) changes as follows:

| Cluster Status | Description |
|:----- |:----------- |
| ScalingWorkerNodesDown | Worker node scale-down in progress, including Pod eviction, node Drain, kubeadm reset, and other operations. |
| Ready | Scale-down completed. The nodes have been successfully removed, and the cluster is in a healthy state. |
| WorkerScalingDownFailed | Scale-down failed. Troubleshooting is required. |

> ![icon](figures/icon-note.gif) **Note:**
> - The scale-down process is asynchronous. After the `BKENode` resource is deleted, the command returns immediately, and the scale-down operation is executed asynchronously in the backend.
> - The scale-down duration depends on the number of Pods running on the nodes and the network conditions. Typically, each node takes 3 to 10 minutes.
> - During the Pod eviction phase, if there are Pods on the node that cannot be evicted (such as DaemonSet Pods, local storage Pods, etc.), the scale-down time may be extended. Pods that cannot be evicted will be forcibly cleaned up after the node is deleted.

### 4. Verify Scale-Down Result

After the scale-down is completed, verify that the nodes have been removed from the cluster.

```bash
# View the BKENode resources in the management cluster (bootstrap node) to confirm that the nodes to be deleted are no longer in the list
kubectl get bn -n bke-cluster

# View the service cluster node list to confirm that the nodes have been removed
kubectl get nodes
```

The nodes to be deleted no longer appear in the node list, and the cluster status returns to `Ready`, indicating that the scale-down is successful.

> ![icon](figures/icon-notice.gif) **Notice:**
> If the cluster status remains at `ScalingWorkerNodesDown` for a long time, or shows `WorkerScalingDownFailed`, refer to [Cluster Deployment Issue Diagnosis Guide](../cluster_installation_guide/appendix/cluster_deployment_issue_diagnosis_guide.md) to troubleshoot the cause. Common issues include Pods on the node that cannot be evicted, node disconnection, or nodes in an abnormal state.

## Next Steps

After completing the cluster scale-down, if you need to scale up the cluster, see [Scaling Up a Service Cluster via Backend (Command Line)](./service_cluster_scale_up_command.md).
