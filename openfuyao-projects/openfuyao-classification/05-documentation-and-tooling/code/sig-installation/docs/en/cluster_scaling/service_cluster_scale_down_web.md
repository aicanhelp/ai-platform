# Scaling Down a Service Cluster via the Web UI

Cluster scale-down refers to removing existing worker nodes from a service cluster to release computing resources. Through the cluster lifecycle management page of the openFuyao management plane, users can easily perform service cluster scale-down operations on the Web UI.

## Prerequisites

- The service cluster to be scaled down is in the "Healthy" state. You can view the cluster status on the cluster lifecycle management list page. For details, see [Installing a Service Cluster on the Management Plane](../cluster_installation_guide/service_cluster_installation/service_cluster_installation_on_the_management_plane.md).
- The scale-down operation must be performed on the bootstrap node or management cluster used when creating the cluster.
- The nodes to be deleted are worker nodes, and no critical services are running on the nodes, or Pod migration preparations have been made.

> ![icon](figures/icon-notice.gif) **Notice:**
> The cluster scale-down operation must be performed on the bootstrap node or management cluster used when creating the cluster. Otherwise, the target cluster cannot be managed.

## Usage Constraints

- During scale-down, only worker nodes can be deleted. Master nodes cannot be deleted through the Web UI. Manually deleting the `BKENode` resource of a Master role will be rejected by Webhook validation. Deletion is only allowed when initiated by the controller or when the node is in the Failed state.
- Once a node is deleted, it cannot be recovered. Perform this operation with caution.
- When the cluster is in an unstable state such as scaling, installing, or upgrading, a new scale-down operation cannot be initiated.
- Pods on the node to be deleted will be forcibly evicted or deleted. Ensure that your services can tolerate the node going offline.

## Procedure

1. Log in to the openFuyao management plane on the bootstrap node or management cluster used to create the cluster.

   Enter "https://*login IP address of the bootstrap node or management cluster*:*Web service port of cluster lifecycle management*" in the browser, and enter the username and password to log in to the cluster lifecycle management page.

   > ![icon](figures/icon-note.gif) **Note:**
   > - When logging in through the bootstrap node, the default Web service port is 30010. When logging in through the management cluster, the default Web service port is 31616.

2. Select "Cluster Lifecycle Management" from the left navigation pane of the openFuyao platform. The page displays the cluster lifecycle management list, including "Cluster Name", "Status", "Number of Nodes", and other information.

3. On the "Cluster Lifecycle Management" list page, click the "Cluster Name" to be scaled down to enter the "Node Details" page.

4. Select a deletion method.

   **Method 1: Batch Deletion**

   1. Select the check box before the "Node Name" to select the worker nodes to be deleted.

      > ![icon](figures/icon-note.gif) **Note:**
      > The check boxes of control plane nodes (Master nodes) are unavailable. Only worker nodes can be selected for deletion.

   2. Click "Delete" above the list.

   **Method 2: Single Deletion**

   1. In the row of the worker node to be deleted, click the delete icon in the "Operation" column.

5. In the displayed "Delete Node" confirmation window, view the information of the node to be deleted.

   The window displays the following messages:
   - "Once a node is deleted, it cannot be recovered. Perform this operation with caution."
   - "Are you sure you want to delete node *node name*?"

6. Select the "I understand that this deletion is irreversible" check box.

7. Click "Delete" to have the system dispatch the scale-down task.

   - The "Cluster Information" log window pops up, displaying the scale-down operation logs in real time.
   - Return to the cluster lifecycle management list page. The "Status" column of the corresponding cluster displays "Scaling Down".

8. Wait until the "Status" column is updated. The cluster scale-down is complete.

   - Scale-down successful: The "Status" column of the corresponding cluster displays "Healthy".
   - Scale-down failed: The "Status" column of the corresponding cluster displays "Scale Failed".

> ![icon](figures/icon-note.gif) **Note:**
> The scale-down operation includes node Drain, Pod eviction, kubeadm reset, and other steps, which takes a long time. Please wait patiently. Before the scale-down is complete, the cluster is not yet stable. Do not perform other operations on the cluster.

## Status Description

During the scale-down process, the cluster status changes as follows:

| Status | Description |
|:----- |:----------- |
| Scaling Down | Worker nodes are being removed from the cluster, including Pod eviction, node Drain, kubeadm reset, and other operations. |
| Scale Failed | An error occurred during the scale-down process, and the node failed to be removed. Troubleshoot and retry. |
| Healthy | The scale-down is complete, the node has been successfully removed, and the cluster is in the healthy state. |

> ![icon](figures/icon-notice.gif) **Notice:**
> If the scale-down fails, the cluster status displays "Scale Failed". Refer to [Cluster Deployment Issue Diagnosis Guide](../cluster_installation_guide/appendix/cluster_deployment_issue_diagnosis_guide.md) to troubleshoot the cause.

## Follow-up Operations

After completing the cluster scale-down, if you need to scale up the cluster, see [Scaling Up a Service Cluster via the Web UI](./service_cluster_scale_up_web.md).
