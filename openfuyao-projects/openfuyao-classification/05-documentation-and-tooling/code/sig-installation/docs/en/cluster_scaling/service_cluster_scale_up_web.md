# Scaling Up a Service Cluster via the Frontend (Web Page)

Cluster scale-up refers to adding new worker nodes to an existing service cluster to increase the cluster's computing resources and scheduling capacity. Through the cluster lifecycle management page of the openFuyao management plane, users can easily perform service cluster scale-up operations on the web interface.

## Prerequisites

- The service cluster to be scaled up is in the "Healthy" state. You can view the cluster status on the cluster lifecycle management list page. For details, see [Installing a Service Cluster on the Management Plane](../cluster_installation_guide/service_cluster_installation/service_cluster_installation_on_the_management_plane.md).
- The scale-up operation must be performed on the bootstrap node or management cluster used when creating the cluster.

> ![icon](figures/icon-notice.gif) **Notice:**
> The cluster scale-up operation must be performed on the bootstrap node or management cluster used when creating the cluster. Otherwise, the target cluster cannot be managed.

- The new nodes to be added to the cluster have been prepared and meet the following requirements:
  - The node can connect to the management cluster network.
  - The node can be accessed via SSH using the root user.
  - The node is a bare-metal OS without any docker or Kubernetes components installed.
  - The node's IP address is not used by any existing cluster.
  - The time difference between the node and the management cluster is no more than 10 seconds.

## Usage Restrictions

- During scale-up, only worker nodes (Worker Node) can be added. Adding Master nodes via the web page is not supported.
- During scale-up, ensure that the nodes used are unused to avoid scale-up failures caused by IP conflicts or environment conflicts.
- Currently, only nodes with IPv4 addresses are supported.
- When the cluster is in an unstable state such as scaling, installing, or upgrading, initiating a new scale-up operation is not allowed.

## Procedure

1. Log in to the openFuyao management plane on the bootstrap node or management cluster used to create the cluster.

   Enter "https://*login IP address of the bootstrap node or management cluster*:*web service port for cluster lifecycle management*" in the browser, and enter the username and password to log in to the cluster lifecycle management page.

   > ![icon](figures/icon-note.gif) **Note:**
   > - When logging in through the bootstrap node, the default web service port is 30010; when logging in through the management cluster, the default web service port is 31616.

2. Select "Cluster Lifecycle Management" from the left navigation pane of the openFuyao platform. The page displays the cluster lifecycle management list information, including "Cluster Name", "Status", "Nodes", and so on.

3. On the "Cluster Lifecycle Management" list page, click the "Cluster Name" to be scaled up to enter the "Node Details" page.

   > ![icon](figures/icon-note.gif) **Note:**
   > The node list information on the node details page is arranged with Master nodes first and Worker nodes after.

4. Click "+ Add Node" in the upper-right corner of the node list information to open the "Add Node" window.

5. In the "Add Node" window, fill in the information of the nodes to be scaled up.

   **Table 1** Node information description

   | Parameter | Description |
   |:---------- |:------- |
   | Node Name | The hostname of the new node. It must comply with Kubernetes naming conventions: lowercase letters, digits, and hyphens (-), 1 to 63 characters in length, and cannot be purely numeric. |
   | IP Address | The IP address of the new node. It must be a valid IPv4 address and not used by any cluster. |
   | Port | The SSH login port number. The default is 22. Value range: 0 to 65535. |
   | Username | The SSH login username. The default is root. |
   | Password | The SSH login password. |

   - Click "Add New Node" to continue adding node rows.
   - Click the delete icon in the "Operation" column to delete an added node row. At least one row must be retained.

6. Click "Validate All". The system will verify the information of all new nodes.

   The verification includes:
   - Whether the node IP address is unique across all clusters.
   - Whether the node name is unique in the target cluster.
   - Whether the node is reachable via SSH.
   - Whether the time difference between the node and the management cluster is within the allowed range (no more than 10 seconds).
   - For a high-availability cluster, whether the load balancer IP is available.

   > ![icon](figures/icon-note.gif) **Note:**
   > - After verification succeeds, the prompt "One-click verification succeeded, please create" is displayed, and the "OK" button becomes available.
   > - If verification fails, an error message is displayed, and the IP address row of the corresponding node is highlighted in red. Please correct the node information according to the error message and verify again.
   > - After node information is changed, "Validate All" must be performed again.

7. After verification passes, click "OK".

   The system issues the scale-up task and a "Cluster Information" log window pops up, displaying the scale-up operation logs in real time.

8. After the logs finish scrolling, click "Exit Log". Return to the cluster lifecycle management list page. The "Status" column of the corresponding cluster displays "Installing".

   > ![icon](figures/icon-note.gif) **Note:**
   > During scale-up, the cluster status switches between "Installing" and "Scaling Up". "Installing" corresponds to the bkeagent push, node environment preparation, and post-processing stages; "Scaling Up" corresponds to the worker node joining the cluster stage.

9. Wait until the "Status" column is updated to complete the cluster scale-up.

   - Scale-up successful: The "Status" column of the corresponding cluster is "Healthy".
   - Scale-up failed: The "Status" column of the corresponding cluster is "Scale Failed".

> ![icon](figures/icon-note.gif) **Note:**
> Do not perform other operations on the cluster while scale-up is incomplete and the cluster is not yet stable.

## Status Description

During the scale-up process, the cluster status changes as follows:

| Status | Description |
|:----- |:----------- |
| Installing | The stage of pushing the bkeagent agent to new nodes, initializing the node environment (installing containerd and other components), and running post scripts. |
| Scaling Up | The worker node is executing kubeadm join to join the service cluster. |
| Scale Failed | An error occurred during scale-up, and the node failed to join the cluster. Troubleshooting and retry are required. |
| Healthy | Scale-up is complete, all new nodes are ready, and the cluster is in a healthy state. |

> ![icon](figures/icon-notice.gif) **Notice:**
> After scale-up fails, the cluster status is displayed as "Scale Failed". Please refer to [Cluster Deployment Issue Diagnosis Guide](../cluster_installation_guide/appendix/cluster_deployment_issue_diagnosis_guide.md) to troubleshoot the cause.

## Follow-up Operations

After completing cluster scale-up, if you need to scale down the cluster, see [Scaling Down a Service Cluster via the Frontend (Web Page)](./service_cluster_scale_down_web.md).
