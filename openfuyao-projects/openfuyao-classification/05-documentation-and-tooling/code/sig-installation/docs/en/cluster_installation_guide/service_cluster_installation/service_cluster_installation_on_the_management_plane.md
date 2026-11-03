# Installing a Service Cluster via the Frontend (Web Page)

## Prerequisites

Before installing a service cluster via the Web interface, ensure that:

- Bootstrap cluster initialization has been completed: for online installation, see [Online Bootstrap Cluster Installation](../bootstrap_cluster_installation/online_bootstrap_cluster_installation.md); for offline installation, see [Offline Bootstrap Cluster Installation](../bootstrap_cluster_installation/offline_bootstrap_cluster_installation.md).
- **The openFuyao management plane has been deployed**. If `--installConsole=false` was used during bootstrap cluster initialization, this guide cannot be used; instead, please use [Installing a Service Cluster via the Backend (Command Line)](./service_cluster_deployment_guide_(linux_backend).md).
- **Medium and large-scale deployment**: You must first complete [Installing a Management Cluster](../management_cluster_installation/management_cluster_deployment_guide.md), and log in to the openFuyao management plane of the **management cluster** to create a service cluster (default address `https://<management-cluster-node-IP>:31616`).
- **Small-scale deployment**: You may skip deploying a separate management cluster and directly log in to the openFuyao management plane of the **bootstrap cluster** to create a service cluster (default address `https://<bootstrap-node-IP>:30010`).

## Installation Steps

1. Log in to the openFuyao management plane used to create the service cluster (bootstrap cluster or management cluster, see [Prerequisites](#prerequisites)).

   - Bootstrap cluster management plane: enter `https://<bootstrap-node-IP>:30010` in the browser.
   - Management cluster management plane: enter `https://<management-cluster-node-IP>:31616` in the browser.

   > ![icon](../figures/icon-note.gif) **Note:**
   > - On first login, the username is admin and the password is test@1234. You need to change the username and password after the first login.
   > - The openFuyao management plane uses the HTTPS protocol; the default port of the bootstrap cluster management plane is 30010, and the default port of the management/service cluster management plane is 31616.

2. In the left navigation pane of the openFuyao platform, select "Cluster Lifecycle Management". The page displays the cluster lifecycle management list, including information such as "Cluster Name", "Status", and "Nodes".

3. Click "Add Cluster" in the upper right corner of the list to enter the "Add Cluster" page.

4. Fill in the relevant fields on the page.

     **Table 1** Field descriptions

     | Area Name        | Description                                                                                                          |
     |:---------|:-------------------------------------------------------------------------------------------------------------------|
     | Basic Information | The name of the service cluster to create and the openFuyao version to install.                                                                  |
     | Master Information | Information about the control plane nodes of the service cluster. Click ![](../figures/node_add.png) to add more control plane nodes. Click ![](../figures/node_delete.png) in the "Operation" column to delete an added node. When the number of master nodes is more than one and is an odd number, configure the `Load Balancer IP Address` to create a high-availability cluster. This address must be an IP address on the same network segment as the master nodes and not yet assigned. |
     | Node Information   | Information about the node nodes of the service cluster.                                                                                              |
     | Advanced Configuration | Configures installation parameters such as image repository, binary/system package repository, and Chart repository. The pre-filled values on the page are the officially recommended configurations of the openFuyao platform: for online installation, they point to the official default repositories; for offline installation, they point to the local repositories configured during bootstrap cluster initialization. |

   > ![icon](../figures/icon-note.gif) **Note:**
   > - To install a high-availability cluster, you need to add an odd number of Master nodes and configure a load balancer IP address, and do not use the bootstrap node as a Master node of the high-availability cluster.
   > - For the add-ons in Advanced Configuration, you need to select `openFuyao Cores`. Only the service cluster installed after selecting this option will have an openFuyao management plane. If you do not select it, you can install it later by following [(Optional) Install openFuyao Management Plane](#optional-install-openfuyao-management-plane).

5. Click "Validate All" in the lower right corner.

6. After validation succeeds, click "Create" below to submit the service cluster creation task.

7. The "Cluster Information" window pops up. Wait until the cluster information finishes scrolling, then return to the cluster list page. The "Status" column shows "Installing". When the "Status" column is updated, the new cluster is created.
   - The "Status" column of the corresponding cluster may show the following states: Installing, Scaling Up, Checking, Unhealthy, Healthy. Among them, "Scaling Up" appears in non-single-Master cluster scenarios; "Checking" appears periodically (this is normal); "Unhealthy" means the cluster is waiting for the corresponding Pods to run normally (it takes some time to wait; if it remains unhealthy for a long time, you need to troubleshoot based on logs). The final status of the cluster is "Healthy" or "Install Failed".
   - If the "Status" column of the corresponding cluster shows "Healthy", the addition is successful.
   - If the "Status" column of the corresponding cluster shows "Install Failed", the addition failed.

   > ![icon](../figures/icon-note.gif) **Note:** <br />
   > - The "Cluster Information" window scrolls through the event logs of the cluster installation process. When the window shows the log line `Reason: ClusterReady, Message: cluster is ready` and no new logs appear, it means the cluster information scrolling is complete and the cluster is installed successfully; if the log line `Reason: ClusterDeployFailed, Message: Cluster deploy failed, process exit` appears, it means the cluster install failed. The scrolling duration depends on the cluster scale and network conditions.
   > If the "Cluster Logs" show error messages that require further troubleshooting, see the relevant content in [FAQ](../appendix/FAQ.md).

8. If `openFuyao Cores` was selected during installation, after the service cluster is successfully installed, you can log in to the openFuyao management plane of the service cluster to perform cluster management operations.

   Enter `https://<service-cluster-IP>:31616` in the browser.<br />

   > ![icon](../figures/icon-note.gif) **Note:**
   >
   > - On first login, the username is admin and the password is test@1234. You need to change the username and password after the first login.
   > - The openFuyao management plane of the service cluster uses the HTTPS protocol, and the default Web service port is 31616.

## (Optional) Install openFuyao Management Plane

If you did **not** select `openFuyao Cores` when creating the service cluster (or for other reasons the target cluster does not have an openFuyao management plane), you can manually install it after the cluster is ready, according to your scenario:

- Online scenario: See [Online openFuyao-system-controller Manual Deployment](../appendix/online_incremental_deployment_guide.md).
- Offline scenario: See [Offline openFuyao-system-controller Manual Deployment](../appendix/offline_incremental_deployment_guide.md).
