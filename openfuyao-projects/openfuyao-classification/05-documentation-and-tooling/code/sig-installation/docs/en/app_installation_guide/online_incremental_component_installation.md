# Online Component Incremental Installation

## Background

In an online environment where the business cluster was installed using the openFuyao community Cluster-API method, new extension components need to be installed in the existing cluster incrementally. There is no need to re-prepare offline artifacts or reinstall the cluster. Simply log in to the openFuyao management plane and pull and deploy openFuyao official extension components (such as NUMA affinity plugin, VictoriaMetrics, etc.) online from the platform's built-in official component source through the application marketplace, with zero additional configuration and ready to deploy out of the box.

The comparison between online incremental installation and [Offline Component Incremental Installation](./offline_incremental_component_installation.md) is as follows:

| Comparison Item | Online Incremental Installation (This Article) | Offline Component Incremental Installation |
| --- | --- | --- |
| Applicable Environment | Cluster nodes can access the public network/official component source. | Cluster nodes cannot access the public network. |
| Artifact Acquisition Method | Pulled online directly from the official component source by the platform during deployment. | Built using bkeadm as an image incremental installation package and imported. |
| Additional Build Environment | Not required. | Build node requires tar, pigz, bkeadm (using the `oci` strategy to prepare incremental packages; Docker not required). |
| Chart Package Source | Synced online from the openFuyao official repository. | Chart package downloaded from the repository and uploaded to the local repository. |
| Image Source | Pulled online from the official image repository during deployment. | `bke registry patch` synchronizes incremental package images to the offline repository. |
| Operation Entry Point | openFuyao management plane application marketplace. | Offline bootstrap node command line + openFuyao management plane application marketplace. |

## Prerequisites

- The business cluster has been [installed online](../cluster_installation_guide/service_cluster_installation/service_cluster_installation_on_the_management_plane.md) using the Cluster-API method.
- Business cluster nodes can access the public network or official component source (official image repository cr.openfuyao.cn, official Chart component source) for online pulling of extension component dependency images and Chart packages. Verification method see [Pre-deployment Verification](#pre-deployment-verification) item 3.
- marketplace-service is running normally in the business cluster.
- Have a login account for the openFuyao management plane.

> ![icon](figures/icon-note.gif) **Note:**<br />
> With a complete openFuyao installation, the application marketplace initially configures two default repositories: local is the user's local repository, and openFuyao is the official repository. Online incremental installation directly uses the extension components published in the openFuyao official repository, without needing to add third-party repositories or upload Chart packages.

## Pre-deployment Verification

Before deploying extension components, it is recommended to verify the environment following these steps.

1. Verify the overall status of the business cluster.

   Log in to any node in the business cluster and execute the following command to confirm that all Pods are in `Running` or `Completed` state.

   ```bash
   kubectl get pod -A
   ```

2. Verify the application marketplace component service status.

   Execute the following command to confirm that marketplace-service related Pods are in `Running` state.

   ```bash
   kubectl get pod -A | grep marketplace
   ```

3. Verify network connectivity from nodes to the official component source.

   Execute the following command on the business cluster node to confirm that the node can access the official image repository. If the command fails, please check the node network, DNS resolution, or proxy configuration first.

   ```bash
   curl -I https://cr.openfuyao.cn
   ```

4. Verify the openFuyao official repository status.

   Log in to the openFuyao management plane, select "Application Market > Repository Configuration" from the left navigation bar, and confirm that the openFuyao official repository exists and its status is normal. Click "Sync" to get the latest official application and extension component list; if synchronization fails, please check whether the repository address is available and the authentication information is correct.

## Deployment Steps

The following uses deploying the logging extension component logging-package as an example.

1. Enter "https://_management-plane-login-IP-address_:_openFuyao-web-service-port_" in the browser, enter the username and password to log in to the openFuyao management plane.

2. Sync the openFuyao official repository.

   Select "Application Market > Repository Config" from the left navigation bar, click the icon in the operation column of the openFuyao official repository row, and select "Sync"; or click the openFuyao official repository name to enter the repository "Details" page and click "Sync". After synchronization is complete, the application list will show the latest official applications and extension components.

   > ![icon](figures/icon-note.gif) **Note:**<br />
   > If the official repository synchronization was already performed during pre-deployment verification, this step can be skipped.

3. Select "Application Market > Applications" from the left navigation bar to enter the "Applications" page. Search for "logging-package" in the search box in the upper right corner to find the logging extension component.

4. Click the application card to enter the application "Details" page. You can select the version to deploy in the "Version" selector, and view the component's detailed description and YAML format parameter information in the "Details" and "Default Parameters" tab pages.

5. On the application "Details" page, click "Deploy" to enter the "Deploy" page.

6. Fill in the application name in the installation information, select the application version information and namespace.

   > ![icon](figures/icon-note.gif) **Note:**<br />
   > - Extension components can only be installed once within the platform. After installation, the "Deploy" button for the extension component in the application details will change to "Manage"; clicking "Manage" will navigate to the corresponding "Extension Component Management" page.
   > - Some application names starting with numbers may cause errors during deployment; see logs for details.

7. Fill in the Helm Chart values information in Values.yaml as needed. If using the platform's preset default parameter group, no modification is needed; proceed directly to the next step.

   > ![icon](figures/icon-note.gif) **Note:**<br />
   > The logging extension component logging-package consists of a logging backend layer, a logging UI layer, and an open-source software layer (log collection component promtail, log storage component Loki, etc.). For component descriptions and Values.yaml parameter configurations, please refer to openFuyao/docs [Logs](https://gitcode.com/openFuyao/docs/blob/master/docs/en/management_plane_operations/observability_center/logs.md).

8. Click "Deploy" to start deploying the extension component.

## Result Verification

1. View deployment results on the management plane.

   After deployment is complete, select "Extension Component Management" from the left navigation bar (for regular applications, select "Application Management") to enter the corresponding page to view the deployed component's name, status, and update time. When the component status is normal, it indicates successful deployment. Click the component name to enter the "Details" page to further view details, YAML, resources, logs, events, monitoring, and other information.

2. View component Pod status from the command line.

   Log in to any node in the business cluster and execute the following command to confirm that all Pods related to the newly deployed extension component are in `Running` or `Completed` state.

   ```bash
   kubectl get pod -A
   ```

3. (Optional) Further verify by component capabilities.

   - Logging component: After successful deployment, select "Logs > Log Query" under "Observability Center" in the left navigation bar of the management plane to view logs, and manage collection sources, collection tasks, and alert rules in "Logs > Log Configuration". For details, see openFuyao/docs [Logs](https://gitcode.com/openFuyao/docs/blob/master/docs/en/management_plane_operations/observability_center/logs.md).

## FAQ

1. Target extension component not found in the application list.

   - The openFuyao official repository has not been synced or synchronization failed. Please go to "Application Market > Repository Configuration", perform "Sync Repository" on the openFuyao official repository, and retry.
   - The repository address is unavailable or the authentication information is incorrect, causing synchronization failure. Please check the repository configuration and re-sync.

2. Pod remains in Pending state after deployment.

   - The number of Pods that can run on each node in the Kubernetes cluster is limited, defaulting to 110. When all nodes reach the limit, new Pods to be created will not be schedulable and will remain in Pending state. Please clean up or scale out nodes in a timely manner to avoid unlimited increase in Pending Pods.
   - Insufficient cluster resources or node network exceptions. Please execute `kubectl describe pod <pod-name> -n <namespace>` to view events and confirm whether there are issues such as insufficient resources or image pull failures.

3. Image pull failure.

   Online deployment requires pulling component dependency images online from the official image repository. Please confirm that the business cluster nodes can access cr.openfuyao.cn, check the node network, DNS resolution, or proxy configuration, and redeploy.

4. Deployment failure with parameter configuration error.

   Incorrect Helm Chart parameter configuration in Values.yaml during deployment may cause deployment failure or the component not running properly. Please enter the application "Details" page to view "Default Parameters", correct the values information according to the component usage guide, and redeploy.

5. The "Deploy" button for the extension component shows "Manage".

   Extension components can only be installed once within the platform. After installation, the "Deploy" button for the extension component in the application details will change to "Manage"; clicking "Manage" will navigate to the corresponding "Extension Component Management" page, where you can view or manage the installed extension component.

6. Some extension components conflict with platform built-in components.

   Some extension components may have port conflicts or functional overlaps with platform built-in components (e.g., monitoring, logging components), and simultaneous installation is not recommended. Before deployment, please review the corresponding extension component's usage documentation, confirm compatibility and installation requirements, and handle conflicts with platform built-in components according to the documentation.
