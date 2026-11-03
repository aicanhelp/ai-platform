# Online Bootstrap Cluster Installation

## Configuration Requirements

#### Minimum Deployment Hardware Configuration

| Node Type | Node Count | CPU | Memory | Disk         |
| -------- | -------- | --- | ---- | ------------ |
| Bootstrap Node | 1        | 2   | 4    | System disk >=100G |
| Management Cluster | 1        | 8   | 16   | System disk >=100G |
| Service Cluster | 1        | 8   | 16   | System disk >=100G |

Note: The bootstrap node can be one of the management cluster nodes.

#### Recommended Hardware Configuration

| Node Type | Node Count     | CPU          | Memory         | Disk                         |
| -------- | ------------ | ------------ | ------------ | ---------------------------- |
| Bootstrap Node | 1            | 2            | 4            | System disk >=100G                 |
| Management Cluster | 3            | 8            | 16           | System disk >=100G<br>Data disk >=300G |
| Service Cluster | Based on actual conditions | Based on actual conditions | Based on actual conditions | Based on actual conditions                 |

## Environment Preparation

It is recommended to prepare two machines to serve as the bootstrap node and the service cluster node respectively. They can also be deployed on the same machine (in this case, K3s and Kubernetes will run simultaneously, which may affect each other and is not recommended). The bootstrap node needs the tar tool installed. You can run the following commands directly according to your operating system:

```bash
# RPM-based (openEuler/CentOS/RHEL, etc.)
yum install -y tar

# Debian-based (Ubuntu/Debian, etc.)
apt-get update && apt-get install -y tar
```

> ![icon](../figures/icon-notice.gif) **Notice:**  
> It is recommended that your node environment does not have any docker or Kubernetes components installed, otherwise version conflicts may occur and cause installation failure. If you need to perform a trial installation in an existing environment, you can download the [environment pre-check tool](../environment_preparation/environment_pre_check_tool_guide.md) to perform environment verification.
> If the service cluster needs to install the openFuyao management plane, the service cluster nodes also need the tar tool installed.

## Installation Steps

1. Download, verify, and install the BKE tool on the bootstrap node. To customize BKE tool parameters, refer to [Configuration Parameters](../appendix/bkeadm_configuration.md).
   
   1.1 Based on the download.sh script

   - Install the latest version of the BKE installation tool: Log in to the bootstrap node as the root user and run the following command to download, verify, and install the latest version of the BKE installation tool on the bootstrap node.

   ```bash
   # Download the download.sh script
   curl -LO https://openfuyao.obs.cn-north-4.myhuaweicloud.com/openFuyao/bkeadm/releases/download/latest/download.sh
   # Download the verification file for download.sh and verify it. If verification succeeds, "-: OK" will be output. If verification fails, contact the openFuyao sig-installation email installation@list.openfuyao.cn or the openFuyao community email contact@openfuyao.cn
   curl -LO https://openfuyao.obs.cn-north-4.myhuaweicloud.com/openFuyao/bkeadm/releases/download/latest/download.sh.sha256
   sha256sum -c <(cat download.sh.sha256) < download.sh
   # Run download.sh to download the BKE installation tool. The sha256sum of the installation tool will be verified during execution
   chmod +x download.sh && ./download.sh
   ```

   - Install a specified version of the BKE installation tool: If you want to download another version, log in to the bootstrap node as the root user and use the following command to specify the version to download.
   
   ```bash
   export VERSION=x.y.z # Replace x.y.z with the actual version you want to download.

   # Download the download.sh script
   curl -LO https://openfuyao.obs.cn-north-4.myhuaweicloud.com/openFuyao/bkeadm/releases/download/latest/download.sh
   # Download the verification file for download.sh and verify it. If verification succeeds, "-: OK" will be output. If verification fails, contact the openFuyao sig-installation email installation@list.openfuyao.cn or the openFuyao community email contact@openfuyao.cn
   curl -LO https://openfuyao.obs.cn-north-4.myhuaweicloud.com/openFuyao/bkeadm/releases/download/latest/download.sh.sha256
   sha256sum -c <(cat download.sh.sha256) < download.sh
   # Run download.sh to download the BKE installation tool. The sha256sum of the installation tool will be verified during execution
   chmod +x download.sh && ./download.sh
   ```

   > ![icon](../figures/icon-note.gif) **Note:**<br/>
   >
   > - For openFuyao v26.06, v26.09 and later versions: x.y.z values (BKE installation tool version numbers) are 26.6.0, 26.9.0 and other subsequent numeric version numbers.
   > - For openFuyao v26.03 and earlier versions: the x.y.z values (BKE installation tool version numbers) correspond to openFuyao versions as follows:
   >
   >    |openFuyao Version|BKE Installation Tool Version|
   >    |--|--|
   >    |v26.03|1.2.2|
   >    |v25.12|1.0.5|
   >    |v25.09|v25.09|
   >    |v25.06|v25.06|
   >    |v25.03|v25.03|

   1.2 Based on binary packages

   If you do not want to download the target version of the BKE installation tool based on the download.sh script, you can also directly download the binary package, extract it, and move the binary file to a system path (such as `/usr/local/bin`). When downloading manually, please follow [Installation Tool Download and Verification](../../cluster_installation_guide/environment_preparation/installation_package_integrity_verification.md) to complete the integrity verification before deployment:

   | Architecture | Download |
   | :--- | :--- |
   | x86_64 | [bkeadm_linux_amd64.tar.gz](https://openfuyao.obs.cn-north-4.myhuaweicloud.com/openFuyao/bkeadm/releases/download/latest/bkeadm_linux_amd64.tar.gz) |
   | ARM | [bkeadm_linux_arm64.tar.gz](https://openfuyao.obs.cn-north-4.myhuaweicloud.com/openFuyao/bkeadm/releases/download/latest/bkeadm_linux_arm64.tar.gz) |

2. Run the following command to initialize the bootstrap node.

   This step deploys a lightweight K3s cluster on the bootstrap node. The cluster will deploy cluster-api, provider-bke, and Pods related to openFuyao installation and deployment.

   > ![icon](../figures/icon-note.gif) **Note:**<br />After running `bke init`, the tool will first display the parameters in effect for this installation (such as `--hostIP`, `--domain`, etc.) and show a confirmation prompt `Confirm the parameters, press Y to continue N will exit. [Y/N]?`: Enter `Y` to continue the installation, or enter `N` to exit the installation. You can also append the `--confirm` parameter to the command to skip this confirmation step.

   **Method 1: Default initialization (deploy the openFuyao management plane)**

   ```bash
   bke init --onlineImage cr.openfuyao.cn/openfuyao/bke-online-installed:latest
   ```

   **Method 2: Minimal initialization (without deploying the openFuyao management plane)**

   To only initialize the bootstrap node without deploying the front-end openFuyao management plane, add `--installConsole=false` to `bke init`. The image source can be either of the two options:

   - **Pull images from the online repository** (same as Method 1, no local installation package specified):

     ```bash
     bke init --onlineImage cr.openfuyao.cn/openfuyao/bke-online-installed:latest --installConsole=false
     ```

   - **Use a local installation package** (optional, for local image acceleration; must also satisfy constraints such as `--otherRepo`, see [Configuration Parameters](../appendix/bkeadm_configuration.md) for details):

     ```bash
     bke init --onlineImage cr.openfuyao.cn/openfuyao/bke-online-installed:latest --installConsole=false --imageFilePath=/tmp/local_amd/image_amd64.tar.gz
     ```

     Where: `imageFilePath` points to the local installation package. For the build method, refer to [Bootstrap Node Initialization Process Optimization](https://gitcode.com/openFuyao/sig-installation/blob/master/design/%E5%BC%95%E5%AF%BC%E8%8A%82%E7%82%B9%E5%AE%89%E8%A3%85%E4%BC%98%E5%8C%96%E5%92%8C%E5%81%A5%E5%BA%B7%E6%A3%80%E6%B5%8B%E6%9C%BA%E5%88%B6/%E5%BC%95%E5%AF%BC%E8%8A%82%E7%82%B9%E5%88%9D%E5%A7%8B%E5%8C%96%E6%B5%81%E7%A8%8B%E4%BC%98%E5%8C%96.md).

   > ![icon](../figures/icon-caution.gif)**Notice:**<br/>
   > Method 2 does not deploy the front-end openFuyao management plane. To create a service cluster later, use the backend command line path instead of the management plane Web operations.

3. Run the following command to view the Pods deployed on the bootstrap cluster.

   ```bash
   kubectl get pod -A
   ```
   
   When `bke init` is executed successfully, the last output will be the `BKE initialization is complete` log line. After all Pods are in `Running` or `Completed` status, the bootstrap cluster installation is complete. Then select the service cluster installation path according to the initialization method:

   - **Method 1 (openFuyao management plane deployed)**: Log in to the openFuyao management plane of the bootstrap node to continue with [Installing the Service Cluster via the Frontend (Web Page)](../service_cluster_installation/service_cluster_installation_on_the_management_plane.md).

     > ![icon](../figures/icon-note.gif) **Note:**<br/>
     > - The login URL for the openFuyao management plane of the bootstrap node is "https://_bootstrap node IP address_:30010".
     > - For the first login, the default username is admin and the default password is test@1234. You need to change the username and password after the first login.
     > - The openFuyao management plane of the bootstrap node uses the HTTPS protocol, and the default web service port is 30010.

   - **Method 2 (openFuyao management plane not deployed, `--installConsole=false`)**: You cannot use the management plane Web to create a service cluster. Please use the backend command line to continue the installation. See [Installing the Service Cluster via the Backend (Command Line)](../service_cluster_installation/service_cluster_deployment_guide_(linux_backend).md).

## Uninstallation

For bootstrap cluster uninstallation (deletion scope, high-risk impact, optional media cleanup, and success criteria), see [Deleting the Bootstrap Cluster via the Backend (Command Line)](../../cluster_uninstallation/delete_bootstrap_cluster_command.md). For deletion sequence and impact description, see [Cluster Uninstallation Overview](../../cluster_uninstallation/overview.md).
