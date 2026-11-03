# Offline Installation of Bootstrap Cluster

## Configuration Requirements

#### Minimal Deployment Hardware Configuration

| Node Type | Node Count | CPU | Memory | Disk |
| -------- | -------- | --- | ---- | ------------ |
| Bootstrap Node | 1 | 2 | 4 | System disk >= 100G |
| Management Cluster | 1 | 8 | 16 | System disk >= 100G |
| Service Cluster | 1 | 8 | 16 | System disk >= 100G |

Note: The bootstrap node can be one of the nodes in the management cluster.

#### Recommended Hardware Configuration

| Node Type | Node Count | CPU | Memory | Disk |
| -------- | ------------ | ------------ | ------------ | ---------------------------- |
| Bootstrap Node | 1 | 2 | 4 | System disk >= 100G |
| Management Cluster | 3 | 8 | 16 | System disk >= 100G<br>Data disk >= 300G |
| Service Cluster | Depending on actual conditions | Depending on actual conditions | Depending on actual conditions | Depending on actual conditions |

## Environment Preparation

- It is recommended to prepare two machines to serve as the bootstrap node and the service cluster node respectively; they can also be co-deployed on the same machine (in this case, K3s and Kubernetes will run simultaneously, which may affect each other and is not recommended). The bootstrap node must have the tar tool installed.

- Prepare a build node with public network access (used to build the offline installation package in the online scenario).
  
  > ![icon](../figures/icon-note.gif) **Note:**<br />
  > For the build node configuration, refer to the bootstrap node related configuration information in the above [Configuration Requirements](#configuration-requirements) section.

- Ensure that this machine does not have a `/bke` directory.
- Install tar, pigz, bkeadm, and Docker.
  
  - Install tar. You can directly execute the following command according to your operating system:

    ```bash
    # RPM-based (openEuler/CentOS/RHEL, etc.)
    yum install -y tar

    # Debian-based (Ubuntu/Debian, etc.)
    apt-get update && apt-get install -y tar
    ```

  - Install pigz. You can directly execute the following command according to your operating system; for the source code, refer to [madler/pigz](https://github.com/madler/pigz).

    ```bash
    # RPM-based (openEuler/CentOS/RHEL, etc.)
    yum install -y pigz

    # Debian-based (Ubuntu/Debian, etc.)
    apt-get update && apt-get install -y pigz
    ```

  - Install Docker. You can directly execute the following command according to your operating system:

    ```bash
    # RPM-based (openEuler/CentOS/RHEL, etc.)
    yum install -y docker
    systemctl enable --now docker

    # Debian-based (Ubuntu/Debian, etc.)
    apt-get update && apt-get install -y docker.io
    systemctl enable --now docker
    ```

  - To download bkeadm, execute the following command for installation. For custom modification of BKE tool related parameter configuration, refer to [Configuration Parameters](../appendix/bkeadm_configuration.md).
    
    - Install the latest version of the BKE installation tool: Log in to the bootstrap node as the root user and execute the following command to download, verify, and install the latest version of the BKE installation tool on the bootstrap node.

      ```bash
      # Download the download.sh script
      curl -LO https://openfuyao.obs.cn-north-4.myhuaweicloud.com/openFuyao/bkeadm/releases/download/latest/download.sh
      # Download the checksum file for download.sh and verify it; on success it outputs -: OK; on failure, please contact the openFuyao sig-installation email installation@list.openfuyao.cn or the openFuyao community email contact@openfuyao.cn
      curl -LO https://openfuyao.obs.cn-north-4.myhuaweicloud.com/openFuyao/bkeadm/releases/download/latest/download.sh.sha256
      sha256sum -c <(cat download.sh.sha256) < download.sh
      # Run download.sh to download the BKE installation tool; the sha256sum of the installation tool will be verified during execution
      chmod +x download.sh && ./download.sh
      ```

    - Install a specified version of the BKE installation tool: If you want to download another version, log in to the bootstrap node as the root user and use the following command to specify the version for download.
  
      ```bash
      export VERSION=x.y.z # Replace x.y.z with the actual version you want to download.

      # Download the download.sh script
      curl -LO https://openfuyao.obs.cn-north-4.myhuaweicloud.com/openFuyao/bkeadm/releases/download/latest/download.sh
      # Download the checksum file for download.sh and verify it; on success it outputs -: OK; on failure, please contact the openFuyao sig-installation email installation@list.openfuyao.cn or the openFuyao community email contact@openfuyao.cn
      curl -LO https://openfuyao.obs.cn-north-4.myhuaweicloud.com/openFuyao/bkeadm/releases/download/latest/download.sh.sha256
      sha256sum -c <(cat download.sh.sha256) < download.sh
      # Run download.sh to download the BKE installation tool; the sha256sum of the installation tool will be verified during execution
      chmod +x download.sh && ./download.sh
      ```

      > ![icon](../figures/icon-note.gif) **Note:**<br/>
      >
      > - For openFuyao v26.06, v26.09 and later versions: x.y.z values (BKE installation tool version numbers) are 26.6.0, 26.9.0 and other subsequent numeric version numbers.
      > - For openFuyao v26.03 and earlier versions: the correspondence between the x.y.z value (BKE installation tool version number) and the openFuyao version is as follows:
      >
      >    |openFuyao Version|BKE Installation Tool Version|
      >    |--|--|
      >    |v26.03|1.2.2|
      >    |v25.12|1.0.5|
      >    |v25.09|v25.09|
      >    |v25.06|v25.06|
      >    |v25.03|v25.03|

## Installation Steps

1. Build the deployment package.

   1.1 After completing the Docker installation, add the following configuration to the Docker configuration file.  

    Edit the /etc/docker/daemon.json file (create one if it does not exist).

   ```json
    {
    "insecure-registries": [     
      "0.0.0.0/0"
    ]    
    }
   ```

   1.2 Execute the following command to restart Docker.

   ```bash
    systemctl restart docker
   ```

   1.3 Build the offline installation deployment package.

    It collects binary files, RPM packages, Chart packages, etc., and finally generates a compressed archive. Core-VersionConfig-latest.yaml can be downloaded by clicking [Download](https://openfuyao.obs.cn-north-4.myhuaweicloud.com/openFuyao/version-config/Core-VersionConfig-latest.yaml).
    Determine the extension components to be installed, and turn on the download switch for the corresponding components in the yaml file (off by default). The following shows enabling the volcano component image download switch; subsequently, when preparing the offline installation package, the images depended on by volcano will be downloaded.

      ```bash
        - architecture:  # openfuyao volcano extension component
            - amd64
            - arm64
          needDownload: true # Switch for whether to download the images required by the volcano extension component; false - off, do not download; true - on, download
          subImages:
            - sourceRepo: cr.openfuyao.cn/openfuyao
              targetRepo: openfuyao
              images:
                - name: volcano-config-website
                  tag:
                    - latest
                - name: volcano-config-service
                  tag:
                    - latest
                - name: numa-tuning-operator
                  tag:
                    - v25.03
                - name: volcano/resource-exporter
                  tag:
                    - latest
                - name: oauth-proxy
                  tag:
                    - latest
                - name: vc-scheduler
                  tag:
                    - latest
            - sourceRepo: docker.io
              targetRepo: /
              images:
                - name: volcanosh/vc-webhook-manager
                  tag:
                    - v1.9.0
                - name: volcanosh/vc-controller-manager
                  tag:
                    - v1.9.0
      ```

    Execute the following command to build the offline installation deployment package.
    
      > ![icon](../figures/icon-note.gif) **Note:**
      > The duration is approximately 1 hour.
      > Errors that occur during the import process have a retry mechanism; errors that do not cause the process to terminate can be ignored.

      ```bash
      rm -rf /bke && bke build -f Core-VersionConfig-latest.yaml -t bke.tar.gz
      ```

2. Perform offline installation on the bootstrap node.

   > ![icon](../figures/icon-caution.gif) **Notice:**
   > If the bootstrap node has container runtimes such as docker or containerd, or related system residual files, the openFuyao configuration for the container runtime will not take effect, resulting in installation failure.
   > <span style="color:red;font-weight:bold;">To ensure successful installation, it is recommended to use a brand-new clean node for installation.</span>

    2.1 Copy the deployment package from the online environment to the bootstrap node in the offline environment (operate according to the actual situation).

    2.2 Execute the following command to extract the deployment package to the root directory.

      ```bash
        rm -rf /bke && tar zxvf <deployment package name eg: bke.tar.gz> -C /
      ```
      > ![icon](../figures/icon-note.gif) **Note:**<br/>
      > - It is recommended that the available storage space on the root directory (`/`) after extraction be no less than 50 GB (installation media extraction, image and binary cache, k3s/containerd data, etc. all occupy the root partition). If the available space is insufficient, initialization may fail or be interrupted midway.
      > - Example of available space check: `df -h /`. For detailed description of the disk item in environment checking, see [Environment Check Configuration Analysis](../appendix/env-check-config-analysis.md).

3. Modify the BKE installation tool name and initialize the bootstrap node.
    
    ```bash
    # Modify the installation tool name
    ARCH=$(uname -m)
    case $ARCH in
    x86_64) ARCH="amd64";;
    aarch64) ARCH="arm64";;
    esac
    mv /usr/local/bin/bkeadm_linux_$ARCH /usr/local/bin/bke
    # Initialize the bootstrap node
    bke init 
    ```

    > ![icon](../figures/icon-note.gif) **Note:**<br />After executing `bke init`, the tool will first display the parameters that take effect for this installation (such as `--hostIP`, `--domain`, etc.), and show the confirmation prompt `Confirm the parameters, press Y to continue N will exit. [Y/N]?`: enter `Y` to continue the installation, or enter `N` to exit the installation. You can also append the `--confirm` parameter to the command to skip this confirmation step.

4. Execute the following command to view the Pods deployed on the bootstrap cluster.

    ```bash
    kubectl get pod -A
    ```

    When `bke init` is executed successfully, the log line `BKE initialization is complete` will be output at the end. After waiting for all Pod statuses to become `Running` or `Completed`, the bootstrap cluster installation is complete. You can log in to the openFuyao management plane of the bootstrap node to continue with the [Service Cluster Installation](../service_cluster_installation/service_cluster_installation_on_the_management_plane.md).

    > ![icon](../figures/icon-note.gif) **Note:**<br/>
    > - The openFuyao management plane login URL for the bootstrap node is "https://_Bootstrap Node IP Address_:30010".
    > - On first login, the default username is admin and the default password is test@1234. You need to change the username and password after the first login.
    > - The openFuyao management plane of the bootstrap node uses the https protocol, and the default web service port is 30010.

## Uninstallation

For bootstrap cluster uninstallation (deletion scope, high-risk impact, optional media cleanup, and success criteria), see [Delete Bootstrap Cluster via Backend (Command Line)](../../cluster_uninstallation/delete_bootstrap_cluster_command.md). For the deletion sequence and impact description, see [Cluster Uninstallation Overview](../../cluster_uninstallation/overview.md).
