# Path Mounting for Cluster Installation

This document describes how to use path mounting for cluster installation and deployment.

## Background Information

Since cluster installation data is stored on the system disk by default, if the system disk size is limited, cluster deployment will also be hindered accordingly.

## Prerequisites

Clusters installed using the openFuyao tool.

## Usage Limitations

The machine has a system disk and a data disk. Below, it is assumed that the system disk is mounted on the `/` directory and the data disk is mounted on the `/home` directory.

## Usage Scenarios

- In the online scenario, K3s bootstrap cluster installation.
- In the offline scenario, K3s bootstrap cluster installation.
- Management and business K8s cluster installation.

## K3s Bootstrap Cluster Installation

openFuyao uses a K3s cluster as the bootstrap cluster, which is a simplified K8s cluster that provides temporary cluster management functionality. The mountable directories provided by the bootstrap node initialization are as follows.

- The global working directory for initialization. This directory provides `OS`-related `RPM` file storage, binary files required for cluster installation (such as containerd, kubelet, kubectl, etc.), and offline image packages.
- The cluster runtime data storage directory `/var/lib`. This directory supports independent disk mounting.

### Online Installation of K3s Bootstrap Cluster

You can specify the global working directory of `bke init` through the file `/opt/BKE_WORKSPACE` or the environment variable `BKE_WORKSPACE`. The environment variable has higher priority. If neither is specified, the default `/bke` directory is used.

This section describes bootstrap node initialization with a custom initialization directory using the file method. Follow the steps below.

1. Set the file content.

   ```bash
   echo "/home" > /opt/BKE_WORKSPACE
   ```

    

2. Mount the `/var/lib` runtime directory.

   The following describes how to create a new partition from `/dev/vdb` and mount it to the `/var/lib` directory.

   ```bash
   # Create partition
   fdisk /dev/vdb
   # Enter the interactive interface
   # Enter n to create a new partition
   # Select partition number (default is fine, e.g., 2)
   # Starting sector (press Enter for default, use the start of remaining space)
   # Ending sector: +20G
   # Enter w to save and exit
   
   # Let the system recognize the new partition and format it
   partprobe /dev/vdb && mkfs.ext4 /dev/vdb2
   
   # Mount directory
   mv /var/lib /var/lib.old
   mkdir /var/lib
   mount /dev/vdb2 /var/lib
   cp -a /var/lib.old/* /var/lib/
   
   # Set automatic mount on boot
   echo "UUID=$(blkid -s UUID -o value /dev/vdb2) /var/lib ext4 defaults 0 0" >> /etc/fstab
   ```

    

3. Download the `bkeadm` binary installation tool.

   ```bash
   # Download the download.sh script
   curl -LO https://openfuyao.obs.cn-north-4.myhuaweicloud.com/openFuyao/bkeadm/releases/download/latest/download.sh
   # Download the checksum file for download.sh and verify; if verification succeeds, it outputs "-: OK"; if verification fails, contact the openFuyao sig-installation email installation@list.openfuyao.cn or openFuyao community email contact@openfuyao.cn
   curl -LO https://openfuyao.obs.cn-north-4.myhuaweicloud.com/openFuyao/bkeadm/releases/download/latest/download.sh.sha256
   sha256sum -c <(cat download.sh.sha256) < download.sh
   # Run download.sh to download the BKE installation tool; the sha256sum of the installation tool is verified during execution
   chmod +x download.sh && ./download.sh
   ```

    

4. Bootstrap node initialization.

   ```bash
   bke init --onlineImage cr.openfuyao.cn/openfuyao/bke-online-installed:latest
   ```

    

### Offline Installation of K3s Bootstrap Cluster

You can specify the global working directory of `bke init` through the file `/opt/BKE_WORKSPACE` or the environment variable `BKE_WORKSPACE`. The environment variable has higher priority. If neither is specified, the default `/bke` directory is used.

For offline installation package construction, refer to [Offline Bootstrap Cluster Installation](../../en/cluster_installation_guide/bootstrap_cluster_installation/offline_bootstrap_cluster_installation.md).

This section describes bootstrap node initialization with a custom initialization directory using the environment variable method. Follow the steps below.

1. Set the environment variable.

   ```bash
   # Set environment variable, effective only in the current window. /home is an example; replace with the actual environment value
   export BKE_WORKSPACE="/home"
   
   # Set environment variable, effective permanently in all windows
   echo 'export BKE_WORKSPACE="/home"' >> ~/.bashrc
   source ~/.bashrc
   ```

    

2. `/var/lib` directory mounting. The operation is similar to step 2 in [Online Installation](#online-installation-of-k3s-bootstrap-cluster).

3. Extract the offline installation package.

   ```bash
   # bke.tar.gz is the offline archive name; replace with the actual directory on the actual environment
   tar -xzvf bke.tar.gz -C $BKE_WORKSPACE
   ```

    

4. Move the `bkeadm` binary installation tool to the system environment directory.

   ```bash
   ARCH=$(uname -m)
   case $ARCH in
   x86_64) ARCH="amd64";;
   aarch64) ARCH="arm64";;
   esac
   mv $BKE_WORKSPACE/usr/local/bin/bkeadm_linux_$ARCH /usr/local/bin/bke
   ```

    

5. Initialize the bootstrap node.

   ```bash
   bke init
   ```

    

## Management and Business K8s Cluster Installation

Mount the `/var/lib` directory on each node, referring to [K3s Bootstrap Cluster Online Installation](#online-installation-of-k3s-bootstrap-cluster), and then use the bootstrap cluster to deploy the management or business cluster.
