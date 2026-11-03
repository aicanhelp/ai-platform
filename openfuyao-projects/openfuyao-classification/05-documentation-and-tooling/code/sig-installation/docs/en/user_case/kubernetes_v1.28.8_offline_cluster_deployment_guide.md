# v1.28.8 Version K8s Cluster Offline Installation Guide

## Background

The K8s version used by the openFuyao community continues to evolve. For example, the released v25.12 version uses K8s version v1.34.3, but there are still clusters using lower K8s versions in the community. This document aims to provide a solution for installing clusters with lower K8s versions.

## Prerequisites

None.

## Restrictions

- Offline installation.
- K8s images and binaries used are community self-built.

## Usage Scenarios

- Bootstrap node is initialized, install v1.28.8 version K8s business cluster.
- Bootstrap node is not initialized, install v1.28.8 version K8s business cluster.

## Procedure

For the two usage scenarios above, they are introduced separately below.

### Bootstrap Node Initialized, Install v1.28.8 Version K8s Business Cluster

Follow the steps below.

1. Build incremental offline image package.

   Below is the configuration file, named component.yaml for later operation instructions.

   ```yaml
   registry:
     imageAddress: hub.oepkgs.net/openfuyao/registry:2.8.1
     architecture:
       - amd64
       - arm64
   repos:
     - architecture:
         - amd64
         - arm64
       needDownload: true
       subImages:
         - sourceRepo: cr.openfuyao.cn/openfuyao
           targetRepo: kubernetes
           images:
             - name: kube-apiserver
               tag:
                 - 1.28.8-of.1
             - name: kube-controller-manager
               tag:
                 - 1.28.8-of.1
             - name: kube-scheduler
               tag:
                 - 1.28.8-of.1
             - name: kube-proxy
               tag:
                 - 1.28.8-of.1
             - name: etcd
               tag:
                 - 3.5.12-of.1
   files:
     - address: https://openfuyao.obs.cn-north-4.myhuaweicloud.com/kubernetes/kubernetes/releases/download/1.28.8-of.1/bin/linux/arm64/
       files:
         - fileName: kubectl
           fileAlias: kubectl-v1.28.8-of.1-arm64
         - fileName: kubelet
           fileAlias: kubelet-v1.28.8-of.1-arm64
     - address: https://openfuyao.obs.cn-north-4.myhuaweicloud.com/kubernetes/kubernetes/releases/download/1.28.8-of.1/bin/linux/amd64/
       files:
         - fileName: kubectl
           fileAlias: kubectl-v1.28.8-of.1-amd64
         - fileName: kubelet
           fileAlias: kubelet-v1.28.8-of.1-amd64
   ```

   Execute the following command to build the incremental offline package.

   ```bash
   # Download the download.sh script
   curl -LO https://openfuyao.obs.cn-north-4.myhuaweicloud.com/openFuyao/bkeadm/releases/download/latest/download.sh
   # Download the checksum file for download.sh and verify; if verification succeeds, it outputs "-: OK"; if verification fails, contact the openFuyao sig-installation email installation@list.openfuyao.cn or openFuyao community email contact@openfuyao.cn
   curl -LO https://openfuyao.obs.cn-north-4.myhuaweicloud.com/openFuyao/bkeadm/releases/download/latest/download.sh.sha256
   sha256sum -c <(cat download.sh.sha256) < download.sh
   # Run download.sh to download the BKE installation tool; the sha256sum of the installation tool is verified during execution
   chmod +x download.sh && ./download.sh
   
   # Build incremental package
   bke build patch -f component.yaml -t component.tar.gz --strategy=oci
   ```

   

2. Synchronize images and binaries to bootstrap node.

   Copy the offline package prepared in step 1 to the bootstrap node and execute the following command to synchronize.

   ```bash
   # Extract offline package
   tar -xzvf component.tar.gz
   
   # Synchronize images and copy binary files
   bke registry patch --source ./component --target 127.0.0.1:40443
   cp ./component/volumes/kubectl-v1.28.8-of.1-arm64 /bke/mount/source_registry/files
   cp ./component/volumes/kubelet-v1.28.8-of.1-arm64 /bke/mount/source_registry/files
   cp ./component/volumes/kubectl-v1.28.8-of.1-amd64 /bke/mount/source_registry/files
   cp ./component/volumes/kubelet-v1.28.8-of.1-amd64 /bke/mount/source_registry/files
   ```

   

3. Upload new version file.

   Execute the following command to download the index file index.yaml.

   ```bash
   curl -O https://openfuyao.obs.cn-north-4.myhuaweicloud.com/openFuyao/version-config/index.yaml
   ```

   The content of index.yaml file is as follows.

   ```yaml
   - openFuyaoVersion: latest
     filePath: ./VersionConfig-latest.yaml
   - openFuyaoVersion: v25.12-rc.2
     filePath: ./Core-VersionConfig-v25.12-rc.2.yaml
   - openFuyaoVersion: v25.12-rc.3
     filePath: ./Core-VersionConfig-v25.12-rc.3.yaml
   - openFuyaoVersion: v25.12
     filePath: ./Core-VersionConfig-v25.12.yaml
   - openFuyaoVersion: v25.12-rc.5
     filePath: ./Core-VersionConfig-v25.12-rc.5.yaml
   - openFuyaoVersion: v26.03-rc.1
     filePath: ./Core-VersionConfig-v26.03-rc.1.yaml
   ```

   Download the corresponding version file. To install v26.03-rc.1 version of openFuyao cluster, execute the following command.

   ```bash
   curl -O https://openfuyao.obs.cn-north-4.myhuaweicloud.com/openFuyao/version-config/Core-VersionConfig-v26.03-rc.1.yaml
   ```

   Modify K8s version related fields and image tags in the version file. Specific modification points are as follows.

   ```yaml
   kubernetesVersion: v1.34.3-of.1 # Modify to v1.28.8-of.1
   etcdVersion: v3.6.7-of.1 # Modify to v3.5.12-of.1
   
   repos:
     - architecture: # k8s components, need updated by Orchestration-engine SIG
         - amd64
         - arm64
       needDownload: true
       isKubernetes: true
       subImages:
         - sourceRepo: cr.openfuyao.cn/openfuyao/
           targetRepo: kubernetes
           images:
             - name: kube-apiserver
               usedPodInfo:
                 - podPrefix: kube-apiserver
                   namespace: kube-system
               tag:
                 - 1.34.3-of.1 # APIServer image tag, modify to 1.28.8-of.1
             - name: kube-controller-manager
               usedPodInfo:
                 - podPrefix: kube-controller-manager
                   namespace: kube-system
               tag:
                 - 1.34.3-of.1 # controller-manager image tag, modify to 1.28.8-of.1
             - name: kube-scheduler
               usedPodInfo:
                 - podPrefix: kube-scheduler
                   namespace: kube-system
               tag:
                 - 1.34.3-of.1 # scheduler image tag, modify to 1.28.8-of.1
             - name: kube-proxy
               usedPodInfo:
                 - podPrefix: kube-proxy
                   namespace: kube-system
               tag:
                 - 1.34.3-of.1 # kube-proxy image tag, modify to 1.28.8-of.1
             - name: etcd
               usedPodInfo:
                 - podPrefix: etcd
                   namespace: kube-system
               tag:
                 - 3.6.7-of.1 # etcd image tag, modify to 3.5.12-of.1
   addons:
     - name: kubeproxy
       version: v1.34.3-of.1 # kube-proxy version, modify to v1.28.8-of.1
   ```

   Log in to the openFuyao management plane on the bootstrap node, select "Cluster Lifecycle Management > Patch Upload" in the left navigation bar, select the modified version file from step 3 and upload.
   > ![](./figures/icon-notice.gif) **Notice:**<br />
   > All images for the business cluster to be installed need to exist in the local image source service.

4. Business cluster installation.

    In the openFuyao management plane, select the corresponding version to install the business cluster, refer to [Installing Business Cluster on Management Plane](../../en/cluster_installation_guide/service_cluster_installation/service_cluster_installation_on_the_management_plane.md).


### Bootstrap Node Not Initialized, Install v1.28.8 Version K8s Business Cluster

Follow the steps below.

1. Prepare version file.
   Execute the following command to download the index file.

   ```bash
   curl -O https://openfuyao.obs.cn-north-4.myhuaweicloud.com/openFuyao/version-config/index.yaml
   ```

   The content of the index file is as follows.

   ```yaml
   - openFuyaoVersion: latest
     filePath: ./VersionConfig-latest.yaml
   - openFuyaoVersion: v25.12-rc.2
     filePath: ./Core-VersionConfig-v25.12-rc.2.yaml
   - openFuyaoVersion: v25.12-rc.3
     filePath: ./Core-VersionConfig-v25.12-rc.3.yaml
   - openFuyaoVersion: v25.12
     filePath: ./Core-VersionConfig-v25.12.yaml
   - openFuyaoVersion: v25.12-rc.5
     filePath: ./Core-VersionConfig-v25.12-rc.5.yaml
   - openFuyaoVersion: v26.03-rc.1
     filePath: ./Core-VersionConfig-v26.03-rc.1.yaml
   ```

   Download the corresponding version file. To install v26.03-rc.1 version of openFuyao cluster, execute the following command.

   ```bash
   curl -O https://openfuyao.obs.cn-north-4.myhuaweicloud.com/openFuyao/version-config/Core-VersionConfig-v26.03-rc.1.yaml
   ```

   Modify K8s version related fields and image tags in the version file. Specific modification points are as follows.

   ```yaml
   kubernetesVersion: v1.34.3-of.1 # Modify to v1.28.8-of.1
   etcdVersion: v3.6.7-of.1 # Modify to v3.5.12-of.1
   
   repos:
     - architecture: # k8s components, need updated by Orchestration-engine SIG
         - amd64
         - arm64
       needDownload: true
       isKubernetes: true
       subImages:
         - sourceRepo: cr.openfuyao.cn/openfuyao/
           targetRepo: kubernetes
           images:
             - name: kube-apiserver
               usedPodInfo:
                 - podPrefix: kube-apiserver
                   namespace: kube-system
               tag:
                 - 1.34.3-of.1 # APIServer image tag, modify to 1.28.8-of.1
             - name: kube-controller-manager
               usedPodInfo:
                 - podPrefix: kube-controller-manager
                   namespace: kube-system
               tag:
                 - 1.34.3-of.1 # controller-manager image tag, modify to 1.28.8-of.1
             - name: kube-scheduler
               usedPodInfo:
                 - podPrefix: kube-scheduler
                   namespace: kube-system
               tag:
                 - 1.34.3-of.1 # scheduler image tag, modify to 1.28.8-of.1
             - name: kube-proxy
               usedPodInfo:
                 - podPrefix: kube-proxy
                   namespace: kube-system
               tag:
                 - 1.34.3-of.1 # kube-proxy image tag, modify to 1.28.8-of.1
             - name: etcd
               usedPodInfo:
                 - podPrefix: etcd
                   namespace: kube-system
               tag:
                 - 3.6.7-of.1 # etcd image tag, modify to 3.5.12-of.1
   files:
     # Download link modify as follows
     # address: https://openfuyao.obs.cn-north-4.myhuaweicloud.com/kubernetes/kubernetes/releases/download/1.28.8-of.1/bin/linux/arm64/
     - address: https://openfuyao.obs.cn-north-4.myhuaweicloud.com/kubernetes/kubernetes/releases/download/1.34.3-of.1/bin/linux/arm64/
       files:
         - fileName: kubectl
           fileAlias: kubectl-v1.34.3-of.1-arm64 # kubectl alias, modify to kubectl-v1.28.8-of.1-arm64
         - fileName: kubelet
           fileAlias: kubelet-v1.34.3-of.1-arm64 # kubelet alias, modify to kubelet-v1.28.8-of.1-arm64
     # Download link modify as follows
     # address: https://openfuyao.obs.cn-north-4.myhuaweicloud.com/kubernetes/kubernetes/releases/download/1.28.8-of.1/bin/linux/amd64/
     - address: https://openfuyao.obs.cn-north-4.myhuaweicloud.com/kubernetes/kubernetes/releases/download/1.34.3-of.1/bin/linux/amd64/
       files:
         - fileName: kubectl
           fileAlias: kubectl-v1.34.3-of.1-amd64 # kubectl alias, modify to kubectl-v1.28.8-of.1-amd64
         - fileName: kubelet
           fileAlias: kubelet-v1.34.3-of.1-amd64 # kubelet alias, modify to kubelet-v1.28.8-of.1-amd64
   addons:
     - name: kubeproxy
       version: v1.34.3-of.1 # kube-proxy version, modify to v1.28.8-of.1
   ```

      

2. Build offline installation package.

   ```bash
   # Download the download.sh script
   curl -LO https://openfuyao.obs.cn-north-4.myhuaweicloud.com/openFuyao/bkeadm/releases/download/latest/download.sh
   # Download the checksum file for download.sh and verify; if verification succeeds, it outputs "-: OK"; if verification fails, contact the openFuyao sig-installation email installation@list.openfuyao.cn or openFuyao community email contact@openfuyao.cn
   curl -LO https://openfuyao.obs.cn-north-4.myhuaweicloud.com/openFuyao/bkeadm/releases/download/latest/download.sh.sha256
   sha256sum -c <(cat download.sh.sha256) < download.sh
   # Run download.sh to download the BKE installation tool; the sha256sum of the installation tool is verified during execution
   chmod +x download.sh && ./download.sh
   
   # Build offline installation package
   bke build -f Core-VersionConfig-v26.03-rc.1.yamls -t bke-v26.03-rc.1.tar.gz
   ```

   Offline installation package preparation reference [Offline Installation Bootstrap Cluster](../../en/cluster_installation_guide/bootstrap_cluster_installation/offline_bootstrap_cluster_installation.md).

3. Bootstrap node initialization.

    Bootstrap node initialization, reference [Offline Installation Bootstrap Cluster](../../en/cluster_installation_guide/bootstrap_cluster_installation/offline_bootstrap_cluster_installation.md).

4. Upload version file.

   Log in to the openFuyao management plane on the bootstrap node, select "Cluster Lifecycle Management > Patch Upload" in the left navigation bar, select the modified version file from step 3 and upload.

5. Business cluster installation.

   In the openFuyao management plane, select the corresponding version to install the business cluster, refer to [Installing Business Cluster on Management Plane](../../en/cluster_installation_guide/service_cluster_installation/service_cluster_installation_on_the_management_plane.md).
