# Offline openFuyao-system-controller Manual Deployment

## Background Information

This document describes how, in an offline scenario, you first build and deploy a `K8s-only` offline package, create a business cluster that does not include `openFuyao-system-controller`, then supplement `openFuyao-system-controller` and its dependent images and Charts through an incremental package, and finally manually deploy the `openFuyao-system-controller` component in the business cluster to bring up the OpenFuyao management plane.

<!-- The overall process is divided into five parts:

1. K8s image package build
2. Install a bootstrap cluster containing only K8s images on the bootstrap node
3. Bring up a business cluster with only K8s from the bootstrap cluster
4. Build and upload the additional openFuyao-system-controller image package
5. Deploy the openFuyao-system-controller component on the business cluster -->

## Prerequisites

The following machines have been prepared:

| Role | Example IP | Purpose |
| --- | --- | --- |
| Build machine | `192.168.200.210` | Build the `K8s-only` offline package and the incremental package |
| Bootstrap node | `192.168.200.217` | Deploy the bootstrap node, providing the image repository and Chart repository |
| Business cluster node | `192.168.200.216` | Create a business cluster with only K8s, and incrementally install the controller |

For machine hardware configuration, refer to the configuration requirements in [Quick Start](https://docs.openfuyao.cn/en/docs/v26.03/quick_start.html#quick-installation); the build machine is mainly used to build the offline package and the incremental package, it is recommended that the disk be no less than 100GB, and no less than 50GB of available space be reserved for storing downloaded content and build artifacts.

The following files have been prepared:

| File | Purpose | Preparation Method |
| --- | --- | --- |
| `K8s-only.yaml` | Build the first-stage `K8s-only` offline package | Refer to the `K8s-only` download configuration in the [Appendix](#appendix) of this document, and adjust according to the actual version |
| `offline-incremental.yaml` | Build the second-stage incremental package | Refer to the incremental component download configuration in the [Appendix](#appendix) of this document, and adjust according to the actual version |
| `openFuyao-system-controller.yaml` | Deploy the business cluster controller component | Later copied and rendered from the bootstrap node in [Section 5.2](#52-prepare-the-openfuyao-system-controller-yaml) |
| `VersionConfig-latest.yaml` | Component version configuration used for `latest` version offline installation | Downloaded to the bootstrap node along with the offline package; for other versions, replace with the corresponding version's file |

The YAML examples in the appendix of this document apply to the `latest` version offline installation scenario. You can check the version configuration file in the offline package to confirm the current version: `VersionConfig-latest.yaml` with `openFuyaoVersion: latest` indicates the `latest` version; for other versions, refer to the image tags, Chart file names, and download addresses in the corresponding version configuration file.

## Usage Restrictions

Offline installation is required.

## Tutorial

### 1. K8s Image Package Build

#### 1.1 Images to Be Retained in K8s-only.yaml

Images actually used by the pure K8s cluster:

```text
deploy.bocloud.k8s:40443/kubernetes/bkeagent-deployer:latest
deploy.bocloud.k8s:40443/kubernetes/calico/cni:v3.31.3
deploy.bocloud.k8s:40443/kubernetes/calico/kube-controllers:v3.31.3
deploy.bocloud.k8s:40443/kubernetes/calico/node:v3.31.3
deploy.bocloud.k8s:40443/kubernetes/coredns:1.12.2-of.1
deploy.bocloud.k8s:40443/kubernetes/etcd:3.6.7-of.1
deploy.bocloud.k8s:40443/kubernetes/kube-apiserver:1.34.3-of.1
deploy.bocloud.k8s:40443/kubernetes/kube-controller-manager:1.34.3-of.1
deploy.bocloud.k8s:40443/kubernetes/kube-proxy:1.34.3-of.1
deploy.bocloud.k8s:40443/kubernetes/kube-scheduler:1.34.3-of.1
deploy.bocloud.k8s:40443/kubernetes/pause:3.9
```

Images actually used by the bootstrap node:

```text
deploy.bocloud.k8s:443/kubernetes/bke-console-service:latest
deploy.bocloud.k8s:443/kubernetes/bke-console-website:latest
deploy.bocloud.k8s:443/kubernetes/bke-manifests:latest
deploy.bocloud.k8s:443/kubernetes/busybox:1.36.1
deploy.bocloud.k8s:443/kubernetes/cluster-api/cluster-api-controller:v1.4.3
deploy.bocloud.k8s:443/kubernetes/cluster-api/kubeadm-bootstrap-controller:v1.4.3
deploy.bocloud.k8s:443/kubernetes/cluster-api/kubeadm-control-plane-controller:v1.4.3
deploy.bocloud.k8s:443/kubernetes/cluster-api-provider-bke:latest
deploy.bocloud.k8s:443/kubernetes/coredns:1.12.2-of.1
deploy.bocloud.k8s:443/kubernetes/ingress-nginx/controller:v1.9.4
deploy.bocloud.k8s:443/kubernetes/ingress-nginx/kube-webhook-certgen:v20231011-8b53cabe0
deploy.bocloud.k8s:443/kubernetes/installer-service:latest
deploy.bocloud.k8s:443/kubernetes/installer-website:latest
deploy.bocloud.k8s:443/kubernetes/oauth-proxy:latest
deploy.bocloud.k8s:443/kubernetes/oauth-server:latest
deploy.bocloud.k8s:443/kubernetes/oauth-webhook:latest
deploy.bocloud.k8s:443/kubernetes/plugin-management-service:latest
deploy.bocloud.k8s:443/kubernetes/user-management-operator:latest
```

The `K8s-only` offline package needs to retain both the pure K8s cluster images and the images and Charts required by the bootstrap node itself. This stage does not include `openFuyao-system-controller` and its subsequent management plane business components.

For the YAML file format, refer to the [Appendix](#appendix) section of this document.

#### 1.2 Build the K8s-only Offline Package

Before building, confirm that the `bke` command is installed on the build machine. If not, you can download and prepare it with the following commands:

```bash
# Download the download.sh script
curl -LO https://openfuyao.obs.cn-north-4.myhuaweicloud.com/openFuyao/bkeadm/releases/download/latest/download.sh
# Download the checksum file for download.sh and verify it; a successful verification outputs -: OK, if verification fails please contact the openFuyao sig-installation email installation@list.openfuyao.cn or the openFuyao community email contact@openfuyao.cn
curl -LO https://openfuyao.obs.cn-north-4.myhuaweicloud.com/openFuyao/bkeadm/releases/download/latest/download.sh.sha256
sha256sum -c <(cat download.sh.sha256) < download.sh
# Run download.sh to download the BKE installation tool; during execution the sha256sum of the installation tool will be verified
chmod +x download.sh && ./download.sh

ARCH=$(uname -m)
case $ARCH in
x86_64) ARCH="amd64";;
aarch64) ARCH="arm64";;
esac

mv /usr/local/bin/bkeadm_linux_$ARCH /usr/local/bin/bke
chmod +x /usr/local/bin/bke
bke version
```

> ![icon](../figures/icon-note.gif) **Note:**<br />
> This document uses the `latest` version as an example. If using a specified openFuyao version, replace `latest` in the download address with the corresponding version, and ensure that the image tags, Chart file names, and version configuration files in `K8s-only.yaml` are replaced accordingly.

Execute on the build machine:

```bash
rm -rf /bke
bke build -f K8s-only.yaml -t K8s-only.tar.gz
```

Transfer the offline package to the bootstrap node:

```bash
scp /root/K8s-only.tar.gz root@192.168.200.217:/root/
```

### 2. Install a Bootstrap Cluster with Only K8s Images on the Bootstrap Node

#### 2.1 Clean Up the Old Environment on the Bootstrap Node

Execute on the bootstrap node:

```bash
bke reset --all --mount
rm -rf /bke
rm -f /usr/local/bin/bke
```

#### 2.2 Extract the K8s-only Offline Package

```bash
tar -zxf /root/K8s-only.tar.gz -C /
ls -lh /bke/volumes/image.tar.gz /bke/volumes/source.tar.gz
```

#### 2.3 Prepare the bke Command

```bash
ARCH=$(uname -m)
case $ARCH in
x86_64) ARCH="amd64";;
aarch64) ARCH="arm64";;
esac
mv /usr/local/bin/bkeadm_linux_$ARCH /usr/local/bin/bke
chmod +x /usr/local/bin/bke
bke version
```

If the current offline package does not contain the `bkeadm_linux_$ARCH` for the corresponding architecture, you need to additionally prepare a `bke` binary file of the same architecture.

#### 2.4 Disconnect from the Network

```bash
sed -i 's/^\s*nameserver/# nameserver/' /etc/resolv.conf
```

#### 2.5 Initialize the Bootstrap Node

```bash
bke init
```

If you need to explicitly specify ports, you can use:

```bash
bke init --kubernetesPort 6443 --imageRepoPort 40443
```

#### 2.6 Check the Bootstrap Node

```bash
kubectl get pods -A
kubectl get pods -A | grep openfuyao-system-controller
```

The second command should produce no output, indicating that the bootstrap node has not deployed `openFuyao-system-controller`.

### 3. Bring Up a Business Cluster with Only K8s from the Bootstrap Cluster

#### 3.1 Create the Business Cluster

Log in to the bootstrap node page (the access address is `https://<bootstrap-node-IP>:30010`, in this example `https://192.168.200.217:30010`) to create a business cluster.

When creating the business cluster, do not add the `openFuyao-system-controller` component.

#### 3.2 Check the Business Cluster

Execute on the business cluster node:

```bash
kubectl get nodes
kubectl get pods -A
crictl images
```

Confirm that the business cluster only contains pure K8s related components, without `openFuyao-system-controller` and OpenFuyao management plane business components.

### 4. Build and Upload the Additional openFuyao-system-controller Image Package

#### 4.1 Build the Incremental Package

After preparing `offline-incremental.yaml` on the build machine, execute:
For the YAML file format, refer to the [Appendix](#appendix) section of this document.

```bash
bke build patch -f offline-incremental.yaml -t patch-log.tar.gz --strategy oci
```
> ![icon](../figures/icon-note.gif) **Note:**<br />This document builds the incremental package in OCI format (`--strategy oci`), which does not depend on Docker.

After extraction, patch-log.tar.gz should contain:

```text
patch-log/
├── manifests.yaml
└── volumes/
    ├── oci-layout/
    └── charts/
        ├── application-management-service-0.0.0-latest.tgz
        ├── console-service-0.0.0-latest.tgz
        └── ...
```

> ![icon](../figures/icon-notice.gif) **Notice:**<br />`bke build patch` only collects binaries, Charts, etc. into the incremental package; it does not import them into the existing system package/binary repository (HTTP repository) or Chart Registry on the bootstrap node. Importing images requires separately executing `bke registry patch`, and binaries and Charts also need to be manually imported according to their respective processes.

#### 4.2 Transfer the Incremental Package to the Bootstrap Node

```bash
scp /root/patch-log.tar.gz root@192.168.200.217:/root/
```

#### 4.3 Import Incremental Images on the Bootstrap Node

Execute on the bootstrap node:

```bash
cd /root
rm -rf patch-log
tar -xzvf patch-log.tar.gz

bke registry patch --source ./patch-log --target 0.0.0.0:40443
```

> ![icon](../figures/icon-notice.gif) **Notice:**<br />`bke registry patch` only imports incremental images into the `40443` image repository; it does not import `volumes/charts/*.tgz`, nor does it import binaries/system packages into the HTTP repository.

Check the images:

```bash
curl -k https://127.0.0.1:40443/v2/kubernetes/openfuyao-system-controller/tags/list
curl -k https://127.0.0.1:40443/v2/kubernetes/application-management-service/tags/list
```

#### 4.4 Sync Incremental Charts on the Bootstrap Node

View the Charts in the incremental package:

```bash
find /root/patch-log -name "*.tgz"
```

Upload Charts to the bootstrap node ChartMuseum:

```bash
find /root/patch-log/volumes/charts -name "*.tgz" -print0 | while IFS= read -r -d '' f; do
  echo "upload $f"
  curl --data-binary "@$f" http://127.0.0.1:38080/api/charts
done
```

If `permission denied` is returned, you need to first handle the mount directory permissions of the bootstrap node ChartMuseum, then re-upload.

Check the Charts:

```bash
curl http://127.0.0.1:38080/index.yaml | grep -E "application-management-service|console-service|console-website|marketplace-service|harbor|monitoring-service|web-terminal-service"
```

### 5. Deploy the openFuyao-system-controller Component on the Business Cluster

#### 5.1 Create patch-config in the Business Cluster
> ![icon](../figures/icon-note.gif) **Note:**<br />
> `VersionConfig-latest.yaml` only applies to the `latest` version offline installation scenario.
> Users can run the following commands on the bootstrap node to confirm the version configuration actually carried by the offline package:
> ```bash
> ls /bke/mount/source_registry/files/patches/
> find /bke/mount/source_registry/files/patches -maxdepth 1 -type f \
>   \( -name "VersionConfig-*.yaml" -o -name "Core-VersionConfig-*.yaml" \) \
>   -exec grep -n "^openFuyaoVersion:" {} \;
> ```
> If the version configuration file is `VersionConfig-latest.yaml` and `openFuyaoVersion` is `latest`, use the example commands below; if it is a specified version such as `v26.06-rc.1`, replace `VersionConfig-latest.yaml` in the commands below with the actual version configuration file name.

Copy the version configuration from the bootstrap node on the business cluster node:

```bash
scp root@192.168.200.217:/bke/mount/source_registry/files/patches/VersionConfig-latest.yaml /tmp/VersionConfig-latest.yaml
```

Create the ConfigMap:

```bash
kubectl create ns openfuyao-system-controller --dry-run=client -o yaml | kubectl apply -f -

kubectl create cm patch-config \
  -n openfuyao-system-controller \
  --from-file=patch-data=/tmp/VersionConfig-latest.yaml \
  --dry-run=client -o yaml | kubectl apply -f -

kubectl get cm patch-config -n openfuyao-system-controller
```

#### 5.2 Prepare the openFuyao-system-controller YAML

Execute on the bootstrap node:

```bash
cp /etc/openFuyao/addons/manifests/kubernetes/openfuyao-system-controller/latest/openfuyao-system-controller.yaml /root/openfuyao-system-controller.yaml
```

Render the template:

```bash
sed -i 's#{{ if .repo }}{{ .repo }}{{ else }}cr.openfuyao.cn/openfuyao/{{ end }}#deploy.bocloud.k8s:40443/kubernetes/#g' /root/openfuyao-system-controller.yaml
sed -i 's#{{ .tagVersion }}#latest#g' /root/openfuyao-system-controller.yaml
sed -i 's#{{ .repo }}#deploy.bocloud.k8s:40443/kubernetes#g' /root/openfuyao-system-controller.yaml
sed -i 's#{{ .helmRepo }}#http://192.168.200.217:38080#g' /root/openfuyao-system-controller.yaml
```

Check whether the template variables have been cleaned up:

```bash
grep -n "{{" /root/openfuyao-system-controller.yaml
```

Check key fields:

```bash
grep -nE "image:|OPENFUYAO_REGISTRY|HELM_REPORITORY_URL|entrypoint.sh" /root/openfuyao-system-controller.yaml
grep -nA2 "OPENFUYAO_REGISTRY\|HELM_REPORITORY_URL" /root/openfuyao-system-controller.yaml
```

Copy the file to the business cluster node:

```bash
scp /root/openfuyao-system-controller.yaml root@192.168.200.216:/tmp/openfuyao-system-controller.yaml
```

#### 5.3 Deploy openFuyao-system-controller

Execute on the business cluster node:

```bash
kubectl apply -f /tmp/openfuyao-system-controller.yaml
kubectl get pods -n openfuyao-system-controller -w
```

#### 5.4 Check the Installation Result

View all components:

```bash
kubectl get pods -A
```

View abnormal Pods:

```bash
kubectl get pods -A | grep -Ev "Running|Completed"
```

If there is no abnormal output, it means that `openFuyao-system-controller` and the management plane components it brought up have been deployed successfully.

### Appendix
> ![icon](../figures/icon-note.gif) **Note:**<br />The yaml files below only apply to the `latest` version offline installation scenario.

Below is the download configuration yaml file for K8s-only containing only K8s.
```yaml
registry:
  imageAddress: hub.oepkgs.net/openfuyao/registry:2.8.1
  architecture:
    - amd64
    - arm64
repos:
  - architecture: # k3s cluster images
      - amd64
      - arm64
    needDownload: true
    subImages:
      - sourceRepo: hub.oepkgs.net/openfuyao
        targetRepo: kubernetes
        imageTrack: ""
        images:
          - name: registry
            tag:
              - 2.8.1
          - name: nginx
            tag:
              - 1.23.0-alpine
          - name: helm/chartmuseum
            tag:
              - v0.16.2
          - name: openebs/nfs-server-alpine
            tag:
              - 0.9.0
          - name: rancher/k3s
            tag:
              - v1.25.16-k3s4
          - name: rancher/mirrored-pause
            tag:
              - 3.6
  - architecture: # K8s components
      - amd64
      - arm64
    needDownload: true
    subImages:
      - sourceRepo: cr.openfuyao.cn/openfuyao
        targetRepo: kubernetes
        images:
          - name: coredns
            tag:
              - 1.12.2-of.1
      - sourceRepo: cr.openfuyao.cn/openfuyao
        targetRepo: kubernetes
        images:
          - name: kube-apiserver
            tag:
              - 1.34.3-of.1
          - name: kube-controller-manager
            tag:
              - 1.34.3-of.1
          - name: kube-scheduler
            tag:
              - 1.34.3-of.1
          - name: kube-proxy
            tag:
              - 1.34.3-of.1
          - name: etcd
            tag:
              - 3.6.7-of.1
      - sourceRepo: hub.oepkgs.net/openfuyao
        targetRepo: kubernetes
        images:
          - name: pause
            tag:
              - 3.9
      - sourceRepo: cr.openfuyao.cn/openfuyao/keepalived # keepalived component
        targetRepo: kubernetes/keepalived
        images:
          - name: keepalived
            tag:
              - 1.3.5
      - sourceRepo: hub.oepkgs.net/openfuyao # haproxy component
        targetRepo: kubernetes
        images:
          - name: haproxy
            tag:
              - 2.1.4
      - sourceRepo: hub.oepkgs.net/openfuyao/calico # calico component
        targetRepo: kubernetes/calico
        images:
          - name: cni
            tag:
              - v3.31.3
          - name: node
            tag:
              - v3.31.3
          - name: kube-controllers
            tag:
              - v3.31.3
  - architecture: # openFuyao components
      - amd64
      - arm64
    needDownload: true
    subImages:
      - sourceRepo: cr.openfuyao.cn/openfuyao # core components
        targetRepo: kubernetes
        images:
          - name: bke-console-website
            tag:
              - latest # For non-latest versions, refer to the actual tag of this component in the corresponding Core-VersionConfig, e.g. v26.06-rc.1 is 1.0.3
          - name: bke-console-service
            tag:
              - latest # For non-latest versions, refer to the actual tag of this component in the corresponding Core-VersionConfig, e.g. v26.06-rc.1 is 1.0.2
          - name: installer-website
            tag:
              - latest # For non-latest versions, refer to the actual tag of this component in the corresponding Core-VersionConfig, e.g. v26.06-rc.1 is 1.2.2
          - name: installer-service
            tag:
              - latest # For non-latest versions, refer to the actual tag of this component in the corresponding Core-VersionConfig, e.g. v26.06-rc.1 is 1.2.3
          - name: upgrade-path
            tag:
              - latest # For non-latest versions, refer to the actual tag of this component in the corresponding Core-VersionConfig
          - name: cluster-api-provider-bke  # kubernetes
            tag:
              - latest # For non-latest versions, refer to the actual tag of this component in the corresponding Core-VersionConfig, e.g. v26.06-rc.1 is 1.3.0
          - name: bke-manifests  #   # kubernetes
            tag:
              - latest # For non-latest versions, refer to the actual tag of this component in the corresponding Core-VersionConfig, e.g. v26.06-rc.1 is 1.2.3
          - name: bkeagent-deployer
            tag:
              - latest # For non-latest versions, refer to the actual tag of this component in the corresponding Core-VersionConfig, e.g. v26.06-rc.1 is 1.3.0
          - name: oauth-proxy
            tag:
              - latest # For non-latest versions, refer to the actual tag of this component in the corresponding Core-VersionConfig, e.g. v26.06-rc.1 is 1.0.2
          - name: oauth-server
            tag:
              - latest # For non-latest versions, refer to the actual tag of this component in the corresponding Core-VersionConfig, e.g. v26.06-rc.1 is 1.0.3
          - name: oauth-webhook
            tag:
              - latest # For non-latest versions, refer to the actual tag of this component in the corresponding Core-VersionConfig, e.g. v26.06-rc.1 is 1.0.3
          - name: plugin-management-service
            tag:
              - latest # For non-latest versions, refer to the actual tag of this component in the corresponding Core-VersionConfig, e.g. v26.06-rc.1 is 1.0.2
          - name: user-management-operator
            tag:
              - latest # For non-latest versions, refer to the actual tag of this component in the corresponding Core-VersionConfig, e.g. v26.06-rc.1 is 1.0.2
          - name: kubectl-openfuyao
            tag:
              - latest # For non-latest versions, refer to the actual tag of this component in the corresponding Core-VersionConfig, e.g. v26.06-rc.1 is 1.0.2
      - sourceRepo: hub.oepkgs.net/openfuyao # busybox component
        targetRepo: kubernetes
        images:
          - name: busybox
            tag:
              - 1.36.1
      - sourceRepo: hub.oepkgs.net/openfuyao/cluster-api # cluster-api component
        targetRepo: kubernetes/cluster-api
        images:
          - name: cluster-api-controller
            tag:
              - v1.4.3
          - name: kubeadm-bootstrap-controller
            tag:
              - v1.4.3
          - name: kubeadm-control-plane-controller
            tag:
              - v1.4.3
      - sourceRepo: hub.oepkgs.net/openfuyao/ingress-nginx # ingress-nginx component
        targetRepo: kubernetes/ingress-nginx
        images:
          - name: controller
            tag:
              - v1.9.4
          - name: kube-webhook-certgen
            tag:
              - v20231011-8b53cabe0

rpms:
debs: []
files:
  - address: https://openfuyao.obs.cn-north-4.myhuaweicloud.com/kubernetes/kubernetes/releases/download/1.34.3-of.1/bin/linux/arm64/
    files:
      - fileName: kubectl
        fileAlias: kubectl-v1.34.3-of.1-arm64
      - fileName: kubelet
        fileAlias: kubelet-v1.34.3-of.1-arm64
  - address: https://openfuyao.obs.cn-north-4.myhuaweicloud.com/kubernetes/kubernetes/releases/download/1.34.3-of.1/bin/linux/amd64/
    files:
      - fileName: kubectl
        fileAlias: kubectl-v1.34.3-of.1-amd64
      - fileName: kubelet
        fileAlias: kubelet-v1.34.3-of.1-amd64
  - address: https://openfuyao.obs.cn-north-4.myhuaweicloud.com/containerd/containerd/releases/download/v2.1.1-origin/
    files:
      - fileName: containerd-v2.1.1-linux-amd64.tar.gz
      - fileName: containerd-v2.1.1-linux-arm64.tar.gz
  - address: https://openfuyao.obs.cn-north-4.myhuaweicloud.com/containernetworking/plugins/releases/download/v1.4.1/
    files:
      - fileName: cni-plugins-linux-amd64-v1.4.1.tgz
      - fileName: cni-plugins-linux-arm64-v1.4.1.tgz
  - address: https://openfuyao.obs.cn-north-4.myhuaweicloud.com/helm/releases/download/v3.14.2/
    files:
      - fileName: helm-v3.14.2-linux-amd64.tar.gz
      - fileName: helm-v3.14.2-linux-arm64.tar.gz
  - address: https://openfuyao.obs.cn-north-4.myhuaweicloud.com/mikefarah/yq/releases/download/v4.43.1
    files:
      - fileName: yq_linux_arm64
      - fileName: yq_linux_amd64
  - address: https://openfuyao.obs.cn-north-4.myhuaweicloud.com/jqlang/jq/releases/download/v1.7.1
    files:
      - fileName: jq-linux-arm64
      - fileName: jq-linux-amd64
  - address: https://openfuyao.obs.cn-north-4.myhuaweicloud.com/cloudflare/cfssl/releases/download/v1.6.4
    files:
      - fileName: cfssl-certinfo_1.6.4_linux_arm64
      - fileName: cfssl-certinfo_1.6.4_linux_amd64
      - fileName: cfssl_1.6.4_linux_arm64
      - fileName: cfssl_1.6.4_linux_amd64
      - fileName: cfssljson_1.6.4_linux_arm64
      - fileName: cfssljson_1.6.4_linux_amd64
  - address: https://openfuyao.obs.cn-north-4.myhuaweicloud.com/opencontainers/runc/releases/download/v1.1.12/
    files:
      - fileName: runc-arm64
      - fileName: runc-amd64
  - address: https://openfuyao.obs.cn-north-4.myhuaweicloud.com/etcd-io/etcd/releases/download/v3.5.6/
    files:
      - fileName: etcdctl-v3.5.6-linux-amd64
      - fileName: etcdctl-v3.5.6-linux-arm64
  - address: https://openfuyao.obs.cn-north-4.myhuaweicloud.com/rpm/releases/download/v0.0.1/
    files:
      - fileName: rpm.tar.gz
  - address: https://openfuyao.obs.cn-north-4.myhuaweicloud.com/nfs/
    files:
      - fileName: nfsshare.tar.gz
  - address: https://openfuyao.obs.cn-north-4.myhuaweicloud.com/openFuyao/bkeadm/releases/download/latest/ # For non-latest versions, refer to files.address in the corresponding Core-VersionConfig, e.g. v26.06-rc.1 is https://openfuyao.obs.cn-north-4.myhuaweicloud.com/openFuyao/bkeadm/releases/download/1.2.3/
    files:
      - fileName: bkeadm_linux_amd64
      - fileName: bkeadm_linux_arm64
patches:
  - address: https://openfuyao.obs.cn-north-4.myhuaweicloud.com/openFuyao/version-config/
    files:
      - fileName: VersionConfig-latest.yaml # For other versions, change to the other version, e.g.: for v26.06-rc.1, change to Core-VersionConfig-v26.06-rc.1.yaml
charts:
  - address: https://openfuyao.obs.cn-north-4.myhuaweicloud.com/charts/releases/download/latest/ # For non-latest versions, refer to charts.address in the corresponding Core-VersionConfig, e.g. v26.06-rc.1 is https://openfuyao.obs.cn-north-4.myhuaweicloud.com/charts/releases/download/
    files:
      - fileName: oauth-webhook-0.0.0-latest.tgz # For non-latest versions, refer to the actual chart file name in the corresponding Core-VersionConfig, e.g. v26.06-rc.1 is oauth-webhook-1.0.3.tgz
      - fileName: oauth-server-0.0.0-latest.tgz # For non-latest versions, refer to the actual chart file name in the corresponding Core-VersionConfig, e.g. v26.06-rc.1 is oauth-server-1.0.3.tgz
      - fileName: plugin-management-service-0.0.0-latest.tgz # For non-latest versions, refer to the actual chart file name in the corresponding Core-VersionConfig, e.g. v26.06-rc.1 is plugin-management-service-1.0.2.tgz
      - fileName: user-management-operator-0.0.0-latest.tgz # For non-latest versions, refer to the actual chart file name in the corresponding Core-VersionConfig, e.g. v26.06-rc.1 is user-management-operator-1.0.2.tgz
      - fileName: installer-service-0.0.0-latest.tgz # For non-latest versions, refer to the actual chart file name in the corresponding Core-VersionConfig, e.g. v26.06-rc.1 is installer-service-1.2.3.tgz
      - fileName: installer-website-0.0.0-latest.tgz # For non-latest versions, refer to the actual chart file name in the corresponding Core-VersionConfig, e.g. v26.06-rc.1 is installer-website-1.2.2.tgz
      - fileName: bke-console-website-0.0.0-latest.tgz # For non-latest versions, refer to the actual chart file name in the corresponding Core-VersionConfig, e.g. v26.06-rc.1 is bke-console-website-1.0.3.tgz
      - fileName: bke-console-service-0.0.0-latest.tgz # For non-latest versions, refer to the actual chart file name in the corresponding Core-VersionConfig, e.g. v26.06-rc.1 is bke-console-service-1.0.2.tgz
```

Below is the image download configuration yaml file for the openFuyao-system-controller incremental component.
```yaml
registry:
  imageAddress: hub.oepkgs.net/openfuyao/registry:2.8.1
  architecture:
    - amd64
    - arm64
repos:
  - architecture: # openFuyao components
      - amd64
      - arm64
    needDownload: true
    subImages:
      - sourceRepo: cr.openfuyao.cn/openfuyao # core components
        targetRepo: kubernetes
        images:
          # For non-latest versions, refer to the actual tag of this component in the corresponding Core-VersionConfig file
          - name: application-management-service
            tag:
              - latest # For other versions, change to the other version, e.g.: for v26.06-rc.1, change to 1.0.2
          - name: bkeagent-launcher
            tag:
              - latest # For other versions, change to the other version, e.g.: for v26.06-rc.1, change to 1.3.0
          - name: console-service
            tag:
              - latest # For other versions, change to the other version, e.g.: for v26.06-rc.1, change to 1.1.0
          - name: console-website
            tag:
              - latest # For other versions, change to the other version, e.g.: for v26.06-rc.1, change to 1.1.0
          - name: marketplace-service
            tag:
              - latest # For other versions, change to the other version, e.g.: for v26.06-rc.1, change to 1.0.3
          - name: monitoring-service
            tag:
              - latest # For other versions, change to the other version, e.g.: for v26.06-rc.1, change to 1.0.4
          - name: openfuyao-system-controller
            tag:
              - latest # For other versions, change to the other version, e.g.: for v26.06-rc.1, change to 1.1.3
          - name: web-terminal-service
            tag:
              - latest # For other versions, change to the other version, e.g.: for v26.06-rc.1, change to 1.0.3
      - sourceRepo: hub.oepkgs.net/openfuyao/brancz # brancz component
        targetRepo: kubernetes/brancz
        images:
          - name: kube-rbac-proxy
            tag:
              - v0.18.0
      - sourceRepo: cr.openfuyao.cn/openfuyao/harbor # harbor component
        targetRepo: kubernetes/harbor
        images:
          - name: chartmuseum-photon
            tag:
              - v2.7.0
          - name: harbor-core
            tag:
              - v2.7.0
          - name: harbor-portal
            tag:
              - v2.7.0
          - name: harbor-registryctl
            tag:
              - v2.7.0
          - name: harbor-db
            tag:
              - v2.7.0
          - name: harbor-jobservice
            tag:
              - v2.7.0
          - name: nginx-photon
            tag:
              - v2.7.0
          - name: redis-photon
            tag:
              - v2.7.0
          - name: registry-photon
            tag:
              - v2.7.0
          - name: redis
            tag:
              - v7.4.2
      - sourceRepo: hub.oepkgs.net/openfuyao/jimmidyson # jimmidyson component
        targetRepo: kubernetes/jimmidyson
        images:
          - name: configmap-reload
            tag:
              - v0.13.0
      - sourceRepo: hub.oepkgs.net/openfuyao/kube-state-metrics # kube-state-metrics component
        targetRepo: kubernetes/kube-state-metrics
        images:
          - name: kube-state-metrics
            tag:
              - v2.12.0
      - sourceRepo: hub.oepkgs.net/openfuyao/metrics-server # metrics-server component
        targetRepo: kubernetes/metrics-server
        images:
          - name: metrics-server
            tag:
              - v0.7.1
      - sourceRepo: hub.oepkgs.net/openfuyao/prometheus-operator # prometheus-operator component
        targetRepo: kubernetes/prometheus-operator
        images:
          - name: prometheus-config-reloader
            tag:
              - v0.74.0
          - name: prometheus-operator
            tag:
              - v0.74.0
      - sourceRepo: hub.oepkgs.net/openfuyao/prometheus  # prometheus component
        targetRepo: kubernetes/prometheus
        images:
          - name: alertmanager
            tag:
              - v0.27.0
          - name: blackbox-exporter
            tag:
              - v0.25.0
          - name: node-exporter
            tag:
              - v1.8.1
          - name: prometheus
            tag:
              - v2.52.0
rpms:
debs: []
files: []
patches: []
charts:
  - address: https://openfuyao.obs.cn-north-4.myhuaweicloud.com/charts/releases/download/latest/ # For non-latest versions, refer to charts.address in the corresponding Core-VersionConfig file
    files:
      - fileName: console-website-0.0.0-latest.tgz # For non-latest versions, refer to charts.files in the corresponding Core-VersionConfig file, e.g. v26.06-rc.1 is console-website-1.1.0.tgz
      - fileName: monitoring-service-0.0.0-latest.tgz # For non-latest versions, refer to the actual chart file name in the Core-VersionConfig, e.g. v26.06-rc.1 is monitoring-service-1.0.4.tgz
      - fileName: console-service-0.0.0-latest.tgz # For non-latest versions, refer to the actual chart file name in the Core-VersionConfig, e.g. v26.06-rc.1 is console-service-1.1.0.tgz
      - fileName: marketplace-service-0.0.0-latest.tgz # For non-latest versions, refer to the actual chart file name in the Core-VersionConfig, e.g. v26.06-rc.1 is marketplace-service-1.0.3.tgz
      - fileName: application-management-service-0.0.0-latest.tgz # For non-latest versions, refer to the actual chart file name in the Core-VersionConfig, e.g. v26.06-rc.1 is application-management-service-1.0.2.tgz
      - fileName: web-terminal-service-0.0.0-latest.tgz # For non-latest versions, refer to the actual chart file name in the Core-VersionConfig, e.g. v26.06-rc.1 is web-terminal-service-1.0.3.tgz
      - fileName: harbor-1.11.4.tgz

```
