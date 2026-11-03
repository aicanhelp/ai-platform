# Custom Repository Configuration

## Background

* Custom repository configuration is supported, including custom configuration of **image repository**, **binary repository**, and **chart repository**.
* One-way authentication access is supported for the image repository and chart repository (containerd and helm perform one-way access to the repository, and use the same CA). The binary repository does not involve authentication access.

 > ![icon](../figures/icon-note.gif) **Note:**<br />The chart repository is currently only used independently by the NUMA component in addons. For private Chart / addon operation steps, see [Guide to Installing NUMA Affinity Extension Using Add-ons](https://gitcode.com/openFuyao/sig-installation/blob/master/docs/en/user_case/numa_affinity_extension_installation_guide_using_add-ons.md).

## Prerequisites

* Users need to provide custom image repository, binary repository, and chart repository.

  * Please import all required images into the private image repository (ensure the namespace, image name, and image tag are uploaded correctly).
  * Import all required binary files into the private binary repository.
  * Import all required chart packages into the private chart repository.

- On the bootstrap node, by convention, upload the repository's root CA to the specified path `/etc/openFuyao`. If the directory does not exist, create it and it is recommended to tighten the directory permissions to **700**; if the directory already exists, do not modify the directory permissions. It is recommended to tighten the `ca.crt` file permissions to **600**. The directory structure is `/etc/openFuyao/ca.crt`.

  ```bash
  if [ ! -d /etc/openFuyao ]; then
    mkdir -p /etc/openFuyao
    chmod 700 /etc/openFuyao
  fi
  # After placing ca.crt at /etc/openFuyao/ca.crt, execute
  chmod 600 /etc/openFuyao/ca.crt
  ```

  > ![icon](../figures/icon-caution.gif) **Notice:**<br />When `/etc/openFuyao` already exists, do not adjust its directory permissions to avoid affecting existing configurations on this path. Although `ca.crt` typically does not contain a private key, it is still recommended to restrict access following the principle of least privilege, allowing only the file owner to read and write, to avoid unnecessary reads by other local accounts.

- For the list of media to be uploaded, see the [Appendix section](./custom_repository_configuration_guide.md#appendix).

## Usage Restrictions

None.

## Operation Steps

> ![icon](../figures/icon-note.gif) **Note:**<br />When creating a management cluster/business cluster, openFuyao carries the configuration through the following custom resources (CRs). For field meanings and default values (official recommended configuration of the openFuyao platform), see [Appendix: cluster-api-provider-bke Configuration Parameters](../appendix/cluster-api-provider-bke_configuration.md).
>
> | Abbreviation | CR Type | kubectl short name / resource name | Purpose |
> | --- | --- | --- | --- |
> | bc | `BKECluster` | `bc` | Describes cluster-level configuration such as version, repository, network, addon, etc. of the cluster to be created |
> | kct | `KubeletConfig` | `kct` | Custom kubelet configuration, referenced by bc's `spec.KubeletConfigRef` |
> | cct | `ContainerdConfig` | `cct` | Custom containerd configuration, referenced by bc's `spec.clusterConfig.cluster.containerdConfigRef` |

1. Bootstrap node initialization.

    Upload the bkeadm binary tool to the bootstrap node.
    Use bke init to complete the bootstrap node initialization. The following is a reference initialization command:

    ``````

      bke init \
      # 1. Image repository
      --otherRepo 192.168.100.17:443/openfuyao \   # Custom image repository (including port + path), note that the recommended namespace is openFuyao
      --domain my.harbor.com \                     # Image repository domain (matches --otherRepo, ensuring certificate and access consistency)
      --imageRepoPort 443 \                        # Image repository port (explicitly specify the port number)
      --imageRepoCAFile /etc/openFuyao/ca.crt \    # Private repository CA certificate (specify the certificate path to read)
      # 2. Binary repository
      --otherSource http://192.168.100.48:8080 \
      # 3. Feature switch: disable console installation
      --installConsole=false \                     # Do not install the BKE built-in console (disabled by default)
      --enableNTP=false \                          # Do not perform NTP synchronization service operations
      # 4. Others
      --confirm                                    # Confirm in advance, no subsequent Y/N confirmation needed
    ``````

2. Create a management cluster/business cluster through bc (BKECluster).

    2.1 Query and edit bc, under `spec.clusterConfig.cluster`, change the image repository (imageRepo), binary/system package repository (httpRepo), and Chart repository (chartRepo) to custom addresses.

    ```bash
    # View existing BKECluster
    kubectl get bc -A
    # Edit the bc of the target cluster (replace placeholders with actual namespace and name)
    kubectl edit bc -n <namespace> <name>
    ```

    Custom repository configuration example (written to `spec.clusterConfig.cluster`):

    ```yaml
    httpRepo:  # Binary/system package repository
      domain: http.bocloud.k8s   # Can keep the default domain, or change to the actual domain
      ip: 192.168.200.173        # Change to the actual http repository IP
      port: "40080"              # Change to the actual http repository port number
      prefix: ""                 # Change to the actual path prefix, can be empty by default
    imageRepo:  # Image repository
      domain: deploy.bocloud.k8s # Change to the actual image repository domain
      ip: 192.168.200.173        # Change to the actual image repository IP
      port: "40443"              # Change to the actual image repository port number
      prefix: openfuyao          # Change to the actual project name
    chartRepo:  # Chart repository
      domain: deploy.bocloud.k8s # Change to the actual chart repository domain; if no domain, IP can be used
      ip: 192.168.200.173        # Change to the actual chart repository IP
      port: "38080"              # Change to the actual chart repository port number
      prefix: ""                 # Change to the actual path prefix, can be empty by default
    ```

    2.2 Create or edit cct (ContainerdConfig), supplement the containerd registry configuration for the private image repository, and reference it in bc.

    ```bash
    # Edit existing ContainerdConfig (replace placeholders with actual namespace and name)
    kubectl edit cct -n <namespace> <name>
    # Or create/update via YAML and then apply
    kubectl apply -f my-containerd.yaml
    ```

    registry configuration example:

    ```yaml
    registry:
      configPath: /etc/containerd/certs.d
      configs:
        "my.harbor.com":            # Image repository domain
          capabilities: ["pull", "resolve"]
          host: my.harbor.com
          skipVerify: false
          plainHTTP: false
          tls:
            caFile: /etc/openFuyao/ca.crt
            insecureSkipVerify: false
    ```

    Reference this cct in bc:

    ```yaml
    spec:
      clusterConfig:
        cluster:
          containerdConfigRef:
            name: bke-containerd      # Consistent with the metadata.name of the ContainerdConfig in the previous step
            namespace: bke-containerd # Consistent with the metadata.namespace of the ContainerdConfig in the previous step
    ```

    2.3 (Optional) Create or edit kct (KubeletConfig), customize the kubelet configuration, and reference it in bc.

    ```bash
    # Edit existing KubeletConfig (replace placeholders with actual namespace and name)
    kubectl edit kct -n <namespace> <name>
    # Or create/update via YAML and then apply
    kubectl apply -f my-kubelet.yaml
    ```

    Reference this kct in bc:

    ```yaml
    spec:
      KubeletConfigRef:
        name: bke-kubelet      # Consistent with the metadata.name of KubeletConfig
        namespace: bke-kubelet # Consistent with the metadata.namespace of KubeletConfig
    ```

    > ![icon](../figures/icon-note.gif) **Note:**<br />For the complete field descriptions and examples of kct / cct, see [Binary Component Custom Configuration (Containerd and kubelet)](../service_management_cluster_configuration/binary_file_custom_configuration_guide.md); for all BKECluster fields, see [Appendix: cluster-api-provider-bke Configuration Parameters](../appendix/cluster-api-provider-bke_configuration.md).

    > ![icon](../figures/icon-caution.gif) **Notice:**<br />In production environments, it is forbidden to set `insecureSkipVerify` to `true`. This configuration skips repository server certificate verification, posing risks of man-in-the-middle attacks, mistakenly connecting to the wrong repository, and pulling tampered content. It may only be used for a short time in temporary troubleshooting or isolated test environments where the risk is explicitly accepted.

## Appendix

### Image List

**1. Basic components for building a K3s cluster, all architectures are `amd64, arm64`.**

| **Image Address**                        | **Tag**      | **Description**                 |
| ----------------------------------------- | ------------- |------------------------|
| cr.openfuyao.cn/openfuyao/registry        | 2.8.1         | Image repository (duplicated with the base image, reused uniformly)     |
| cr.openfuyao.cn/openfuyao/nginx_yum_repo  | 1.23.0-alpine | YUM source proxy service (based on Nginx)    |
| cr.openfuyao.cn/openfuyao/chartmuseum     | v0.16.2       | Helm Chart repository service        |
| cr.openfuyao.cn/openfuyao/nfs-server-alpine | 0.9.0         | NFS storage service (lightweight Alpine version) |
| cr.openfuyao.cn/openfuyao/k3s             | v1.25.16-k3s4 | K3s cluster core binary image          |
| cr.openfuyao.cn/openfuyao/mirrored-pause  | 3.6           | Container pause image (K3s container lifecycle management)   |

**2. K8s core component images (needDownload: true)**

Covering K8s control plane, network, load balancing and other core functions, all architectures are `amd64, arm64`.

| **Image Address**                                  | **Tag**   | **Description**                     |
| ----------------------------------------------- | ---------- |----------------------------|
| cr.openfuyao.cn/openfuyao/kubernetes/coredns    | v1.10.1    | K8s DNS service (CoreDNS)        |
| cr.openfuyao.cn/openfuyao/kubernetes/pause      | 3.9        | K8s container base image (version differentiated from K3s)       |
| cr.openfuyao.cn/openfuyao/kube-apiserver        | of-v1.33.1 | K8s API server (openFuyao customized version)  |
| cr.openfuyao.cn/openfuyao/kube-controller-manager | of-v1.33.1 | K8s controller manager (openFuyao customized version)   |
| cr.openfuyao.cn/openfuyao/kube-scheduler        | of-v1.33.1 | K8s scheduler (openFuyao customized version)      |
| cr.openfuyao.cn/openfuyao/kube-proxy            | of-v1.33.1 | K8s network proxy (openFuyao customized version)     |
| cr.openfuyao.cn/openfuyao/etcd                  | of-v3.5.21 | Etcd database (K8s cluster storage, openFuyao customized version) |
| cr.openfuyao.cn/openfuyao/keepalived/keepalived | 1.3.5      | High availability VIP service (Keepalived)      |
| cr.openfuyao.cn/openfuyao/haproxy/haproxy       | 2.1.4      | Load balancing service (HAProxy)            |
| cr.openfuyao.cn/openfuyao/calico/cni            | v3.27.3    | Calico CNI network plugin (container network interface)    |
| cr.openfuyao.cn/openfuyao/calico/node           | v3.27.3    | Calico node agent (network policy enforcement)        |
| cr.openfuyao.cn/openfuyao/calico/kube-controllers | v3.27.3    | Calico controller (network resource management)         |
| cr.openfuyao.cn/kubernetes/pause                | 3.9        | K8s container base image (version differentiated from K3s)       |

**3. bke tool itself**

| **Image Address**                                             | **Tag** | **Description**                  |
| ---------------------------------------------------------- | -------- |-------------------------|
| cr.openfuyao.cn/openfuyao/cluster-api-provider-bke         | latest   | BKE cluster API provider (CAPI compatible) |
| cr.openfuyao.cn/openfuyao/bke-manifests                    | latest   | BKE resource manifest image              |
| cr.openfuyao.cn/openfuyao/cluster-api/cluster-api-controller | v1.4.3   | CAPI cluster controller              |
| cr.openfuyao.cn/openfuyao/cluster-api/kubeadm-bootstrap-controller | v1.4.3   | CAPI Kubeadm bootstrap controller      |
| cr.openfuyao.cn/openfuyao/cluster-api/kubeadm-control-plane-controller | v1.4.3   | CAPI Kubeadm control plane controller    |

**4. Extension components**

**4.1 Volcano scheduling extension**

All architectures are `amd64, arm64`, used for container scheduling optimization (such as NUMA affinity, resource scheduling).

| **Image Address**                                    | **Tag** | **Description**                    |
| ------------------------------------------------- | -------- |---------------------------|
| cr.openfuyao.cn/openfuyao/volcano-config-website  | latest   | Volcano configuration page              |
| cr.openfuyao.cn/openfuyao/volcano-config-service  | latest   | Volcano configuration service              |
| cr.openfuyao.cn/openfuyao/numa-affinity-webhook   | latest   | NUMA affinity Webhook service        |
| cr.openfuyao.cn/openfuyao/numa-tuning-operator    | v25.03   | NUMA resource tuning Operator         |
| cr.openfuyao.cn/openfuyao/volcano/resource-exporter | latest   | Volcano resource metrics exporter           |
| cr.openfuyao.cn/openfuyao/oauth-proxy             | latest   | OAuth authentication proxy (reused by multiple components)         |
| cr.openfuyao.cn/openfuyao/vc-scheduler            | latest   | Volcano scheduler core             |
| docker.io/volcanosh/vc-webhook-manager            | v1.9.0   | Volcano Webhook manager (official image) |
| docker.io/volcanosh/vc-controller-manager         | v1.9.0   | Volcano controller manager (official image)      |
| docker.io/alpine/openssl                          | 3.5.0    | OpenSSL tool (basic dependency, Alpine version) |
| cr.openfuyao.cn/openfuyao/kubectl-openfuyao       | latest   | openFuyao customized Kubectl tool    |

### **Binary File List**

**1. K8s core operation tools**

Used for command-line management and node operation of K8s clusters, all provide `amd64`/`arm64` dual-architecture support.

| **Component Name**               | **Architecture** | **File Type** | **Description**                               |
| -------------------------- | ---------- | ------------ |--------------------------------------|
| kubectl-v1.33.1-amd64      | amd64      | Single binary   | K8s cluster command-line tool (openFuyao customized v1.33.1 version) |
| kubectl-v1.33.1-arm64      | arm64      | Single binary   | Same as above (adapted for ARM architecture nodes)                      |
| kubelet-v1.33.1-amd64      | amd64      | Single binary   | K8s node agent (maintains node-to-cluster communication, v1.33.1 version)       |
| kubelet-v1.33.1-arm64      | arm64      | Single binary   | Same as above (adapted for ARM architecture nodes)                      |
| etcdctl-v3.5.6-linux-amd64 | amd64      | Single binary   | Etcd database command-line tool (manages cluster storage, v3.5.6 version)      |
| etcdctl-v3.5.6-linux-arm64 | arm64      | Single binary   | Same as above (adapted for ARM architecture nodes)                      |

**2. Container runtime core components**

Underlying binaries supporting container start/stop and resource management, including the runtime main body and core dependencies.

| **Component Name**                         | **Architecture** | **File Type** | **Description**                            |
| ------------------------------------ | ---------- | ------------ |-----------------------------------|
| containerd-v2.1.1-linux-amd64.tar.gz | amd64      | Archive       | Container runtime (core component, v2.1.1 version, archive contains binary and config)  |
| containerd-v2.1.1-linux-arm64.tar.gz | arm64      | Archive       | Same as above (adapted for ARM architecture nodes)                   |
| runc-amd64                           | amd64      | Single binary   | Lightweight container runtime (containerd dependency, creates/destroys containers) |
| runc-arm64                           | arm64      | Single binary   | Same as above (adapted for ARM architecture nodes)                   |

**3. Container Network Plugin (CNI)**

Used to configure container networking (such as network isolation, cross-node communication), in archive format (containing multiple CNI plugin binaries).

| **Component Name**                       | **Architecture** | **File Type** | **Description**                                        |
| ---------------------------------- | ---------- | ------------ |-----------------------------------------------|
| cni-plugins-linux-amd64-v1.4.1.tgz | amd64      | Archive       | CNI network plugin collection (v1.4.1 version, includes bridge, host-local and other plugins) |
| cni-plugins-linux-arm64-v1.4.1.tgz | arm64      | Archive       | Same as above (adapted for ARM architecture nodes)                               |

**4. Package management and configuration processing tools**

Used for Helm Chart management, YAML/JSON parsing and other operations tasks.

| **Component Name**                    | **Architecture** | **File Type** | **Description**                                  |
| ------------------------------- | ---------- | ------------ |-----------------------------------------|
| helm-v3.14.2-linux-amd64.tar.gz | amd64      | Archive       | Helm package management tool (v3.14.2 version, manages K8s application Charts)  |
| helm-v3.14.2-linux-arm64.tar.gz | arm64      | Archive       | Same as above (adapted for ARM architecture nodes)                         |
| yq_linux_amd64                  | amd64      | Single binary   | YAML data processing tool (query/modify YAML files, v4.43.1 version) |
| yq_linux_arm64                  | arm64      | Single binary   | Same as above (adapted for ARM architecture nodes)                         |
| jq-linux-amd64                  | amd64      | Single binary   | JSON data processing tool (parse/filter JSON, v1.7.1 version)     |
| jq-linux-arm64                  | arm64      | Single binary   | Same as above (adapted for ARM architecture nodes)                         |

**5. Auxiliary archives used for RPM package management and NFS share configuration (containing related binaries or scripts).**

| **Component Name**    | **Architecture**             | **File Type** | **Description**                   |
| --------------- | ---------------------- |----------|--------------------------|
| rpm.tar.gz      | Universal (archive contains multiple architectures) | Archive      | RPM package management related tool collection (adapted for RPM-based operating systems) |
| nfsshare.tar.gz | Universal (archive contains multiple architectures) | Archive      | NFS share service configuration tool (enables cluster storage sharing)    |
| charts.tar.gz   | Required               | Archive      | charts package file                |

**6. BKE cluster-specific management tools**

Initialization and operations tools for openFuyao BKE clusters, with dual-architecture support.

| **Component Name**       | **Architecture** | **File Type** | **Description**                  |
| ------------------ | ---------- | ------------ |-------------------------|
| bkeadm_linux_amd64 | amd64      | Single binary   | BKE cluster management tool (initialize, upgrade, operate clusters) |
| bkeadm_linux_arm64 | arm64      | Single binary   | Same as above (adapted for ARM architecture nodes)         |

Note: The cfssl tool is used in openFuyao core components, but is not used in the co-creation scenario.

**Complete Binary File List**

```yaml
files:
  - address: https://openfuyao.obs.cn-north-4.myhuaweicloud.com/kubernetes/kubernetes/releases/download/of-v1.33.1/bin/linux/arm64/
    files:
      - kubectl-v1.33.1-arm64
  - address: https://openfuyao.obs.cn-north-4.myhuaweicloud.com/kubernetes/kubernetes/releases/download/of-v1.33.1/bin/linux/amd64/
    files:
      - kubectl-v1.33.1-amd64
  - address: https://openfuyao.obs.cn-north-4.myhuaweicloud.com/kubernetes/kubernetes/releases/download/of-v1.33.1/bin/linux/arm64/
    files:
      - kubelet-v1.33.1-arm64
  - address: https://openfuyao.obs.cn-north-4.myhuaweicloud.com/kubernetes/kubernetes/releases/download/of-v1.33.1/bin/linux/amd64/
    files:
      - kubelet-v1.33.1-amd64
  - address: https://openfuyao.obs.cn-north-4.myhuaweicloud.com/containerd/containerd/releases/download/v2.1.1-origin/
    files:
      - containerd-v2.1.1-linux-amd64.tar.gz
      - containerd-v2.1.1-linux-arm64.tar.gz
  - address: https://openfuyao.obs.cn-north-4.myhuaweicloud.com/containernetworking/plugins/releases/download/v1.4.1/
    files:
      - cni-plugins-linux-amd64-v1.4.1.tgz
      - cni-plugins-linux-arm64-v1.4.1.tgz
  - address: https://openfuyao.obs.cn-north-4.myhuaweicloud.com/helm/releases/download/v3.14.2/
    files:
      - helm-v3.14.2-linux-amd64.tar.gz
      - helm-v3.14.2-linux-arm64.tar.gz
  - address: https://openfuyao.obs.cn-north-4.myhuaweicloud.com/mikefarah/yq/releases/download/v4.43.1
    files:
      - yq_linux_arm64
      - yq_linux_amd64
  - address: https://openfuyao.obs.cn-north-4.myhuaweicloud.com/jqlang/jq/releases/download/v1.7.1
    files:
      - jq-linux-arm64
      - jq-linux-amd64
  - address: https://openfuyao.obs.cn-north-4.myhuaweicloud.com/cloudflare/cfssl/releases/download/v1.6.4
    files:
      - cfssl-certinfo_1.6.4_linux_arm64
      - cfssl-certinfo_1.6.4_linux_amd64
      - cfssl_1.6.4_linux_arm64
      - cfssl_1.6.4_linux_amd64
      - cfssljson_1.6.4_linux_arm64
      - cfssljson_1.6.4_linux_amd64
  - address: https://openfuyao.obs.cn-north-4.myhuaweicloud.com/opencontainers/runc/releases/download/v1.1.12/
    files:
      - runc-arm64
      - runc-amd64
  - address: https://openfuyao.obs.cn-north-4.myhuaweicloud.com/etcd-io/etcd/releases/download/v3.5.6/
    files:
      - etcdctl-v3.5.6-linux-amd64
      - etcdctl-v3.5.6-linux-arm64
  - address: https://openfuyao.obs.cn-north-4.myhuaweicloud.com/rpm/releases/download/v0.0.1/
    files:
      - rpm.tar.gz
  - address: https://openfuyao.obs.cn-north-4.myhuaweicloud.com/nfs/
    files:
      - nfsshare.tar.gz
  - address: https://openfuyao.obs.cn-north-4.myhuaweicloud.com/charts/releases/download/latest/
    files:
      - charts.tar.gz
  - address: https://openfuyao.obs.cn-north-4.myhuaweicloud.com/openFuyao/bkeadm/releases/download/latest/
    files:
      - bkeadm_linux_amd64
      - bkeadm_linux_arm64
```

### Chart Package List

Currently only involves the NUMA component.

```yaml
  - address: https://openfuyao.obs.cn-north-4.myhuaweicloud.com/charts/releases/download/latest/
    files:
      - numa-affinity-package-0.0.0-latest.tgz
```
