# Kunpeng Accelerator Engine (KAE) Device Plugin User Guide

## What is KAE

KAE is Kunpeng Accelerator Engine, a capability carried in Kunpeng 920, 920+ series processors. Users need to load License to enable KAE certificate loading method see [Pre-installation Preparation Certificate Loading](https://www.hikunpeng.com/document/detail/en/kunpengaccel/kae/usermanual/kunpengaccel_06_0008.html).

KAE is mainly used for encryption/decryption acceleration and compression/decompression acceleration, providing a set of interfaces completely compatible with openSSL, enabling applications using openSSL to load kae dynamic engine through openssl engine mechanism, thereby offloading encryption/decryption operations to hardware for acceleration.

KAE architecture explanation please see [KAE Architecture Explanation](https://www.hikunpeng.com/document/detail/en/kunpengaccel/kae/usermanual/kunpengaccel_16_0002.html).

## KAE Device Plugin Description

KAE Device Plugin is used to expose KAE virtualization devices to cluster layer, enabling device resources to be recognized and managed by k8s, openFuyao clusters; KAE Device Plugin based on k8s native Device Plugin framework,implements gRPC Server for communication between Device Plugin and node Kubelet, thereby implementing device discovery, registration and resource allocation scheduling.

KAE Device Plugin architecture please see [KAE Device Plugin Architecture](https://gitee.com/openfuyao/enhancements/tree/master/design/feature/KAE%E9%B2%B2%E9%B9%8F%E5%8A%A0%E9%80%9F%E7%89%B9%E6%80%A7%E8%AF%B4%E6%98%8E%E4%B9%A6)

## Getting Started

Before starting to install KAE Device Plugin, please confirm your cluster has **at least one** node meeting the following requirements:

1. Physical machine node server carries Kunpeng 920/920+ series chips, and KAE license has been loaded; or virtual machine node host server carries Kunpeng 920/920+ series chips, and license has been loaded; in virtual machine node usage scenario, device virtualization and binding have been completed.

2. Node has installed KAE driver software, installation guide please see [KAE Driver Installation Guide](https://www.hikunpeng.com/document/detail/en/kunpengaccel/kae/usermanual/kunpengaccel_06_0012.html )

After confirming your node environment meets requirements, you can deploy this plugin through application marketplace.

KAE Device Plugin runs as DaemonSet daemon process (node label: accelerator/kae.hpre), after successful installation, all nodes with node label (accelerator/kae) will run instances of this daemon process, thereby performing device resource discovery, registration and scheduling.

### Using KAE Acceleration in Cluster

Current KAE HPRE encryption/decryption supports OpenSSL 1.1.1.1x version asymmetric encryption/decryption. For services using OpenSSL 1.1.1.1x version encryption/decryption in cluster, can configure as follows:

1. Edit workload needing KAE acceleration in cluster such as `nginx_deployment.yaml`:

    ```shell
    kubectl edit deployments nginx_deployment -n <namespace>
    ```

2. Add `resources` field, add resource request for KAE:

    ```yaml
    resources:
      requests:
        openfuyao.com/kae.hpre: 1
    ```

3. Mount openssl configuration file to container image:

    ```yaml
      volumeMounts:
        - name: openssl-conf
          mountPath: /<container_path>/openssl.cnf
    volumes:
      - name: openssl-conf
        hostPath:
        path: /<host_path>/openssl.cnf
    ```

    `/<host_path>` is configuration file host machine path, `/<container_path>` is container internal path.

    `openssl.cnf` file content is:

    ```text
    openssl_conf=openssl_def
    [openssl_def]
    engines=engine_section
    [engine_section]
    kae=kae_section
    [kae_section]
    engine_id=kae
    dynamic_path=/usr/local/lib/engines-1.1/kae.so
    KAE_CMD_ENABLE_ASYNC=1
    KAE_CMD_ENABLE_SM3=1
    KAE_CMD_ENABLE_SM4=1
    default_algorithms=ALL
    init=1
    ```

    Can also mount above file to container through configMap method.

4. Set environment variable:

    ```yaml
    env:
      - name: OPENSSL_CONF
        value: /<container_path>/openssl.cnf
    ```

5. Save edit content, takes effect after Pod rescheduled.

### Uninstalling, Rolling Back, Upgrading KAE Device Plugin

Component uninstallation, upgrade, rollback can all be done through application marketplace, or through Helm command line for uninstallation, rollback and upgrade.

## Local Build

### Image Build

#### Build Parameters

- `GOPRIVATE`: Configure Go language private repository, equivalent to `GOPRIVATE` environment variable.
- `COMMIT`: Hash value of current git commit.
- `VERSION`: Component version.
- `SOURCE_DATE_EPOCH`: Timestamp of image rootfs.

#### Build Commands

- Build and push to specified OCI repository.

  <details open>
  <summary>Using <code>docker</code></summary>

  ```bash
  docker buildx build . -f <path/to/dockerfile> \
      -o type=image,name=<oci/repository>:<tag>,oci-mediatypes=true,rewrite-timestamp=true,push=true \
      --platform=linux/amd64,linux/arm64 \
      --provenance=false \
      --build-arg=GOPRIVATE=gopkg.openfuyao.cn \
      --build-arg=COMMIT=$(git rev-parse HEAD) \
      --build-arg=VERSION=0.0.0-latest \
      --build-arg=SOURCE_DATE_EPOCH=$(git log -1 --pretty=%ct)
  ```

  </details>
  <details>
  <summary>Using <code>nerdctl</code></summary>

  ```bash
  nerdctl build . -f <path/to/dockerfile> \
      -o type=image,name=<oci/repository>:<tag>,oci-mediatypes=true,rewrite-timestamp=true,push=true \
      --platform=linux/amd64,linux/arm64 \
      --provenance=false \
      --build-arg=GOPRIVATE=gopkg.openfuyao.cn \
      --build-arg=COMMIT=$(git rev-parse HEAD) \
      --build-arg=VERSION=0.0.0-latest \
      --build-arg=SOURCE_DATE_EPOCH=$(git log -1 --pretty=%ct)
  ```

  </details>

  Where `<path/to/dockerfile>` is Dockerfile path `./build/Dockerfile`, `<oci/repository>` is image address, `<tag>` is image tag.

- Build and export OCI Layout to local tarball.

  <details open>
  <summary>Using <code>docker</code></summary>

  ```bash
  docker buildx build . -f <path/to/dockerfile> \
      -o type=oci,name=<oci/repository>:<tag>,dest=<path/to/oci-layout.tar>,rewrite-timestamp=true \
      --platform=linux/amd64,linux/arm64 \
      --provenance=false \
      --build-arg=GOPRIVATE=gopkg.openfuyao.cn \
      --build-arg=COMMIT=$(git rev-parse HEAD) \
      --build-arg=VERSION=0.0.0-latest \
      --build-arg=SOURCE_DATE_EPOCH=$(git log -1 --pretty=%ct)
  ```

  </details>
  <details>
  <summary>Using <code>nerdctl</code></summary>

  ```bash
  nerdctl build . -f <path/to/dockerfile> \
      -o type=oci,name=<oci/repository>:<tag>,dest=<path/to/oci-layout.tar>,rewrite-timestamp=true \
      --platform=linux/amd64,linux/arm64 \
      --provenance=false \
      --build-arg=GOPRIVATE=gopkg.openfuyao.cn \
      --build-arg=COMMIT=$(git rev-parse HEAD) \
      --build-arg=VERSION=0.0.0-latest \
      --build-arg=SOURCE_DATE_EPOCH=$(git log -1 --pretty=%ct)
  ```

  </details>

  Where `<path/to/dockerfile>` is Dockerfile path `./build/Dockerfile`, `<oci/repository>` is image address, `<tag>` is image tag, `path/to/oci-layout.tar` is tar package path.

- Build and export image rootfs to local directory.

  <details open>
  <summary>Using <code>docker</code></summary>

  ```bash
  docker buildx build . -f <path/to/dockerfile> \
      -o type=local,dest=<path/to/output>,platform-split=true \
      --platform=linux/amd64,linux/arm64 \
      --provenance=false \
      --build-arg=GOPRIVATE=gopkg.openfuyao.cn \
      --build-arg=COMMIT=$(git rev-parse HEAD) \
      --build-arg=VERSION=0.0.0-latest
  ```

  </details>
  <details>
  <summary>Using <code>nerdctl</code></summary>

  ```bash
  nerdctl build . -f <path/to/dockerfile> \
      -o type=local,dest=<path/to/output>,platform-split=true \
      --platform=linux/amd64,linux/arm64 \
      --provenance=false \
      --build-arg=GOPRIVATE=gopkg.openfuyao.cn \
      --build-arg=COMMIT=$(git rev-parse HEAD) \
      --build-arg=VERSION=0.0.0-latest
  ```

  </details>

  Where `<path/to/dockerfile>` is Dockerfile path `./build/Dockerfile`, `path/to/output` is local directory path.

### Helm Chart Build

- Package Helm Chart.

  ```bash
  helm package <path/to/chart> -u \
      --version=0.0.0-latest \
      --app-version=openFuyao-v25.09
  ```

  Where `<path/to/chart>` is Chart folder path.

- Push Chart package to specified OCI repository.

  ```bash
  helm push <path/to/chart.tgz> oci://<oci/repository>:<tag>
  ```

  Where `<path/to/chart.tgz>` is Chart package path, `<oci/repository>` is Chart package push address, `<tag>` is Chart package tag.

## License

[MulanPSL-2.0](http://license.coscl.org.cn/MulanPSL2)
