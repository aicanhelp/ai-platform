# Introduction

bkeadm is a Kubernetes-based container orchestration tool designed to provide users with convenient and efficient container orchestration and application deployment capabilities.

## Image Building

### Build Parameters

- `GOPRIVATE`: Configure Go language private repository, equivalent to the `GOPRIVATE` environment variable.
- `COMMIT`: Hash value of the current git commit.
- `VERSION`: Component version.
- `SOURCE_DATE_EPOCH`: Timestamp of the image rootfs.

### Build Commands

Run `make test-preflight` before every build or release. This gate verifies
that the embedded `preflight.yaml` policy loads, validates, and compiles into
the runtime Bundle. Do not run the following `docker buildx` or `nerdctl
build` commands if verification fails.

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

  Where `<path/to/dockerfile>` is the Dockerfile path, `<oci/repository>` is the image address, and `<tag>` is the image tag.

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

  Where `<path/to/dockerfile>` is the Dockerfile path, `<oci/repository>` is the image address, `<tag>` is the image tag, and `path/to/oci-layout.tar` is the tar package path.

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
  <<details>
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

  Where `<path/to/dockerfile>` is the Dockerfile path and `path/to/output` is the local directory path.

## Cluster Installation

### Online Installation

1. Download and automatically install bkeadm.

    ```shell
    # Method 1: Quick download
    curl -sfL https://openfuyao.obs.cn-north-4.myhuaweicloud.com/openFuyao/bkeadm/releases/download/latest/download.sh | bash
    ```
    ```shell
   # Method 2: Verify complete download of file
   ## Download download.sh script file
   curl -LO https://openfuyao.obs.cn-north-4.myhuaweicloud.com/openFuyao/bkeadm/releases/download/latest/download.sh
   ## Download verification file for download.sh and verify (optional). Successful verification will output -: OK. Failed verification requires contacting openFuyao community maintainers to investigate the reason
   curl -LO https://openfuyao.obs.cn-north-4.myhuaweicloud.com/openFuyao/bkeadm/releases/download/latest/download.sh.sha256
   sha256sum -c <(cat download.sh.sha256) < download.sh
   ## Run download.sh file to download bke installation tool. sha256sum of the installation tool will be verified during execution
   chmod +x download.sh && ./download.sh
    ```

2. Initialize bootstrap node.

    - This function will deploy a lightweight K3s cluster on the bootstrap node. The cluster will deploy cluster-api, provider-bke, and Pods related to openFuyao installation and deployment.

    ```shell
    bke init --otherRepo cr.openfuyao.cn/openfuyao/bke-online-installed:latest
    ```

### Offline Installation

1. Build offline installation deployment package, refer to building deployment package.

2. Copy deployment package to offline environment bootstrap node.

3. Extract deployment package to root directory.

    ```shell
    rm -rf /bke && tar zxvf <deployment package name eg: bke.tar.gz> -C /
    ```

    > Requires root directory storage space greater than 29GB after extraction, otherwise initialization will fail.

4. Rename bke installation tool and initialize bootstrap node.

    - Requires bootstrap node to be clean, without docker, containerd, and other components pre-installed.

    ```shell
    # Rename installation tool
    ARCH=$(uname -m)
    case $ARCH in
    x86_64) ARCH="amd64";;
    aarch64) ARCH="arm64";;
    esac
    mv /usr/local/bin/bkeadm_linux_$ARCH /usr/local/bin/bke
   
    # Initialize bootstrap node
    bke init 
    ```

## Building Deployment Package

- Requires build machine to have tar, pigz tools and bkeadm installed.
- Requires online deployment dependency environment to have docker and buildx pre-installed, refer to [docker official documentation](https://docs.docker.com/engine/install/) for installation.
- Requires offline deployment artifact environment to have docker pre-installed.
- After completing docker installation, add the following configuration to docker configuration file:
    - Edit `/etc/docker/daemon.json`, add the following configuration to docker configuration file

    ```shell
    "insecure-registries": [
    "deploy.bocloud.k8s:40443",
    "0.0.0.0/0"
    ],
    ```

    - Restart docker after modification.

    ```shell
    systemctl restart docker
    ```

### Building Online Deployment Dependencies

- Collect binary files, rpm packages, chart packages, etc., and finally generate an image.

   ```shell
   rm -rf /bke && bke build online-image -f online-artifacts.yaml --arch amd64,arm64 -t cr.openfuyao.cn/openfuyao/bke-online-installed:latest
   ```

### Building Offline Deployment Artifacts

- Collect binary files, rpm packages, chart packages, etc., and finally generate a compressed package. For offline-artifacts.yaml, please use [offline-artifacts.yaml file in assets folder](./assets/offline-artifacts.yaml)

    ```shell
    rm -rf /bke && bke build -f offline-artifacts.yaml -t bke.tar.gz
    ```

## Other Commands

- Image sync command reference as follows:

    ```shell
    bke registry sync --dest-tls-verify --src-tls-verify  --multi-arch --source  deploy.bocloud.k8s:40443/openfuyao/cluster-api-provider-bke:latest    --target deploy.bocloud.k8s:40443/kubernetes/cluster-api-provider-bke:latest
    ```
