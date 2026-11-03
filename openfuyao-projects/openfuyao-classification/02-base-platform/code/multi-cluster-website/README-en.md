# multi-cluster-website

## Feature Introduction

openFuyao multi-cluster management as an extension component aims to provide efficient and flexible Kubernetes cluster cross-environment management, which can be installed to the openFuyao platform through the application marketplace. It supports:

Cluster list, providing a cluster list interface to view and manage the status and basic information of all clusters.
Cluster lifecycle management, simplifying the process of cluster expansion and destruction, supporting users to manage new clusters, adjust the scale of existing clusters, update cluster configurations, or safely cancel management of clusters that are no longer needed.
View cluster credentials: Supports users to use cluster credentials for quick access to clusters.
Cluster label management, supports users to quickly edit cluster labels and other operations.
Service cross-cluster access, provides cross-cluster access and other functions, supporting users to access services on managed clusters through a main cluster entry.

This project is the frontend implementation of the multi-cluster component.

Belongs to sig: [sig-container-platform](https://gitcode.com/openFuyao/sig-container-platform)

## Local Build

### Image Build

#### Build Parameters

- `COMMIT`: Hash value of the current git commit.
- `VERSION`: Component version.
- `SOURCE_DATE_EPOCH`: Timestamp of the image rootfs.

#### Build Commands

- Build and push to specified OCI repository.

  <details open>
  <summary>Using <code>docker</code></summary>

  ```bash
  docker buildx build . -f <path/to/dockerfile> \
      -o type=image,name=<oci/repository>:<tag>,oci-mediatypes=true,rewrite-timestamp=true,push=true \
      --platform=linux/amd64,linux/arm64 \
      --provenance=false \
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
      --build-arg=COMMIT=$(git rev-parse HEAD) \
      --build-arg=VERSION=0.0.0-latest \
      --build-arg=SOURCE_DATE_EPOCH=$(git log -1 --pretty=%ct)
  ```

  </details>

  Where `<path/to/dockerfile>` is the Dockerfile path `./build/Dockerfile`, `<oci/repository>` is the image address, and `<tag>` is the image tag.

- Build and export OCI Layout to local tarball.

  <details open>
  <summary>Using <code>docker</code></summary>

  ```bash
  docker buildx build . -f <path/to/dockerfile> \
      -o type=oci,name=<oci/repository>:<tag>,dest=<path/to/oci-layout.tar>,rewrite-timestamp=true \
      --platform=linux/amd64,linux/arm64 \
      --provenance=false \
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
      --build-arg=COMMIT=$(git rev-parse HEAD) \
      --build-arg=VERSION=0.0.0-latest \
      --build-arg=SOURCE_DATE_EPOCH=$(git log -1 --pretty=%ct)
  ```

  </details>

  Where `<path/to/dockerfile>` is the Dockerfile path `./build/Dockerfile`, `<oci/repository>` is the image address, `<tag>` is the image tag, and `path/to/oci-layout.tar` is the tar package path.

- Build and export image rootfs to local directory.

  <details open>
  <summary>Using <code>docker</code></summary>

  ```bash
  docker buildx build . -f <path/to/dockerfile> \
      -o type=local,dest=<path/to/output>,platform-split=true \
      --platform=linux/amd64,linux/arm64 \
      --provenance=false \
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
      --build-arg=COMMIT=$(git rev-parse HEAD) \
      --build-arg=VERSION=0.0.0-latest
  ```

  </details>

  Where `<path/to/dockerfile>` is the Dockerfile path `./build/Dockerfile` and `path/to/output` is the local directory path.

### Helm Chart Build

- Package Helm Chart.

  ```bash
  helm package <path/to/chart> -u \
      --version=0.0.0-latest \
      --app-version=openFuyao-v25.09
  ```

  Where `<path/to/chart>` is the Chart folder path.

- Push Chart package to specified OCI repository.

  ```bash
  helm push <path/to/chart.tgz> oci://<oci/repository>:<tag>
  ```

  Where `<path/to/chart.tgz>` is the Chart package path, `<oci/repository>` is the Chart package push address, and `<tag>` is the Chart package tag.

## Installation Instructions

### Prerequisites

The multi-cluster management component is installed as an extension component to the openFuyao platform. The hardware, software, and network requirements are consistent with the overall installation requirements of the openFuyao platform.

### Start Installation

1. Log in to the openFuyao platform management interface.
2. Select "Application Marketplace > Application List" in the left navigation bar of the openFuyao platform to enter the "Application Marketplace" interface.
3. Check the "Extension Component" application type on the left to view all extension components. Or enter "multi-cluster-service" in the "Search Box".
4. Click the "multi-cluster-service" card to enter the extension component "Details" interface.
5. Click the "Deploy" button to enter the "Deploy" interface.
6. Enter "Application Name", select "Installation Version" and "Namespace".
7. Enter the values information to be deployed in the parameter configuration Values.yaml.
8. Click the "Deploy" button to complete the deployment.
9. Click "Extension Component Management" in the left navigation bar to manage the multi-cluster extension component.
> **Note:**
> - After cluster node restart, the karmada service may become unavailable, please operate node restart with caution.
> - If node restart causes the karmada service to become unavailable, please uninstall the multi-cluster component and reinstall it.
> - The current multi-cluster management component can only be installed as an extension component on the openFuyao container platform, and does not support independent helm-chart deployment for now.

## Usage Instructions

For detailed usage instructions, please refer to the [User Guide](https://docs.openfuyao.cn/en/docs/v26.03/user_guide/multi_cluster_management.html)
