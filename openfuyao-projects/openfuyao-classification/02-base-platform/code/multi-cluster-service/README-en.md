# multi-cluster-service

## Feature Introduction

openFuyao multi-cluster management as an extension component aims to provide efficient and flexible Kubernetes cluster cross-environment management, which can be installed to the openFuyao platform through the application market. It supports:

Cluster list, providing a cluster list interface to view and manage the status and basic information of all clusters.
Cluster lifecycle management, simplifying the process of cluster expansion and destruction, supporting users to onboard new clusters, adjust the scale of existing clusters, update cluster configurations, or securely cancel clusters that are no longer needed.
View cluster credentials, supporting users to use cluster credentials for quick access to clusters.
Cluster label management, supporting users to quickly edit cluster labels and other operations.
Cross-cluster service access, providing cross-cluster access and other functions, supporting users to access services on onboarded clusters from one main cluster entry.

This project is the backend implementation of the multi-cluster component.

Belonging to sig: [sig-container-platform](https://gitcode.com/openFuyao/sig-container-platform)

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

  Where `<path/to/dockerfile>` is the Dockerfile path `./build/Dockerfile`, `<oci/repository>` is the image address, and `<tag>` is the image tag.

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

  Where `<path/to/dockerfile>` is the Dockerfile path `./build/Dockerfile`, `<oci/repository>` is the image address, `<tag>` is the image tag, and `path/to/oci-layout.tar` is the tar package path.

- Build and export image rootfs.

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

  Where `<path/to/dockerfile>` is the Dockerfile path `./build/Dockerfile` and `path/to/output` is the local directory path.

### Helm Chart Build

- Package Helm Chart.

  ```bash
  helm package <path/to/chart> -u \
      --version=0.0.0-latest \
      --app-version=openFuyao-v25.09
  ```

  Where `<path/to/chart>` is the Chart folder path

- Push Chart package to specified OCI repository.

  ```bash
  helm push <path/to/chart.tgz> oci://<oci/repository>:<tag>
  ```

  Where `<path/to/chart.tgz>` is the Chart package path, `<oci/repository>` is the Chart package push address, and `<tag>` is the Chart package tag.

## Installation Instructions

### Prerequisites

The multi-cluster management component is installed as an extension component to the openFuyao platform. The requirements for hardware, software, and network are consistent with the overall installation requirements of the openFuyao platform.

### Start Installation

1. Login to the openFuyao platform management interface.
2. In the left navigation bar of the openFuyao platform, select "Application Market > Application List" to enter the "Application Market" interface.
3. Select the application type "Extension Components" on the left to view all extension components. Or enter "multi-cluster-service" in the "Search box".
4. Click the "multi-cluster-service" card to enter the extension component "Details" interface.
5. Click the "Deploy" button to enter the "Deployment" interface.
6. Enter the "Application Name", select "Installation Version" and "Namespace".
7. Enter the values information to be deployed in the Values.yaml of parameter configuration.
8. Click the "Deploy" button to complete the deployment.
9. Click "Extension Component Management" in the left navigation bar to manage the multi-cluster extension component.
> **Note:**
> - After a cluster node restart, the karmada service may be unavailable. Please operate node restart with caution.
> - If the karmada service becomes unavailable due to a node restart, please uninstall the multi-cluster component and reinstall it.
> - Currently, the multi-cluster management component can only be installed as an extension component on the openFuyao container platform. Standalone helm chart deployment is not yet supported.

## Usage Instructions

For detailed usage instructions, please refer to [User Guide](https://docs.openfuyao.cn/en/docs/v26.03/user_guide/multi_cluster_management.html)

## Security Capabilities

openFuyao v26.06 and earlier versions only provide basic security features: mutual TLS (mTLS) between services and component-level user authentication and authorization (RBAC). Users and integration partners need to adapt and harden these capabilities according to their own deployment environments (e.g., introducing cert-manager, setting up unified certificates, deploying authentication middleware, and configuring NetworkPolicy to narrow the reachable scope). The current release focuses on delivering core functionality; security hardening is the responsibility of users. Security capabilities will be gradually planned and rolled out in subsequent versions.