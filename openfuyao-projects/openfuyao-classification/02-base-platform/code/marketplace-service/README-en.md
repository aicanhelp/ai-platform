# marketplace-service

## Feature Introduction

The openFuyao application marketplace mainly supports displaying applications and managing application repositories, providing convenient deployment services for platform extension capabilities and components.

### Application Scenarios

The openFuyao application marketplace is widely used for repository management and application deployment operations by users and administrators, providing multiple practical application scenarios:

- Platform extension capabilities: Significantly enhance the overall functionality of the platform by deploying openFuyao extension components.
- Quick deployment: Users can directly upload Helm packages to the openFuyao platform for quick deployment and testing operations.

### Capability Scope

- Application deployment: Users can select required applications in the application marketplace for deployment and view deployment status in subsequent management.
- Extension capability deployment: Select extension components through the application marketplace to help the platform enhance functionality and performance.
- Repository management: Support adding, deleting, and managing application repositories, facilitating quick access to new application repositories and flexible selection of deployable applications.
- Local repository: Support uploading multiple applications and multiple versions locally without uploading packages to the repository.

## Usage

The specific usage of the openFuyao application marketplace includes two major parts: using the application list and using package management. The former supports repository viewing, repository synchronization, repository addition, and other functions, while the latter supports managing package versions and adding packages.

### Prerequisites

- Kubernetes 1.28 and Helm 3.14.2 are installed in the cluster.
- marketplace-service is running normally in the cluster.

### View Repository

Repositories are used to store and manage applications and extension components, providing a centralized resource management entry. View basic information and resources contained in the repository through the "Repository Details" interface.

### Sync Repository

Used to update application or extension component information in the repository to ensure resource information is up-to-date.

### Add Repository

The add repository function is used to connect new application or extension component repositories to the openFuyao platform for unified management by openFuyao, and can also connect multiple types of resources.

### Manage Package Versions

Package management is used for version management of packages within the openFuyao platform, including adding new versions or deleting old versions.

### Add Package

Used to upload new application packages or component packages in the platform to provide support for subsequent version management and resource deployment.

## Local Build

### Image Build

#### Build Commands

- Build and push to specified OCI repository.

  <details open>
  <summary>Using <code>docker</code></summary>

  ```bash
  docker buildx build . -f <path/to/dockerfile> \
      -o type=image,name=<oci/repository>:<tag>,oci-mediatypes=true,rewrite-timestamp=true,push=true \
      --platform=linux/amd64,linux/arm64 \
      --provenance=false \
  ```

  </details>
  <details>
  <summary>Using <code>nerdctl</code></summary>

  ```bash
  nerdctl build . -f <path/to/dockerfile> \
      -o type=image,name=<oci/repository>:<tag>,oci-mediatypes=true,rewrite-timestamp=true,push=true \
      --platform=linux/amd64,linux/arm64 \
      --provenance=false \
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
  ```

  </details>
  <details>
  <summary>Using <code>nerdctl</code></summary>

  ```bash
  nerdctl build . -f <path/to/dockerfile> \
      -o type=oci,name=<oci/repository>:<tag>,dest=<path/to/oci-layout.tar>,rewrite-timestamp=true \
      --platform=linux/amd64,linux/arm64 \
      --provenance=false \
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
  ```

  </details>
  <details>
  <summary>Using <code>nerdctl</code></summary>

  ```bash
  nerdctl build . -f <path/to/dockerfile> \
      -o type=local,dest=<path/to/output>,platform-split=true \
      --platform=linux/amd64,linux/arm64 \
      --provenance=false \
  ```

  </details>

  Where `<path/to/dockerfile>` is the Dockerfile path and `path/to/output` is the local directory path.

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

## Security Capabilities

openFuyao v26.06 and earlier versions only provide basic security features: mutual TLS (mTLS) between services and component-level user authentication and authorization (RBAC). Users and integration partners need to adapt and harden these capabilities according to their own deployment environments (e.g., introducing cert-manager, setting up unified certificates, deploying authentication middleware, and configuring NetworkPolicy to narrow the reachable scope). The current release focuses on delivering core functionality; security hardening is the responsibility of users. Security capabilities will be gradually planned and rolled out in subsequent versions.