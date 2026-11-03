# application-management-service

## Feature Introduction

Application management is an application management tool provided by the openFuyao platform, offering complete deployment, update, and uninstallation functions. The current version supports application installation through Chart packages using Helm. In application management, the entire application list can be displayed and subsequent O&M management operations can be performed.

### Application Scenarios

Application management currently supports complete O&M operations for Helm format release applications.

- Application information acquisition: Acquires application name, status, update time information in list form, supports filtering and sorting, and the detail page provides more information such as yaml, resource list, events, monitoring, etc.
- Application lifecycle management: Performs installation, uninstallation, upgrade, and rollback operations on various applications installed in the cluster.
- Supports installation after uploading various forms of Helm Chart packages, and modification of related setting parameters for applications.

### Capability Scope

- Basic information: Supports displaying application name, status, creation time, application template, template version, and update time. The application list page supports supports sorting by name and update time, and filtering based on installation status.
- O&M management: Supports application upgrade, and uninstallation. The upgrade function only supports applications installed through the openFuyao application marketplace. Supports application resource display (such as logs, events, etc.), and configuration modification (YAML form).

## Usage

In the left navigation bar of the openFuyao platform, click "Application Management" to enter the "Application Management" interface, where you can view deployed application instances, including application name, status, and update time. Click "Application Name" to enter the "Application Details" interface to view detailed information about the application, including details, YAML, resources, logs, events, and monitoring.

### Application Operations

#### Prerequisites

- Kubernetes version 1.28 or above is installed in the cluster.
- Helm version 3.14.2 or above is installed in the cluster.
- marketplace-service is running normally in the cluster.

#### Usage Limitations

The software format to be installed is Helm Chart, currently only supports installation of software uploaded to the repository.

#### Application Upgrade

1. In the "Application Details" interface, click "Operation" and select "Upgrade" to enter the "Application Upgrade" interface.
2. Click the "Version Information" dropdown box to select the specified version to upgrade.
3. Click "Confirm" to complete the upgrade.

#### Application Uninstallation and Rollback

1. Application uninstallation: Supports uninstallation of installed applications.
2. Application rollback: Supports rollback to historically installed versions, identified by version number and sequence number.

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