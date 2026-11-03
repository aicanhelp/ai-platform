
# Online/Offline Colocation

Online/Offline colocation is a complete set of cloud-native hybrid deployment and resource overselling solutions provided by openFuyao. Through a unified colocation management platform and user-friendly visualization interface, it supports online/offline business hybrid deployment, ensuring scheduling of online business during peak usage periods while enabling offline business to use oversold resources during online business low-peak periods, significantly improving cluster resource utilization and business deployment density. This repository is the online/offline colocation backend component of openFuyao, providing service interfaces externally, including colocation monitoring information interfaces, and colocation node management addition, removal, and colocation scheduling policy configuration.

## Feature Introduction

### Core Capabilities

Provides basic colocation capabilities and advanced colocation features in online/offline colocation:

**Basic Colocation Capabilities**:

- **Business Multi-QoS Level Management**: Supports three-level QoS classification: LLS (high latency sensitive), LS (latency sensitive), and BE (best effort).
- **Colocation Scenario Scheduling Enhancement**: Ensures that offline business is scheduled to nodes using oversold resources.
- **Node Colocation Engine Based on Rubik**: Supports configuring water level eviction functionality.
- **Unified Configuration Management**: Uses unified configmap for colocation configuration management (node management, volcano, and rubik configuration management).
- **Oversold Resource Management**: Oversold agent deployment scope and oversold resource calculation correction.
- **Colocation Basic Monitoring**: Provides basic colocation monitoring capabilities.

**Advanced Colocation Features**:

- **NUMA Topology-Aware Scheduling Enhancement**: Supports NUMA affinity scheduling optimization.
- **Non-Intrusive Colocation Based on NRI Mechanism**: Utilizes containerd NRI mechanism to implement container resource management.
- **CPU Elastic Throttling**: Supports CPU resource dynamic adjustment.
- **Memory Asynchronous Reclamation**: Memory reclamation mechanism based on QoS level.
- **Memory Bandwidth Limitation**: Limits BE-level Pod's occupation of memory bandwidth and CPU cache.
- **PSI Interference Detection**: Interference detection and handling based on system pressure metrics.

### Performance Improvement Effects

Based on performance data verified through actual testing:

- **Resource Utilization Improvement**: In test scenarios, resource utilization is improved by 30%-50%, with QPS decrease not exceeding 5%.
- **NUMA Affinity Optimization**: After enabling NUMA affinity NRI plugin, in test scenarios, average latency is reduced by 40%+, and throughput is improved by 70%+.
- **Colocation Stability**: Through QoS classification and water level eviction mechanisms, ensures stable performance of online business.

### Capability Scope

This feature consists of the following core components (code repositories), providing a complete colocation solution:

+ **colocation-management**: Colocation workload admission control and oversold resource management.
+ **colocation-service**: Colocation monitoring information interface and colocation management backend service.
+ **colocation-website**: Colocation statistics visualization and configuration management frontend interface.

Main functions include:

+ Supports priority-level scheduling and load balancing scheduling for businesses with different QoS levels.
+ Supports CPU and memory QoS suppression of online business by offline business on a single machine.
+ Supports eviction and rescheduling of offline business based on CPU/memory water level on a single machine.
+ Supports advanced colocation features such as CPU elastic throttling, memory asynchronous reclamation, memory bandwidth limitation, and PSI interference detection.
+ Provides cluster-level and node-level colocation resource monitoring and visualization display.
+ Supports viewing node colocation capabilities, configuring colocation policy parameters through Web interface.
+ Provides user-friendly way to view and enable/disable advanced colocation features.
+ Supports one-click enable/disable of colocation nodes, with real-time viewing of operation results.

## Installation Instructions

### Prerequisites

1. **System Environment Requirements**
   - Kubernetes v1.21 and above version.
   - containerd v1.7.0 and above version.
   - kube-prometheus v1.19 and above version.
   - volcano v1.9.0 and above version.

2. **Install Volcano Scheduler**
   
   ```shell
   helm repo add volcano-sh https://volcano-sh.github.io/helm-charts
   helm repo update
   helm install volcano volcano-sh/volcano --version 1.9.0 -n volcano-system --create-namespace
   ```

3. **Configure Kubelet** (if supporting LS-level Pod core binding)
   - Configure CPU management policy as static.
   - Enable NUMA affinity policy.

4. **Configure containerd NRI Function**
   Edit `/etc/containerd/config.toml` to add NRI configuration:
   
   ```toml
   [plugins."io.containerd.nri.v1.nri"]
   disable = false
   disable_connections = false
   plugin_config_path="/etc/nriconf.d"
   plugin_path="/opt/nri/plugins"
   ```

### Start Installation

#### Installation Through openFuyao Platform

1. Login to openFuyao platform, select ""Application Market > Applications".
2. Search for "colocation-package" extension component.
3. Click **Deploy**, configure application name, version, and namespace.
4. Configure deployment parameters in Values.yaml.
5. Complete deployment.
6. Manage this component in **Extension Management**.

#### Standalone Deployment

1. Pull helm package:
   
   ```shell
   helm pull oci://cr.openfuyao.cn/charts/colocation-package --version xxx
   ```

2. Extract and modify configuration:
   
   ```shell
   tar -zxvf colocation-package-xxx.tgz
   vim colocation-package/values.yaml
   ```

3. Install:
   
   ```shell
   helm install colocation-package ./
   ```

### openFuyao Platform Usage

After deployment is completed, you can access the following functions through "Computing Power Optimization Center > Online/Offline Colocation" on the openFuyao platform:

- **Overview**: View the online/offline colocation workflow.
- **Colocation Policy Configuration**: Manage colocation nodes and configure scheduling policy parameters.
- **Colocation Monitoring**: View cluster-level and node-level colocation monitoring data.

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

## Security Capability Description

openFuyao v26.06 and earlier versions only provide basic security features: mutual authentication between services (mTLS) and component-level user authentication and authorization (RBAC) require users/integration partners to adapt and harden according to their own deployment environment (such as introducing cert-manager, implementing unified certificate management, deploying authentication middleware, configuring NetworkPolicy to restrict access scope, etc.). The current focus is on basic feature delivery; security protection needs to be strengthened by users themselves, and security capabilities will be gradually planned and implemented in subsequent versions.