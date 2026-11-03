# NPU Operator

## Feature Introduction
Kubernetes provides access to special hardware resources (such as Ascend NPU) through device plugins. However, configuring and managing nodes with these hardware resources requires configuring multiple software components (such as drivers, container runtimes, or other libraries), which are complex, difficult to install, and error-prone. NPU Operator uses the Operator Framework in Kubernetes to automatically manage all software components required to configure Ascend devices. These components include Ascend drivers and firmware, enabling full-process cluster operation, supporting cluster job scheduling, operation and maintenance monitoring, fault recovery, and other functions of the MindCluster device plugin. By installing the corresponding components, NPU resource management, optimized scheduling of workloads, and containerization support for training and inference tasks can be achieved, enabling AI jobs to be deployed and run in container form on NPU devices.

## Documentation

- [Architecture Overview](docs/design/architecture.md) (in Chinese)
- [CRD Reference](docs/design/crd-reference.md) (in Chinese)
- [Reconcile Logic](docs/design/reconcile.md) (in Chinese)
- [Documentation Index & Maintenance Guide](docs/README.md) (in Chinese)
- [Changelog](CHANGELOG.md) (in Chinese)
- [Contributing Guide](CONTRIBUTING.md) (in Chinese)

## Installation Deployment
For deployment methods, please refer to the [NPU Operator User Guide](https://docs.openfuyao.cn/en/docs/v26.03/user_guide/npu_operator.html).

## Local Build

### Image Builden

#### Build Parameters

- `GOPRIVATE`: Configure Go language private repository, equivalent to the `GOPRIVATE` environment variable.
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

### NPU Exporter ServiceMonitor

The regular exporter ServiceMonitor is disabled by default, independently of its workload. To enable Prometheus Operator integration, set Helm values:

```yaml
exporter:
  enabled: true
  serviceMonitor:
    enabled: true
```

The corresponding CR fields are `spec.exporter.managed` and `spec.exporter.serviceMonitor.enabled`. Install the `monitoring.coreos.com/v1` ServiceMonitor CRD and create the `monitoring` namespace first. Configure Prometheus to select this ServiceMonitor.

Disabling the switch or exporter management removes only the ServiceMonitor owned by the current CR. Existing installations using it must explicitly enable the switch during upgrade. Update the NPUClusterPolicy CRD together with the Operator image to support the new field.
