# NPU Container Toolkit

## Feature Introduction
NPU Container Toolkit is a deployment tool for installing ascend-docker-runtime. ascend-docker-runtime is a container engine plugin that provides NPU containerization support for all AI workloads, enabling users to run AI workloads smoothly on Ascend devices in the form of Docker containers.

## Installation Deployment
This component is one of the components installed by openFuyao NPU Operator features and can also be installed independently. For deployment methods, please refer to [NPU Operator User Guide](https://docs.openfuyao.cn/en/docs/v26.03/user_guide/npu_operator.html) and [Component Deployment yaml](https://gitcode.com/openFuyao/npu-operator/tree/master/assets/oci-runtime).

## Local Build

### Image Build

#### Build Parameters

- `COMMIT`: Hash value of current git commit.
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
      --build-arg=COMMIT=$(git rev-parse HEAD) \
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
      --build-arg=COMMIT=$(git rev-parse HEAD) \
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
