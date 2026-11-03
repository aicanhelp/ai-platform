# Online/Offline Colocation Extension Frontend

Online/Offline colocation is a complete cloud-native hybrid deployment and resource overselling solution provided by openFuyao. Through a unified colocation management platform and user-friendly visualization interface, it supports online/offline business hybrid deployment, ensuring scheduling of online business during peak usage periods while enabling offline business to use oversold resources during online business low-peak periods, significantly improving cluster resource utilization and business deployment density. This repository is the openFuyao online/offline colocation frontend component, supporting unified colocation strategy configuration and colocation monitoring viewing functions.

## Feature Introduction

For details, see https://gitcode.com/openFuyao/colocation-service/blob/master/README.md

## Installation Instructions

For details, see https://gitcode.com/openFuyao/colocation-service/blob/master/README.md

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
      -o type=local,dest=<path;to/output>,platform-split=true \
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