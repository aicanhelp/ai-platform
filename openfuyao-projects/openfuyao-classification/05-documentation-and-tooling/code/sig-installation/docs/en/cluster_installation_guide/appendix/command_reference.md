# Command Reference

This document contains the command reference for Cluster-API installation and deployment.

## Build Installation Package

- Command

  ```shell
  bke build 
  ```

- Command Function

   Build the installation package, collecting binary files, RPM packages, chart packages, etc. to construct the deployment package.

- Command Format

  1. Build the deployment package based on a YAML file.

      ```shell
     bke build -f <yaml file path, eg: offline-artifacts.yaml> -t <target file path, eg: bke.tar.gz>
     ```

  2. View command help.

     ```shell
     bke build -h
     ```

- Parameter Description

  | Parameter Name | Description                         |
  |---------------|-----------------------------|
  | `-f`, `--file` | Specifies the configuration file path.               |
  | `-h`, `--help` | Displays command help information.               |
  | `-t`, `--target` | Specifies the output path for the packaged BKE file.        |
  | `--skip-tls-verify` | When YAML is not configured, image sync skips TLS verification. |

- Execution User

   This command is executed by the root user.

- Prerequisites

   bkeadm has been installed.

- Notes

   None

## Build Incremental Package

- Command

  ```shell
  bke build patch
  ```

- Command Function

   Build a BKE incremental package (images, binaries, Charts, etc.) according to a YAML file, outputting a `tar.gz` that can be distributed offline.

- Command Format

  1. Build an incremental package in OCI format (can work without Docker).

     ```shell
     bke build patch -f <yaml, eg: offline-incremental.yaml> -t <output, eg: patch-log.tar.gz> --strategy oci
     ```

  2. Skip image sync TLS verification when YAML is not configured.

     ```shell
     bke build patch -f offline-incremental.yaml -t patch-log.tar.gz --strategy oci --skip-tls-verify
     ```

  3. View command help.

     ```shell
     bke build patch -h
     ```

- Parameter Description

  | Parameter Name | Description |
  |---------------|-----------------------------|
  | `-f`, `--file` | Incremental package configuration file path (required). |
  | `-t`, `--target` | Incremental package output path. |
  | `-s`, `--strategy` | Image sync strategy; specify as `oci` when building an incremental package. |
  | `--skip-tls-verify` | Skip source repository TLS verification when YAML is not configured. |
  | `-h`, `--help` | Displays command help information. |

- Execution User

   This command is executed by the root user.

- Prerequisites

   bkeadm has been installed.

- Notes

   `bke build patch` only collects binaries, Charts, etc. into the incremental package; it does not import them into the existing system packages/binary repository (HTTP repository) or Chart Registry on the bootstrap node. Importing images requires separately executing `bke registry patch`, and binaries and Charts must also be manually imported according to their respective procedures.

## Image Sync

- Command

  ```shell
  bke registry sync
  bke registry patch
  ```

- Command Function

   - `bke registry sync`: Sync a single image, batch images, or an entire repository between two image repositories by data block.
   - `bke registry patch`: Import images from the incremental directory produced by `bke build patch` and extracted, into the target image repository (typically the bootstrap node `40443`).

- Command Format

  1. Sync a single image (multi-architecture).

     ```shell
     bke registry sync --source docker.io/library/busybox:1.35 --target 127.0.0.1:40443/library/busybox:1.35 --multi-arch
     ```

  2. Sync a single image (specified architecture, mutually exclusive with `--multi-arch`).

     ```shell
     bke registry sync --source docker.io/library/busybox:1.35 --target 127.0.0.1:40443/library/busybox:1.35 --arch arm64
     ```

  3. Batch sync by list.

     ```shell
     bke registry sync --source docker.io/library -f image-list.txt --target 127.0.0.1:40443/library
     ```

  4. Repository-level sync.

     ```shell
     bke registry sync --source 192.168.200.240:40443 --target 192.168.200.245:40443 --sync-repo
     ```

  5. Sync the built incremental image directory.

     ```shell
     tar -xzvf patch-log.tar.gz
     bke registry patch --source ./patch-log --target 127.0.0.1:40443
     ```

  6. View command help.

     ```shell
     bke registry sync -h
     bke registry patch -h
     ```

- Parameter Description

   **`bke registry sync`**

   | Parameter Name | Description |
   |---------------|-----------------------------|
   | `--source` | Source image/repository address (required). |
   | `--target` | Target image/repository address (required). |
   | `-f`, `--file` | Batch image list file, one `name:tag` per line. |
   | `--multi-arch` | Sync multi-architecture manifest. |
   | `--arch` | Sync only the specified architecture; mutually exclusive with `--multi-arch`. |
   | `--src-tls-verify` | Source repository TLS verification. |
   | `--dest-tls-verify` | Target repository TLS verification. |
   | `--sync-repo` | Repository-level full repository sync. |
   | `-h`, `--help` | Displays command help information. |

   **`bke registry patch`**

   | Parameter Name | Description |
   |---------------|-----------------------------|
   | `--source` | Incremental package extracted directory (required). |
   | `--target` | Target image repository address, default `127.0.0.1:40443`. |
   | `-h`, `--help` | Displays command help information. |

- Execution User

   This command is executed by the root user.

- Prerequisites

   1. bkeadm has been installed.
   2. The target image repository is accessible (the common port for the bootstrap node local repository is `40443`).
   3. Before using `registry patch`, you must first complete the incremental package build, transfer, and extraction.

- Notes

   1. Recommended chain: `bke build patch` → transfer and extract → `bke registry patch`; use `bke registry sync` to supplement the repository with a single external image.
   2. `bke registry patch` only imports incremental images; it does not import `volumes/charts/*.tgz`, nor does it import binaries/system packages into the HTTP repository.

## View Image Repository Contents

- Command

  ```shell
  bke registry view
  ```

- Command Function

   View information such as image paths, tags, and architectures in the image repository; optionally export the image list.

- Command Format

  1. View the image repository (recommended: do not write the protocol, it will automatically use HTTPS and skip certificate verification, suitable for local self-signed repositories).

     ```shell
     bke registry view 127.0.0.1:40443
     ```

  2. Filter by prefix and limit the number of tags displayed per image.

     ```shell
     bke registry view 127.0.0.1:40443 --prefix kubernetes --tags 3
     ```

  3. Export the image list to the current directory.

     ```shell
     bke registry view 127.0.0.1:40443 --export
     ```

  4. View command help.

     ```shell
     bke registry view -h
     ```

- Parameter Description

  | Parameter Name | Description |
  |---------------|-----------------------------|
  | Positional parameter (repository address) | Image repository address (required), for example `127.0.0.1:40443`. |
  | `--prefix` | Filter by image path prefix. |
  | `--tags` | Maximum number of tags to display per image, default `3`. |
  | `--export` | Export the image list to the current directory; the file name is automatically generated as `{architecture}_image-list.txt` (e.g. `amd64_image-list.txt`), and cannot be customized. |
  | `-h`, `--help` | Displays command help information. |

- Execution User

   This command is executed by the root user.

- Prerequisites

   1. bkeadm has been installed.
   2. The target image repository is accessible.

- Notes

   1. The repository address for `bke registry view` should **not** be written as `https://127.0.0.1:40443`. When `https://` is written manually, system certificate verification is performed, and local self-signed repositories will report unauthenticated certificates; when no protocol is specified, the tool automatically adds `https://` and skips certificate verification.
   2. `--tags` defaults to `3`; whether or not you write `--tags 3`, the effect is the same; when you need to view more tags, increase this value.
   3. `--export` cannot specify the export file name; if you need a custom name after export, you can rename it yourself.

## Initialize Bootstrap Node

- Command

  ```shell
  bke init
  ```

- Command Function

   Initialize the bootstrap node, including node checks, repository startup, cluster installation, and other operations.

- Command Format

  1. Initialize the bootstrap node.

     ```shell
     bke init
     ```

  2. Initialize the bootstrap node and deploy the management cluster.

     ```shell
     bke init --file <cluster configuration file path, eg: /root/openfuyao/bkecluster.yaml>
     ```

  3. Specify the installation image and use the default image source.

     ```shell
     bke init --onlineImage <deploy image, eg: harbor.openfuyao.com/openfuyao/bke-online-installed:latest>
     ```

  4. Customize the image source and software source.

     ```shell
     bke init --otherRepo <other repo, eg: harbor.openfuyao.com/openfuyao> --otherSource <other source, eg: http://192.168.1.120:40080>
     ```

  5. View command help.

     ```shell
     bke init -h
     ```

- Parameter Description

  | Parameter Name                | Description                                                                 |
  |:----------------------|:-------------------------------------------------------------------|
  | `-f`, `--file`        | Specifies the `bkecluster.yaml` configuration file path (required).                          |
  | `-h`, `--help`        | Displays command help information.                                                   |
  | `--chartRepoPort`     | Sets the Helm Chart repository port (default: `38080`).                           |
  | `--clusterAPI`        | Specifies the `cluster-api-bke` version (default: `v25.03`).                       |
  | `--confirm`           | Skip confirmation prompts and execute directly.                                             |
  | `--hostIP`            | Specifies the IP address of the local Kubernetes API Server (required).                   |
  | `--kubernetesPort`    | Sets the local Kubernetes API port (default: `36443`).                       |
  | `--ntpServer`         | Configures the NTP server (default: `cn.pool.ntp.org:123`, `local` means use local service). |
  | `--otherRepo`         | Specifies the image repository address (e.g. Helm Chart, container images, etc.).                        |
  | `--otherSource`       | Custom private source address for system software package installation.                                    |
  | `--runtime`           | Select the container runtime (`docker` or `containerd`, default: `containerd`).      |
  | `-s`, `--runtimeStorage` | Sets the container runtime storage directory (default: `/var/lib/docker` or `/var/lib/containerd`). |
  | `--yumRepoPort`       | Sets the YUM repository port (default: `40080`).                                 |

- Execution User

   This command is executed by the root user.

- Prerequisites

   1. bkeadm has been installed.
   2. The cluster configuration file has been prepared.

- Notes

   None.

## Cluster Installation

- Command

  ```shell
  bke cluster
  ```

- Command Function

   Cluster lifecycle management.

- Command Format

  1. Cluster installation and deployment.

     ```shell
     bke cluster create -f <cluster configuration file path, eg: /root/openfuyao/master.yaml> -n <node configuration file path, eg: /root/openfuyao/nodes.yaml>
     ```

  2. View command help.

     ```shell
     bke cluster -h
     ```

- Parameter Description

  | Parameter Name      | Description                   |
  |:----------| :--------------------- |
  | `-f`, `--file` | Specifies the `bkecluster.yaml` configuration file path (required).|
  | `-n`, `--node` | Specifies the `bkenode.yaml` configuration file path (required).|
  | `-h`, `--help` | Command help.             |

- Execution User

   This command is executed by the root user.

- Prerequisites

   1. bkeadm has been installed.
   2. The cluster configuration file has been prepared.

- Notes

   None.

## Cluster Uninstallation

- Command

  ```shell
  bke reset
  ```

- Command Function

   Clean up bootstrap node services and restore nodes to the initial state before BKE installation. For the deletion scope, high-risk impacts, and success criteria of each parameter, see [Cluster Uninstallation Overview](../../cluster_uninstallation/overview.md) and [Delete Bootstrap Cluster via Backend (Command Line)](../../cluster_uninstallation/delete_bootstrap_cluster_command.md).

- Command Format

  1. Delete the local kubernetes service.

     ```shell
     bke reset
     ```

  2. Clear the bootstrap node and mounted directories.

     ```shell
     bke reset --mount
     ```

  3. Clear node containers and the runtime environment.

     ```shell
     bke reset --all
     ```

  4. Clear node services and skip the deletion confirmation.

     ```shell
     bke reset --all --confirm
     ```

  5. View command help.

     ```shell
     bke reset -h
     ```

- Parameter Description

  | Parameter Name       | Description                         |
  |---------------|-----------------------------|
  | `--all`       | Restore nodes.                     |
  | `--confirm`   | Skip all deletion confirmations.               |
  | `-h`, `--help`| Displays reset command help information.           |
  | `--mount`     | Remove the extraction directory and related services.         |

- Execution User

   This command is executed by the root user.

- Prerequisites

   1. bkeadm has been installed.
   2. The cluster has been prepared.

- Notes

   None.
