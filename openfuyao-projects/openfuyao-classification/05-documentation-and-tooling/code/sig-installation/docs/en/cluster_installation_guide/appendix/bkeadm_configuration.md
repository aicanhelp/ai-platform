# bkeadm Configuration Parameters

## Quick Reference for Common Configuration Scenarios

**Table 1** Quick reference for common configuration scenarios

| Configuration Scenario | Key Configuration Items | Reference Section                                                                                              |
|---|---|---------------------------------------------------------------------------------------------------|
| Initialize or deploy a cluster using YAML. | `bke init --file/-f` | `bke init Configuration Items` in this document; for YAML fields, see [cluster-api-provider-bke configuration parameters](cluster-api-provider-bke_configuration.md). |
| Initialize the bootstrap node and specify local repository ports. | `--imageRepoPort`, `--chartRepoPort`, `--yumRepoPort`, `--kubernetesPort`, `--consolePort` | `bke init Configuration Items` in this document.                                                                                  |
| Install online or specify a private repository. | `--onlineImage`, `--otherRepo`, `--otherSource`, `--otherChart` | `bke init Configuration Items` in this document.                                                                                  |
| Use a local image package to accelerate initialization. | `--imageFilePath`, `--otherRepo`, `--installConsole=false` | `bke init Configuration Items` in this document.                                                                                  |
| Create a cluster or nodes. | `cluster create --file/-f`, `cluster create --nodes/-n` | `bke cluster Configuration Items` in this document.                                                                               |
| Build an offline package or an incremental package. | `build --file/-f`, `build --target/-t`, `build patch --file/-f`, `build patch --target/-t` | `bke build and Build YAML Configuration` in this document.                                                                           |
| Synchronize, view, or import image repository contents. | `registry sync`, `registry view`, `registry patch` related parameters | `bke registry Configuration Items` in this document.                                                                              |

## bkeadm

### bkeadm Command Overview

**Table 2** bkeadm command overview configuration table

| Configuration Item Name                                                    | Default Value | Description | Setting Method |
|-------------------------------------------------------| --- | --- | --- |
| `bke --kubeconfig`                                    | None | Kubernetes kubeconfig path; a global parameter, required. | CLI persistent flag |
| `bke --doc`                                           | None | Prints command documentation; optional. | CLI persistent flag |
| `bke init/reset/start/version/registry/build/cluster` | None | Subcommands actually mounted under the current root command. | Code registration |

### `bke init` Configuration Items

**Table 3** bke init configuration items table

| Configuration Item Name | Default Value | Description | Setting Method |
| --- | --- | --- | --- |
| `--file/-f` | None | Specifies the BKECluster YAML; once provided, the repo, NTP, and other configurations within it are read and can be deployed directly; required; for configurable YAML fields, see [cluster-api-provider-bke configuration parameters](cluster-api-provider-bke_configuration.md). | CLI |
| `--domain` | `deploy.bocloud.k8s` | Local image repository domain; when empty, PreRun still fills in the default value; optional. | CLI |
| `--imageRepoPort` | `40443` | Local image repository port; optional; corresponds to a different local service than `--chartRepoPort` and `--yumRepoPort`; when customizing, they must not overlap. | CLI |
| `--hostIP` | None | Bootstrap node IP address; when empty, the egress/intranet IP address is automatically detected; required. | CLI/Auto-detection |
| `--kubernetesPort` | `36443` | Port exposed by the local management cluster Kubernetes API; optional. | CLI |
| `--consolePort` | `30010` | bke console NodePort; optional. | CLI |
| `--yumRepoPort` | `40080` | Local system package source (HTTP repo) service port, optional; independent from image repository port and chart repo port, must avoid conflicts when customized. | CLI |
| `--chartRepoPort` | `38080` | Local chart repo port, optional; independent from image repository port and yum repo port, must avoid conflicts when customized. | CLI |
| `--ntpServer` | `cn.pool.ntp.org:123` | NTP server; when the value is `local`, the local NTP service is started; optional. | CLI |
| `--onlineImage` | None | Online installation image that provides binaries and system packages; required. | CLI |
| `--otherRepo` | None | Private image repository prefix, for example `cr.openfuyao.cn/openfuyao`; required. | CLI/YAML reverse-derivation |
| `--otherSource` | None | Private system package source; must start with `http://`; required. | CLI/YAML reverse-derivation |
| `--otherChart` | None | Private Helm chart repository; required. | CLI/YAML reverse-derivation |
| `--clusterAPI` | `latest` | Fallback value for the cluster-api-provider-bke/manifests version; optional. | CLI/Version ConfigMap |
| `--confirm` | `false` | Skips interactive confirmation; optional. | CLI |
| `--oFVersion/-v` | `latest` | Target openFuyao version; optional. | CLI |
| `--versionUrl` | `https://openfuyao.obs.cn-north-4.myhuaweicloud.com/openFuyao/version-config/` | Online version configuration index URL; optional. | CLI |
| `--installConsole` | `true` | Whether to install bkeconsole; optional. | CLI |
| `--enableNTP` | `true` | Whether to enable NTP settings; when disabled, the NTP field in the generated configuration can be empty; optional. | CLI |
| `--imageRepoTLSVerify` | `true` | Image repository TLS verification switch; optional. | CLI |
| `--imageRepoCAFile` | None | Image repository CA file; required. | CLI |
| `--imageRepoUsername` | None | Image repository username; required. | CLI |
| `--imageRepoPassword` | None | Image repository password; required. | CLI |
| `--imageFilePath` | None | Local image tar package path; required; must be `.tar.gz/.tgz` and the file must exist; only used in the online installation optimization scenario where `--otherRepo` is specified; when used, `--otherRepo` must not be empty, and `--installConsole=false` must be set. | CLI |
| `--agentHealthPort` | `58080` | bkeagent health check listening port; must be 0-65535; optional. | CLI |

> **Note:** The local system package source service corresponding to `--yumRepoPort` has different names in different configuration locations but is actually the same service (default port `40080`):
>
> | Configuration Location | Name |
> | --- | --- |
> | `bke init` CLI | `--yumRepoPort` |
> | Cluster CRD | `spec.clusterConfig.cluster.httpRepo.port` |
> | Documentation | "HTTP repo" / "http repo" / "system package source" |
>
> Where "yum" refers to the consumer side (this service also serves as the node's yum/apt system package source), and "http" refers to the transport side (currently provides file distribution via HTTP protocol); in offline deployment, `--otherSource` points to this service.

Port relationships and conflict checking: `--imageRepoPort`, `--chartRepoPort`, and `--yumRepoPort` are used for the local image repository, chart repo, and yum/http repo respectively, and are three independent service ports. When customizing any port, you must avoid the other two ports, as well as `--kubernetesPort`, `--consolePort`, and any ports already occupied by other processes on the host. The current initialization preflight checks whether a port is already occupied on the host and filters out ports of existing BKE management containers; however, it does not explicitly check whether these parameters are filled with the same port. If configured with duplicates, the preflight may pass, but subsequent service startup or port publishing will fail. You need to adjust them to different ports based on the port occupation information in the startup logs and retry.

### `bke cluster` Configuration Items

**Table 4** bke cluster configuration items table

| Configuration Item Name | Default Value | Description | Setting Method |
| --- | --- | --- | --- |
| `cluster list` | None | Lists clusters; no dedicated flag; optional. | CLI |
| `cluster create --file/-f` | None | BKECluster YAML; required. | CLI |
| `cluster create --nodes/-n` | None | BKENode YAML; required. | CLI |

### `bke build` and Build YAML Configuration

**Table 5** bke build and build YAML configuration items table

| Configuration Item Name | Default Value | Description                          | Setting Method |
| --- | --- |-----------------------------| --- |
| `build --file/-f` | None | Build configuration file; required.                   | CLI |
| `build --target/-t` | None | Output BKE package path; required.               | CLI |
| `build --skip-tls-verify` | `false` | When YAML is not configured, image synchronization skips TLS verification; optional. | CLI |
| `build patch --file/-f` | None | Incremental package configuration file; required.                  | CLI |
| `build patch --target/-t` | None | Incremental package output path; required.                  | CLI |
| `build patch --strategy/-s` | `oci` (recommended by documentation) | Image synchronization strategy; when building an incremental package, specify `oci` (can work without Docker). | CLI |
| `build patch --skip-tls-verify` | `false` | Skips source repository TLS verification when YAML is not configured; optional. | CLI |

### `bke registry` Configuration Items

**Table 6** bke registry configuration items table

| Configuration Item Name | Default Value | Description                          | Setting Method |
| ----------------------------------- | --- |-----------------------------| --- |
| `registry sync --file/-f` | None | Batch image list file, one `name:tag` per line; required.   | CLI |
| `registry sync --multi-arch` | `false` | Synchronizes multi-architecture manifests; optional.          | CLI |
| `registry sync --source` | None | Source image/repository; required.                   | CLI |
| `registry sync --target` | None | Target image/repository; required.                  | CLI |
| `registry sync --src-tls-verify` | `false` | Source repository TLS verification; optional; leaving TLS verification disabled by default poses security risks. In production environments, it is recommended to explicitly set it to `true`, and only disable it when the source repository's trusted certificate chain is unavailable or when using an intranet self-signed repository. | CLI |
| `registry sync --dest-tls-verify` | `false` | Target repository TLS verification; optional; leaving TLS verification disabled by default poses security risks. In production environments, it is recommended to explicitly set it to `true`, and only disable it when the target repository's trusted certificate chain is unavailable or when using an intranet self-signed repository. | CLI |
| `registry sync --arch` | None | Specifies the architecture; mutually exclusive with `--multi-arch`: `--multi-arch` means synchronizing the complete multi-architecture manifest, while `--arch` means synchronizing only the specified architecture; specifying both at the same time will cause the command to error and exit before execution; required. | CLI |
| `registry sync --sync-repo` | `false` | Repository-level synchronization; optional.                   | CLI |
| `registry sync --doc` | None | Hint; optional.                      | CLI |
| `registry view --prefix` | None | Image path prefix when viewing the repository; optional.             | CLI |
| `registry view --tags` | `3` | Maximum number of tags to display per image; optional; the default is `3`.          | CLI |
| `registry view --export` | `false` | Exports the image list to the current directory; optional; the file name is automatically generated as `{architecture}_image-list.txt` and cannot be customized.              | CLI |
| `registry patch --source` | None | Incremental package directory; required.                    | CLI |
| `registry patch --target` | `127.0.0.1:40443` | Target image repository address; optional.               | CLI |

Examples:

#### Synchronizing a Single Image

```bash
   bke registry sync --source docker.io/library/busybox:1.35 --target 192.168.200.240:40443/library/busybox:1.35 --multi-arch
```

Success criteria: The output contains `Sync image docker.io/library/busybox:1.35 to 192.168.200.240:40443/library/busybox:1.35 success`, indicating that the source image has been copied to the target repository; if the source image is a multi-architecture image, you may also see process logs for synchronizing architecture-specific images first, followed by manifest creation.

   Exception handling: When the output contains `Sync image ... failed`, `by registry failed`, or certificate, network, or authentication-related errors, first check whether the source/target repository addresses are accessible, whether the image tag exists, and whether the repository credentials are correct; in production environments, also confirm whether you need to explicitly add `--src-tls-verify=true`, `--dest-tls-verify=true`, and whether the target repository certificate chain is trusted.

#### Synchronizing Multiple Images

```bash
   cat > image-list.txt <<'EOF'
   busybox:1.28
   alpine:3.14
   EOF

   bke registry sync --source docker.io/library -f image-list.txt --target 192.168.200.240:40443/library
```

Success criteria: The output displays synchronization logs per image, and finally outputs `Image list sync completed: total=2, success=2, failed=0`. Here `total` indicates the number of images in the file, `success` indicates the number of successfully synchronized images, and `failed=0` indicates that there are no failures.

   Exception handling: If `failed` is greater than 0, it means at least one image in the list failed to synchronize; the output will also contain the corresponding `Sync image ... failed` for that image. You need to check each failed image to verify whether the `name:tag` in `image-list.txt` is correct, whether the source repository has that tag, whether the target repository has write permissions, and whether the network, TLS, and authentication configurations meet the requirements.

#### Viewing the Image List of the Current Image Repository

```bash
   bke registry view 192.168.200.240:40443 --prefix kubernetes --tags 3
```

Success criteria: The output displays image information in a table with column headers including `IMAGE`, `TAGS`, `ARCHITECTURE`, `CREATE TIME`, and `SIZE`. Being able to see image names, tags, architectures, creation times, and sizes matching `--prefix kubernetes` indicates that repository access and manifest parsing succeeded.

   Exception handling: If the output is `no repositories found`, it means the repository is accessible but has no images to display; if `View repositories ... failed`, `get tags failed`, `unmarshal tags failed`, or `view failed` appears, you need to check whether the repository address is accessible, whether the repository supports the Docker Registry V2 API, whether `--prefix` filtered out the target images, and whether the authentication and network configurations are correct.

> ![icon](../figures/icon-note.gif) **Note:**<br />When viewing a local self-signed image repository, **do not** manually prepend `https://` (e.g., `https://127.0.0.1:40443`), otherwise system certificate verification will be triggered and may report the certificate as unauthenticated. It is recommended to omit the protocol and use `127.0.0.1:40443` directly: the tool will automatically append `https://` and skip certificate verification. `--export` exports to `{architecture}_image-list.txt` based on the architecture; a custom file name cannot be specified. For more command details, see [Command Reference](command_reference.md).

#### Repository-Level Synchronization

```bash
   bke registry sync  --source 192.168.200.240:40443  --target 192.168.200.245:40443  --sync-repo
```

Success criteria: The output first displays the synchronization process for each image, and finally outputs `Repo sync completed: total=<count>, success=<count>, failed=0` and `Sync repo 192.168.200.240:40443 to 192.168.200.245:40443 success`. Here `failed=0` indicates that there are no failed images in this repository-level synchronization.

   Exception handling: If reading the repository catalog fails, `Failed to get repo catalog`, `Failed to unmarshal repo catalog`, or `No repositories found` will appear; if some images fail to synchronize, `Failed to sync image ...` will appear, and the failure count will be reflected in the summary. In this case, first confirm that the source repository `/v2/_catalog` is accessible, then check each failed image for tag, architecture image suffix, target repository write permissions, repository capacity, network connectivity, and TLS/authentication configuration.

### `bke start` Configuration Items

**Table 7** bke start configuration items table

| Configuration Item Name | Default Value | Description | Setting Method |
| --- | --- | --- | --- |
| `start ntpserver --systemd` | `false` | Starts the NTP service using systemd; optional. | CLI |
| `start ntpserver --foreground` | `false` | Starts the NTP service in the foreground; optional. | CLI |

### `bke reset` Configuration Items


**Table 8** bke reset configuration items table

| Configuration Item Name | Default Value | Description | Setting Method |
|---|---|---|---|
| `reset` | None | Cleans up containers and directories related to the local Kubernetes/k3s transition cluster; does not trigger interactive confirmation. | CLI |
| `reset --all` | `false` | On top of a normal `reset`, further clears node container services, NTP services, container runtimes, networks, and related residual files; when `--confirm` is not specified, the user is required to confirm before execution; if the user cancels, execution does not continue; optional. | CLI |
| `reset --mount` | `false` | Deletes the extraction directory; when used alone, it also cleans up container services, NTP services, and software source configurations, and does not trigger interactive confirmation; when used with `--all`, it also removes BKE-managed system configurations and restores the timezone; optional. | CLI |
| `reset --confirm` | `false` | Only used in the `--all` scenario; skips the pre-execution confirmation of `bke reset --all` or `bke reset --all --mount`; does not change the behavior of a normal `reset` or `--mount` alone; optional. | CLI |
