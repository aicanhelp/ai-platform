# 命令参考

本文档是Cluster-API安装部署的命令参考内容。

## 构建安装部署包

- 命令

  ```shell
  bke build 
  ```

- 命令功能

   构建安装部署包，收集二进制文件、rpm包、chart包等进行部署包的构建。

- 命令格式

  1. 根据yaml构建部署包。

     ```shell
     bke build -f <yaml file path, eg: offline-artifacts.yaml> -t <target file path, eg: bke.tar.gz>
     ```

  2. 查看命令帮助。

     ```shell
     bke build -h
     ```

- 参数说明

  | 参数名称       | 描述                         |
  |---------------|-----------------------------|
  | `-f`, `--file` | 指定配置文件路径。               |
  | `-h`, `--help` | 显示命令帮助信息。               |
  | `-t`, `--target` | 指定打包BKE文件输出路径。        |
  | `--skip-tls-verify` | YAML 未配置时，镜像同步跳过 TLS 校验。 |

- 执行用户

  该命令是由root用户执行。

- 使用前提

  已安装好bkeadm。

- 注意事项

  无

## 构建增量包

- 命令

  ```shell
  bke build patch
  ```

- 命令功能

  按 YAML 构建 BKE 增量包（镜像、二进制、Chart 等），输出可离线分发的 `tar.gz`。

- 命令格式

  1. 以 OCI 格式构建增量包（可不依赖 Docker）。

     ```shell
     bke build patch -f <yaml, eg: offline-incremental.yaml> -t <output, eg: patch-log.tar.gz> --strategy oci
     ```

  2. YAML 未配置时跳过镜像同步 TLS 校验。

     ```shell
     bke build patch -f offline-incremental.yaml -t patch-log.tar.gz --strategy oci --skip-tls-verify
     ```

  3. 查看命令帮助。

     ```shell
     bke build patch -h
     ```

- 参数说明

  | 参数名称 | 描述 |
  |---------------|-----------------------------|
  | `-f`, `--file` | 增量包配置文件路径（必填）。 |
  | `-t`, `--target` | 增量包输出路径。 |
  | `-s`, `--strategy` | 镜像同步策略；构建增量包时指定为 `oci`。 |
  | `--skip-tls-verify` | YAML 未配置时跳过源仓 TLS 校验。 |
  | `-h`, `--help` | 显示命令帮助信息。 |

- 执行用户

  该命令是由root用户执行。

- 使用前提

  已安装bkeadm。

- 注意事项

  `bke build patch` 只负责把二进制、Chart 等收集进增量包，不会把它们导入引导节点已有的系统包/二进制仓库（HTTP 仓）或 Chart Registry；导入镜像需另行执行 `bke registry patch`，二进制与 Chart 也需按对应流程手动导入。

## 镜像同步

- 命令

  ```shell
  bke registry sync
  bke registry patch
  ```

- 命令功能

  - `bke registry sync`：在两个镜像仓库之间按数据块同步单个镜像、批量镜像或整仓。
  - `bke registry patch`：将 `bke build patch` 产出并解压后的增量目录中的镜像导入目标镜像仓（通常为引导节点 `40443`）。

- 命令格式

  1. 同步单个镜像（多架构）。

     ```shell
     bke registry sync --source docker.io/library/busybox:1.35 --target 127.0.0.1:40443/library/busybox:1.35 --multi-arch
     ```

  2. 同步单个镜像（指定架构，与 `--multi-arch` 互斥）。

     ```shell
     bke registry sync --source docker.io/library/busybox:1.35 --target 127.0.0.1:40443/library/busybox:1.35 --arch arm64
     ```

  3. 按列表批量同步。

     ```shell
     bke registry sync --source docker.io/library -f image-list.txt --target 127.0.0.1:40443/library
     ```

  4. 仓库级同步。

     ```shell
     bke registry sync --source 192.168.200.240:40443 --target 192.168.200.245:40443 --sync-repo
     ```

  5. 同步构建后的增量镜像目录。

     ```shell
     tar -xzvf patch-log.tar.gz
     bke registry patch --source ./patch-log --target 127.0.0.1:40443
     ```

  6. 查看命令帮助。

     ```shell
     bke registry sync -h
     bke registry patch -h
     ```

- 参数说明

  **`bke registry sync`**

  | 参数名称 | 描述 |
  |---------------|-----------------------------|
  | `--source` | 源镜像/仓库地址（必填）。 |
  | `--target` | 目标镜像/仓库地址（必填）。 |
  | `-f`, `--file` | 批量镜像列表文件，每行 `name:tag`。 |
  | `--multi-arch` | 同步多架构 manifest。 |
  | `--arch` | 只同步指定架构；与 `--multi-arch` 互斥。 |
  | `--src-tls-verify` | 源仓库 TLS 校验。 |
  | `--dest-tls-verify` | 目标仓库 TLS 校验。 |
  | `--sync-repo` | 仓库级整仓同步。 |
  | `-h`, `--help` | 显示命令帮助信息。 |

  **`bke registry patch`**

  | 参数名称 | 描述 |
  |---------------|-----------------------------|
  | `--source` | 增量包解压目录（必填）。 |
  | `--target` | 目标镜像仓地址，默认 `127.0.0.1:40443`。 |
  | `-h`, `--help` | 显示命令帮助信息。 |

- 执行用户

  该命令是由root用户执行。

- 使用前提

  1. 已安装bkeadm。
  2. 目标镜像仓可访问（引导节点本地仓常见端口为 `40443`）。
  3. 使用 `registry patch` 前需先完成增量包构建、传输并解压。

- 注意事项

  1. 推荐链路：`bke build patch` → 传输并解压 → `bke registry patch`；单个外部镜像补仓使用 `bke registry sync`。
  2. `bke registry patch` 只导入增量镜像，不会导入 `volumes/charts/*.tgz`，也不会导入二进制/系统包到 HTTP 仓。

## 镜像仓内容查看

- 命令

  ```shell
  bke registry view
  ```

- 命令功能

  查看镜像仓中的镜像路径、tag、架构等信息；可选导出镜像列表。

- 命令格式

  1. 查看镜像仓（推荐：不写协议，自动 HTTPS 并跳过证书校验，适用于本地自签仓）。

     ```shell
     bke registry view 127.0.0.1:40443
     ```

  2. 按前缀过滤，并限制每个镜像展示的 tag 数量。

     ```shell
     bke registry view 127.0.0.1:40443 --prefix kubernetes --tags 3
     ```

  3. 导出镜像列表到当前目录。

     ```shell
     bke registry view 127.0.0.1:40443 --export
     ```

  4. 查看命令帮助。

     ```shell
     bke registry view -h
     ```

- 参数说明

  | 参数名称 | 描述 |
  |---------------|-----------------------------|
  | 位置参数（仓库地址） | 镜像仓地址（必填），例如 `127.0.0.1:40443`。 |
  | `--prefix` | 按镜像路径前缀过滤。 |
  | `--tags` | 每个镜像最多展示的 tag 数量，默认 `3`。 |
  | `--export` | 导出镜像列表到当前目录；文件名自动生成为 `{架构}_image-list.txt`（如 `amd64_image-list.txt`），不可自定义文件名。 |
  | `-h`, `--help` | 显示命令帮助信息。 |

- 执行用户

  该命令是由root用户执行。

- 使用前提

  1. 已安装bkeadm。
  2. 目标镜像仓可访问。

- 注意事项

  1. `bke registry view` 的仓库地址**不要**写成 `https://127.0.0.1:40443`。手写 `https://` 时会走系统证书校验，本地自签仓会报证书未认证；不写协议时工具会自动补 `https://` 并跳过证书校验。
  2. `--tags` 默认即为 `3`，写不写 `--tags 3` 效果相同；需要查看更多 tag 时可加大该值。
  3. `--export` 不能指定导出文件名，导出后如需自定义名称可自行重命名。

## 初始化引导节点

- 命令

  ```shell
  bke init
  ```

- 命令功能

  初始化引导节点，包括节点检查、仓库启动、集群安装等操作。

- 命令格式

  1. 初始化引导节点。

     ```shell
     bke init
     ```

  2. 初始化引导节点并部署管理集群。

     ```shell
     bke init --file <cluster configuration file path, eg: /root/openfuyao/bkecluster.yaml>
     ```

  3. 指定安装镜像并使用默认镜像源。

     ```shell
     bke init --onlineImage <deploy image, eg: harbor.openfuyao.com/openfuyao/bke-online-installed:latest>
     ```

  4. 自定义镜像源和软件源。

     ```shell
     bke init --otherRepo <other repo, eg: harbor.openfuyao.com/openfuyao> --otherSource <other source, eg: http://192.168.1.120:40080>
     ```

  5. 查看命令帮助。

     ```shell
     bke init -h
     ```

- 参数说明

  | 参数名称                | 描述                                                                 |
  |:----------------------|:-------------------------------------------------------------------|
  | `-f`, `--file`        | 指定`bkecluster.yaml`配置文件路径（必填）。                          |
  | `-h`, `--help`        | 显示命令帮助信息。                                                   |
  | `--chartRepoPort`     | 设置Helm Chart仓库端口（默认：`38080`）。                           |
  | `--clusterAPI`        | 指定`cluster-api-bke`版本（默认：`v25.03`）。                       |
  | `--confirm`           | 跳过确认提示，直接执行。                                             |
  | `--hostIP`            | 指定本地Kubernetes API Server的IP地址（必填）。                   |
  | `--kubernetesPort`    | 设置本地Kubernetes API端口（默认：`36443`）。                       |
  | `--ntpServer`         | 配置NTP服务器（默认：`cn.pool.ntp.org:123`，`local`表示使用本地服务）。 |
  | `--otherRepo`         | 指定镜像仓库地址（如Helm Chart、容器镜像等）。                        |
  | `--otherSource`       | 自定义系统软件包安装的私有源地址。                                    |
  | `--runtime`           | 选择容器运行时（`docker`或`containerd`，默认：`containerd`）。      |
  | `-s`, `--runtimeStorage` | 设置容器运行时存储目录（默认：`/var/lib/docker`或`/var/lib/containerd`）。 |
  | `--yumRepoPort`       | 设置YUM仓库端口（默认:`40080`）。                                 |

- 执行用户

  该命令是由root用户执行。

- 使用前提

  1. 已安装bkeadm。
  2. 已准备好集群配置文件。

- 注意事项

  无。

## 集群安装

- 命令

  ```shell
  bke cluster
  ```

- 命令功能

  集群的生命周期管理。

- 命令格式

  1. 集群的安装部署。

     ```shell
     bke cluster create -f <cluster configuration file path, eg: /root/openfuyao/master.yaml> -n <node configuration file path, eg: /root/openfuyao/nodes.yaml>
     ```

  2. 查看命令帮助。

     ```shell
     bke cluster -h
     ```

- 参数说明

  | 参数名称      | 描述                   |
  |:----------| :--------------------- |
  | `-f`，`--file` | 指定`bkecluster.yaml`配置文件路径（必填）。|
  | `-n`，`--node` | 指定`bkenode.yaml`配置文件路径（必填）。|
  | `-h`，`--help` | 命令帮助。             |

- 执行用户

  该命令是由root用户执行。

- 使用前提

  1. 已安装bkeadm。
  2. 已准备好集群配置文件。

- 注意事项

  无。

## 集群卸载

- 命令

  ```shell
  bke reset
  ```

- 命令功能

  清理引导节点服务并将节点恢复至未安装 BKE 的初始状态。各参数删除范围、高危影响与成功判据见[集群卸载总览](../../cluster_uninstallation/overview.md)与[通过后台（命令行）删除引导集群](../../cluster_uninstallation/delete_bootstrap_cluster_command.md)。

- 命令格式

  1. 删除本地kubernetes服务。

     ```shell
     bke reset
     ```

  2. 清除引导节点及挂载目录。

     ```shell
     bke reset --mount
     ```

  3. 清空节点容器及运行时环境。

     ```shell
     bke reset --all
     ```

  4. 清除节点服务并跳过删除确认。

     ```shell
     bke reset --all --confirm
     ```

  5. 查看命令帮助。

     ```shell
     bke reset -h
     ```

- 参数说明

  | 参数名称       | 描述                         |
  |---------------|-----------------------------|
  | `--all`       | 恢复节点。                     |
  | `--confirm`   | 跳过所有删除确认。               |
  | `-h`, `--help`| 显示重置命令帮助信息。           |
  | `--mount`     | 移除解压目录和相关服务。         |

- 执行用户

  该命令是由root用户执行。

- 使用前提

  1. 已安装bkeadm。
  2. 已准备好集群。

- 注意事项

  无。
