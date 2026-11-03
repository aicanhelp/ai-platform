# bkeadm配置参数

## 常见配置场景速查

**表1** 常见配置场景速查表

| 配置场景 | 主要配置项 | 参考章节                                                                                              |
|---|---|---------------------------------------------------------------------------------------------------|
| 用YAML初始化或部署集群。 | `bke init --file/-f` | 本文`bke init配置项`；YAML字段见[cluster-api-provider-bke配置参数](cluster-api-provider-bke_configuration.md)。 |
| 初始化引导节点并指定本地仓库端口。 | `--imageRepoPort`、`--chartRepoPort`、`--yumRepoPort`、`--kubernetesPort`、`--consolePort` | 本文`bke init配置项`。                                                                                  |
| 在线安装或指定私有仓库。 | `--onlineImage`、`--otherRepo`、`--otherSource`、`--otherChart` | 本文`bke init配置项`。                                                                                  |
| 使用本地镜像包加速初始化。 | `--imageFilePath`、`--otherRepo`、`--installConsole=false` | 本文`bke init配置项`。                                                                                  |
| 创建集群或节点。 | `cluster create --file/-f`、`cluster create --nodes/-n` | 本文`bke cluster配置项`。                                                                               |
| 构建离线包或增量包。 | `build --file/-f`、`build --target/-t`、`build patch --file/-f`、`build patch --target/-t` | 本文`bke build与构建YAML配置`。                                                                           |
| 同步、查看或导入镜像仓库内容。 | `registry sync`、`registry view`、`registry patch`相关参数 | 本文`bke registry配置项`。                                                                              |

## bkeadm

### bkeadm命令总览

**表2** bkeadm命令总览配置表

| 配置项名称                                                    | 缺省值 | 说明 | 设置方式 |
|-------------------------------------------------------| --- | --- | --- |
| `bke --kubeconfig`                                    | 无 | Kubernetes kubeconfig路径，全局参数，必填。 | CLI persistent flag |
| `bke --doc`                                           | 无 | 打印命令文档，非必填。 | CLI persistent flag |
| `bke init/reset/start/version/registry/build/cluster` | 无 | 当前root命令实际挂载的子命令。 | 代码注册 |

### `bke init`配置项

**表3** bke init配置项表

| 配置项名称 | 缺省值 | 说明 | 设置方式 |
| --- | --- | --- | --- |
| `--file/-f` | 无 | 指定BKECluster YAML；传入后会读取其中repo、NTP等配置并可直接部署，必填；YAML可配置字段见[cluster-api-provider-bke配置参数](cluster-api-provider-bke_configuration.md)。 | CLI |
| `--domain` | `deploy.bocloud.k8s` | 本机镜像仓库域名；为空时PreRun仍回填缺省值，非必填。 | CLI |
| `--imageRepoPort` | `40443` | 本地镜像仓库端口，非必填；与`--chartRepoPort`、`--yumRepoPort`分别对应不同本地服务，自定义时需要彼此避开。 | CLI |
| `--hostIP` | 无 | 启动节点IP地址；为空时自动探测出口/内网IP地址，必填。 | CLI/自动探测 |
| `--kubernetesPort` | `36443` | 本地管理集群Kubernetes API暴露端口，非必填。 | CLI |
| `--consolePort` | `30010` | bke console NodePort，非必填。 | CLI |
| `--yumRepoPort` | `40080` | 本地系统包源（HTTP 仓）服务端口，非必填；与镜像仓库端口、chart repo端口相互独立，自定义时需要彼此避开。 | CLI |
| `--chartRepoPort` | `38080` | 本地chart repo端口，非必填；与镜像仓库端口、yum repo端口相互独立，自定义时需要彼此避开。 | CLI |
| `--ntpServer` | `cn.pool.ntp.org:123` | NTP服务器；值为`local`时启动本机NTP服务，非必填。 | CLI |
| `--onlineImage` | 无 | 在线安装镜像，提供二进制和系统包，必填。 | CLI |
| `--otherRepo` | 无 | 私有镜像仓库前缀，例如`cr.openfuyao.cn/openfuyao`，必填。 | CLI/YAML反推 |
| `--otherSource` | 无 | 私有系统包源，必须以`http://`开头，必填。 | CLI/YAML反推 |
| `--otherChart` | 无 | 私有Helm chart仓库，必填。 | CLI/YAML反推 |
| `--clusterAPI` | `latest` | cluster-api-provider-bke/manifests版本回退值，非必填。 | CLI/版本ConfigMap |
| `--confirm` | `false` | 跳过交互确认，非必填。 | CLI |
| `--oFVersion/-v` | `latest` | openFuyao目标版本，非必填。 | CLI |
| `--versionUrl` | `https://openfuyao.obs.cn-north-4.myhuaweicloud.com/openFuyao/version-config/` | 在线版本配置索引地址，非必填。 | CLI |
| `--installConsole` | `true` | 是否安装bkeconsole，非必填。 | CLI |
| `--enableNTP` | `true` | 是否启用NTP设置；关闭后生成配置里的NTP可为空，非必填。 | CLI |
| `--imageRepoTLSVerify` | `true` | 镜像仓库TLS校验开关，非必填。 | CLI |
| `--imageRepoCAFile` | 无 | 镜像仓库CA文件，必填。 | CLI |
| `--imageRepoUsername` | 无 | 镜像仓库用户名，必填。 | CLI |
| `--imageRepoPassword` | 无 | 镜像仓库密码，必填。 | CLI |
| `--imageFilePath` | 无 | 本地镜像tar包路径，必填；要求`.tar.gz/.tgz`且文件存在；仅用于指定`--otherRepo`的在线安装优化场景，使用时`--otherRepo`不能为空，并且必须设置`--installConsole=false`。 | CLI |
| `--agentHealthPort` | `58080` | bkeagent健康检查侦听端口，要求0-65535，非必填。 | CLI |

> **说明：** `--yumRepoPort`对应的本地系统包源服务在不同配置位置名称不同，实际为同一个服务（默认端口`40080`）：
>
> | 配置位置 | 名称 |
> | --- | --- |
> | `bke init` CLI | `--yumRepoPort` |
> | Cluster CRD | `spec.clusterConfig.cluster.httpRepo.port` |
> | 文档表述 | “HTTP 仓”/“http repo”/“系统包源” |
>
> 其中“yum”指消费侧（该服务同时作为节点的yum/apt系统包源），“http”指传输侧（当前以http协议提供文件分发）；离线部署时`--otherSource`指向的正是该服务。

端口关系与冲突检查：`--imageRepoPort`、`--chartRepoPort`、`--yumRepoPort`分别用于本地镜像仓库、chart repo、yum/http repo，是三个独立服务端口。自定义任一端口时，需要避开另外两个端口，以及`--kubernetesPort`、`--consolePort`和宿主机上已被其他进程占用的端口。当前初始化预检会检查端口是否已被宿主机占用，并会过滤已存在的BKE管理容器端口；但不会显式检查这几个参数之间是否填写成相同端口。如果配置重复，预检可能通过，后续服务启动或端口发布时会失败，需要根据启动日志中的端口占用信息调整为不同端口后重试。

### `bke cluster`配置项

**表4** bke cluster配置项表

| 配置项名称 | 缺省值 | 说明 | 设置方式 |
| --- | --- | --- | --- |
| `cluster list` | 无 | 列出集群；无专属flag，非必填。 | CLI |
| `cluster create --file/-f` | 无 | BKECluster YAML，必填。 | CLI |
| `cluster create --nodes/-n` | 无 | BKENode YAML，必填。 | CLI |

### `bke build`与构建YAML配置

**表5** bke build与构建YAML配置项表

| 配置项名称 | 缺省值 | 说明                          | 设置方式 |
| --- | --- |-----------------------------| --- |
| `build --file/-f` | 无 | 构建配置文件，必填。                   | CLI |
| `build --target/-t` | 无 | 输出BKE包路径，必填。               | CLI |
| `build --skip-tls-verify` | `false` | YAML未配置时，镜像同步跳过TLS校验，非必填。 | CLI |
| `build patch --file/-f` | 无 | 增量包配置文件，必填。                  | CLI |
| `build patch --target/-t` | 无 | 增量包输出路径，必填。                  | CLI |
| `build patch --strategy/-s` | `oci`（文档推荐） | 镜像同步策略；构建增量包时指定为`oci`（可不依赖 Docker）。 | CLI |
| `build patch --skip-tls-verify` | `false` | YAML 未配置时跳过源仓 TLS 校验，非必填。 | CLI |

### `bke registry`配置项

**表6** bke registry配置项表

| 配置项名称 | 缺省值 | 说明                          | 设置方式 |
| ----------------------------------- | --- |-----------------------------| --- |
| `registry sync --file/-f` | 无 | 批量镜像列表文件，每行`name:tag`，必填。   | CLI |
| `registry sync --multi-arch` | `false` | 同步多架构manifest，非必填。          | CLI |
| `registry sync --source` | 无 | 源镜像/仓库，必填。                   | CLI |
| `registry sync --target` | 无 | 目标镜像/仓库，必填。                  | CLI |
| `registry sync --src-tls-verify` | `false` | 源仓库TLS校验，非必填；默认关闭TLS校验存在安全风险，生产环境建议显式设置为`true`，仅在确认源仓库证书可信链不可用或使用内网自签仓库时关闭。 | CLI |
| `registry sync --dest-tls-verify` | `false` | 目标仓库TLS校验，非必填；默认关闭TLS校验存在安全风险，生产环境建议显式设置为`true`，仅在确认目标仓库证书可信链不可用或使用内网自签仓库时关闭。 | CLI |
| `registry sync --arch` | 无 | 指定架构，和`--multi-arch`互斥：`--multi-arch`表示同步完整多架构manifest，`--arch`表示只同步指定架构，两者同时指定时命令会在执行前报错退出，必填。 | CLI |
| `registry sync --sync-repo` | `false` | 仓库级同步，非必填。                   | CLI |
| `registry sync --doc` | 无 | 提示，非必填。                      | CLI |
| `registry view --prefix` | 无 | 查看仓库时的镜像路径前缀，非必填。             | CLI |
| `registry view --tags` | `3` | 每个镜像最多展示的 tag 数量，非必填；默认即为 `3`。          | CLI |
| `registry view --export` | `false` | 导出镜像列表到当前目录，非必填；文件名自动生成为 `{架构}_image-list.txt`，不可自定义。              | CLI |
| `registry patch --source` | 无 | 增量包目录，必填。                    | CLI |
| `registry patch --target` | `127.0.0.1:40443` | 目标镜像仓库地址，非必填。               | CLI |

举例:

#### 同步单个镜像

```bash
   bke registry sync --source docker.io/library/busybox:1.35 --target 192.168.200.240:40443/library/busybox:1.35 --multi-arch
```

成功判断：回显中出现`Sync image docker.io/library/busybox:1.35 to 192.168.200.240:40443/library/busybox:1.35 success`，表示源镜像已经复制到目标仓库；如果源镜像是多架构镜像，也可能先看到按架构镜像同步、再创建manifest的过程日志。

   异常处理：回显中出现`Sync image ... failed`、`by registry failed`或证书、网络、认证相关错误时，优先检查源/目标仓库地址是否可访问、镜像tag是否存在、仓库凭据是否正确；生产环境还应确认是否需要显式增加`--src-tls-verify=true`、`--dest-tls-verify=true`，以及目标仓库证书链是否可信。

#### 同步多个镜像

```bash
   cat > image-list.txt <<'EOF'
   busybox:1.28
   alpine:3.14
   EOF

   bke registry sync --source docker.io/library -f image-list.txt --target 192.168.200.240:40443/library
```

成功判断：回显中会按镜像输出同步日志，并在最后输出`Image list sync completed: total=2, success=2, failed=0`。其中`total`表示文件中的镜像数量，`success`表示成功同步数量，`failed=0`表示没有失败项。

   异常处理：如果`failed`大于0，说明列表中至少有镜像同步失败；回显中会同时出现对应镜像的`Sync image ... failed`。需要根据失败镜像逐项检查`image-list.txt`中的`name:tag`是否正确、源仓库是否存在该tag、目标仓库是否有写入权限，以及网络、TLS、认证配置是否满足要求。

#### 查看当前镜像仓库的镜像列表

```bash
   bke registry view 192.168.200.240:40443 --prefix kubernetes --tags 3
```

成功判断：回显会以表格展示镜像信息，表头包含`IMAGE`、`TAGS`、`ARCHITECTURE`、`CREATE TIME`、`SIZE`。能看到符合`--prefix kubernetes`的镜像名称、tag、架构、创建时间和大小，表示仓库访问和manifest解析成功。

   异常处理：如果回显为`no repositories found`，表示仓库可访问但没有可展示的镜像；如果出现`View repositories ... failed`、`get tags failed`、`unmarshal tags failed`或`view failed`，需要检查仓库地址是否可访问、仓库是否支持Docker Registry V2 API、`--prefix`是否过滤掉了目标镜像，以及认证、网络配置是否正确。

> ![说明](../figures/icon-note.gif) **说明：**<br />查看本地自签镜像仓时**不要**手写 `https://`（如 `https://127.0.0.1:40443`），否则会走系统证书校验并可能报证书未认证。推荐不写协议，直接使用 `127.0.0.1:40443`：工具会自动补 `https://` 并跳过证书校验。`--export` 会按架构导出为 `{架构}_image-list.txt`，不能指定自定义文件名。更多命令说明见[命令参考](command_reference.md)。

#### 仓库级同步

```bash
   bke registry sync  --source 192.168.200.240:40443  --target 192.168.200.245:40443  --sync-repo
```

成功判断：回显中会先输出逐个镜像的同步过程，最后输出`Repo sync completed: total=<数量>, success=<数量>, failed=0`和`Sync repo 192.168.200.240:40443 to 192.168.200.245:40443 success`。其中`failed=0`表示本次仓库级同步没有失败镜像。

   异常处理：如果仓库目录读取失败，会出现`Failed to get repo catalog`、`Failed to unmarshal repo catalog`或`No repositories found`；如果部分镜像同步失败，会出现`Failed to sync image ...`，并在汇总中体现失败数量。此时先确认源仓库`/v2/_catalog`可访问，再按失败镜像检查tag、架构镜像后缀、目标仓库写权限、仓库容量、网络连通性和TLS/认证配置。

### `bke start`配置项

**表7** bke start配置项表

| 配置项名称 | 缺省值 | 说明 | 设置方式 |
| --- | --- | --- | --- |
| `start ntpserver --systemd` | `false` | 用systemd启动NTP服务，非必填。 | CLI |
| `start ntpserver --foreground` | `false` | 前台启动NTP服务，非必填。 | CLI |

### `bke reset`配置项


**表8** bke reset配置项表

| 配置项名称 | 缺省值 | 说明 | 设置方式 |
|---|---|---|---|
| `reset` | 无 | 清理本地Kubernetes/k3s过渡集群相关容器和目录；不触发交互确认。 | CLI |
| `reset --all` | `false` | 在普通`reset`基础上进一步清空节点容器服务、NTP服务、容器运行时、网络和相关残留文件；未加`--confirm`时会在执行前要求用户确认，用户取消则不继续执行，非必填。 | CLI |
| `reset --mount` | `false` | 删除解压目录；单独使用时还会清理容器服务、NTP服务和软件源配置，且不触发交互确认；与`--all`同用时还会移除BKE管理的系统配置并恢复时区，非必填。 | CLI |
| `reset --confirm` | `false` | 仅用于`--all`场景，跳过`bke reset --all`或`bke reset --all --mount`的执行前确认；不改变普通`reset`或单独`--mount`的行为，非必填。 | CLI |
