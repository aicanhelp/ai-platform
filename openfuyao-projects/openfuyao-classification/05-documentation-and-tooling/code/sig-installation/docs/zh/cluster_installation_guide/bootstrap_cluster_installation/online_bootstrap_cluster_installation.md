# 在线安装引导集群

## 配置要求

#### 最小化部署硬件配置

| 节点类型 | 节点数量 | CPU | 内存 | 硬盘         |
| -------- | -------- | --- | ---- | ------------ |
| 引导节点 | 1        | 2   | 4    | 系统盘>=100G |
| 管理集群 | 1        | 8   | 16   | 系统盘>=100G |
| 业务集群 | 1        | 8   | 16   | 系统盘>=100G |

注：引导节点可以是管理集群的一个节点。

#### 推荐硬件配置

| 节点类型 | 节点数量     | CPU          | 内存         | 硬盘                         |
| -------- | ------------ | ------------ | ------------ | ---------------------------- |
| 引导节点 | 1            | 2            | 4            | 系统盘>=100G                 |
| 管理集群 | 3            | 8            | 16           | 系统盘>=100G<br>数据盘>=300G |
| 业务集群 | 实际情况而定 | 实际情况而定 | 实际情况而定 | 实际情况而定                 |

## 环境准备

推荐准备两台机器，分别承担引导节点与业务集群节点角色；也可共部署于同一台机器（此时将同时运行 K3s 与 Kubernetes，可能相互影响，不推荐）。引导节点需安装 tar 工具。可按操作系统直接执行以下命令：

```bash
# RPM系（openEuler/CentOS/RHEL等）
yum install -y tar

# Debian系（Ubuntu/Debian等）
apt-get update && apt-get install -y tar
```

> ![须知](../figures/icon-notice.gif) **须知：**  
> 建议您的节点环境未安装任何docker与Kubernetes组件，否则可能会发生版本冲突导致安装失败。如需在存量环境试验性安装，可下载[环境前置检查工具](../environment_preparation/environment_pre_check_tool_guide.md)进行环境校验。
> 若业务集群需要安装openFuyao管理面，那么业务集群的节点也需要安装tar工具。

## 安装步骤

1. 在引导节点上下载、校验并安装 BKE 工具，如需自定义修改 BKE 工具相关参数配置可参考[配置参数](../appendix/bkeadm_configuration.md)。
   
   1.1 基于download.sh 脚本

   - 安装最新版本BKE安装工具：使用root用户登录引导节点，执行如下命令，在引导节点上下载、校验并安装最新版本的 BKE 安装工具。

   ```bash
   # 下载 download.sh 脚本
   curl -LO https://openfuyao.obs.cn-north-4.myhuaweicloud.com/openFuyao/bkeadm/releases/download/latest/download.sh
   # 下载 download.sh 的校验文件并进行校验；校验成功会输出 -: OK，校验失败请联系 openFuyao sig-installation邮箱installation@list.openfuyao.cn或openFuyao 社区邮箱 contact@openfuyao.cn
   curl -LO https://openfuyao.obs.cn-north-4.myhuaweicloud.com/openFuyao/bkeadm/releases/download/latest/download.sh.sha256
   sha256sum -c <(cat download.sh.sha256) < download.sh
   # 运行 download.sh 下载 BKE 安装工具，执行过程中会校验安装工具的 sha256sum
   chmod +x download.sh && ./download.sh
   ```

   - 安装指定版本BKE安装工具：若想下载其他版本，请使用root用户登录引导节点后使用如下命令指定版本下载。
  
   ```bash
   export VERSION=x.y.z # x.y.z根据您想下载的其他实际版本替换。

   # 下载 download.sh 脚本
   curl -LO https://openfuyao.obs.cn-north-4.myhuaweicloud.com/openFuyao/bkeadm/releases/download/latest/download.sh
   # 下载 download.sh 的校验文件并进行校验；校验成功会输出 -: OK，校验失败请联系 openFuyao sig-installation邮箱installation@list.openfuyao.cn或openFuyao 社区邮箱 contact@openfuyao.cn
   curl -LO https://openfuyao.obs.cn-north-4.myhuaweicloud.com/openFuyao/bkeadm/releases/download/latest/download.sh.sha256
   sha256sum -c <(cat download.sh.sha256) < download.sh
   # 运行 download.sh 下载 BKE 安装工具，执行过程中会校验安装工具的 sha256sum
   chmod +x download.sh && ./download.sh
   ```

   > ![输入图片说明](../figures/icon-note.gif) **说明：**<br/>
   >
   > - 对于openFuyao v26.06、v26.09及后续版本：x.y.z取值（BKE 安装工具版本号）为26.6.0、26.9.0及其他后续数字版本号。
   > - 对于openFuyao v26.03及之前版本：x.y.z取值（BKE 安装工具版本号）与openFuyao版本对应关系如下：
   >
   >    |openFuyao版本|BKE 安装工具版本|
   >    |--|--|
   >    |v26.03|1.2.2|
   >    |v25.12|1.0.5|
   >    |v25.09|v25.09|
   >    |v25.06|v25.06|
   >    |v25.03|v25.03|

   1.2 基于二进制软件包

   如不基于download.sh 脚本下载目标版本的 BKE 安装工具，也可直接下载二进制包后解压，并将二进制文件移动至系统路径（如 `/usr/local/bin`）；手动下载时请按[安装工具下载及校验](../../cluster_installation_guide/environment_preparation/installation_package_integrity_verification.md)完成完整性校验后再部署：

   | 架构 | 下载 |
   | :--- | :--- |
   | x86_64 | [bkeadm_linux_amd64.tar.gz](https://openfuyao.obs.cn-north-4.myhuaweicloud.com/openFuyao/bkeadm/releases/download/latest/bkeadm_linux_amd64.tar.gz) |
   | ARM | [bkeadm_linux_arm64.tar.gz](https://openfuyao.obs.cn-north-4.myhuaweicloud.com/openFuyao/bkeadm/releases/download/latest/bkeadm_linux_arm64.tar.gz) |

2. 执行如下命令，初始化引导节点。

   该步骤会在引导节点上部署一个轻量级的K3s集群，集群会部署cluster-api、provider-bke以及openFuyao安装部署相关的Pod。

   > ![输入图片说明](../figures/icon-note.gif) **说明：**<br />执行 `bke init` 后，工具会先回显本次安装生效的参数（如 `--hostIP`、`--domain` 等），并出现 `Confirm the parameters, press Y to continue N will exit. [Y/N]?` 的确认提示：输入 `Y` 继续安装，输入 `N` 退出安装。也可在命令后追加 `--confirm` 参数跳过该确认步骤。

   **方式一：默认初始化（部署openFuyao管理面）**

   ```bash
   bke init --onlineImage cr.openfuyao.cn/openfuyao/bke-online-installed:latest
   ```

   **方式二：最小集初始化（不部署openFuyao管理面）**

   仅初始化引导节点、不部署前端 openFuyao 管理面时，在 `bke init` 中增加 `--installConsole=false`。镜像来源可二选一：

   - **从在线仓库拉取镜像**（与方式一相同，不指定本地安装包）：

     ```bash
     bke init --onlineImage cr.openfuyao.cn/openfuyao/bke-online-installed:latest --installConsole=false
     ```

   - **使用本地安装包**（可选，用于本地镜像加速；需同时满足 `--otherRepo` 等约束，详见[配置参数](../appendix/bkeadm_configuration.md)）：

     ```bash
     bke init --onlineImage cr.openfuyao.cn/openfuyao/bke-online-installed:latest --installConsole=false --imageFilePath=/tmp/local_amd/image_amd64.tar.gz
     ```

     其中：`imageFilePath` 指向本地安装包，构建方式参考[引导节点初始化流程优化](https://gitcode.com/openFuyao/sig-installation/blob/master/design/%E5%BC%95%E5%AF%BC%E8%8A%82%E7%82%B9%E5%AE%89%E8%A3%85%E4%BC%98%E5%8C%96%E5%92%8C%E5%81%A5%E5%BA%B7%E6%A3%80%E6%B5%8B%E6%9C%BA%E5%88%B6/%E5%BC%95%E5%AF%BC%E8%8A%82%E7%82%B9%E5%88%9D%E5%A7%8B%E5%8C%96%E6%B5%81%E7%A8%8B%E4%BC%98%E5%8C%96.md)。

   > ![注意](../figures/icon-caution.gif)**注意：**<br/>
   > 方式二不会部署前端 openFuyao 管理面。后续创建业务集群请走后台命令行路径，不要再按管理面 Web 操作。

3. 执行如下命令，查看引导集群上部署的Pod。

   ```bash
   kubectl get pod -A
   ```
   
   `bke init` 执行成功时，最后会输出 `BKE initialization is complete` 日志行。待所有 Pod 状态为 `Running` 或 `Completed` 后，则引导集群安装完毕。后续按初始化方式选择业务集群安装路径：

   - **方式一（已部署openFuyao管理面）**：登录引导节点的openFuyao管理面，继续进行[通过前台（Web页面）安装业务集群](../service_cluster_installation/service_cluster_installation_on_the_management_plane.md)。

     > ![说明](../figures/icon-note.gif) **说明：**<br/>
     > - 引导节点的openFuyao管理面登录网址为“https://_引导节点IP地址_:30010”。
     > - 首次登录时，默认用户名为admin，默认密码为test@1234。首次登录后需要修改用户名和密码。
     > - 引导节点的openFuyao管理面使用https协议，web服务端口默认为30010。

   - **方式二（未部署openFuyao管理面，`--installConsole=false`）**：无法使用管理面Web创建业务集群，请改用后台命令行继续安装，参见[通过后台（命令行）安装业务集群](../service_cluster_installation/service_cluster_deployment_guide_(linux_backend).md)。

## 卸载

引导集群卸载（删除范围、高危影响、可选介质清理与成功判据）见[通过后台（命令行）删除引导集群](../../cluster_uninstallation/delete_bootstrap_cluster_command.md)。删除顺序与影响说明见[集群卸载总览](../../cluster_uninstallation/overview.md)。
