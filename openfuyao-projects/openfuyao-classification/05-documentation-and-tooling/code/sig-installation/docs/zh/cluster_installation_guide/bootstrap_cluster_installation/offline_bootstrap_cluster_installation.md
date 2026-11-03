# 离线安装引导集群

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

- 推荐准备两台机器，分别承担引导节点与业务集群节点角色；也可共部署于同一台机器（此时将同时运行 K3s 与 Kubernetes，可能相互影响，不推荐）。引导节点需安装 tar 工具。

- 准备一台可访问公网的构建节点（用于在线场景中构建离线安装包）。
  
  > ![说明](../figures/icon-note.gif) **说明：**<br />
  > 构建节点配置可参考上述[配置要求](#配置要求)章节的引导节点相关配置信息。

- 确保这台机器没有`/bke`目录 。
- 安装tar、pigz、bkeadm和Docker。
  
  - 安装tar。可按操作系统直接执行以下命令：

    ```bash
    # RPM系（openEuler/CentOS/RHEL等）
    yum install -y tar

    # Debian系（Ubuntu/Debian等）
    apt-get update && apt-get install -y tar
    ```

  - 安装pigz。可按操作系统直接执行以下命令；源码可参考[madler/pigz](https://github.com/madler/pigz)。

    ```bash
    # RPM系（openEuler/CentOS/RHEL等）
    yum install -y pigz

    # Debian系（Ubuntu/Debian等）
    apt-get update && apt-get install -y pigz
    ```

  - 安装Docker。可按操作系统直接执行以下命令：

    ```bash
    # RPM系（openEuler/CentOS/RHEL等）
    yum install -y docker
    systemctl enable --now docker

    # Debian系（Ubuntu/Debian等）
    apt-get update && apt-get install -y docker.io
    systemctl enable --now docker
    ```

  - bkeadm下载可执行如下命令进行安装，如需自定义修改BKE工具相关参数配置可参考[配置参数](../appendix/bkeadm_configuration.md)。
    
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

## 安装步骤

1. 构建部署包。

   1.1 完成Docker安装后，在Docker配置文件中增加以下配置。  

    编辑/etc/docker/daemon.json文件（如果没有就创建一个）。

   ```json
    {
    "insecure-registries": [     
      "0.0.0.0/0"
    ]    
    }
   ```

   1.2 执行如下命令，重启Docker。

   ```bash
    systemctl restart docker
   ```

   1.3 构建离线安装部署包。

    收集二进制文件、RPM包、Chart包等最后生成一个压缩包，Core-VersionConfig-latest.yaml可单击[下载](https://openfuyao.obs.cn-north-4.myhuaweicloud.com/openFuyao/version-config/Core-VersionConfig-latest.yaml)。
    确定需要安装的扩展组件，将yaml文件中对应的组件的下载开关打开（默认关闭）。下面展示开启volcano组件镜像下载开关，后续离线安装包制备时就会下载volcano所依赖的镜像。

      ```bash
        - architecture:  # openfuyao volcano 扩展组件
            - amd64
            - arm64
          needDownload: true # volcano 扩展组件所需镜像是否下载开关，false-关闭不下载，true-开启下载
          subImages:
            - sourceRepo: cr.openfuyao.cn/openfuyao
              targetRepo: openfuyao
              images:
                - name: volcano-config-website
                  tag:
                    - latest
                - name: volcano-config-service
                  tag:
                    - latest
                - name: numa-tuning-operator
                  tag:
                    - v25.03
                - name: volcano/resource-exporter
                  tag:
                    - latest
                - name: oauth-proxy
                  tag:
                    - latest
                - name: vc-scheduler
                  tag:
                    - latest
            - sourceRepo: docker.io
              targetRepo: /
              images:
                - name: volcanosh/vc-webhook-manager
                  tag:
                    - v1.9.0
                - name: volcanosh/vc-controller-manager
                  tag:
                    - v1.9.0
      ```

    执行如下命令，构建离线安装部署包。
   
      > ![说明](../figures/icon-note.gif) **说明：**
      > 持续时长大约1小时。
      > 导包过程中出现报错会有重试机制，非导致过程终止的报错可忽略。

      ```bash
      rm -rf /bke && bke build -f Core-VersionConfig-latest.yaml -t bke.tar.gz
      ```

2. 进行引导节点离线安装。

   > ![注意](../figures/icon-caution.gif) **注意：**
   > 引导节点存在docker、containerd等容器运行时或相关系统残留文件会导致openFuyao对容器运行时的配置无法生效，从而导致安装失败。
   > <span style="color:red;font-weight:bold;">为确保安装成功，建议使用全新干净节点进行安装。</span>

    2.1 将部署包从在线环境中复制到离线环境引导节点（根据实际情况操作）。

    2.2 执行如下命令，解压部署包到根目录。

      ```bash
        rm -rf /bke && tar zxvf <部署包名字 eg: bke.tar.gz> -C /
      ```
      > ![说明](../figures/icon-note.gif) **说明：**<br/>
      > - 建议解压后根目录（`/`）可用存储空间不少于 50 GB（安装介质解压、镜像与二进制缓存、k3s/containerd 数据等均占用根分区）。可用空间不足时，初始化可能失败或中途中断。
      > - 可用空间检查示例：`df -h /`。环境检查中磁盘项的详细说明见[环境检查配置分析](../appendix/env-check-config-analysis.md)。

3. 修改BKE安装工具名称，并初始化引导节点。
   
    ```bash
    # 修改安装工具名称
    ARCH=$(uname -m)
    case $ARCH in
    x86_64) ARCH="amd64";;
    aarch64) ARCH="arm64";;
    esac
    mv /usr/local/bin/bkeadm_linux_$ARCH /usr/local/bin/bke
    # 初始化引导节点
    bke init 
    ```

    > ![输入图片说明](../figures/icon-note.gif) **说明：**<br />执行 `bke init` 后，工具会先回显本次安装生效的参数（如 `--hostIP`、`--domain` 等），并出现 `Confirm the parameters, press Y to continue N will exit. [Y/N]?` 的确认提示：输入 `Y` 继续安装，输入 `N` 退出安装。也可在命令后追加 `--confirm` 参数跳过该确认步骤。

4. 执行如下命令，可以查看引导集群上部署的Pod。

    ```bash
    kubectl get pod -A
    ```

    执行 `bke init` 成功时，最后会输出 `BKE initialization is complete` 日志行。等待所有 Pod 状态为 `Running` 或 `Completed` 后，则引导集群安装完毕。可登录引导节点的openFuyao管理面，继续进行[业务集群的安装](../service_cluster_installation/service_cluster_installation_on_the_management_plane.md)。

    > ![说明](../figures/icon-note.gif) **说明：**<br/>
    > - 引导节点的openFuyao管理面登录网址为“https://_引导节点IP地址_:30010”。
    > - 首次登录时，默认用户名为admin，默认密码为test@1234。首次登录后需要修改用户名和密码。
    > - 引导节点的openFuyao管理面使用https协议，web服务端口默认为30010。

## 卸载

引导集群卸载（删除范围、高危影响、可选介质清理与成功判据）见[通过后台（命令行）删除引导集群](../../cluster_uninstallation/delete_bootstrap_cluster_command.md)。删除顺序与影响说明见[集群卸载总览](../../cluster_uninstallation/overview.md)。
