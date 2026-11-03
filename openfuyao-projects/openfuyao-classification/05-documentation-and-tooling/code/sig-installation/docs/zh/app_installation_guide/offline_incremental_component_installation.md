# 离线组件增量安装

## 背景

离线环境使用了openFuyao社区的Cluster-API方式安装了业务集群，后续需要在已有的集群中安装新的扩展组件，做到增量安装，而无须重新制备离线制品包再进行集群的安装。

## 环境准备

- 准备一台可访问公网的构建节点（用于在线场景中构建组件增量安装包）。
- 安装tar、pigz和bkeadm。本文增量包制备使用`oci`策略，不需要安装Docker。
    - tar可通过系统包管理器安装，例如 `yum install -y tar` 或 `apt-get update && apt-get install -y tar`。
    - pigz可通过系统包管理器安装，例如 `yum install -y pigz` 或 `apt-get update && apt-get install -y pigz`；源码可参考[madler/pigz](https://github.com/madler/pigz)。
    - bkeadm下载可执行如下命令进行安装。
      ```bash
      # 下载 download.sh 脚本
      curl -LO https://openfuyao.obs.cn-north-4.myhuaweicloud.com/openFuyao/bkeadm/releases/download/latest/download.sh
      # 下载 download.sh 的校验文件并进行校验；校验成功会输出 -: OK，校验失败请联系 openFuyao sig-installation邮箱installation@list.openfuyao.cn或openFuyao 社区邮箱 contact@openfuyao.cn
      curl -LO https://openfuyao.obs.cn-north-4.myhuaweicloud.com/openFuyao/bkeadm/releases/download/latest/download.sh.sha256
      sha256sum -c <(cat download.sh.sha256) < download.sh
      # 运行 download.sh 下载 BKE 安装工具，执行过程中会校验安装工具的 sha256sum
      chmod +x download.sh && ./download.sh
      ```

## 安装步骤

按照如下步骤进行组件的增量安装：

1. 镜像增量安装包制备。

   使用`oci`策略构建增量包：不拉起临时镜像仓库，将组件依赖镜像同步到包内的OCI Layout目录并打包（可不依赖 Docker）。

   1.1 准备组件依赖镜像的yaml文件，下面举例是openFuyao社区的logging-packages组件的镜像下载配置yaml文件。

      ```yaml
      registry:
        imageAddress: cr.openfuyao.cn/openfuyao/registry:2.8.1
        architecture:
          - amd64
          - arm64
      repos:
        - architecture: # openfuyao logging 扩展组件
            - amd64
            - arm64
          needDownload: true # logging 扩展组件所需镜像是否下载开关，false-关闭不下载，true-开启下载
          subImages:
            - sourceRepo: cr.openfuyao.cn/openfuyao
              targetRepo: openfuyao
              images:
                - name: logging-operator
                  tag:
                    - latest
                - name: logging-website
                  tag:
                    - latest
                - name: oauth-proxy
                  tag:
                    - latest
            - sourceRepo: docker.io
              targetRepo: /
              images:
                - name: grafana/loki
                  tag:
                    - 2.9.1
                - name: grafana/promtail
                  tag:
                    - 2.9.3
                - name: bats/bats
                  tag:
                    - 1.8.2
      rpms:
      debs: []
      files: []
      charts: []
      ```

   1.2 制备镜像增量安装包，完成后会得到镜像增量安装包patch-log.tar.gz压缩文件。使用`oci`策略构建：不拉起临时镜像仓库，将组件依赖镜像同步到包内的OCI Layout目录并打包（可不依赖 Docker）。

      ```bash
      bke build patch -f offline.yaml -t patch-log.tar.gz --strategy oci
      ```

2. 将离线的镜像增量安装包复制到离线引导节点。

   在在线构建节点上执行如下命令，将镜像增量安装包复制到离线引导节点（根据实际情况选择拷贝方式）。

   ```bash
   scp patch-log.tar.gz root@<离线引导节点IP>:/root/
   ```

3. 上传Chart包到应用仓库的local仓库。

   离线场景无法从openFuyao官方Chart仓在线拉取扩展组件，需要先从openFuyao官网下载目标扩展组件的Chart包，再导入到离线业务集群应用仓库的local仓库。

   3.1 从[openFuyao官网](https://www.openfuyao.cn)下载目标扩展组件的Chart包（如logging-package的Chart包）：鼠标移动至官网“下载”页签，单击“应用中心”，在“请输入Chart名称”搜索栏中搜索目标扩展组件（如logging-package），单击后进入组件详情页，在“版本列表”中获取所需版本的Chart包。

   3.2 浏览器中输入“https://_管理面的登录IP地址_:_openFuyao的web服务端口_”，输入用户名和密码登录离线业务集群的openFuyao管理面。

   3.3 左侧导航栏选择“应用市场 > 仓库配置”，进入“仓库配置”界面，确认应用仓库中存在local仓库。

   3.4 切换至“包管理”标签页，单击“添加包”，选择已下载的Chart包并上传，单击“确定”完成添加。

   > ![输入图片说明](figures/icon-note.gif) **说明：**<br />支持上传.tgz、.tar.gz格式的Chart包。由于网络波动等原因，添加包后包同步可能不及时，包内组件不会立即展示在应用列表中。可在“包管理”标签页单击右侧的“同步”手动同步到local仓库，同步完成后再进行组件安装。

4. 进行组件安装。

   4.1 离线引导节点镜像同步。

      ```bash
      # 解压安装包
      tar -xzvf patch-log.tar.gz
      # 镜像同步
      bke registry patch --source ./patch-log --target 0.0.0.0:40443
      ```

   4.2 业务集群的openFuyao管理面应用管理安装组件。

      登录openFuyao管理面，左侧导航栏选择“应用市场 > 应用列表”，在右上角搜索框中搜索目标扩展组件（如logging-package），可以看到对应的扩展组件。

      单击应用卡片，进入应用“详情”界面。可在“版本”选择器中选择需要部署的版本，并在“详细信息”和“默认参数”标签页中查看组件的详细说明与YAML格式参数信息。

      单击“部署”，在安装信息中填写应用名称，选择应用版本信息和命名空间；在Values.yaml中按需填写Helm Chart的values信息，若使用平台预设的默认参数组，无须修改。

      单击“部署”，开始部署扩展组件。部署完成后，扩展组件安装结果的确认方式可参考[在线组件增量安装](./online_incremental_component_installation.md)的“结果验证”章节。
