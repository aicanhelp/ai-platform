# 在线组件增量安装

## 背景

在线环境使用openFuyao社区的Cluster-API方式安装了业务集群，后续需要在已有的集群中安装新的扩展组件，做到增量安装。无须重新制备离线制品包，也无须重新安装集群，直接登录openFuyao管理面，通过应用市场从平台内置的官方组件源在线拉取并部署openFuyao官方扩展组件（如NUMA亲和插件、VictoriaMetrics等），零额外配置，开箱可部署。

在线增量安装与[离线组件增量安装](./offline_incremental_component_installation.md)的对照关系如下：

| 对比项 | 在线增量安装（本篇） | 离线组件增量安装 |
| --- | --- | --- |
| 适用环境 | 集群节点可访问公网/官方组件源。 | 集群节点无法访问公网。 |
| 制品获取方式 | 部署时由平台直接从官方组件源在线拉取。 | 使用bkeadm构建镜像增量安装包并导入。 |
| 额外构建环境 | 无须额外准备。| 构建节点需安装tar、pigz、bkeadm（使用`oci`策略制备增量包，不需要Docker）。 |
| Chart包来源 | openFuyao官方仓库在线同步。 | 从仓库下载Chart包后上传至local仓库。 |
| 镜像来源 | 部署时在线拉取官方镜像仓镜像。 | `bke registry patch`将增量包镜像同步到离线仓库。 |
| 操作入口 | openFuyao管理面应用市场。 | 离线引导节点命令行 + openFuyao管理面应用市场。 |

## 前提条件

- 业务集群已通过Cluster-API方式[在线安装](../cluster_installation_guide/service_cluster_installation/service_cluster_installation_on_the_management_plane.md)完毕。
- 业务集群节点可访问公网或官方组件源（官方镜像仓cr.openfuyao.cn、官方Chart组件源），用于在线拉取扩展组件依赖的镜像与Chart包。校验方法参见[部署前校验](#部署前校验)第3项。
- 业务集群中marketplace-service运行正常。
- 拥有openFuyao管理面的登录账号。

> ![输入图片说明](figures/icon-note.gif) **说明：**<br />
> openFuyao完整安装的情况下，应用市场初始配置两个默认仓库：local为用户本地仓库，openFuyao为官方仓库。在线增量安装直接使用openFuyao官方仓库中已发布的扩展组件，无须添加第三方仓库，也无须上传Chart包。

## 部署前校验

部署扩展组件前，建议按如下步骤校验环境。

1. 校验业务集群整体状态。

   登录业务集群任一节点，执行如下命令，确认所有Pod处于`Running`或`Completed`状态。

   ```bash
   kubectl get pod -A
   ```

2. 校验应用市场组件服务状态。

   执行如下命令，确认marketplace-service相关Pod处于`Running`状态。

   ```bash
   kubectl get pod -A | grep marketplace
   ```

3. 校验节点到官方组件源的网络连通性。

   在业务集群节点上执行如下命令，确认节点可以访问官方镜像仓。若命令执行失败，请先检查节点网络、DNS解析或代理配置。

   ```bash
   curl -I https://cr.openfuyao.cn
   ```

4. 校验openFuyao官方仓库状态。

   登录openFuyao管理面，左侧导航栏选择“应用市场 > 仓库配置”，确认openFuyao官方仓库已存在且状态正常。单击“同步”获取最新的官方应用与扩展组件列表；若同步失败，请检查仓库地址是否可用、认证信息是否正确。

## 部署步骤

下面以部署日志扩展组件logging-package为例进行说明。

1. 浏览器中输入“https://_管理面的登录IP地址_:_openFuyao的web服务端口_”，输入用户名和密码登录openFuyao管理面。

2. 同步openFuyao官方仓库。

   左侧导航栏选择“应用市场 > 仓库配置”，在openFuyao官方仓库所在行的操作列单击图标，选择“同步仓库”；或单击openFuyao官方仓库名称进入仓库“详情”界面，单击“同步”。同步完成后，应用列表即为最新的官方应用与扩展组件。

   > ![输入图片说明](figures/icon-note.gif) **说明：**<br />
   > 若部署前校验中已执行官方仓库同步，可跳过此步。

3. 左侧导航栏选择“应用市场 > 应用列表”，进入“应用列表”界面。在右上角搜索框中搜索“logging-package”，可以看到日志扩展组件。

4. 单击应用卡片，进入应用“详情”界面。可在“版本”选择器中选择需要部署的版本，并在“详细信息”和“默认参数”标签页中查看组件的详细说明与YAML格式参数信息。

5. 在应用“详情”界面单击“部署”，进入“部署”界面。

6. 在安装信息中填写应用名称，选择应用版本信息和命名空间。

   > ![输入图片说明](figures/icon-note.gif) **说明：**<br />
   > - 扩展组件在平台内只能安装一次，安装完成后，应用详情内的扩展组件“部署”会变成“管理”，单击“管理”会跳转至对应的“扩展组件管理”中。
   > - 部分应用名称以数字开头在部署时可能出错，具体情况见日志。

7. 在Values.yaml中按需填写Helm Chart的values信息。若使用平台预设的默认参数组，无须修改，直接进入下一步。

   > ![输入图片说明](figures/icon-note.gif) **说明：**<br />
   > 日志扩展组件logging-package由日志后端层、日志UI层与开源软件层（日志采集组件promtail、日志存储组件Loki等）组成，组件说明与Values.yaml参数配置请参见openFuyao/docs[《日志》](https://gitcode.com/openFuyao/docs/blob/master/docs/zh/management_plane_operations/observability_center/logs.md)。

8. 单击“部署”，开始部署扩展组件。

## 结果验证

1. 在管理面查看部署结果。

   部署完成后，左侧导航栏选择“扩展组件管理”（常规应用选择“应用管理”），进入对应界面查看已部署组件的名称、状态、更新时间。当组件状态正常时，表示部署成功。单击组件名称进入“详情”界面，可进一步查看详情、YAML、资源、日志、事件、监控等信息。

2. 在命令行查看组件Pod状态。

   登录业务集群任一节点，执行如下命令，确认新部署扩展组件相关的Pod均处于`Running`或`Completed`状态。

   ```bash
   kubectl get pod -A
   ```

3. （可选）按组件能力进一步验证。

   - 日志组件：部署成功后，可在管理面左侧导航栏“观测中心”中选择“日志 > 日志查询”查看日志，并在“日志 > 日志配置”中管理采集源、采集任务与告警规则，详见openFuyao/docs[《日志》](https://gitcode.com/openFuyao/docs/blob/master/docs/zh/management_plane_operations/observability_center/logs.md)。

## 常见问题

1. 应用列表中搜索不到目标扩展组件。

   - openFuyao官方仓库未同步或同步失败。请进入“应用市场 > 仓库配置”，对openFuyao官方仓库执行“同步仓库”后重试。
   - 仓库地址不可用或认证信息不正确导致同步失败。请检查仓库配置后重新同步。

2. 部署后Pod一直处于Pending状态。

   - Kubernetes集群中各节点上能运行的Pod数量受限，默认为110个。所有节点上的Pod均达到限制数量后，新的待创建Pod将无法调度而一直处于Pending状态。请及时清理或扩容节点，避免Pending状态的Pod无限增加。
   - 集群资源不足或节点网络异常。请执行`kubectl describe pod <pod-name> -n <namespace>`查看事件，确认是否存在资源不足、镜像拉取失败等问题。

3. 镜像拉取失败。

   在线部署需要从官方镜像仓在线拉取组件依赖镜像。请确认业务集群节点可访问cr.openfuyao.cn，检查节点网络、DNS解析或代理配置后重新部署。

4. 部署失败，提示参数配置错误。

   部署时Values.yaml中的Helm Chart参数配置错误可能导致部署失败或组件无法正常运行。请进入应用“详情”界面查看“默认参数”，按组件使用指导修正values信息后重新部署。

5. 扩展组件的“部署”按钮显示为“管理”。

   扩展组件在平台内只能安装一次。安装完成后，应用详情内的扩展组件“部署”会变成“管理”，单击“管理”会跳转至对应的“扩展组件管理”中，可查看或管理已安装的扩展组件。

6. 部分扩展组件与平台自带组件存在冲突。

   部分扩展组件与平台自带组件可能存在端口冲突或功能重叠（如监控、日志类组件），不建议同时安装。部署前请查阅对应扩展组件的使用文档，确认兼容性与安装要求，并按文档指引处理与平台自带组件的冲突。
