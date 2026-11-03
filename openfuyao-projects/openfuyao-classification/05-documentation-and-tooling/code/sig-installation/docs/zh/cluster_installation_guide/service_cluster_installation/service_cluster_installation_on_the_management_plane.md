# 通过前台（Web页面）安装业务集群

## 前提条件

在通过Web界面安装业务集群前，请确认：

- 已完成引导集群初始化：在线参见[在线安装引导集群](../bootstrap_cluster_installation/online_bootstrap_cluster_installation.md)，离线参见[离线安装引导集群](../bootstrap_cluster_installation/offline_bootstrap_cluster_installation.md)。
- **已部署openFuyao管理面**。若引导集群初始化时使用了`--installConsole=false`，则无法使用本指南，请改用[通过后台（命令行）安装业务集群](./service_cluster_deployment_guide_(linux_backend).md)。
- **中大规模部署**：须先完成[安装管理集群](../management_cluster_installation/management_cluster_deployment_guide.md)，并登录**管理集群**的openFuyao管理面创建业务集群（默认地址 `https://<管理集群节点IP>:31616`）。
- **小规模部署**：可不单独部署管理集群，直接登录**引导集群**的openFuyao管理面创建业务集群（默认地址 `https://<引导节点IP>:30010`）。

## 安装步骤

1. 登录用于创建业务集群的openFuyao管理面（引导集群或管理集群，见[前提条件](#前提条件)）。

   - 引导集群管理面：浏览器中输入“https://<引导节点IP地址>:30010”。
   - 管理集群管理面：浏览器中输入“https://<管理集群节点IP地址>:31616”。

   > ![说明](../figures/icon-note.gif) **说明：** 
   > - 首次登录时，用户名为admin，密码为test@1234。首次登录后需要修改用户名和密码。
   > - openFuyao管理面使用https协议；引导集群管理面默认端口为30010，管理/业务集群管理面默认端口为31616。

2. 在openFuyao平台左侧导航栏中选择“集群生命周期管理”。界面展示集群生命周期管理的列表，包括“集群名称”、“状态”、“节点数”等信息。

3. 单击列表右上方的“+ 新增集群”，进入“新增集群”界面。

4. 填写界面相关内容。

    **表1** 界面相关说明

    | 区域名称     | 说明                                                                                                                 |
    |:---------|:-------------------------------------------------------------------------------------------------------------------|
    | 基本信息     | 创建业务集群名称以及要安装的openFuyao版本。                                                                                         |
    | Master信息 | 业务集群的控制平面节点的相关信息。单击![](../figures/node_add.png)可继续增加控制平面节点。单击“操作”列的![](../figures/node_delete.png)可删除已添加的节点。当master节点数量多于一个且为奇数时，配置`负载均衡器IP地址`创建高可用集群，此地址要求是和master节点同网段且未被分配使用的IP地址。 |
    | Node信息   | 业务集群的node节点信息。                                                                                                     |
    | 高级配置     | 配置镜像仓库、二进制/系统包仓库、Chart 仓库等安装参数。界面预填值即为 openFuyao 平台官方推荐配置：在线安装时指向官方默认仓库；离线安装时指向引导集群初始化时配置的本地仓库。 |

   > ![说明](../figures/icon-note.gif) **说明：**
   > - 如果要安装高可用集群，需要添加奇数个Master节点，并配置负载均衡器IP地址，且不要使用引导节点作为高可用集群的Master节点。
   > - 高级配置中的附加件需要选择`openFuyao Cores`，选择此选项后安装的业务集群才会有openFuyao管理面。若不选择，可在安装完成后按[（可选）安装openFuyao管理面](#可选安装-openfuyao-管理面)补装。

5. 单击右下角的“一键校验”。

6. 校验成功后，单击下方的“创建”，即可下发业务集群的创建任务。

7. 界面弹出“集群信息”窗口。待集群信息滚动完成后，返回集群列表界面，“状态”列显示为“安装中”，待“状态”列字段更新完毕，完成新增集群。
   - 对应集群“状态”列字段会出现以下状态，安装中、扩容中、检查中、异常、健康。其中“扩容中”在非单Master的集群场景下会出现，“检查中”是会周期出现（属于正常现象），“异常”是集群在等待对应Pod正常运行（需要等待一定时间，若长时间处于异常状态，需要根据日志进行排查）。集群最终状态是“健康”或“安装失败”。
   - 对应集群“状态”列字段为“健康”，代表新增成功。
   - 对应集群“状态”列字段为“安装失败”，代表新增失败。

   > ![说明](../figures/icon-note.gif) **说明：** <br />
   > - “集群信息”窗口滚动显示集群安装过程的事件日志。当窗口出现`Reason: ClusterReady, Message: cluster is ready`日志行且不再滚动新日志时，表示集群信息滚动完成，集群安装成功；若出现 `Reason: ClusterDeployFailed, Message: Cluster deploy failed, process exit`日志行，表示集群安装失败。滚动耗时视集群规模与网络情况而定。
   > 如果“集群日志”显示报错信息，需要进一步定位解决，可参考[FAQ](../appendix/FAQ.md)相关内容。
   
8. 若安装时选择了`openFuyao Cores`，业务集群安装成功后可以登录业务集群的openFuyao管理面进行集群管理相关的操作。

   浏览器中输入“https://<业务集群IP地址>:31616”。<br />

   > ![说明](../figures/icon-note.gif) **说明：** 
   >
   > - 首次登录时，用户名为admin，密码为test@1234。首次登录后需要修改用户名和密码。
   > - 业务集群的openFuyao管理面使用https协议，Web服务端口默认为31616。

## （可选）安装 openFuyao 管理面

若创建业务集群时 **未** 选择 `openFuyao Cores`（或其他原因导致目标集群上没有 openFuyao 管理面），可在集群就绪后按场景手动补装：

- 在线场景：参见[在线 openFuyao-system-controller 手动部署](../appendix/online_incremental_deployment_guide.md)。
- 离线场景：参见[离线 openFuyao-system-controller 手动部署](../appendix/offline_incremental_deployment_guide.md)。
