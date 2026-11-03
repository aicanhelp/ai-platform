# 安装须知

## 支持环境

该章节介绍openFuyao安装工具支持的操作系统。

### 操作系统

**表1** 支持操作系统信息

| 操作系统  | 发行版本 | 内核版本 | 架构       |
| --------- |-------| ---------- | ---------- |
| openEuler | 20.03 | 4.19.* | ARM64、x86_64 |
| openEuler | 22.03 | 5.10.* | ARM64、x86_64 |
| openEuler | 24.03 | 6.6.* | ARM64、x86_64 |
| Ubuntu    | 22.04 | 5.15.* | ARM64、x86_64 |
| UOS       | 20 1070e | 4.19.* | ARM64、x86_64 |
| UOS       | 20 1070e | 5.10.* | ARM64、x86_64 |

> ![输入图片说明](../figures/icon-note.gif) **说明：**<br />其他版本操作系统暂未测试，可能会出现不在预期内的问题。
>
> 社区基于5.10内核进行功能开发和自验证，在openEuler环境进行兼容性测试，内核版本包含4.19、5.10、5.15和6.6。
>
> openFuyao集群目前仅支持IPv4场景，暂不支持IPv6。

## 注意事项

- 推荐准备两台机器，分别承担引导节点与业务集群节点角色；也可共部署于同一台机器（此时将同时运行 K3s 与 Kubernetes，可能相互影响，不推荐）。
- 引导节点的openFuyao管理面首次登录时，用户名为admin，密码为test@1234。首次登录后需要修改用户名和密码。使用的https协议，web服务端口默认为30010。
- 业务集群节点的openFuyao管理面首次登录时，用户名为admin，密码为test@1234。首次登录后需要修改用户名和密码。使用的https协议，web服务端口默认为31616。
- 建议您的节点环境为干净的操作系统环境，未安装任何docker与Kubernetes组件，否则可能会发生版本冲突导致安装失败。如需在存量环境试验性安装，可下载[环境前置检查工具](environment_pre_check_tool_guide.md)进行环境校验。
- 待安装集群节点的时间需要确保和引导节点的时间一致，时间差上限为10s。
- 可选：如需使用自定义CA证书来签发Kubernetes集群内所有组件的证书，以满足企业级安全合规要求，可在引导节点初始化之前配置自定义证书。详细操作请参考[自定义集群证书使用指导](../bootstrap_cluster_configuration/using_custom_cluster_certificates.md)。
- 可选：如需配置自定义仓库进行安装请参见[自定义仓库使用指导](../bootstrap_cluster_configuration/custom_repository_configuration_guide.md)。
- 可选：如需对节点打标签并在addon中编辑标签亲和请参见[节点标签亲和部署使用指导](../service_management_cluster_configuration/node_label_affinity_deployment_guide.md)。
- 可选：如需containerd和kubelet自定义配置请参见[二进制文件自定义使用指导](../service_management_cluster_configuration/binary_file_custom_configuration_guide.md)。
- 可选：如需通过addon安装自定义扩展件请参见[addon使用指导](../service_management_cluster_configuration/using_add-ons.md)。
