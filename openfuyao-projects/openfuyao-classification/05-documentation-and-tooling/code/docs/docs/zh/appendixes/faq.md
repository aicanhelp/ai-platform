---
sidebar_position: 4
---

# FAQ

## 1. 在包管理页面添加包后未正常显示新加入包

可能是网络波动等因素造成的同步失败，手动单击“同步”即可。

## 2. 添加远端仓库报错unable to find index.yaml, please provide correct ChartMuseum project URL

远端仓库链接填写不正确，需要确保URL指定了harbor仓内具体的项目名，如openFuyao官方仓`https://helm.openfuyao.cn`指定了项目“openfuyao-catalog”。

## 3. 在集群内邀请平台管理员加入，对其设置cluster角色后对应权限不生效

如果用户已经是platform admin角色，再对其设置cluster级角色后权限仍然是platform admin权限。platform admin角色权限优先级最高。

## 4. cluster admin用户无法邀请其他用户加入集群

主要原因是该用户没有platform admin权限，需要先为该用户配置platform admin角色后才可以邀请其他用户。

## 5. 应用卸载后再次部署时部署失败

应用卸载需要一定的时间，立刻单击部署可能因未完全卸载完成导致部署失败，建议等待几分钟后再次部署。

## 6. 使用火狐浏览器访问openFuyao平台日志模块时出现异常

当使用火狐浏览器访问openFuyao平台日志模块，出现页面无法打开或者数据显示异常时，请手动清理浏览器缓存即可。

## 7. 浏览器与服务器的时间差异可能导致部分监控数据受到影响

当前浏览器与服务器之间存在时间差异，可能会出现数据延迟或错位。为避免持续的影响，建议检查本地设备时间设置，并确保与服务器时间同步。调整浏览器时间后，变更生效可能需要一段时间。

## 8. 如何修改openFuyao管理面暴露端口

当前openFuyao管理面默认端口号为31616，如果要修改默认端口号为其他端口，v25.03版本请按如下步骤处理。
> ![输入图片说明](figures/icon-note.gif) **说明：**<br />此步骤以安装包fuyao-v25.03-openEuler-22.03-amd64为例，其他安装包处理步骤类似。

- 操作步骤

  1. 下载openFuyao安装包完成后，解压安装包内asset目录下的shell脚本压缩包。

     ```shell
     cd asset
     tar -zxvf fuyao-openEuler-22.03-amd64.tar.gz
     ```

  2. 编辑`fuyao-openEuler-22.03-amd64/ingress/ingress-nginx/resource/ingress-nginx.yaml`文件，在372行左右找到`nodePort: 31616`字样，修改31616为其他端口号，修改处如下。

     ```shell
     367 - appProtocol: https
     368   name: https
     369   port: 443
     370   protocol: TCP
     371   targetPort: https
     372   nodePort:  31616 # 修改项
     ```

  3. 完成修改后删除原`fuyao-openEuler-22.03-amd64.tar.gz`，并压缩`fuyao-openEuler-22.03-amd64`为`fuyao-openEuler-22.03-amd64.tar.gz`。

     ```shell
     rm -f fuyao-openEuler-22.03-amd64.tar.gz
     tar -czvf fuyao-openEuler-22.03-amd64.tar.gz ./fuyao-openEuler-22.03-amd64/*
     ```

  4. 至此已经完成端口号修改，可继续按照[《安装指导》](https://gitcode.com/openFuyao/sig-installation/blob/master/docs/zh/cluster_installation_guide/read_before_installation/installation_process.md)安装集群，完成集群安装后需使用修改后的端口号访问openFuyao管理面。

## 9. Pod拉起失败，kubectl logs查看日志时出现failed to create fsnotify watcher: too many open files

可能是因为系统默认的fs.inotify.max_user_instances太小，在pod所在节点重新设置此值即可。若为多节点集群安装部署，可为全部节点重新设置fs.inotify.max_user_instances，以避免报错。

方案一：临时设置fs.inotify.max_user_instances值。

```bash
sudo sysctl fs.inotify.max_user_instances=8192
```

> ![输入图片说明](figures/icon-note.gif)**说明**：
> 
> 重新进入控制台时需再次设置。

方案二：永久保存fs.inotify.max_user_instances值。

```bash
echo fs.inotify.max_user_instances=8192| tee -a /etc/sysctl.conf && sudo sysctl -p
```

## 10. 根分区下空间不足但其他分区空间仍有剩余

对于根分区空间不足而其他分区有多余空间的情况，可以考虑将容器相关目录迁移到其他分区。例如，由一个仍有剩余空间的分区挂载在/home，则可以按照如下步骤进行迁移：

1. 在`/home`下创建目录。

   ```bash
   mkdir /home/containerd
   ```

2. 按顺序停止kubelet和containerd。

   ```bash
   systemctl stop kubelet
   systemctl stop containerd
   ```

3. 修改containerd配置文件。

   先找到containerd使用的配置文件，默认为/etc/containerd/config.toml。
   编辑文件，记录原本的root的目录，修改为`root = "/home/containerd"`。

4. 将原有文件迁移到新建的文件夹下。

   ```bash
   cp -r ${original_root}/* /home/containerd/
   ```

5. 重新按顺序启动containerd和kubelet。

   ```bash
   systemctl start containerd
   systemctl start kubelet
   ```

> ![输入图片说明](figures/icon-note.gif)**说明**：<br />
> 该方案在生产场景的可行性和稳定性尚未经过验证，**不推荐**在生产环境使用。

