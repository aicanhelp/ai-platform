/******************************************************************
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain n copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package root

import "fmt"

func (op *Options) PrintDoc() {
	content := `
** 该指令将提示一些常用命令及常规运维手段，详细使用请参考BKE在线文档 **

1. bke init 指令将初始化引导节点，默认安装的运行时为containerd

2. 目前boc4.0仅支持在docker上运行，如果采用allinone模式，则需要使用 bke init --runtime=docker指令

3. bke init 指令将在节点上启动五个容器服务，分别为镜像仓库服务、chat仓库服务、yum源服务、NFS服务以及kubernetes服务

4. bke init中ntpserver启动原理是，当用户bke init --ntpServer=local 或者 提供的ntpserver地址不可用时，将在本机启动一个ntpserver服务

5. bke init 启动的所有服务

6. bke start可以创建bke init启动的服务

7. bke reset 仅仅移除kubernetes容器，在通过bke init快速拉起，适用于k3s服务出问题时快速重建，bke reset --all将清空节点包括容器运行时等

8. bke init拉起的kubernetes是一个容器内运行的k3s，我们使用它作为初始k8s安装portal集群，装完之后即可销毁 bke reset

9. kubernetes服务的kubeconfig文件存在于 /etc/rancher/k3s/k3s.yaml ， 可以使用指令 kubectl --kubeconfig=/etc/rancher/k3s/k3s.yaml get pod -A 查看k3s集群内信息

10. 当发现k3s集群不健康时，尽可执行bke reset && bke init 重建，无需深究修复k3s，它只是一个过渡服务

11. bke init 生成的bkecluster配置文件/bke/cluster/bkecluster.yaml，修改该配置文件并执行bke cluster create -f /bke/cluster/bkecluster.yaml

12. bke cluster create -f xxx 指令是向k3s集群提交了一个bc的crd文件，k3s集群内的cluster-api-bke会监听该crd资源的变化并启动部署集群

13. bke cluster create 会持续输出日志，这个日志实质是获取的k3s bc那个crd的事件，可以随时中断并不会影响集群部署

14. 中断bke cluster输出后可用通过 kubectl get event -n bke-cluster -w 指令获取事件输出

15. cluster-api-bke安装集群的方式是，先向目标节点发送一个bkeagent，通过systemctl status bkeagent查看状态，cluster-api-bke发送指令bkeagent去执行

16. bkegaent服务的日志在/var/log/openFuyao/bkeagent.log，通过查看该日志可以获知当前节点在干啥

17. 当目标k8s集群安装完成后，开始安装bkecluster.yaml中的addons组件，其中网络组件是阻塞安装即必须容器启动成功才会执行接下来组件的安装

18. 附加组件中有一个bocoperator的服务，当该服务安装到portal集群后，他将启动boc服务的安装，可以通过查看它的日志来获取boc安装进度

19. 基于现场的复杂情况，有时pod无法running成功，可以迅速修改集群内的yaml让pod running，operator服务判断组件是否成功的唯一标准就是pod running状态

20. 各个组件的安装yaml是一个sidecar名为manifests附加到了bocoperator以及cluster-api-bke pod中，若想永久修改yaml内容需要重新打包镜像并替换该容器
`

	fmt.Print(content)
}
