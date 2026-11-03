# 基于UB共享内存的容器热迁移

## 特性介绍

本特性旨在提供秒级容器热迁移能力，在不停止容器的情况下，将容器从一个节点迁移到另外一个节点，做到秒级切换，上层业务不中断，无感知，提高业务可用性，并通过结合使用UB资源池化和高速互联总线特性的能力，实现更细粒度、更灵活的编排调度能力，均衡集群各节点的负载占用，提升整体集群的资源利用率。

### 应用场景

- 作为集群管理员，在节点配置热迁移的节点资源利用率水位线，在节点出现资源占用较高时，自动识别可热迁移服务，通过下发热迁移CR，将部分业务迁移至其他资源充足节点，保证集群整体稳定运行。

- 当集群各节点资源使用率普遍偏高时，新Pod可能因单节点资源不足而无法调度，尽管集群总体剩余资源仍能满足其需求。此时，可通过集群管理组件自动识别支持迁移的服务，并对其进行动态调度，从而在合适节点上腾出足够资源，确保新Pod成功调度并运行。

### 能力范围

- 热迁移只支持单集群内同构节点间热迁移。
- 不修改K8s代码，新增热迁移的workload类型，不支持对存量workload进行迁移。
- 当前不支持IP地址不变更。
- 当前不支持大页内存和共享内存的增量加载。

### 亮点特征

基于容器热迁移能力，结合UB超节点中提供的资源池化和高速互联总线等特性，动态感知节点资源变化，触发集群Pod的重均衡热迁移，达到容器脏页200MB内存规格下，热迁移停机时长<1.5s。

### 实现原理

#### 容器热迁移

通过K8s的声明机制，增加热迁移CRD，新增热迁移Controller与Agent组件，来调谐热迁移的CR，Agent组件需要与运行时交互，下发`checkpoint/restore`接口，运行时调用CRIU进行真正的`checkpoint`和`restore`。

  **新增组件：**

  - Migrate Controller：热迁移控制器，控制整体的迁移流程，对CR进行调谐，根据当前状态下发下一步的操作，容器运行，主备选主运行。
  - Migrate Agent：热迁移过程的履行组件，与运行时交互，在源节点进行`checkpoint`，传输数据到目标节点，在目标节点进行`restore`，并上报步骤履行状态，容器运行，daemonset部署。

​        **需要适配开源组件：**

​            Containerd：容器运行时，支持容器冻结，支持增量checkpoint与restore。

​            Runc：适配增量checkpoint与restore接口。

​            CRIU：实现增量的checkpoint与restore。

#### UB调度

在容器场景实现内存共享，需要将UBS Memory的接口暴露给容器，容器进程需要显式调用UBS Memory接口进行共享内存的创建、加载、读写操作，进程通过mmap方式加载使用共享内存，不受cgroup的内存子系统限制。

通过K8s的CSI扩展机制，实现Pod使用共享内存的生命周期管理。通过声明自定义资源，支持业务自定义共享内存配额，业务调用共享内存的创建、删除接口时，主动拦截请求，调用服务端做配额校验与更新。在K8s的云管理系统中，需要新增CSI组件，做共享内存设备挂载与配额校验，同时与MXE交互，实现端到端容器内存共享的流程。

  **新增组件：**

  MatrixController：超节点内存资源控制器，负责根据各物理节点压力来决策节点是否进行内存借用或归还、创建容器热迁移任务等操作。

  - 实现新的策略，来筛选当前集群中哪些节点上的哪些Pod需要被迁移。
  - 实现新的Evictor，来对这些需要被迁移的Pod进行筛选目标节点，并下发热迁移CR，触发Pod热迁移。

  MatrixAgent：内存资源操作履行器。

  - 上报节点压力，从当前节点获取内存资源使用详情，内存借用信息（numa维度）等，创建或更新CR进行上报。
  - 根据controller下发的CR，对接MXE接口，履行节点内存操作，上报CR履行状态。

#### 交互流程

1. 指标上报，matrix agent通过定时的获取节点/NUMA的内存使用情况，构造CR信息上报至kube-apiserver。

2. 压力感知决策，MatrixController启动后，根据节点资源利用率CR更新来判断是否启用均衡策略的执行，根据配置的策略获取节点信息和节点Pod信息，筛选需要重调度的Pod，发送给扩展的Evictor；Evictor通过调度器的filter、score插件来筛选热迁移Pod可调度的节点，创建热迁移CR。

3. Migrate Controller感知迁移任务后开始下发首次checkpoint，更新源Pod的annotation字段，标记为开始checkpoint。

4. 源节点上的Migrate Agent在检测到Pod需要执行checkpoint后，会从Pod的注解（annotation）中获取相关信息，并调用容器运行时发起checkpoint操作。在此过程中，Agent会预先申请共享内存，供CRIU直接将dump数据写入其中；而目标节点上的CRIU则从同一共享内存中读取数据进行恢复，从而要求源节点与目标节点处于同一共享域内。最后，Agent会更新Pod的迁移状态。

5. Controller将根据Pod中迁移状态的变化，更新CR为下一步骤（目标节点restore），下发下一阶段的restore，如果目标Pod不存在则先创建目标节点的临时Pod，同时annotation中增加热迁移的信息。

6. 目标节点Kubelet检测到新创建Pod，开始履行创建容器，调用Agent代理的CRI接口（避免修改kubelet代码），Agent中根据annotation中的信息，指定镜像使用checkpoint的镜像地址，下发到运行时进行创建容器，成功后，更新Pod的迁移状态。

7. Controller将根据Pod中迁移状态的变化，判断是否需要增量的checkpoint（脏页比例过高，未达到系统设定的阈值）还是final-checkpoint，更新源Pod annotation。

8. 源节点上的Migrate Agent将根据传入参数，调用容器运行时执行增量checkpoint或final-checkpoint（该操作将冻结进程），随后更新源Pod的迁移状态。

9. Controller在final-checkpoint完成后，下发last-restore，更新目标节点的Pod annotation。

10. 目标节点上的Migrate Agent将根据传入参数，调用容器运行时执行final-restore操作。完成后，Agent会更新目标Pod的迁移状态，最终使Pod进入正常运行状态。

11. 最终由controller做清理和恢复，如删除源Pod，删除目标Pod的annotation等信息。


## 安装

### 前置条件

- 已安装K8s集群。
- 已安装UB仿真/HCC场景。
- 已安装[openFuyao/containerd](http://gitcode.com/openFuyao/containerd/tree/of-v2.1.1)，代码仓分支：of-v2.1.1。
- 已安装[openFuyao/runc](http://gitcode.com/openFuyao/runc/tree/of-v1.1.10)，代码仓分支：of-v1.1.10。
- 已安装[criu](http://github.com/checkpoint-restore/criu/tree/v4.2)，代码仓分支：v4.2。

### 软件部署

1. 获取容器热迁移场景所需的软件包：LiveMigrate-${version}-linux.zip。

2. 部署MatrixAgent插件。

   2.1 在K8s的master/worker节点，在/openFuyao/live_migrate/目录下，均解压MatrixAgent的镜像。

      ```
      unzip -d image BeiMing-KubernetesPlugin-MatrixAgent-image-${version}.image.zip
      cd image 
      gunzip -c beiming-kubernetesplugin-matrixagent-image-${version}-aarch64.tar.gz > beiming-kubernetesplugin-matrixagent-image-${version}-aarch64.tar
      ```

   2.2 在K8s的master/worker节点，在/openFuyao/live_migrate/目录下，均导入MatrixAgent的镜像。

      ```
      ctr -n k8s.io image import beiming-kubernetesplugin-matrixagent-image-${version}-aarch64.tar
      ```

   2.3 在K8s的master/worker节点确认是否导入成功（可选）。

      执行以下命令，若返回信息中包含beiming-kubernetesplugin-matrixagent-image:，则表示导入成功。

      ```
      ctr -n k8s.io image list | grep beiming-kubernetesplugin-matrixagent-image
      ```

   2.4 在master节点的/openFuyao/live_migrate/目录下，解压MatrixAgent的chart包。

      ```
      unzip -d chart BeiMing-KubernetesPlugin-MatrixAgent-chart-${version}.image.zip
      cd chart
      tar -xvf beiming-kubernetesplugin-matrixagent-chart-${version}-noarch.tar.gz
      ```
   
      2.4.1 解压后chart内的目录结构类似如下所示。

      ```
      |-- f55fdasfdsafdsafdsafdsa1fgfd24sfsdggfposx1232fdsf113.json(文件名随机生成)
      |-- beiming-kubernetesplugin-matrixagent-chart-${version}-noarch.tar.gz
      |-- 19fdafsdafdsamfasdfapiuixnyfdteuozpmnwdsfqdsafsafdsa(文件夹名随机生成)
      |-- manifest.json
      |-- repositories
      ```
      
      2.4.2 进入到beiming-kubernetesplugin-matrixagent-chart-${version}-noarch.tar.gz解压后目录，chart文件夹内唯一的文件夹中。

      ```
      cd 19fdafsdafdsamfasdfapiuixnyfdteuozpmnwdsfqdsafsafdsa
      ```

      目录结构如下所示：

      ```
      |-- json
      |-- layer.tar.gz
      |-- VERSION
      ```

   2.5 部署MatrixAgent服务，仅需在master节点执行。
      
      2.5.1 检查helm部署历史。

      ```
      helm list --namespace kube-system | grep m
      ```
      
      2.5.2 如果`返回信息`中包含ma kube-system，则该环境之前部署过MatrixAgent服务，需执行以下命令进行清理。

      ```
      helm uninstall ma -n kube-system
      ```
      
      2.5.3 请在上述目录下执行以下命令进行部署。

      ```
      helm install ma layer.tar.gz -n kube-system
      ```

3. 部署MatrixController插件。

   3.1 在K8s的master/worker节点，在/openFuyao/live_migrate/目录下，均解压MatrixController的镜像。

      ```
      unzip -d imageController BeiMing-KubernetesPlugin-MatrixController-image-${version}.image.zip
      cd imageController 
      gunzip -c beiming-kubernetesplugin-matrixcontroller-image-${version}-aarch64.tar.gz > beiming-kubernetesplugin-matrixcontroller-image-${version}-aarch64.tar
      ```

   3.2 在K8s的master节点，在/openFuyao/live_migrate/目录下，均导入MatrixController的镜像。

      ```
      ctr -n k8s.io image import beiming-kubernetesplugin-matrixcontroller-image-${version}-aarch64.tar
      ```

   3.3 在K8s的master/worker节点确认是否导入成功（可选）。

      执行以下命令，若`返回信息`中包含beiming-kubernetesplugin-matrixcontroller-image:，则表示导入成功。

      ```
      ctr -n k8s.io image list | grep beiming-kubernetesplugin-matrixcontroller-image
      ```

   3.4 在master节点的/openFuyao/live_migrate/目录下，解压MatrixController的Chart包。
      
      ```
      unzip -d chartController BeiMing-KubernetesPlugin-MatrixController-chart-${version}.image.zip
      cd chartController
      tar -xvf beiming-kubernetesplugin-matrixcontroller-chart-${version}-noarch.tar.gz
      ```
       
      3.4.1 解压后chart内的目录结构类似如下所示：

      ```
      |-- apouixgnue,zhurewlkc0smxg3ube4wytb9epiknz8adfa2q3fd41a.json(文件名随机生成)
      |-- beiming-kubernetesplugin-matrixcontroller-chart-${version}-noarch.tar.gz
      |-- 32fdzcdsfdlfsmfasdfapiuixnyfdteuozpmnwdspamnzierzqw(文件夹名随机生成)
      |-- manifest.json
      |-- repositories
      ```

      3.4.2 进入到beiming-kubernetesplugin-matrixcontroller-chart-${version}-noarch.tar.gz解压后目录，chart文件夹内唯一的文件夹中。
         
      ```
      cd 32fdzcdsfdlfsmfasdfapiuixnyfdteuozpmnwdspamnzierzqw
      ```

      目录结构如下所示：

      ```
      |-- json
      |-- layer.tar.gz
      |-- VERSION
      ```

   3.5 部署MatrixController服务，仅需在master节点执行。
      
      3.5.1 检查helm部署历史。

      ```
      helm list --namespace kube-system | grep m
      ```

      3.5.2 如果`返回信息`中包含mc kube-system，则该环境之前部署过MatrixController服务，需执行以下命令进行清理。

      ```
      helm uninstall mc -n kube-system
      ```

      3.5.3 请在上述目录下执行以下命令进行部署。

      ```
      helm install mc layer.tar.gz -n kube-system
      ```

4. 上传软件包并解压，在K8s的master/worker节点均执行。

   ```
   # /openFuyao/live_migrate 为LiveMigrate_v1.0_linux.zip软件包解压路径
   mkdir -p /openFuyao/live_migrate
   unzip LiveMigrate_1.0_linux.zip -d /openFuyao/live_migrate
   ```

5. 执行如下命令，部署crd资源。

   ```
   kubectl apply -f migrate-crd.yaml
   ```

6. 部署migrate-agent插件。

   6.1 在K8s的master/worker节点，在/openFuyao/live_migrate目录下，均导入migrate-agent镜像。

      ```
      ctr -n k8s.io image import migrate-agent-v1.0.tar
      ```

   6.2 在K8s的master/worker节点确认是否导入成功（可选）。

      执行以下命令，若`返回信息`中包含migrate-agent，则表示导入成功。

      ```
      ctr -n k8s.io image list | grep migrate-agent
      ```

   6.3 在/openFuyao/live_migrate目录下，执行以下命令进行部署：

      ```
      kubectl apply -f migrate-agent_deploy.yaml
      ```

   6.4 migrate-agent启动完成后，修改kubelet配置，将container-runtime-endpoint改为cri代理的sock地址。

      ```
      vi /usr/lib/systemd/system/kubelet.service
      ```

      将`--container-runtime-endpoint`的值改为：

      ```
      --container-runtime-endpoint=unix:///var/run/cri-proxy.sock
      ```

   6.5 修改完成后，执行以下命令，重启kubelet。

      ```
      systemctl daemon-reload
      systemctl restart kubelet
      ```

7. 部署migrate-controller插件。

   7.1 在K8s的master节点，在/openFuyao/live_migrate目录下，均执行以下命令，导入migrate-controller镜像。

      ```
      ctr -n k8s.io image import migrate-controller-v1.0.tar
      ```

   7.2 在K8s的master节点确认是否导入成功（可选）。

      执行以下命令， 若`返回信息`中包含migrate-controller，则表示导入成功。

      ```
      ctr -n k8s.io image list | grep migrate-controller
      ```

   7.3 在/openFuyao/live_migrate目录下，执行以下命令进行部署：

      ```
      kubectl apply -f migrate-controller_deploy.yaml
      ```

## 使用容器热迁移

### 背景信息

使用本特性前，需要熟悉Kubernetes集群基本操作命令（如kubectl get pods， kubectl describe node）。


### 使用限制

- 热迁移支持基于内核态的Checkpoint&Restore。
- 依赖UB超节点的高级调度能力，包括：
  - 拓扑域（Topology Domain）感知调度。
  - 超节点维度的资源池化与调度。
  - 基于实时负载的动态建模与智能迁移决策。
### 操作步骤

1. 在/openFuyao/live_migrate目录下，执行如下命令，安装热迁移测试Pod（已有热迁移测试Pod可跳过本步骤）。

   ``` 
   kubectl apply -f mysql_stateful.yaml
   ```

2. 源Pod启动完成后，编辑迁移文件migrate.yaml，根据实际情况，填写pod、location、transportType以及dataPath字段信息。

   ```
   # migrate.yaml
   apiVersion: dhac.livemigrate.livemigrate/v1
   kind: LiveMigrateController
   metadata:
     labels:
       app.kubernetes.io/name: livemigratecontroller
       app.kubernetes.io/instance: livemigratecontroller-sample
       app.kubernetes.io/part-of: live-migration
       app.kubernetes.io/managed-by: kustomize
       app.kubernetes.io/created-by: live-migration
     name: livemigratecontroller-sample
   spec:
     # TODO(user)：Add fields here
     pod: livemigratestateful-mysql-0-2wx4d   # 热迁移Pod名称，根据实际情况修改
     srcNode: master                          # 热迁移节点名称，根据实际情况修改
     dstNode: worker                          # 热迁移目标节点，根据实际情况修改
     location: "xxxxxx"                       # checkpoint镜像数据存储目录，根据实际情况修改
     transportType: "ssh"                     # 数据传输方式（支持ssh, shareStorage、pageServer）
     dataPath: "xxxxxx"                       # 本地存储的传输数据文件目录，根据实际情况修改
   ```

   > ![输入图片说明](../figures/icon-note.gif) **说明：**<br />  
   >
   >  确保location和dataPath目录存在且migrate-agent插件对目录有读写权限，否则会导致热迁移失败。
   > 

3. 执行如下命令，触发迁移任务。

   ``` 
   kubectl apply migrate.yaml
   ```

4. 执行如下命令，查看迁移结果：

   ```
   kubectl get pod -A
   ```

   当迁移完成后，源Pod被成功删除，新Pod正常创建，且迁移前后数据保持一致，即表示本次迁移成功。