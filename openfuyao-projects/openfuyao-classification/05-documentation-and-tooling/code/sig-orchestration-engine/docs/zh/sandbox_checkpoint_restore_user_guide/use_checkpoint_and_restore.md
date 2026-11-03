# Agent沙箱Checkpoint与Restore

## 特性介绍

Checkpoint（检查点）与Restore（恢复）是OpenKruise Agents提供的沙箱快照与恢复能力。依托Checkpoint可对运行中的Sandbox捕获包含内存、文件系统的完整快照，再通过Restore基于已有快照快速拉起全新Sandbox实例，实现业务环境的保存与复用。

### 应用场景

- **AI Agent会话恢复**：Agent执行到中间状态后创建快照，后续可从快照恢复继续执行，避免重复初始化和重复消耗Token。
- **环境模板化复用**：将已安装好依赖、配置好的Sandbox环境制作成快照，快速批量克隆出多个相同环境的Sandbox。
- **开发调试**：在关键节点创建快照，出现问题时可快速回退到已知状态。
- **弹性伸缩**：结合SandboxSet预热池，基于快照快速扩容出带有业务状态的Sandbox实例。

### 能力范围

- 对运行中的Sandbox创建全量快照（内存+文件系统）。
- 支持多种存储后端（Local、S3、MinIO、CSI、NFS）持久化快照归档。
- 基于快照快速恢复出新的Sandbox实例，恢复后的Sandbox包含快照时的完整内存和文件系统状态。
- 支持快照自动过期（TTL）和手动删除（自动清理远端归档）。
- 通过E2B SDK兼容API触发Checkpoint和Restore。

### 亮点特征

- **声明式管理**：通过Kubernetes CRD（Checkpoint）声明式管理快照生命周期，与K8s生态无缝集成。
- **VM级快照**：基于Kata Containers的VM级隔离，快照包含完整虚拟机状态，恢复后进程、网络连接等均可保持原有状态。
- **多存储后端**：支持Local、S3、MinIO、CSI、NFS五种存储后端，满足不同场景的持久化需求。
- **数据完整性校验**：快照产物包含SHA256校验和，恢复时自动验证，确保数据未被篡改。
- **SandboxTemplate联动**：创建Checkpoint时自动生成SandboxTemplate，恢复时无需手动重建Pod配置。
- **E2B SDK兼容**：通过E2B SDK可直接使用Checkpoint ID触发恢复，降低集成成本。

### 基本概念

**表1** 基本概念

| 概念 | 说明 |
|------|------|
| Checkpoint | 快照操作的CRD载体，定义目标Sandbox、持久化内容、TTL等参数，跟踪快照生命周期（Pending→Creating→Succeeded/Failed）。 |
| Restore | 从已有Checkpoint恢复出新Sandbox的过程，通过在Sandbox上设置`agents.kruise.io/restore-from`注解触发。 |
| SandboxTemplate | 快照关联的Pod模板，创建Checkpoint时自动生成，保存源Sandbox的PodSpec、VolumeClaimTemplates等信息，Restore时用于构建新Sandbox。 |
| sandbox-node-agent | 节点级DaemonSet组件，负责与containerd/kata-shim通信执行快照创建、归档上传、快照下载等操作。 |
| ArchiveLocation | 快照归档在存储后端中的URI标识（如`local:///var/lib/checkpoints/...`、`s3://bucket/...`）。 |

### 实现原理

#### Checkpoint创建流程

Checkpoint Controller通过gRPC与节点上的sandbox-node-agent通信，完成快照创建。

```
用户创建Checkpoint CR
       │
       ▼
Controller: 解析源Sandbox和Pod
       │
       ▼
Controller: 检测运行时类型（仅支持kata-qemu）
       │
       ▼
Controller: Preflight检查node-agent可用性
       │
       ▼
Controller: gRPC调用node-agent CreateCheckpoint
       │
       ▼
node-agent: containerd→kata-shim创建快照归档
       │
       ▼
node-agent: 上传归档到存储后端+写入metadata.json
       │
       ▼
Controller: 轮询GetCheckpointStatus直至COMPLETED
       │
       ▼
Checkpoint Phase→Succeeded，ArchiveLocation已填充
```

#### Restore恢复流程

```
用户创建Sandbox（带restore-from注解）
       │
       ▼
Sandbox Controller: 检测restore-from注解
       │
       ▼
Controller: 获取Checkpoint CR，校验状态（Succeeded+ArchiveLocation+RuntimeType）
       │
       ▼
Controller: 构建Restore Pod，注入归档位置、运行时类型等注解
       │
       ▼
kata-shim: 启动时读取Pod注解，gRPC调用node-agent DownloadSnapshot
       │
       ▼
node-agent: 从存储后端下载归档，校验SHA256，解压到/var/lib/kata-restore/<pod-uid>/
       │
       ▼
kata-shim: 基于快照恢复VM，Pod进入Running
       │
       ▼
Controller: Sandbox Phase→Running，异步清理节点本地快照文件
```

### 与相关特性的关系

**表2** 与相关特性的关系

| 特性 | 关系 |
|------|------|
| Sandbox | Checkpoint的目标对象；Restore的产物是新的Sandbox。 |
| SandboxTemplate | 创建Checkpoint时自动生成，保存源Sandbox的Pod模板；Restore时Controller从SandboxTemplate获取PodSpec。 |
| SandboxSet | 可结合Checkpoint预热Sandbox池，加速Restore后的Sandbox获取。 |
| SandboxClaim | 可声明式获取已Restore的Sandbox。 |
| sandbox-node-agent | Checkpoint/Restore的节点级执行组件，提供快照创建、归档上传、快照下载等服务。 |

### 相关实例

通过E2B SDK可直接使用Checkpoint ID触发恢复：

```python
from e2b import Sandbox

# templateID 设为 Checkpoint 名称，自动触发 Restore
sandbox = Sandbox(template="my-checkpoint-name")
sandbox.wait_ready()

print(f"Sandbox ID: {sandbox.sandbox_id}")
```

sandbox-manager内部自动完成以下步骤：

1. 检测到`templateID`对应一个Succeeded Checkpoint。
2. 查找关联的SandboxTemplate。
3. 创建Sandbox CR（带`restore-from`注解）。
4. Controller接管后续Restore流程。
5. 等待Sandbox进入Running状态。

## 安装

### 前提条件

1. Kubernetes集群为v1.26及以上版本。
2. 已部署OpenKruise Agents（包括agent-sandbox-controller、sandbox-manager）。
3. Sandbox使用Kata Containers运行时（`kata-qemu`）。
4. 已启用Checkpoint和Restore Feature Gate。

### 开始安装

1. 启用Feature Gate

    Checkpoint和Restore功能默认关闭，需要在controller启动参数中手动开启：

    ```
    --feature-gates=SandboxCheckpoint=true,SandboxRestore=true
    ```

    修改已有部署：

    ```bash
    kubectl edit deploy -n sandbox-system agent-sandbox-controller
    ```

    在容器的`args`中找到`--feature-gates`参数，添加`SandboxCheckpoint=true,SandboxRestore=true`。

2. 部署sandbox-node-agent

    sandbox-node-agent以DaemonSet形式运行在每个需要执行快照和恢复的节点上。根据部署环境选择在线部署或离线部署方式。

    **方式一：在线部署（推荐）**

    在可访问公网及镜像仓库的环境下，直接使用一键部署脚本完成部署：

    ```bash
    ./hack/deploy.sh
    ```

    脚本将自动完成ServiceAccount创建、镜像拉取与DaemonSet部署。部署完成后跳至“验证部署”步骤。

    **方式二：离线部署**

    在内网或无公网环境下，需手动构建镜像并部署。

    2.1 离线构建sandbox-node-agent：

    ```bash
    go build -o bin/node-agent ./cmd/node-agent
    docker build -t your-registry/node-agent:latest -f Dockerfile.node-agent .
    docker push your-registry/node-agent:latest
    ```

    2.2 创建ServiceAccount：

    ```bash
    kubectl apply -f config/rbac/node_agent_sa.yaml
    ```

    2.3 修改`config/node-agent/daemonset.yaml`中的镜像地址并部署：

    ```yaml
    containers:
      - name: agent
        image: your-registry/node-agent:latest
    ```

    ```bash
    kubectl apply -f config/node-agent/daemonset.yaml
    ```

    **验证部署**

    ```bash
    kubectl get daemonset sandbox-node-agent -n sandbox-system
    kubectl get pods -n sandbox-system -l app=sandbox-node-agent -o wide
    kubectl logs -n sandbox-system -l app=sandbox-node-agent --tail=20
    ```

    正常启动后应看到：

    ```
    Starting node agent port=9090 workDir=/var/lib/node-agent storageBackend=local ...
    Node agent listening port=9090
    ```

    **表3** node-agent启动参数说明

    | 参数 | 缺省值 | 说明 |
    |------|--------|------|
    | `--grpc-port` | `9090` | gRPC服务端口。 |
    | `--work-dir` | `/var/lib/node-agent` | 工作目录。 |
    | `--storage-backend` | `local` | 存储后端类型：`local`、`s3`、`minio`、`csi`、`nfs`。 |
    | `--checkpoint-dir` | - | 本地快照目录（设置后绕过storage-backend）。 |
    | `--containerd-mode` | `real` | containerd模式：`real`（连接socket）或`mock`（测试用）。 |
    | `--max-concurrent-downloads` | `2` | 最大并发快照下载数。 |
    | `--snapshot-root` | `/var/lib/kata-restore` | Kata恢复约定路径根目录。 |

3. 配置存储后端

    node-agent支持多种存储后端来保存快照归档，通过`--storage-backend`参数指定，各后端通过环境变量配置。使用`--storage-backend`时，必须**不设置**`--checkpoint-dir`参数。如果同时设置了`--checkpoint-dir`，node-agent会走直写路径，完全绕过storage-backend配置。

    3.1 Local后端

    **表4** Local后端环境变量

    | 环境变量 | 缺省值 | 必填 | 说明 |
    |----------|--------|------|------|
    | `STORAGE_LOCAL_BASE_DIR` | `/var/lib/checkpoints/local` | 否 | 本地存储根目录 |

    执行如下命令配置Local后端环境变量：

    ```bash
    kubectl set env daemonset/sandbox-node-agent -n sandbox-system \
      STORAGE_LOCAL_BASE_DIR=/var/lib/checkpoints
    ```

    Checkpoint产物归档路径：`<STORAGE_LOCAL_BASE_DIR>/<namespace>/<checkpointID>/checkpoint.tar.gz`

    3.2 CSI后端

    **表5** CSI后端环境变量

    | 环境变量 | 缺省值 | 必填 | 说明 |
    |----------|--------|------|------|
    | `STORAGE_CSI_VOLUME_PATH` | - | 是 | CSI卷挂载路径 |

    执行如下命令配置CSI后端环境变量：

    ```bash
    kubectl set env daemonset/sandbox-node-agent -n sandbox-system \
      STORAGE_CSI_VOLUME_PATH=/mnt/csi-checkpoints
    ```

    Checkpoint产物归档路径：`<STORAGE_CSI_VOLUME_PATH>/checkpoints/<namespace>/<checkpointID>/checkpoint.tar.gz`

    3.3 NFS后端

    **表6** NFS后端环境变量

    | 环境变量 | 缺省值 | 必填 | 说明 |
    |----------|--------|------|------|
    | `STORAGE_NFS_MOUNT_PATH` | - | 是 | NFS挂载路径（需挂载到所有节点） |

    执行如下命令配置NFS后端环境变量：

    ```bash
    kubectl set env daemonset/sandbox-node-agent -n sandbox-system \
      STORAGE_NFS_MOUNT_PATH=/mnt/nfs-checkpoints
    ```

    Checkpoint产物归档路径：`<STORAGE_NFS_MOUNT_PATH>/checkpoints/<namespace>/<checkpointID>/checkpoint.tar.gz`

    3.4 S3后端

    **表7** S3后端环境变量

    | 环境变量 | 缺省值 | 必填 | 说明 |
    |----------|--------|------|------|
    | `STORAGE_S3_ENDPOINT` | - | 是 | S3服务端点 |
    | `STORAGE_S3_ACCESS_KEY_ID` | - | 是 | 访问密钥ID |
    | `STORAGE_S3_SECRET_ACCESS_KEY` | - | 是 | 访问密钥 |
    | `STORAGE_S3_BUCKET_NAME` | - | 是 | Bucket名称 |
    | `STORAGE_S3_REGION` | `us-east-1` | 否 | 区域 |
    | `STORAGE_S3_USE_SSL` | `true` | 否 | 是否使用SSL |
    | `STORAGE_S3_PREFIX` | `checkpoints` | 否 | 对象键前缀 |

    执行如下命令配置S3后端环境变量：

    ```bash
    kubectl set env daemonset/sandbox-node-agent -n sandbox-system \
      STORAGE_S3_ENDPOINT=s3.amazonaws.com \
      STORAGE_S3_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE \
      STORAGE_S3_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY \
      STORAGE_S3_BUCKET_NAME=my-checkpoints
    ```

    Checkpoint产物归档路径：`s3://<bucket>/<prefix>/<namespace>/<checkpointID>/checkpoint.tar.gz`

    3.5 MinIO后端

    **表8** MinIO后端环境变量

    | 环境变量 | 缺省值 | 必填 | 说明 |
    |----------|--------|------|------|
    | `STORAGE_MINIO_ENDPOINT` | - | 是 | MinIO服务端点 |
    | `STORAGE_MINIO_ACCESS_KEY_ID` | - | 是 | 访问密钥ID |
    | `STORAGE_MINIO_SECRET_ACCESS_KEY` | - | 是 | 访问密钥 |
    | `STORAGE_MINIO_BUCKET_NAME` | - | 是 | Bucket名称 |
    | `STORAGE_MINIO_REGION` | `us-east-1` | 否 | 区域 |
    | `STORAGE_MINIO_USE_SSL` | `false` | 否 | 是否使用SSL |
    | `STORAGE_MINIO_PREFIX` | `checkpoints` | 否 | 对象键前缀 |

    执行如下命令配置MinIO后端环境变量：

    ```bash
    kubectl set env daemonset/sandbox-node-agent -n sandbox-system \
      STORAGE_MINIO_ENDPOINT=minio:9000 \
      STORAGE_MINIO_ACCESS_KEY_ID=minioadmin \
      STORAGE_MINIO_SECRET_ACCESS_KEY=minioadmin \
      STORAGE_MINIO_BUCKET_NAME=checkpoints
    ```

    Checkpoint产物归档路径：`minio://<bucket>/<prefix>/<namespace>/<checkpointID>/checkpoint.tar.gz`
    
4. 部署Kata

    4.1 安装Kata Containers
    
    ```bash
    # 方式一：使用kata-deploy（推荐）
    KATA_VERSION="3.30.0"
    kubectl apply -f "https://github.com/kata-containers/kata-containers/releases/download/${KATA_VERSION}/kata-deploy-base.yaml"
    kubectl apply -f "https://github.com/kata-containers/kata-containers/releases/download/${KATA_VERSION}/kata-deploy-stable.yaml"

    # 等待kata-deploy完成
    kubectl -n kube-system wait --for=condition=Ready pod -l name=kata-deploy --timeout=300s

    # 验证RuntimeClass
    kubectl get runtimeclass kata-qemu
    # 期望输出: NAME       HANDLER     AGE
    #           kata-qemu  kata-qemu  ...

    # 方式二：手动安装（如果kata-deploy不可用）
    # 下载预编译包
    wget "https://github.com/kata-containers/kata-containers/releases/download/${KATA_VERSION}/kata-static-${KATA_VERSION}-x86_64.tar.xz"
    sudo tar -xvf kata-static-${KATA_VERSION}-x86_64.tar.xz -C /

    # 验证kata安装
    /opt/kata/bin/kata-runtime kata-env | head -20
    
    ```
     4.2 配置containerd
    ```bash
    # 确保containerd 配置支持kata
    sudo mkdir -p /etc/containerd

    # 如果还没有config.toml，生成默认配置
    if [ ! -f /etc/containerd/config.toml ]; then
        sudo containerd config default | sudo tee /etc/containerd/config.toml
    fi

    # 添加kata运行时配置（如果kata-deploy没有自动配置）
    sudo tee -a /etc/containerd/config.toml <<'EOF'

    [plugins."io.containerd.grpc.v1.cri".containerd.runtimes.kata-qemu]
      runtime_type = "io.containerd.kata-qemu.v2"
      privileged_without_host_devices = true
      runtime_path = "/opt/kata/bin/containererd-shim-kata-v2"
      pod_annnotations = ["io.katacontainers.*", "agents.kruise.io/*"]

    [plugins."io.containerd.grpc.v1.cri".containerd.runtimes.kata-qemu.options]
      ConfigPath = "/opt/kata/share/defaults/kata-containers/configuration-qemu.toml"
    EOF
    
    # 重启 containerd
    sudo systemctl restart containerd

    # 验证 kata 运行时可用
    crictl --runtime-endpoint unix:///var/run/containerd/containerd.sock info 2>/dev/null | head -5

    ```

    4.3 编译自定义kata-shim

    ```bash
    # 克隆 kata-containers（使用包含 Checkpoint/Restore 实现的分支）
    cd /opt
    git clone <your-kata-repo-url> kata-containers
    cd kata-containers
    git checkout kata-shim-v2 # 包含 Checkpoint/Restore 实现的分支

    # 生成 config-settings.go
    cd src/runtime
    cp pkg/katautils/config-settings.go.in pkg/katautils/config-settings.go
    sed -i \
        -e 's|@RUNTIME_NAME@|containerd-shim-kata-v2|g' \
        -e 's|@PROJECT_NAME@|Kata Containers|g' \
        -e 's|@PROJECT_TYPE@|kata|g' \
        -e 's|@PROJECT_URL@|https://github.com/kata-containers|g' \
        -e 's|@PROJECT_ORG@|kata-containers|g' \
        -e 's|@PKGRUNDIR@|/run/kata-containers|g' \
        -e 's|@COMMIT@|HEAD|g' \
        -e 's|@VERSION@|3.30.0|g' \
        -e 's|@CONFIG_PATH@|/opt/kata/share/defaults/kata-containers/configuration.toml|g' \
        -e 's|@SYSCONFIG@|/etc/kata-containers/configuration.toml|g' \
        pkg/katautils/config-settings.go

    # 编译
    GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go build \
        -o /tmp/containerd-shim-kata-v2 \
        ./cmd/containerd-shim-kata-v2

    echo "编译完成: /tmp/containerd-shim-kata-v2"

    # 验证二进制包含Checkpoint/Restore相关符号
    strings /tmp/containerd-shim-kata-v2 | grep -E "Checkpointing sandbox|Detected restore snapshot|restore-ready"
    # 期望输出:
    #   Checkpointing sandbox
    #   Detected restore snapshot, will restore from snapshot
    #   restore-ready

     ```
    
    4.4 安装自定义kata-shim
    
    ```bash
    # 备份原始 shim
    sudo cp /opt/kata/bin/containerd-shim-kata-v2 /opt/kata/bin/containerd-shim-kata-v2.bak
    sudo cp /usr/local/bin/containerd-shim-kata-v2 /usr/local/bin/containerd-shim-kata-v2.bak 2>/dev/null || true

    # 替换为自定义 shim
    sudo cp /tmp/containerd-shim-kata-v2 /opt/kata/bin/containerd-shim-kata-v2
    sudo cp /tmp/containerd-shim-kata-v2 /usr/local/bin/containerd-shim-kata-v2
    sudo chmod +x /opt/kata/bin/containerd-shim-kata-v2
    sudo chmod +x /usr/local/bin/containerd-shim-kata-v2

    # 验证版本
    /opt/kata/bin/containerd-shim-kata-v2 --version 2>&1 | head -3

     ```

## 使用Checkpoint和Restore

### 前提条件

- 已完成上述安装步骤（Feature Gate已启用、sandbox-node-agent已部署）。
- 已存在处于`Running`状态的Sandbox（用于创建Checkpoint）。
- 对于Restore：已存在处于`Succeeded`状态的Checkpoint，且包含`ArchiveLocation`和`RuntimeType`。
- 对于Restore：已创建对应的SandboxTemplate（创建Checkpoint时自动生成）。

### 背景信息

- Checkpoint通过Checkpoint CR（短名称`cp`）声明式管理，Controller通过gRPC与节点上的sandbox-node-agent通信。
- Restore通过在Sandbox上设置`agents.kruise.io/restore-from`注解触发，Controller自动创建带有归档信息的Restore Pod。
- 快照创建由kata-shim在VM层面完成，归档文件由node-agent上传到配置的存储后端。
- 恢复时kata-shim通过node-agent下载快照到节点本地`/var/lib/kata-restore/<pod-uid>/`目录，再从快照恢复VM。

### 使用限制

- 仅支持`kata-qemu`运行时，不支持其他RuntimeClass。
- 不支持跨VMM类型Restore（如QEMU快照不能恢复到Firecracker）。
- Checkpoint与Restore必须在同一命名空间内。
- Checkpoint创建超时时间为10分钟。
- 快照下载失败时Pod保持Pending状态，Sandbox持续处于Restoring，不会自动降级到冷启动。
- 使用`--storage-backend`时不能同时设置`--checkpoint-dir`，否则会绕过存储后端配置。

### 操作步骤

#### 创建Checkpoint

1. 确认目标Sandbox处于`Running`状态：

    ```bash
    kubectl get sandbox -n default
    ```

2. 执行`kubectl apply -f checkpoint.yaml`创建Checkpoint资源：

    ```yaml
    # checkpoint.yaml
    apiVersion: agents.kruise.io/v1alpha1
    kind: Checkpoint
    metadata:
      name: my-checkpoint
      namespace: default
    spec:
      sandboxName: my-sandbox          # 目标Sandbox名称（必填）
      keepRunning: true                # 快照后Sandbox是否继续运行，默认true
      persistentContents:              # 持久化内容
        - memory                       # 内存快照
        - filesystem                   # 文件系统快照
      ttlAfterFinished: 1d             # 完成后自动过期时间（支持s/m/h/d格式）
      targetContainer: sandbox         # 目标容器名，默认"sandbox"
    ```

3. 查看Checkpoint状态：

    ```bash
    kubectl get cp -n default
    kubectl describe cp my-checkpoint -n default
    ```

    输出示例：

    ```
    NAME             STATUS      AGE
    my-checkpoint    Succeeded   2m
    ```

    Checkpoint状态流转：

    ```
    Pending → Creating → Succeeded / Failed
    ```

    Conditions变化：

    **表9** Conditions变化

    | 阶段 | Progressing | Archived | Ready |
    |------|-------------|----------|-------|
    | Creating | True (AgentInvoked) | - | - |
    | Succeeded | 已移除 | True (UploadComplete) | True (CheckpointComplete) |
    | Failed | 已移除 | - | - |

    常见错误场景：

    **表10** Checkpoint常见错误场景

    | 场景 | Message示例 |
    |------|--------------|
    | 源Sandbox不存在 | `SourceSandboxNotFound` |
    | 运行时检测失败 | `RuntimeDetectionFailed` |
    | node-agent不可用 | Preflight检查失败，Controller自动重试 |
    | 创建超时（10分钟） | `Timeout` |

#### 从Checkpoint恢复（Restore）

**通过kubectl创建Restore Sandbox：**

1. 执行`kubectl apply -f restore-sandbox.yaml`创建Sandbox，通过`agents.kruise.io/restore-from`注解指定Checkpoint名称：

    ```yaml
    # restore-sandbox.yaml
    apiVersion: agents.kruise.io/v1alpha1
    kind: Sandbox
    metadata:
      name: my-restored-sandbox
      namespace: default
      annotations:
        agents.kruise.io/restore-from: my-checkpoint    # 指向已完成的Checkpoint
    spec:
      templateRef:
        name: my-sandbox-template                       # 引用SandboxTemplate
      template:
        spec:
          runtimeClassName: kata-qemu                   # 必须使用kata运行时
          containers:
            - name: sandbox
              image: nginx:stable-alpine3.20
              ports:
                - containerPort: 80
          restartPolicy: Never
    ```

    Controller检测到`restore-from`注解后，会自动创建RestorePod并注入以下注解：

    **表11** RestorePod注解说明

    | Pod注解 | 说明 |
    |----------|------|
    | `agents.kruise.io/archive-location` | 快照归档位置（来自Checkpoint.Status.ArchiveLocation） |
    | `agents.kruise.io/restore-checkpoint-ref` | Checkpoint名称 |
    | `agents.kruise.io/restore-runtime-type` | 运行时类型（如`kata-qemu`） |
    | `agents.kruise.io/restore-vmm-type` | VMM类型（如`qemu`，从RuntimeType中提取） |

    **通过E2B API触发Restore：**

    ```python
    from e2b import Sandbox

    sandbox = Sandbox(template="my-checkpoint-name")
    sandbox.wait_ready()
    print(f"Sandbox ID: {sandbox.sandbox_id}")
    ```

2. 查看Restore状态：

    ```bash
    kubectl get sandbox my-restored-sandbox -n default
    ```

    输出示例：

    ```
    NAME                    PHASE       AGE
    my-restored-sandbox     Running     2m
    ```

    查看详细信息：

    ```bash
    kubectl describe sandbox my-restored-sandbox -n default
    ```

    输出示例：

    ```
    Name:         my-restored-sandbox
    Namespace:    default
    Annotations:  agents.kruise.io/restore-from: my-checkpoint
    Status:
      Phase:                    Running
      Restore From Checkpoint:  my-checkpoint
      Restore Snapshot Path:    /var/lib/kata-restore/abc-123-def
      Restore Completion Time:  2026-06-03T10:30:00Z
      Node Name:                k8s-master
      Sandbox Ip:               10.244.0.5
    Conditions:
      Type        Status   Reason             Message
      ----        ------   ------             -------
      Restoring   False    RestoreCompleted   Sandbox restored from checkpoint my-checkpoint
      Ready       True     PodReady           Sandbox is ready
    ```

    Restore状态流转：

    ```
    Sandbox创建 → Restoring（Pod创建中）→ Running（Restore完成）
    ```

    异常场景：

    **表12** Restore异常场景

    | 场景 | Sandbox Phase | Message示例 |
    |------|---------------|--------------|
    | Checkpoint不存在 | `Failed` | `Checkpoint my-checkpoint not found` |
    | Checkpoint未成功 | `Failed` | `Checkpoint my-checkpoint is not Succeeded (phase: Failed)` |
    | Checkpoint无ArchiveLocation | `Failed` | `Checkpoint my-checkpoint has no ArchiveLocation` |
    | Checkpoint无RuntimeType | `Failed` | `Checkpoint my-checkpoint has no RuntimeType` |
    | 快照下载失败 | `Restoring`（持续） | Pod保持Pending，Sandbox持续Restoring |

### 后续操作

- Restore完成后，可通过kubectl或E2B API正常使用恢复后的Sandbox。
- Controller会异步清理节点本地的快照文件（`/var/lib/kata-restore/<pod-uid>/`）。
- 可基于恢复后的Sandbox再次创建Checkpoint，实现快照链。

### 相关操作

- 查看快照产物

  Checkpoint成功后，获取归档位置：

  ```bash
  kubectl get cp my-checkpoint -n default -o jsonpath='{.status.archiveLocation}'
  ```

  根据配置的存储后端类型，前往对应位置查看产物：

  **表13** 快照产物查看方式

  | 存储后端 | 查看方式 |
  |----------|----------|
  | `local` | 登录节点，查看`--checkpoint-dir`目录（如`/var/lib/checkpoints`） |
  | `s3` / `minio` | 通过S3控制台或CLI查看对应bucket |
  | `nfs` | 挂载NFS共享路径后查看 |
  | `csi` | 挂载CSI卷后查看 |

- 删除Checkpoint

  ```bash
  kubectl delete cp my-checkpoint -n default
  ```

  删除Checkpoint时Controller会自动清理远端归档文件。

- 设置自动过期（TTL）

  在`spec.ttlAfterFinished`中设置过期时间，Checkpoint完成后将被自动删除：

  ```yaml
  spec:
    ttlAfterFinished: 30m    # 30分钟后自动删除
  ```

  支持的格式：

  **表14** TTL时间格式

  | 格式 | 示例 | 说明 |
  |------|------|------|
  | 秒 | `30s` | 30秒 |
  | 分钟 | `30m` | 30分钟 |
  | 小时 | `2h` | 2小时 |
  | 天 | `1d` | 1天 |

## 附录

### Checkpoint Spec字段参考

**表15** Checkpoint Spec字段

| 字段 | 类型 | 必填 | 缺省值 | 说明 |
|------|------|------|--------|------|
| `sandboxName` | string | 是 | - | 目标Sandbox名称 |
| `podName` | string | 否 | - | 目标Pod名称（与sandboxName二选一） |
| `keepRunning` | bool | 否 | `true` | 快照后Sandbox是否继续运行 |
| `persistentContents` | []string | 否 | - | 持久化内容：`memory`、`filesystem` |
| `ttlAfterFinished` | string | 否 | - | 完成后自动过期时间 |
| `targetContainer` | string | 否 | `sandbox` | 目标容器名称 |
| `backend` | string | 否 | `containerd` | 后端类型：`containerd`或`envd` |
| `runtimeOptions` | object | 否 | - | 运行时特定选项 |

### Checkpoint Status字段参考

**表16** Checkpoint Status字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `phase` | string | 当前阶段：`Pending`、`Creating`、`Succeeded`、`Failed`、`Terminating` |
| `archiveLocation` | string | 快照归档URI（如`local:///var/lib/checkpoints/...`） |
| `archiveSize` | int64 | 归档文件大小（字节） |
| `checksum` | string | 归档文件SHA256校验和 |
| `nodeName` | string | 执行快照的节点名称 |
| `containerID` | string | 容器ID |
| `runtimeType` | string | 运行时类型（如`kata-qemu`） |
| `vmmType` | string | VMM类型（如`qemu`） |
| `completionTime` | time | 完成时间 |
| `conditions` | []Condition | 生命周期条件（Progressing、Archived、Ready） |

### Restore Status字段参考

**表17** Restore Status字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `restoreFromCheckpoint` | string | 源Checkpoint名称 |
| `restoreSnapshotPath` | string | 节点本地快照路径（`/var/lib/kata-restore/<pod-uid>`） |
| `restoreCompletionTime` | time | Restore完成时间 |

### node-agent必需的主机路径挂载

**表18** node-agent主机路径挂载

| 路径 | 用途 |
|------|------|
| `/run/containerd` | 连接containerd socket |
| `/var/lib/node-agent` | Agent工作目录 |
| `/var/lib/checkpoints` | 快照存储目录 |
| `/var/lib/kata-restore` | Kata恢复约定路径 |

### 快照产物结构

Checkpoint成功后，存储后端会写入以下目录结构：

```
<storage-root>/
└── <namespace>/
    └── <checkpointID>/
        ├── checkpoint.tar.gz         # 快照归档文件
        ├── checkpoint.tar.gz.sha256  # SHA256校验和文件
        └── metadata.json             # 快照元数据
```

**metadata.json**包含以下字段：

**表19** metadata.json字段

| 字段 | 说明 |
|------|------|
| `CheckpointID` | Checkpoint CR名称 |
| `Namespace` | Kubernetes命名空间 |
| `ContainerID` | 容器ID |
| `NodeName` | 执行快照的节点名称 |
| `RuntimeType` | 运行时类型（如`kata-qemu`） |
| `CreationTime` | 快照创建时间 |
| `ArchiveSize` | 归档文件大小（字节） |
| `ChecksumSHA256` | 归档文件的SHA256校验和 |

**数据完整性验证**：

```bash
# 计算实际文件的SHA256
sha256sum /var/lib/checkpoints/<namespace>/<cpID>/checkpoint.tar.gz

# 读取记录的校验和
cat /var/lib/checkpoints/<namespace>/<cpID>/checkpoint.tar.gz.sha256

# 比对两者是否一致
```

如果校验和不一致，说明文件可能已损坏或被篡改，不应使用该快照进行恢复。

## FAQ

1. Checkpoint创建后一直处于Pending状态

    检查Feature Gate是否已启用（`SandboxCheckpoint=true`），以及sandbox-node-agent是否在目标节点运行。

2. Checkpoint创建失败，状态为Failed

    **可能原因**

    - 源Sandbox不存在或未运行（`SourceSandboxNotFound`）
    - 运行时检测失败，未使用kata-qemu（`RuntimeDetectionFailed`）
    - node-agent不可用或Preflight检查失败
    - 快照创建超时（10分钟）

    **解决办法**

    通过`kubectl describe cp <name>`查看`Message`字段定位原因，检查node-agent日志：`kubectl logs -n sandbox-system -l app=sandbox-node-agent`。

3. Restore后Sandbox一直处于Restoring状态

    **现象描述**

    Sandbox持续处于`Restoring`阶段，Pod保持Pending状态。

    **可能原因**

    - 快照下载失败（归档文件不存在或存储后端不可达）
    - kata-shim恢复失败

    **解决办法**

    检查Pod Events（`kubectl describe pod <pod-name>`）和node-agent日志。快照下载失败时Pod不会自动降级到冷启动，需手动排查存储后端连通性。

4. 能否跨VMM类型恢复（如QEMU快照恢复到Firecracker）

    不支持。Restore时必须使用与Checkpoint相同的运行时类型（`RuntimeType`）。
