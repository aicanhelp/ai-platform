# victoriametrics-controller

victoriametrics-controller用于在K8s集群上快速部署、卸载VictoriaMetrics监控栈。

## 镜像构建

### 环境准备

若您使用`nerdctl`进行构建，可以忽略这一步。

使用`Docker`进行构建时，为支持多架构镜像构建，需要额外安装插件`buildx`以扩展`Docker`的构建功能。具体安装方式参见[buildx官方安装指导文档](https://github.com/docker/buildx)。

### 构建命令

#### 构建并推送到指定OCI仓库

**使用docker**

```bash
docker buildx build . -f <path/to/dockerfile> \
    -o type=image,name=<oci/repository>:<tag>,oci-mediatypes=true,rewrite-timestamp=true,push=true \
    --platform=linux/amd64,linux/arm64 \
    --provenance=false 
```

**使用nerdctl**

```bash
nerdctl build . -f <path/to/dockerfile> \
    -o type=image,name=<oci/repository>:<tag>,oci-mediatypes=true,rewrite-timestamp=true,push=true \
    --platform=linux/amd64,linux/arm64 \
    --provenance=false
```

其中，`<path/to/dockerfile>`为Dockerfile路径，`<oci/repository>`为镜像地址，`<tag>`为镜像tag。

#### 构建并导出OCI Layout到本地tarball

**使用docker**

```bash
docker buildx build . -f <path/to/dockerfile> \
    -o type=oci,name=<oci/repository>:<tag>,dest=<path/to/oci-layout.tar>,rewrite-timestamp=true \
    --platform=linux/amd64,linux/arm64 \
    --provenance=false
```

**使用nerdctl**

```bash
nerdctl build . -f <path/to/dockerfile> \
    -o type=oci,name=<oci/repository>:<tag>,dest=<path/to/oci-layout.tar>,rewrite-timestamp=true \
    --platform=linux/amd64,linux/arm64 \
    --provenance=false
```

其中，`<path/to/dockerfile>`为Dockerfile路径，`<oci/repository>`为镜像地址，`<tag>`为镜像tag，`path/to/oci-layout.tar`为tar包路径。

#### 构建并导出镜像rootfs到本地目录

**使用docker**

```bash
docker buildx build . -f <path/to/dockerfile> \
    -o type=local,dest=<path/to/output>,platform-split=true \
    --platform=linux/amd64,linux/arm64 \
    --provenance=false
```

**使用nerdctl**

```bash
nerdctl build . -f <path/to/dockerfile> \
    -o type=local,dest=<path/to/output>,platform-split=true \
    --platform=linux/amd64,linux/arm64 \
    --provenance=false
```

其中，`<path/to/dockerfile>`为Dockerfile路径，`path/to/output`为本地目录路径。

## 部署

### 使用bke部署集群

参考：[victoriametrics_controller使用指导](https://gitcode.com/openFuyao/sig-installation/blob/openFuyao-v26.03/docs/zh/user_guide/cluster_installation_deployment/victoriametrics_controller_user_guide.md)

### 非bke部署集群

1. 编辑`deploy.yaml`中镜像地址和参数配置。
   
   修改`spec.template.spec.initContainers.image`和`spec.template.Containers.image`为构建镜像地址`<oci/repository>:<tag>`，如下。
   ```
   initContainers:
      - name: installer
        image: <oci/repository>:<tag>
        imagePullPolicy: Always

    ...

   containers:
      - name: victoriametric-controller
        env:
          - name: OPENFUYAO_REGISTRY
            value: "cr.openfuyao.cn/openfuyao"
        securityContext:
          privileged: true
        image: <oci/repository>:<tag>
        imagePullPolicy: Always
   ```
   修改`spec.template.spec.initContainers.env`字段下的vimtoriametrics相关配置。
   ```
    - name: OPENFUYAO_REGISTRY
      value: "cr.openfuyao.cn/openfuyao"
    - name: VERSION
      value: "0.58.2"
    - name: useVMSingle
      value: "false"
    - name: vmSingleStorageSize
      value: "50Gi"
    - name: vmAgentAllowStatefulSet
      value: "true"
    - name: vmAgentCpuCount
      value: "0.5"
    - name: vmAgentMemorySize
      value: "2Gi"
    - name: vmAgentStorageSize
      value: "10Gi"
    - name: vmAgentReplicaCount
      value: "1"
    - name: vmAgentShareCount
      value: "2"

    ...
   ```

2. 执行以下命令，应用配置文件。
   
   ```
   kubectl apply -f deployment.yaml
   ```
   
   等待victoriametrics完成安装。