# ub-network-bandwidth-operator

UB 网络带宽 QoS Operator

## 编译准备

1. 安装 make、gcc、docker

   可通过 yum install 进行安装
2. 安装 go，版本 >= 1.24.0

   访问官方下载地址 <https://go.dev/dl/go1.24.0.linux-amd64.tar.gz> 获取发布包，执行如下命令解压到环境中：
   ```bash
   tar zxvf go1.24.0.linux-amd64.tar.gz -C /usr/local
   export PATH=/usr/local/go/bin:$PATH
   ```

## 构建

在代码根目录下执行构建

```bash
# 构建二进制
make build

# 构建镜像
make image

# 打包 Helm chart
make chart

# 构建二进制和镜像
make
```

## 部署

### 本地构建产物部署

1. 构建镜像和 Helm Chart

   ```bash
   export VERSION=latest
   export ARCH=aarch64
   export CHART_VERSION=0.0.0-latest

   TAG=${VERSION} make image
   make chart
   ```

2. 将构建中产生的镜像导入集群中
   ```bash
   ctr -n k8s.io image import _output/image/ub-network-bandwidth-${VERSION}-${ARCH}.tar
   ```
3. 使用 Helm Chart 部署
   ```bash
   helm install ub-network-bandwidth-operator _output/chart/ub-network-bandwidth-operator-${CHART_VERSION}.tgz \
     -n ub-network-system \
     --create-namespace
   ```

也可以使用 example 下的 yaml 文件部署，部署前请确认其中的镜像仓库和版本号与本地构建产物一致：

```bash
kubectl create namespace ub-network-system

kubectl apply -f example/ubs-bandwidth-config.yaml
kubectl apply -f example/ub-network-bandwidth-operator.yaml
```

### OCI 仓拉取部署

可以直接从 `cr.openfuyao.cn` 拉取镜像和 Helm Chart 后部署。部署前请确保：

- Kubernetes 集群已安装并可通过 `kubectl` 访问。
- Helm 已安装。
- UBSE 已部署。

#### 1. 设置版本变量

**执行节点：** 所有 Kubernetes 节点和控制节点。

```bash
export VERSION=latest
export OCI_VERSION=0.0.0-latest
export NAMESPACE=ub-network-system
export RELEASE_NAME=ub-network-bandwidth-operator
```

`VERSION` 为镜像 tag，`OCI_VERSION` 为 Helm Chart 版本。两者需要与 OCI 仓发布件版本保持一致。

#### 2. 拉取镜像

**执行节点：** 所有 Kubernetes 节点，包括控制节点和工作节点。

containerd 环境执行：

```bash
ctr -n k8s.io images pull cr.openfuyao.cn/openfuyao/ub-network-bandwidth-operator:${VERSION}
```

Docker 环境执行：

```bash
docker pull cr.openfuyao.cn/openfuyao/ub-network-bandwidth-operator:${VERSION}
```

#### 3. 拉取 Helm Chart

**执行节点：** 控制节点。

```bash
helm pull oci://cr.openfuyao.cn/charts/ub-network-bandwidth-operator --version ${OCI_VERSION}
```

命令执行后会在当前目录生成 `ub-network-bandwidth-operator-${OCI_VERSION}.tgz`。

#### 4. Helm 部署

**执行节点：** 控制节点。

```bash
helm install ${RELEASE_NAME} ./ub-network-bandwidth-operator-${OCI_VERSION}.tgz \
  -n ${NAMESPACE} \
  --create-namespace \
  --set image.repository=cr.openfuyao.cn/openfuyao/ub-network-bandwidth-operator \
  --set image.tag=${VERSION}
```

如果需要在部署时覆盖默认带宽配置，可以通过 Helm values 设置 `config.bandwidths`：

```bash
helm install ${RELEASE_NAME} ./ub-network-bandwidth-operator-${OCI_VERSION}.tgz \
  -n ${NAMESPACE} \
  --create-namespace \
  --set image.repository=cr.openfuyao.cn/openfuyao/ub-network-bandwidth-operator \
  --set image.tag=${VERSION} \
  --set 'config.bandwidths[0].priority=0' \
  --set 'config.bandwidths[0].bandwidth=25'
```

#### 5. 验证部署

**执行节点：** 控制节点。

```bash
kubectl get pod -n ${NAMESPACE} -l app.kubernetes.io/instance=${RELEASE_NAME}
kubectl get cm ubs-bandwidth-config -n ${NAMESPACE}
```

预期结果：

- 每个可调度节点上都有 `ub-network-bandwidth-operator` Pod，且状态为 `Running`。
- `ubs-bandwidth-config` ConfigMap 已创建。

#### 6. 升级和卸载

升级到新版本：

```bash
helm upgrade ${RELEASE_NAME} ./ub-network-bandwidth-operator-${OCI_VERSION}.tgz \
  -n ${NAMESPACE} \
  --set image.repository=cr.openfuyao.cn/openfuyao/ub-network-bandwidth-operator \
  --set image.tag=${VERSION}
```

卸载：

```bash
helm uninstall ${RELEASE_NAME} -n ${NAMESPACE}
```

## 配置说明

通过修改 ConfigMap 来配置带宽 QoS 策略：

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: ubs-bandwidth-config
  namespace: ub-network-system
data:
  config.yaml: |
    bandwidths:
    - priority: 0
      bandwidth: 25
```

### 配置参数说明

`bandwidths` 的配置含义和校验规则以设计文档为准，详见
[配置模型](docs/design-ub-network-bandwidth.md#配置模型) 和
[Validation 规则](docs/design-ub-network-bandwidth.md#validation-规则)。
其中 `bandwidth` 必须为正整数，不能为 0，也不支持小数或字符串。

## 健康检查

Operator 提供了健康检查端点：

```bash
# 健康检查
curl http://<operator-pod-ip>:8089/healthz

# 就绪检查
curl http://<operator-pod-ip>:8089/readyz
```

UB网络带宽设置
