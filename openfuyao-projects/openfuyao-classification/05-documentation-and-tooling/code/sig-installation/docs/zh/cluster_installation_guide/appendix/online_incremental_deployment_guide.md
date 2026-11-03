# 在线openFuyao-system-controller手动部署

## 背景信息

本文档介绍在在线场景下，在一个未通过 BKE 自动部署 `openFuyao-system-controller` 的 Kubernetes 环境中，手动准备 `patch-config`、渲染并部署 `openFuyao-system-controller`，从而拉起 openFuyao 管理面组件。

在线场景中，`openFuyao-system-controller` 和管理面组件直接从在线镜像仓库拉取镜像，并从在线 Helm 仓库拉取 Chart。

## 前提条件

已准备以下机器：

| 角色 | 示例 IP | 用途 |
| --- | --- | --- |
| 业务集群节点 | `192.168.200.216` | 已存在 Kubernetes 集群，手动部署 `openFuyao-system-controller` 并拉起管理面 |

机器硬件配置可参考[快速入门](https://docs.openfuyao.cn/zh/docs/v26.03/quick_start.html#快速安装)中的配置要求。

已准备以下信息：

| 信息 | 用途 | 准备方式 |
| --- | --- | --- |
| `openFuyao-system-controller.yaml` | 部署业务集群 controller 组件 | 后续在 [3 章节](#3-准备-openfuyao-system-controller-yaml) 从本机模板目录拷贝并渲染生成 |
| `VersionConfig-latest.yaml` | `latest` 版本安装使用的组件版本配置 | 本文示例从 `/bke/mount/source_registry/files/patches/VersionConfig-latest.yaml` 读取 |

> ![输入图片说明](../figures/icon-note.gif) **说明：**<br />本文以 `latest` 版本为例。可通过版本配置文件确认当前版本：如果使用的是 `VersionConfig-latest.yaml`，且文件中的 `openFuyaoVersion` 字段为 `latest`，则为 `latest` 版本；如果使用的是 `Core-VersionConfig-v26.06-rc.1.yaml` 等指定版本配置文件，则为对应版本。非 `latest` 版本请将镜像 tag、Chart 仓库地址和版本配置文件替换为对应版本的实际值。

## 使用限制

需要在线安装，业务集群节点需能够访问以下地址：

- 在线镜像仓库：`cr.openfuyao.cn/openfuyao`
- 在线 Helm 仓库：`https://helm.openfuyao.cn/_core`

## 使用教程

### 1. 检查目标环境

#### 1.1 确认未安装管理面组件

在业务集群节点执行：

```bash
kubectl get ns openfuyao-system-controller openfuyao-system monitoring ingress-nginx
kubectl get deploy -A | grep -E "openfuyao|console|oauth|marketplace|monitoring"
kubectl get nodes
```

如果 `openfuyao-system-controller`、`openfuyao-system`、`monitoring`、`ingress-nginx` 等命名空间或管理面组件已经存在，请先确认当前环境是否需要重新部署，避免覆盖已有管理面。

#### 1.2 确认未安装 BKE/CAPI 控制面

```bash
kubectl get crd | grep -E "bke|cluster.x-k8s"
kubectl get pods -n cluster-system
```

如果环境中没有 BKE/CAPI 相关 CRD，不能直接 apply `BKECluster` 类型资源。本文使用手动部署 `openfuyao-system-controller` 的方式安装管理面。

#### 1.3 确认部署模板存在

```bash
ls /etc/openFuyao/addons/manifests/kubernetes/openfuyao-system-controller/latest/openfuyao-system-controller.yaml
grep -n "{{" /etc/openFuyao/addons/manifests/kubernetes/openfuyao-system-controller/latest/openfuyao-system-controller.yaml
```

模板中通常包含如下变量，不能直接 apply：

```text
{{ .repo }}
{{ .helmRepo }}
{{ .tagVersion }}
```

### 2. 创建 patch-config

#### 2.1 确认版本配置文件

在业务集群节点执行，确认用于创建 `patch-config` 的版本配置文件已存在：

```bash
ls /bke/mount/source_registry/files/patches/
grep -n "^openFuyaoVersion:" /bke/mount/source_registry/files/patches/VersionConfig-latest.yaml
```

如果业务集群节点不存在 `/bke/mount/source_registry/files/patches/VersionConfig-latest.yaml`，请先将版本配置文件准备到该路径。本文以 `latest` 版本为例，可执行如下命令下载：

```bash
mkdir -p /bke/mount/source_registry/files/patches
curl -fLo /bke/mount/source_registry/files/patches/VersionConfig-latest.yaml \
  https://openfuyao.obs.cn-north-4.myhuaweicloud.com/openFuyao/version-config/VersionConfig-latest.yaml
```

如果 `openFuyaoVersion` 为 `latest`，可继续使用本文示例；如果是其他版本，请替换为对应版本配置文件，并同步替换后续命令中的文件名。

#### 2.2 创建 namespace 和 ConfigMap

```bash
kubectl create ns openfuyao-system-controller --dry-run=client -o yaml | kubectl apply -f -

kubectl create cm patch-config \
  -n openfuyao-system-controller \
  --from-file=patch-data=/bke/mount/source_registry/files/patches/VersionConfig-latest.yaml \
  --dry-run=client -o yaml | kubectl apply -f -

kubectl get cm patch-config -n openfuyao-system-controller
```

### 3. 准备 openFuyao-system-controller YAML

#### 3.1 拷贝部署模板

```bash
cp /etc/openFuyao/addons/manifests/kubernetes/openfuyao-system-controller/latest/openfuyao-system-controller.yaml /tmp/openfuyao-system-controller.yaml
```

#### 3.2 渲染在线参数

```bash
sed -i 's#{{ if .repo }}{{ .repo }}{{ else }}cr.openfuyao.cn/openfuyao/{{ end }}openfuyao-system-controller:{{ .tagVersion }}#cr.openfuyao.cn/openfuyao/openfuyao-system-controller:latest#g' /tmp/openfuyao-system-controller.yaml
sed -i 's#value: "{{ .repo }}"#value: "cr.openfuyao.cn/openfuyao"#g' /tmp/openfuyao-system-controller.yaml
sed -i 's#value: "{{ .helmRepo }}"#value: "https://helm.openfuyao.cn/_core"#g' /tmp/openfuyao-system-controller.yaml
```

渲染后的关键配置如下：

```yaml
image: cr.openfuyao.cn/openfuyao/openfuyao-system-controller:latest
OPENFUYAO_REGISTRY: cr.openfuyao.cn/openfuyao
HELM_REPORITORY_URL: https://helm.openfuyao.cn/_core
```

检查模板变量是否已清理：

```bash
grep -n "{{" /tmp/openfuyao-system-controller.yaml
```

如果无输出，表示模板变量已清理完成。

检查关键字段：

```bash
grep -nE "image:|OPENFUYAO_REGISTRY|HELM_REPORITORY_URL|entrypoint.sh" /tmp/openfuyao-system-controller.yaml
grep -nA2 "OPENFUYAO_REGISTRY\|HELM_REPORITORY_URL" /tmp/openfuyao-system-controller.yaml
```

### 4. 部署 openFuyao-system-controller

在业务集群节点执行：

```bash
kubectl apply -f /tmp/openfuyao-system-controller.yaml
kubectl get pods -n openfuyao-system-controller -w
```

当 Pod 处于 `Init:0/1` 时，表示 installer initContainer 正在执行安装流程。

查看安装日志：

```bash
kubectl logs -n openfuyao-system-controller deploy/openfuyao-system-controller -c installer -f
```

日志中应能看到在线镜像仓库和在线 Helm 仓库地址：

```text
cr.openfuyao.cn/openfuyao
https://helm.openfuyao.cn/_core
```

### 5. 检查安装结果

#### 5.1 查看全量组件

```bash
kubectl get pods -A
```

#### 5.2 查看管理面组件

```bash
kubectl get pods -n openfuyao-system
kubectl get pods -n ingress-nginx
kubectl get pods -n monitoring
```

成功时核心组件应为 `Running` 或 `Completed`，包括：

- `console-service`
- `console-website`
- `local-harbor`
- `marketplace-service`
- `monitoring-service`
- `oauth-server`
- `oauth-webhook`
- `plugin-management-service`
- `user-management-operator`
- `web-terminal-service`

#### 5.3 查看异常 Pod

```bash
kubectl get pods -A | grep -Ev "Running|Completed"
```

如果无异常输出，表示 `openFuyao-system-controller` 及其拉起的管理面组件已部署完成。

<!-- ### 6. 卸载与恢复安装验证（可选）

#### 6.1 生成 install 和 uninstall YAML

```bash
cp /tmp/openfuyao-system-controller.yaml /tmp/openfuyao-system-controller-install-online.yaml
cp /tmp/openfuyao-system-controller.yaml /tmp/openfuyao-system-controller-uninstall-online.yaml

sed -i 's#sh /home/openfuyao-system/entrypoint.sh -o install#sh /home/openfuyao-system/entrypoint.sh -o uninstall#g' /tmp/openfuyao-system-controller-uninstall-online.yaml
grep -n "entrypoint.sh" /tmp/openfuyao-system-controller-uninstall-online.yaml
```

确认输出中包含：

```text
sh /home/openfuyao-system/entrypoint.sh -o uninstall
```

#### 6.2 执行 uninstall

```bash
kubectl apply -f /tmp/openfuyao-system-controller-uninstall-online.yaml
kubectl rollout restart deploy/openfuyao-system-controller -n openfuyao-system-controller
kubectl logs -n openfuyao-system-controller deploy/openfuyao-system-controller -c installer -f
```

验证卸载结果：

```bash
kubectl get pods -n openfuyao-system
kubectl get pods -n ingress-nginx
kubectl get pods -n monitoring
kubectl get ns | grep -E "openfuyao-system|monitoring|ingress-nginx|session-secret"
```

当 `openfuyao-system` 相关组件被删除或 namespace 不存在时，表示卸载流程生效。

#### 6.3 恢复 install

```bash
kubectl apply -f /tmp/openfuyao-system-controller-install-online.yaml
kubectl rollout restart deploy/openfuyao-system-controller -n openfuyao-system-controller
kubectl logs -n openfuyao-system-controller deploy/openfuyao-system-controller -c installer -f
```

验证恢复结果：

```bash
kubectl get pods -n openfuyao-system-controller
kubectl get pods -n openfuyao-system
kubectl get pods -n ingress-nginx
kubectl get pods -n monitoring
kubectl get pods -A | grep -E "CrashLoopBackOff|ImagePullBackOff|ErrImagePull|Error|Pending"
```

当 `openfuyao-system-controller` 恢复为 `1/1 Running`，`openfuyao-system`、`ingress-nginx`、`monitoring` 主要组件恢复 Running，且异常 Pod 检查无输出时，表示在线场景下重新安装恢复成功。 -->
