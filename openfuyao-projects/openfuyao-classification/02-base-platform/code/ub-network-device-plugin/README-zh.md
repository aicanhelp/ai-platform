# ub-network-device-plugin

UB网络设备插件。

## 编译准备

1. 安装make、gcc、docker。

   可通过 yum install进行安装。

2. 安装 go，版本 >= 1.24.0。

   访问官方下载地址 https://go.dev/dl/go1.24.0.linux-arm64.tar.gz 获取发布包，执行如下命令解压到环境中：
   
   ```bash
   tar zxvf go1.24.0.linux-arm64.tar.gz -C /usr/local
   export PATH=/usr/local/go/bin:$PATH
   ```

## 构建
在代码根目录下执行构建。

```bash
# 构建二进制
make build

# 构建镜像
make image

# 构建二进制和镜像
make
```

## 部署
1. 将构建中产生的二进制产物导入集群中。

   ```bash
   ctr -n k8s.io image import ub-network-device-plugin-v1.0.1.tar
   ```
2. 请参考example下的yaml文件示例，修改其中的镜像仓库和版本号后，在环境中执行应用。

   ```bash
   kubectl apply -f ub-network-device-plugin.yaml
   ```

## 设备使用

在containers中按照如下示例配置resources。

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: urma-test
  namespace: test
spec:
  containers:
  - name: urma-test
    image: busybox:latest
    resources:
      requests:
        cpu: "1"
        memory: "128Mi"
        unifiedbus.com/ub_net_device: "1"
      limits:
        cpu: "1"
        memory: "128Mi"
        unifiedbus.com/ub_net_device: "1"
```
