# console-website-dev 开发 Pod 使用指南

## 0. 前置要求

在现有 openFuyao 集群的节点上进行开发

## 1. 创建开发 Pod

在仓库根目录执行，自动创建 Pod、等待就绪并安装 npm 依赖：

```bash
bash ./hack/dev-pod.sh create
```

> Pod 以 `node:20-slim` 镜像启动，通过 `hostPath` 将本仓库代码挂载到容器内 `/app`。
> 如果本机 hostname 与集群节点名不一致，需设置环境变量 `DEV_POD_NODE` 指定节点名。

## 2. 启用开发流量

将 Service 选择器指向开发 Pod，并开启 NodePort 30001：

```bash
bash ./hack/dev-pod.sh activate
```

## 3. 启动开发服务器

执行命令启动开发服务器

```bash
# 前台运行（可实时查看日志，Ctrl+C 停止）
bash ./hack/dev-pod.sh start

# 后台运行（需要 kill 终止进程）
bash ./hack/dev-pod.sh start -d
```

由于 openFuyao 认证机制，执行后需先访问 `节点IP:31616` 登录管理面，登录完成后再访问 `节点IP:30001` 进行开发调试。

## 4. 构建与预览

```bash
# 构建生产产物
bash ./hack/dev-pod.sh build

# 预览构建产物
bash ./hack/dev-pod.sh preview
bash ./hack/dev-pod.sh preview -d   # 后台运行
```

## 5. 更新依赖

`package.json` 变更后，在 Pod 内重新安装依赖：

```bash
bash ./hack/dev-pod.sh install
```

## 6. 结束开发

```bash
# 停止所有 Vite 进程
bash ./hack/dev-pod.sh kill

# 恢复 Service 指向正式 Pod，关闭 NodePort
bash ./hack/dev-pod.sh deactivate

# 删除开发 Pod
bash ./hack/dev-pod.sh delete
```

## 命令速查

| 命令 | 说明 |
|------|------|
| `create` | 创建 Pod 并自动 `npm install` |
| `delete` | 删除 Pod |
| `activate` | Service 指向开发 Pod + 开启 NodePort 30001 |
| `deactivate` | 恢复 Service 指向正式 Pod + 关闭 NodePort |
| `install` | 在 Pod 内执行 `npm install` |
| `start [-d]` | 启动 Vite 开发服务器，`-d` 后台运行 |
| `build [-d]` | 执行 `npm run build`，`-d` 后台运行 |
| `preview [-d]` | 启动 Vite 预览服务器，`-d` 后台运行 |
| `kill` | 杀掉 Pod 内所有 Vite 进程 |
