# 日志扩展前端

## 项目运行

```bash
# 安装依赖
npm install

# 运行 localhost:7777
npm run start

# 构建
npm run build
```

## 镜像构建

```bash
docker build -t cr.openfuyao.cn/openfuyao/logging-website:latest -f ./build/Dockerfile .
```

## helm部署

```bash
helm install {release} ./chart -n {namespace}
```
