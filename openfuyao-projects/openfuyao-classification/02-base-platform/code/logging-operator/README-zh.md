# 日志扩展组件

## 特性介绍

- 高效的日志收集和管理能力

  支持用户根据自定义的筛选条件对各种服务和容器进行实时日志数据收集和查询。自定义的筛选条件包括但不限于命名空间、服务名称、容器名称、关键字、筛选时间、日志采集路径文件名称以及日志类型级别等。用户可以自行组合多种条件实现精准的查询能力。

- 精准的日志定位能力

  支持用户在选定某个确定的日志源后（例如具体的容器，具体的日志路径等），精准定位一段细粒度时间跨度的所有日志。提供通过自定义时间查询上下文日志功能，使用户在定位到某个关键日志后，可以根据自身需求查询上下文日志。

- 多元的日志采集能力

  支持用户设置日志采集路径从而实现对用户自定义应用所产生的日志进行采集。在不更改默认配置的情况下新增日志采集源。用户可以自定义日志采集工作，允许日志采集组件在用户指定的路径进行日志采集，并可以通过具体的“路径+文件名称”的方式在查询界面感知到该应用的日志。

- 敏感的日志告警能力

  支持用户安装日志组件时，创建日志告警规则，并提供日志告警规则模板以及预置的日志告警规则。在系统采集的日志触发该告警规则后将告警发送给告警组件，实现日志监控+告警+定位的一站式故障检测系统。

- 日志导出/下载能力

  支持用户在日志组件中导出用户想要保存的日志内容。当前版本支持用户在进行一系列筛选操作后导出筛选结果的日志，支持用户通过日志详情界面时间筛选得到单个容器的详情日志。

### 实现原理

日志组件作为openFuyao的生态扩展组件由三部分组成：日志后端层，日志UI层，开源软件层。

- 日志后端层

  logging-operator后端以微服务的形式部署在集群中。后端组件提供日志的查询筛选接口、日志采集组源的查询接口、日志告警规则的查询接口、日志采集工作的增删改查接口、日志详情上下文的查询接口等，为日志扩展组件提供核心能力。

  logging-operator的后端处理逻辑是接收到前台的调用请求后，根据具体的请求内容创建Loki组件使用的查询语句LogQL，并且将这个LogQL拼接成调用Loki http接口的查询请求，从而获得Loki中的查询和筛选结果。之后再根据具体的返回逻辑进行拼接等操作，并最终返回给原调用接口。采集源的配置功能是通过调用k8s的client go接口来对k8s集群资源进行增删改查等操作。

- 日志UI层

  日志的前端UI组件作为一个具有扩展性的可插拔组件独立部署在openFuyao平台中。UI界面在openFuyao平台中进行安装部署之后，日志部分的界面会自动集成在fuyao-website的侧边工具导航栏中。用户可以直接通过平台管理面界面的左树菜单导航栏选择进行日志组件的使用。当集群中没有安装openFuyao平台时，日志UI也可以和日志后端，日志开源软件集成部署为独立日志应用，这种情况下日志UI层是一个独立的界面。

- 日志开源软件层

  日志的开源软件主要为日志系统提供底层的支撑能力，主要由日志采集组件（promtail）和日志存储组件（Loki）组成。日志采集组件以daemonset的方式部署在每一个节点上，根据采集规则将采集源中的日志发送到日志存储组件。日志存储组件Loki接收到http接口后，在日志数据库中执行查询语句并给出查询结果。

## 安全能力说明

openFuyao v26.06及往期版本仅提供基础安全功能：服务间双向认证（mTLS）与组件级用户认证鉴权（RBAC）的需使用者/集成伙伴根据自身部署环境自行适配与加固（如引入cert-manager、做统一证书、部署鉴权中间件、配置NetworkPolicy收缩可达范围等）。当前聚焦基础功能交付，安全防护需用户自行加强，后续版本安全能力会逐步规划落地。

## 安装

参考：https://docs.openfuyao.cn/docs

## 本地构建

### 镜像构建

#### 构建参数

- `GOPRIVATE`：配置Go语言私有仓库，相当于`GOPRIVATE`环境变量。
- `COMMIT`：当前git commit的哈希值。
- `VERSION`：组件版本。
- `SOURCE_DATE_EPOCH`：镜像rootfs的时间戳。

#### 构建命令

- 构建并推送到指定OCI仓库。

  <details open>
  <summary>使用<code>docker</code></summary>

  ```bash
  docker buildx build . -f <path/to/dockerfile> \
      -o type=image,name=<oci/repository>:<tag>,oci-mediatypes=true,rewrite-timestamp=true,push=true \
      --platform=linux/amd64,linux/arm64 \
      --provenance=false \
      --build-arg=GOPRIVATE=gopkg.openfuyao.cn \
      --build-arg=COMMIT=$(git rev-parse HEAD) \
      --build-arg=VERSION=0.0.0-latest \
      --build-arg=SOURCE_DATE_EPOCH=$(git log -1 --pretty=%ct)
  ```

  </details>
  <details>
  <summary>使用<code>nerdctl</code></summary>

  ```bash
  nerdctl build . -f <path/to/dockerfile> \
      -o type=image,name=<oci/repository>:<tag>,oci-mediatypes=true,rewrite-timestamp=true,push=true \
      --platform=linux/amd64,linux/arm64 \
      --provenance=false \
      --build-arg=GOPRIVATE=gopkg.openfuyao.cn \
      --build-arg=COMMIT=$(git rev-parse HEAD) \
      --build-arg=VERSION=0.0.0-latest \
      --build-arg=SOURCE_DATE_EPOCH=$(git log -1 --pretty=%ct)
  ```

  </details>

  其中，`<path/to/dockerfile>`为Dockerfile路径，`<oci/repository>`为镜像地址，`<tag>`为镜像tag。

- 构建并导出OCI Layout到本地tarball。

  <details open>
  <summary>使用<code>docker</code></summary>

  ```bash
  docker buildx build . -f <path/to/dockerfile> \
      -o type=oci,name=<oci/repository>:<tag>,dest=<path/to/oci-layout.tar>,rewrite-timestamp=true \
      --platform=linux/amd64,linux/arm64 \
      --provenance=false \
      --build-arg=GOPRIVATE=gopkg.openfuyao.cn \
      --build-arg=COMMIT=$(git rev-parse HEAD) \
      --build-arg=VERSION=0.0.0-latest \
      --build-arg=SOURCE_DATE_EPOCH=$(git log -1 --pretty=%ct)
  ```

  </details>
  <details>
  <summary>使用<code>nerdctl</code></summary>

  ```bash
  nerdctl build . -f <path/to/dockerfile> \
      -o type=oci,name=<oci/repository>:<tag>,dest=<path/to/oci-layout.tar>,rewrite-timestamp=true \
      --platform=linux/amd64,linux/arm64 \
      --provenance=false \
      --build-arg=GOPRIVATE=gopkg.openfuyao.cn \
      --build-arg=COMMIT=$(git rev-parse HEAD) \
      --build-arg=VERSION=0.0.0-latest \
      --build-arg=SOURCE_DATE_EPOCH=$(git log -1 --pretty=%ct)
  ```

  </details>

  其中，`<path/to/dockerfile>`为Dockerfile路径，`<oci/repository>`为镜像地址，`<tag>`为镜像tag，`path/to/oci-layout.tar`为tar包路径。

- 构建并导出镜像rootfs到本地目录。

  <details open>
  <summary>使用<code>docker</code></summary>

  ```bash
  docker buildx build . -f <path/to/dockerfile> \
      -o type=local,dest=<path/to/output>,platform-split=true \
      --platform=linux/amd64,linux/arm64 \
      --provenance=false \
      --build-arg=GOPRIVATE=gopkg.openfuyao.cn \
      --build-arg=COMMIT=$(git rev-parse HEAD) \
      --build-arg=VERSION=0.0.0-latest
  ```

  </details>
  <details>
  <summary>使用<code>nerdctl</code></summary>

  ```bash
  nerdctl build . -f <path/to/dockerfile> \
      -o type=local,dest=<path/to/output>,platform-split=true \
      --platform=linux/amd64,linux/arm64 \
      --provenance=false \
      --build-arg=GOPRIVATE=gopkg.openfuyao.cn \
      --build-arg=COMMIT=$(git rev-parse HEAD) \
      --build-arg=VERSION=0.0.0-latest
  ```

  </details>

  其中，`<path/to/dockerfile>`为Dockerfile路径，`path/to/output`为本地目录路径。

### Helm Chart构建

- 打包Helm Chart。

  ```bash
  helm package <path/to/chart> -u \
      --version=0.0.0-latest \
      --app-version=openFuyao-v25.09
  ```

  其中，`<path/to/chart>`为Chart文件夹路径。

- 推送Chart包到指定OCI仓库。

  ```bash
  helm push <path/to/chart.tgz> oci://<oci/repository>:<tag>
  ```

  其中，`<path/to/chart.tgz>`为Chart包路径，`<oci/repository>`为Chart包推送地址，`<tag>`为Chart包tag。

