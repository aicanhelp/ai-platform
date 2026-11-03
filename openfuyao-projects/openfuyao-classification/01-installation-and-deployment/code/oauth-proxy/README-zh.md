# openFuyao认证鉴权边车组件

本组件用于openFuyao的扩展组件认证鉴权，以边车的形式随扩展组件一同部署。

本项目基于 [openshift/oauth-proxy](https://github.com/openshift/oauth-proxy) 进行fork，并进行了以下修改：

- 删除了以下不需要的文件：
  - /vendor 文件夹
  - /Dockerfile
  - /Makefile
  - /OWNERS
  - /contrib/ 文件夹
  - /dependencymagnet.go
  - /dist.sh
  - /front.png
  - /htpasswd.go
  - /htpasswd_test.go
  - /oauth-proxy.spec
  - /templates.go
  - /templates_test.go
  - /test/e2e/proxy_test.go
  - /test/e2e/util.go
  - /providers/openshift/ 文件夹
- 增加了以下文件。
  - OAT.yaml
  - /build 文件夹用于构建。
  - /charts 文件夹用于helm-chart打包。
  - /fuyaoerrors 定义error。
  - /providers/openfuyao/ 文件夹 openFuyao的provider实现。
  - /request/ 文件夹请求函数的实现。
- 修改了以下文件用于匹配 /providers/openfuyao中认证鉴权provider的实现。
  - .gitignore
  - /api/api.go
  - /cookie/cookies.go
  - /go.mod
  - /go.sum
  - /http.go
  - /main.go
  - /oauthproxy.go
  - /oauthproxy_test.go
  - /options.go
  - /options_test.go
  - /util/util.go
