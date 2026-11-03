# openFuyao Authentication and Authorization Sidecar Component

This component is used for openFuyao extension component authentication and authorization, deployed as a sidecar together with extension components.

This project is forked based on [openshift/oauth-proxy](https://github.com/openshift/oauth-proxy) and has undergone the following modifications:

- Deleted the following unnecessary files:
   - /vendor folder
   - /Dockerfile
   - /Makefile
   - /OWNERS
   - /contrib/ folder
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
   - /providers/openshift/ folder
- Added the following files.
   - OAT.yaml
   - /build folder for building.
   - /charts folder for helm-chart packaging.
   - /fuyaoerrors defines errors.
   - /providers/openfuyao/ folder openFuyao provider implementation.
   - /request/ folder request function implementation.
- Modified the following files to match the authentication and authorization provider implementation in /providers/openfuyao.
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
