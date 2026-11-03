VERSION ?= v1.0.0
# Image URL to use all building/pushing image targets
IMG ?= controller:latest

ENVTEST_K8S_VERSION ?= 1.22

GOBIN := $(shell command -v go >/dev/null 2>&1 && go env GOBIN 2>/dev/null)
ifeq ($(GOBIN),)
GOBIN := $(shell command -v go >/dev/null 2>&1 && go env GOPATH 2>/dev/null)/bin
endif

.PHONY: all
all: build

COMMIT_ID=$(shell git rev-parse HEAD)
BUILD_TIME=$(shell date +'%Y-%m-%dT%H:%M')
HOST_ARCH=$(shell go env GOHOSTARCH)
HOST_OS=$(shell go env GOHOSTOS)

PKG = gopkg.openfuyao.cn/bkeagent

PLATFORMS ?= amd64 arm64


# Build binary
.PHONY: build
build:
	@echo $(BUILD_TIME) > build/time
	# build BKEAgent binary ${HOST_OS}/${HOST_ARCH}
	@echo LatestCommit: `git log --pretty='%s%b%B' -n 1`
	@CGO_ENABLED=0 GOOS=linux GOARCH=$(HOST_ARCH) go build \
	   -trimpath \
	   -ldflags="-s -w \
	             -X ${PKG}/version.GitCommitID=${COMMIT_ID}  \
	   			 -X ${PKG}/version.Architecture=${HOST_OS}/${HOST_ARCH} \
   				 -X ${PKG}/version.BuildTime=${BUILD_TIME} \
   				 -X ${PKG}/version.Version=${VERSION}" \
   	   -o bin/bkeagent-${HOST_ARCH} .
	@sudo cp bin/bkeagent-amd64 /bkeagent && sudo cp bin/bkeagent-* / && sudo cp bin/bkeagent-* ../cluster-api-provider-bke/bin/

release: bin release-binaries
	# build launcher image
	cd launcher && make release AGENT_VERSION=$(VERSION) AGENT_COMMIT_ID=$(COMMIT_ID) AGENT_BUILD_TIME=$(BUILD_TIME)


release-binaries: clean
	$(foreach PLATFORM,$(PLATFORMS), echo -n "$(PLATFORM)..."; make build HOST_ARCH=$(PLATFORM);)

##@ 帮助系统

.PHONY: help
help: ## 显示此帮助信息
	@printf "\n\033[1m使用方法:\033[0m\n"
	@printf "  make \033[36m<目标>\033[0m\n\n"
	@printf "\033[1m可用目标:\033[0m\n"
	@awk 'BEGIN {FS = ":.*##"} \
		/^[a-zA-Z0-9_-]+:.*?##/ { \
			printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2 \
		} \
		/^##@/ { \
			printf "\n\033[1m%s\033[0m\n", substr($$0, 5) \
		}' $(MAKEFILE_LIST)
	@printf "\n"

##@ 开发工具

.PHONY: manifests
manifests: controller-gen ## 生成 Kubernetes 清单 (RBAC, CRD, Webhook)
	@echo "🛠️  生成 Kubernetes 清单文件..."
	@$(CONTROLLER_GEN) \
		rbac:roleName=manager-role \
		crd \
		webhook \
		paths="./api/..." \
		output:crd:artifacts:config=config/crd/bases
	@echo "✅ 清单文件生成完成"

.PHONY: generate
generate: controller-gen ## 生成 DeepCopy 相关方法实现
	@echo "🛠️  生成 DeepCopy 方法..."
	@$(CONTROLLER_GEN) \
		object:headerFile="hack/boilerplate.go.txt" \
		paths="./api/..."
	@echo "✅ 代码生成完成"

.PHONY: fmt vet
fmt: ## 格式化 Go 代码
	@echo "Running go fmt..."
	@go fmt ./...

vet: ## 静态代码分析
	@echo "Running go vet..."
	@go vet ./...

.PHONY: test
test: manifests generate fmt vet envtest ## 运行测试套件
	@echo "Running tests with K8s $(ENVTEST_K8S_VERSION)..."
	@KUBEBUILDER_ASSETS="$$($(ENVTEST) use $(ENVTEST_K8S_VERSION) -p path)"
	go test ./... -coverprofile cover.out

##@ Kubernetes CRD 管理

# 部署配置
IGNORE_NOT_FOUND ?= false
KUBECONFIG ?= $(HOME)/.kube/config

.PHONY: validate-environment
validate-environment:
	@if [ ! -f "$(KUBECONFIG)" ]; then \
		echo "错误: 找不到 kubeconfig 文件 [$(KUBECONFIG)]"; \
		exit 1; \
	fi

.PHONY: install
install: manifests kustomize validate-environment ## 安装 CRDs 到 Kubernetes 集群
	@echo "🚀 开始安装 CRDs..."
	@$(KUSTOMIZE) build config/crd | kubectl --kubeconfig="$(KUBECONFIG)" apply -f -
	@echo "✅ CRDs 安装完成"

.PHONY: uninstall
uninstall: manifests kustomize validate-environment ## 从集群卸载 CRDs (使用 IGNORE_NOT_FOUND=true 忽略不存在的资源)
	@echo "🧹 开始卸载 CRDs (IGNORE_NOT_FOUND=$(IGNORE_NOT_FOUND))..."
	@$(KUSTOMIZE) build config/crd | kubectl --kubeconfig="$(KUBECONFIG)" delete --ignore-not-found=$(IGNORE_NOT_FOUND) -f -
	@echo "✅ CRDs 卸载完成"


CONTROLLER_GEN = $(shell pwd)/bin/controller-gen
.PHONY: controller-gen
controller-gen: ## Download controller-gen locally if necessary.
	$(call go-get-tool,$(CONTROLLER_GEN),sigs.k8s.io/controller-tools/cmd/controller-gen@v0.9.0)

KUSTOMIZE = $(shell pwd)/bin/kustomize
.PHONY: kustomize
kustomize: ## Download kustomize locally if necessary.
	$(call go-get-tool,$(KUSTOMIZE),sigs.k8s.io/kustomize/kustomize/v3@v3.8.7)

ENVTEST := $(abspath bin/setup-envtest)

.PHONY: envtest
envtest: ## 安全地下载 envtest-setup 工具
	@if [ ! -f "$(ENVTEST)" ]; then
	echo "Downloading setup-envtest...";
	export GOBIN=$(dir $(ENVTEST));
	go install sigs.k8s.io/controller-runtime/tools/setup-envtest@latest;
	fi

# Make sure Docker supports multi-schema image compilation
docker-build:
	@build/run-in-docker.sh ARCH=$(ARCH) COMMIT_ID=`git rev-parse HEAD` build/build.sh

buildx:
	@docker run --privileged --rm tonistiigi/binfmt --uninstall qemu-* && \
	docker run --privileged --rm tonistiigi/binfmt --install all && \
	docker buildx rm mybuilder && \
	docker buildx create --use --name mybuilder --driver-opt image=dockerpracticesig/buildkit:master && \
	docker buildx inspect mybuilder --bootstrap && \
	docker buildx ls

clean:
	@rm -rf bin/bke*

.PHONY: bin
bin:
	if [ ! -d bin ]; then mkdir bin; fi

