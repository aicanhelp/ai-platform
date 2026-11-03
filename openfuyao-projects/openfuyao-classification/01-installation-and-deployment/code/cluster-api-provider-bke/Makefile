# 顶层 Makefile - 统一调用各组件的构建
# 
# 使用方法:
#   make              - 构建所有组件
#   make capbke       - 只构建 capbke controller
#   make bkeagent     - 只构建 bkeagent
#   make help         - 显示所有可用目标

VERSION ?= v1.0.0

.PHONY: all
all: capbke bkeagent ## 构建所有组件

.PHONY: help
help: ## 显示此帮助信息
	@printf "\n\033[1m统一构建系统 - cluster-api-provider-bke\033[0m\n\n"
	@printf "\033[1m使用方法:\033[0m\n"
	@printf "  make \033[36m<目标>\033[0m\n\n"
	@printf "\033[1m可用目标:\033[0m\n"
	@awk 'BEGIN {FS = ":.*##"} \
		/^[a-zA-Z0-9_-]+:.*?##/ { \
			printf "  \033[36m%-25s\033[0m %s\n", $$1, $$2 \
		} \
		/^##@/ { \
			printf "\n\033[1m%s\033[0m\n", substr($$0, 5) \
		}' $(MAKEFILE_LIST)
	@printf "\n\033[1m子 Makefile 帮助:\033[0m\n"
	@printf "  make capbke-help        - 显示 capbke Makefile 帮助\n"
	@printf "  make bkeagent-help      - 显示 bkeagent Makefile 帮助\n"
	@printf "\n"

##@ 组件构建

.PHONY: capbke
capbke: ## 构建 capbke controller
	$(MAKE) -f Makefile.capbke build VERSION=$(VERSION)

.PHONY: bkeagent
bkeagent: ## 构建 bkeagent
	$(MAKE) -f Makefile.bkeagent build VERSION=$(VERSION)

##@ 代码生成

.PHONY: manifests
manifests: ## 生成所有组件的 Kubernetes 清单
	$(MAKE) -f Makefile.capbke manifests
	$(MAKE) -f Makefile.bkeagent manifests

.PHONY: generate
generate: controller-gen ## 生成所有 API 的 DeepCopy 方法
	$(CONTROLLER_GEN) object:headerFile="hack/boilerplate.go.txt" paths="./api/..."

##@ Docker 镜像构建

.PHONY: docker-build
docker-build: docker-build-capbke docker-build-bkeagent ## 构建所有 Docker 镜像

.PHONY: docker-build-capbke
docker-build-capbke: ## 构建 capbke controller 镜像
	$(MAKE) -f Makefile.capbke docker-build VERSION=$(VERSION)

.PHONY: docker-build-bkeagent
docker-build-bkeagent: ## 构建 bkeagent 镜像
	$(MAKE) -f Makefile.bkeagent docker-build VERSION=$(VERSION)

##@ 发布

.PHONY: release
release: release-capbke release-bkeagent ## 发布所有组件

.PHONY: release-capbke
release-capbke: ## 发布 capbke controller
	$(MAKE) -f Makefile.capbke release VERSION=$(VERSION)

.PHONY: release-bkeagent
release-bkeagent: ## 发布 bkeagent
	$(MAKE) -f Makefile.bkeagent release VERSION=$(VERSION)

##@ 清理

.PHONY: clean
clean: ## 清理所有构建产物
	$(MAKE) -f Makefile.capbke clean
	$(MAKE) -f Makefile.bkeagent clean

##@ 帮助

.PHONY: capbke-help
capbke-help: ## 显示 capbke Makefile 帮助
	$(MAKE) -f Makefile.capbke help

.PHONY: bkeagent-help
bkeagent-help: ## 显示 bkeagent Makefile 帮助
	$(MAKE) -f Makefile.bkeagent help

##@ 开发工具

## Location to install dependencies to
LOCALBIN ?= $(shell pwd)/bin
$(LOCALBIN):
	mkdir -p $(LOCALBIN)

CONTROLLER_GEN ?= $(LOCALBIN)/controller-gen-$(CONTROLLER_TOOLS_VERSION)
CONTROLLER_TOOLS_VERSION ?= v0.19.0

.PHONY: controller-gen
controller-gen: $(CONTROLLER_GEN) ## Download controller-gen locally if necessary.
$(CONTROLLER_GEN): $(LOCALBIN)
	$(call go-install-tool,$(CONTROLLER_GEN),sigs.k8s.io/controller-tools/cmd/controller-gen,$(CONTROLLER_TOOLS_VERSION))

.PHONY: kustomize
kustomize: ## 下载 kustomize 工具
	$(MAKE) -f Makefile.capbke kustomize

##@ 部署

.PHONY: deploy
deploy: ## 部署 capbke controller 到集群
	$(MAKE) -f Makefile.capbke deploy

.PHONY: dev-deploy
dev-deploy: ## 开发环境部署
	$(MAKE) -f Makefile.capbke dev-deploy

# go-install-tool will 'go install' any package with custom target and name of binary, if it doesn't exist
# $1 - target path with name of binary (ideally with version)
# $2 - package url which can be installed
# $3 - specific version of package
define go-install-tool
@[ -f $(1) ] || { \
set -e; \
package=$(2)@$(3) ;\
echo "Downloading $${package}" ;\
GOBIN=$(LOCALBIN) go install $${package} ;\
mv "$$(echo "$(1)" | sed "s/-$(3)$$//")" $(1) ;\
}
endef
