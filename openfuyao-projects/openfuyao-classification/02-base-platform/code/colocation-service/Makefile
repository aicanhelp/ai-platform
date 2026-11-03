export GO111MODULE=on

all: generate vet fmt verify test

# 运行代码生成：Deepcopy函数, CRDs和docs
.PHONY: generate
generate: update-codegen

.PHONY: update-codegen
update-codegen:
	chmod +x hack/*.sh
	hack/update-codegen.sh

# 运行go fmt格式化代码
fmt:
	go fmt ./...

# 运行go vet静态检查
vet:
	go vet ./...

# 运行go test
test:
	go test -race -cover ./pkg/apis/...

# 安装CRDs和样例CRs
.PHONY: install
install: crd example

# 安装CRDs
.PHONY: crd
crd:
	kubectl kustomize config/crds | kubectl apply -f -

# 安装CRs
.PHONY: example
example:
	hack/install-examples.sh

# 删除安装的CRDs和CRs.
.PHONY: uninstall
uninstall:
	hack/delete-crds.sh

# 生成文档
.PHONY: docs
docs:
	hack/make-docs.sh

export BUILDX_CONTEXT = colocation-builder
export BUILDX_PLATFORMS = linux/amd64,linux/arm64

# 验证是否支持Docker Buildx.
.PHONY: image.buildx.verify
image.buildx.verify:
	docker version
	$(eval PASS := $(shell docker buildx --help | grep "docker buildx" ))
	@if [ -z "$(PASS)" ]; then \
		echo "Cannot find docker buildx, please install first."; \
		exit 1;\
	else \
		echo "===========> Support docker buildx"; \
		docker buildx version; \
	fi

# 设置multi-arch docker buildx环境变量
.PHONY: image.multiarch.setup
image.multiarch.setup: image.buildx.verify
    # 确保qemu在binfmt_misc中
	@if [ "$(shell uname)" == "Linux" ]; then \
		docker run --rm --privileged multiarch/qemu-user-static --reset -p yes; \
	fi
    # 设置buildx环境变量：重建
	docker buildx rm $(BUILDX_CONTEXT) || :
	docker buildx create --use --name $(BUILDX_CONTEXT) --platform "${BUILDX_PLATFORMS}"