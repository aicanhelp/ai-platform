REGISTRY ?= registry.cn-hangzhou.aliyuncs.com/bocloud

#BKE_REGISTRY ?= 192.168.2.140:5001/kubernetes
BKE_REGISTRY ?= 172.28.100.36:40443/kubernetes

DEV_REGISTRY ?= deploy.bocloud.k8s:40443/kubernetes


VERSION ?= latest

IMAGE_NAME ?= bkeagent-launcher


# 版本镜像指定镜像仓库
RELEASE_IMG ?= $(REGISTRY)/$(IMAGE_NAME):$(VERSION)
# 推送镜像到BKE仓库
BKE_REGISTRY_IMG ?= $(BKE_REGISTRY)/$(IMAGE_NAME):$(VERSION)
# 推送镜像到本地仓库
LOCAL_DEV_IMG ?= $(DEV_REGISTRY)/$(IMAGE_NAME):$(VERSION)


AGENT_VERSION ?= unknown
AGENT_COMMIT_ID ?= unknown
AGENT_ARCH ?= unknown
AGENT_BUILD_TIME=$(shell cat ../build/time)

COMMIT=$(shell git rev-parse HEAD)
DATE=$(shell date +'%Y-%m-%d %H:%M:%SZ')
ARCH=$(shell go env GOHOSTARCH)
HOSTOS=$(shell go env GOHOSTOS)

EMPTY :=
SPACE := $(EMPTY) $(EMPTY)
COMMA := ,
PLATFORMS ?=amd64 arm64


.PHONY: all
all: build docker-build push migrate

.PHONY: build
build:
	@echo build $(ARCH) binary
	@echo agent build time: ${AGENT_BUILD_TIME}
	@CGO_ENABLED=0 GOOS=linux GOARCH=$(ARCH) go build \
		-trimpath \
		-ldflags "-X 'main.gitCommitId=${COMMIT}' \
				  -X 'main.architecture=${HOSTOS}/${ARCH}' \
		          -X 'main.buildTime=${DATE}' \
		          -X 'main.version=${VERSION}' \
		          -X 'main.agentVersion=${AGENT_VERSION}' \
		          -X 'main.agentCommitId=${AGENT_COMMIT_ID}' \
				  -X 'main.agentBuildTime=${AGENT_BUILD_TIME}' \
		          -X 'main.agentArch=${HOSTOS}/${ARCH}'" \
		-o ./bin/launcher-$(ARCH) .

.PHONY: docker-build
docker-build: cp
	@docker build --build-arg TARGETARCH=$(ARCH) -t $(REGISTRY)/$(IMAGE_NAME):$(VERSION) .

push:
	@docker push $(REGISTRY)/$(IMAGE_NAME):$(VERSION)

release: bin clean cp
	echo "Building binaries..."
	$(foreach PLATFORM,$(PLATFORMS), make build ARCH=$(PLATFORM) AGENT_VERSION=$(AGENT_VERSION) AGENT_COMMIT_ID=$(AGENT_COMMIT_ID);)

	@docker buildx build \
		--push \
		--progress plain \
		--platform $(subst $(SPACE),$(COMMA),$(PLATFORMS)) \
 		-t $(RELEASE_IMG) .
	make migrate
	# release external image: $(RELEASE_IMG)
	# release internal image: $(BKE_REGISTRY_IMG)


migrate: migrate-image migrate-image-dev

.PHONY: migrate-image
migrate-image: ## Migrate image to BKE registry
	@bke registry sync --dest-tls-verify --source $(RELEASE_IMG) --target $(BKE_REGISTRY_IMG) --multi-arch

.PHONY: migrate-image-dev
migrate-image-dev: ## Migrate image to local registry
	@bke registry sync --dest-tls-verify --source $(RELEASE_IMG) --target $(LOCAL_DEV_IMG) --multi-arch

package: cp
	@scripts/package.sh $(REGISTRY)/$(IMAGE_NAME) $(VERSION)

.PHONY: cp
cp:
	cp ../bin/bkeagent-* ./bin

.PHONY: clean
clean:
	@rm -rf ./bin/*


.PHONY: bin
bin:
	if [ ! -d bin ]; then mkdir bin; fi