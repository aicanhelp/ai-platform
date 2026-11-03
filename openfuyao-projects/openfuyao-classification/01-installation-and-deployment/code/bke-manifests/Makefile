
#REPOSITORY ?= 192.168.2.140:5001/kubernetes
REPOSITORY ?= deploy.bocloud.k8s:40443/kubernetes
REPOSITORY2=registry.cn-hangzhou.aliyuncs.com/bocloud

NAME ?= addons
VERSION ?= latest
ARCH ?= linux/arm64,linux/amd64

TIME_STAMP = $(shell date +%Y%m%d%H%M)

REPOSITORY_IMAGE ?= $(REPOSITORY)/$(NAME):$(VERSION)
REPOSITORY2_IMAGE ?= $(REPOSITORY2)/$(NAME):$(VERSION)
DEV_REPOSITORY_IMAGE ?= $(REPOSITORY)/$(NAME):$(VERSION)-$(TIME_STAMP)
DEV_REPOSITORY2_IMAGE ?= $(REPOSITORY2)/$(NAME):$(VERSION)-$(TIME_STAMP)
# git info
COMMIT_ID=$(shell git rev-parse HEAD)
BUILD_TIME=$(shell date +'%Y-%m-%dT%H:%M')
HOST_ARCH=$(shell go env GOHOSTARCH)
HOST_OS=$(shell go env GOHOSTOS)
BUILD_TAG=$(shell date +'%Y%m%d%H%M')




.PHONY: all
all: build

.PHONY: docker-build
docker-build: record-build-info
	docker build \
		--no-cache \
		--network host \
		--platform=linux/${HOST_ARCH} \
		--build-arg COMMIT_ID=${COMMIT_ID} \
		--build-arg BUILD_TIME=${BUILD_TIME} \
		--build-arg TARGETARCH=${HOST_ARCH} \
		--build-arg TARGETOS=${HOST_OS} \
		--build-arg VERSION=${VERSION} \
		-t ${REPOSITORY}/addons:${VERSION}-${HOST_ARCH} .

.PHONY: docker-push
docker-push:
	docker push ${REPOSITORY}/addons:${VERSION}-${HOST_ARCH}


.PHONY: docker
docker: docker-build docker-push
	echo ${REPOSITORY}/addons:${VERSION}-${HOST_ARCH}


.PHONY: release
release:
	docker buildx build -t $(REPOSITORY2_IMAGE) --platform=$(ARCH) . --push

.PHONY: package
package:
	@scripts/package.sh $(REPOSITORY)/addons $(VERSION)

.PHONY: release-image
release-image: record-build-info
	@docker buildx build --build-arg COMMIT_ID=${COMMIT_ID} --build-arg BUILD_TIME=${BUILD_TIME} -t $(REPOSITORY2_IMAGE) --platform=$(ARCH) . --push
	@bke registry sync --dest-tls-verify --source $(REPOSITORY2_IMAGE) --target $(REPOSITORY_IMAGE) --multi-arch
	# release external image: $(REPOSITORY2_IMAGE)
	# release internal image: $(REPOSITORY_IMAGE)

.PHONY: dev-release
dev-release: record-build-info
	@docker buildx build --build-arg COMMIT_ID=${COMMIT_ID} --build-arg BUILD_TIME=${BUILD_TIME} -t $(DEV_REPOSITORY2_IMAGE) --platform=$(ARCH) . --push
	@bke registry sync --dest-tls-verify --source $(DEV_REPOSITORY2_IMAGE) --target $(DEV_REPOSITORY_IMAGE) --multi-arch
	# dev release external image: $(DEV_REPOSITORY_IMAGE)
	# dev release internal image: $(DEV_REPOSITORY2_IMAGE)


.PHONY: cp
cp: # used for local development
    # copy manifests to /manifests dir
	@sudo rm -rf /manifests && sudo cp -r . /manifests && sudo chown -R $(shell id -u):$(shell id -g) /manifests

.PHONY: record-build-info
record-build-info:
	# 清空文件内容
	> BUILD_INFO
	echo "🤯 Version=${VERSION}" >> BUILD_INFO
	echo "🤔 GitCommitId=${COMMIT_ID}" >> BUILD_INFO
	echo "👉 Architecture=${HOST_ARCH}" >> BUILD_INFO
	echo "⏲ BuildTime=${BUILD_TIME}" >> BUILD_INFO

buildx:
	@docker run --privileged --rm tonistiigi/binfmt --uninstall qemu-* && \
	docker run --privileged --rm tonistiigi/binfmt --install all && \
	docker buildx rm mybuilder && \
	docker buildx create --use --name mybuilder --driver-opt image=dockerpracticesig/buildkit:master && \
	docker buildx inspect mybuilder --bootstrap && \
	docker buildx ls