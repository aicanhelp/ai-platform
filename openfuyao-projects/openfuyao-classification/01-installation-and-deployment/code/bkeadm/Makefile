
#
GOLANG=1.19.x
ARCH ?= linux/amd64,linux/arm64
timestamp = `date "+%Y-%m-%d"`
version = "v1.0.0"
COMMIT_ID ?= $(shell git rev-parse HEAD)


LDFLAGS = -s -w \
	-X main.gitCommitId=$(COMMIT_ID) \
	-X main.architecture=$(shell go env GOHOSTOS)/$(shell go env GOHOSTARCH) \
	-X main.timestamp=$(timestamp) \
	-X main.ver=$(version)

# Get the currently used golang install path (in GOPATH/bin, unless GOBIN is set)
ifeq (,$(shell go env GOBIN))
GOBIN=$(shell go env GOPATH)/bin
else
GOBIN=$(shell go env GOBIN)
endif

all: build
	@ echo "bin/bke"

test:
	go test .

.PHONY: test-preflight
test-preflight:
	cd preflight && go test ./...
	cd preflight && go test -race ./...
	go test ./pkg/preflightbridge
	go test ./pkg/preflightconfig
	go test ./pkg/preflightbundlecompiler
	go test ./bootstrap/preflightlogenv ./utils/log
	go test . -run TestEmbeddedPreflightPolicyCompiles
	go test ./cmd -run 'Preflight|ReleaseBundle'

# Build binary
.PHONY: build
build:
	@echo LatestCommit: `git log --pretty='%s%b%B' -n 1`
	@go build -ldflags="-s -w -X main.gitCommitId=`git rev-parse HEAD` -X main.architecture=`go env GOHOSTOS`/`go env GOHOSTARCH` \
	-X main.timestamp=$(timestamp) -X main.ver=$(version)" -o bin/bke .

PLATFORMS ?= amd64 arm64
.PHONY: release # Build a multi-arch binary
release:
	@echo "Building binaries...."
	@set -e; for platform in $(PLATFORMS); do \
		echo -n "$$platform..."; \
		ARCH=$$platform $(MAKE) docker-build; \
	done

# Make sure Docker supports multi-schema image compilation
docker-build:
	#@build/run-in-docker.sh ARCH=$(ARCH) COMMIT_ID=`git rev-parse HEAD` VERSION=$(version) TIMESTAMP=$(timestamp) build/build.sh
	CGO_ENABLED=0 GOARCH=$(ARCH) go build \
		-tags "remote exclude_graphdriver_btrfs btrfs_noversion exclude_graphdriver_devicemapper containers_image_openpgp" \
		-ldflags "$(LDFLAGS)" \
		-o bin/bke_$(ARCH) .

buildx:
	@docker run --privileged --rm tonistiigi/binfmt --uninstall qemu-* && \
	docker run --privileged --rm tonistiigi/binfmt --install all && \
	docker buildx rm mybuilder || true && \
	docker buildx create --use --name mybuilder && \
	docker buildx inspect mybuilder --bootstrap && \
	docker buildx ls

docker:
	@docker build -t registry.cn-hangzhou.aliyuncs.com/bocloud/bkeadm:latest .
	@docker push registry.cn-hangzhou.aliyuncs.com/bocloud/bkeadm:latest

clean:
	@rm -rf bin/bke*
