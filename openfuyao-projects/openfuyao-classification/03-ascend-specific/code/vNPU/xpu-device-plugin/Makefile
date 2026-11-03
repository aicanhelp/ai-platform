
.PHONY: all
.DEFAULT_GOAL := all

EXTLDFLAGS := \
	-Wl,-z,relro \
	-Wl,-z,now \
	-Wl,-z,noexecstack

all: vnpu xpu-client-tool

vnpu:
	export CGO_LDFLAGS_ALLOW='-Wl,--unresolved-symbols=ignore-in-object-files' && \
		export CGO_CFLAGS='-fcommon' && export CGO_CPPFLAGS='-fcommon' && \
		go build \
			-tags vnpu \
			-buildmode=pie \
			-ldflags="-s -w -linkmode 'external' -extldflags '$(EXTLDFLAGS)'" \
			-o npu-device-plugin \
			./cmd

xpu-client-tool:
	export CGO_LDFLAGS_ALLOW='-Wl,--unresolved-symbols=ignore-in-object-files' && \
	export CGO_CFLAGS='-D_FORTIFY_SOURCE=2 -O2' && \
		go build \
			-tags vnpu \
			-buildmode=pie \
			-ldflags="-s -w -linkmode 'external' -extldflags '$(EXTLDFLAGS)'" \
			-o xpu-client-tool \
			./client

test: vnpu
	go test -gcflags="all=-l" -count=1 -cover ./...

test-race: vnpu
	go test -race -gcflags="all=-l" -count=1 ./...

coverage: vnpu
	go test -gcflags="all=-l" -count=1 -coverprofile=coverage.out ./...
	go tool cover -html=coverage.out -o coverage.html

clean:
	rm -rf ./xpu-client-tool ./npu-device-plugin
