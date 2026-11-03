
SHELL := /usr/bin/env bash -o errexit -o pipefail -o nounset

CUR_DIR := $(shell pwd)
OUTPUT_DIR=$(CUR_DIR)/_output

.PHONY: build init clean all image

all: clean init build image

init:
	mkdir -p $(OUTPUT_DIR)

build:
	ROOT_DIR=${CUR_DIR} hack/build.sh

image:
	ROOT_DIR=${CUR_DIR} hack/build_image.sh

clean:
	@rm -rf $(OUTPUT_DIR)


