SHELL := /usr/bin/env bash -o errexit -o pipefail -o nounset

CUR_DIR := $(shell pwd)
OUTPUT_DIR=$(CUR_DIR)/_output
CHART_DIR=$(CUR_DIR)/charts/ub-network-bandwidth-operator
CHART_OUTPUT_DIR=$(OUTPUT_DIR)/chart
HELM ?= helm

.PHONY: build init clean all image chart

all: clean init build image

init:
	mkdir -p $(OUTPUT_DIR)

build:
	ROOT_DIR=${CUR_DIR} hack/build.sh

image:
	ROOT_DIR=${CUR_DIR} hack/build_image.sh

chart:
	mkdir -p $(CHART_OUTPUT_DIR)
	$(HELM) package $(CHART_DIR) --destination $(CHART_OUTPUT_DIR)

clean:
	@rm -rf $(OUTPUT_DIR)
