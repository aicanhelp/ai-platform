########################################
# 1. Variables & Configuration
########################################

# Versions
GATEWAY_API_VERSION ?= v1.2.1
IGW_CHART_VERSION   ?= v1.1.0
# Istio version tag - pinned to a dev version that supports inference extension
ISTIO_TAG           := $(shell curl -s https://storage.googleapis.com/istio-build/dev/1.28-dev)

# Infrastructure
NAMESPACE           ?= dev-router
ISTIO_NAMESPACE     ?= istio-system
GATEWAY_PROVIDER    ?= istio

# Tools
CONTAINER_RUNTIME := $(shell command -v nerdctl >/dev/null 2>&1 && echo "nerdctl" && exit 0; command -v docker >/dev/null 2>&1 && echo "docker" && exit 0; command -v podman >/dev/null 2>&1 && echo "podman" && exit 0; echo "unknown")
ISTIOCTL          ?= istioctl

# Images
PLATFORMS         ?= linux/arm64
COMMIT            := $(shell git rev-parse HEAD)
BUILD_REF         := $(shell git rev-parse --abbrev-ref HEAD)-$(shell date +%Y%m%d%H%M%S)
EPP_IMAGE         := hermes-router/epp:dev

# Model Defaults
MODEL             ?= Qwen/Qwen3-8B
MODEL_NAME        := $(shell printf '%s' '$(MODEL)' | tr '/A-Z' '-a-z')
RELEASE_NAME      ?= vllm-pd

# Export for sub-shells
export CONTAINER_RUNTIME

########################################
# 2. Infrastructure Deployment (Foundations)
########################################

.PHONY: check-tools
check-tools:
	@echo "[INFO] Checking required tools..."
	@command -v kubectl >/dev/null 2>&1 || { echo "[ERROR] kubectl not found"; exit 1; }
	@command -v helm >/dev/null 2>&1 || { echo "[ERROR] helm not found"; exit 1; }
	@command -v $(ISTIOCTL) >/dev/null 2>&1 || { echo "[ERROR] istioctl not found"; exit 1; }
	@echo "[OK] All tools found."

.PHONY: create-namespace
create-namespace:
	@echo "[INFO] Ensuring namespace $(NAMESPACE) exists..."
	@if ! kubectl get namespace $(NAMESPACE) >/dev/null 2>&1; then \
	  kubectl create namespace $(NAMESPACE); \
	  echo "[OK] Namespace $(NAMESPACE) created."; \
	else \
	  echo "[OK] Namespace $(NAMESPACE) already exists."; \
	fi

.PHONY: install-crds
install-crds:
	@echo "[INFO] Installing Gateway API CRDs ($(GATEWAY_API_VERSION))..."
	kubectl apply --server-side -f https://github.com/kubernetes-sigs/gateway-api/releases/download/$(GATEWAY_API_VERSION)/standard-install.yaml
	@echo "[INFO] Installing Inference Extension CRDs..."
	kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/releases/download/$(IGW_CHART_VERSION)/manifests.yaml

.PHONY: install-istio
install-istio: check-tools
	@echo "[INFO] Installing Istio with Inference Extension enabled (TAG=$(ISTIO_TAG))..."
	$(ISTIOCTL) install -y --set tag=$(ISTIO_TAG) --set hub=gcr.io/istio-testing --set values.pilot.env.ENABLE_GATEWAY_API_INFERENCE_EXTENSION=true
	@echo "[OK] Istio installed."

.PHONY: build-epp
build-epp:
	@echo "[INFO] Building EPP image ($(EPP_IMAGE))..."
	$(CONTAINER_RUNTIME) -n k8s.io build \
		--platform=$(PLATFORMS) \
		--build-arg TARGETOS=linux \
		--build-arg TARGETARCH=arm64 \
		--build-arg COMMIT=$(COMMIT) \
		--build-arg BUILD_REF=$(BUILD_REF) \
		-t $(EPP_IMAGE) \
		-f Dockerfile.epp \
		.

.PHONY: deploy-infrastructure
deploy-infrastructure: check-tools create-namespace install-crds install-istio build-epp
	@echo "[SUCCESS] Infrastructure deployed."

########################################
# 3. Application Deployment (Inference Stack)
########################################

.PHONY: deploy-inferencepool
deploy-inferencepool:
	@echo "[INFO] Updating Helm dependencies..."
	helm dependency update examples/1_pd_bucket/charts/hermes-router
	@echo "[INFO] Deploying InferencePool (vllm-$(MODEL_NAME))..."
	helm upgrade --install vllm-$(MODEL_NAME) \
	  examples/1_pd_bucket/charts/hermes-router \
	  --namespace $(NAMESPACE) --create-namespace \
	  -f examples/1_pd_bucket/profiles/epp-random-pd-bucket.yaml \
	  --set inferencepool.provider.name=$(GATEWAY_PROVIDER)

.PHONY: deploy-inference-gateway
deploy-inference-gateway:
	@echo "[INFO] Deploying Inference Gateway..."
	helm upgrade --install inference-gateway examples/1_pd_bucket/charts/gateway \
	  -n $(NAMESPACE) --create-namespace \
	  --set gateway.className=$(GATEWAY_PROVIDER)
	@echo "[INFO] Displaying configured routes:"
	@kubectl get httproute -n $(NAMESPACE)

.PHONY: deploy-all
deploy-all: deploy-infrastructure deploy-inferencepool deploy-inference-gateway
	@echo "[SUCCESS] Full environment deployed."

########################################
# 4. Debugging & Testing
########################################

.PHONY: deploy-vllm-aggregated
deploy-vllm-aggregated:
	@if [ -z "$$HF_TOKEN" ]; then echo "[ERROR] HF_TOKEN is not set"; exit 1; fi
	@echo "[INFO] Deploying vLLM (Aggregated)..."
	helm upgrade --install vllm examples/1_pd_bucket/charts/vllm \
		-n ${NAMESPACE} --create-namespace \
		--set hf.createSecret=true \
		--set hf.secretName=hf-token \
		--set hf.secretKey=HF_TOKEN \
		--set hf.token=$$HF_TOKEN \
		--set modelServer.rootCachePath=/home/llm_cache

.PHONY: deploy-vllm-pd
deploy-vllm-pd:
	@if [ -z "$$HF_TOKEN" ]; then echo "[ERROR] HF_TOKEN is not set"; exit 1; fi
	@echo "[INFO] Deploying vLLM (PD Disaggregated) - Release: $(RELEASE_NAME)..."
	helm upgrade --install $(RELEASE_NAME) examples/1_pd_bucket/charts/vllm-pd \
		-n ${NAMESPACE} --create-namespace \
		--set modelServer.name=$(MODEL)

.PHONY: port-forward
port-forward:
	@echo "[INFO] Setting up port-forward to Gateway..."
	@GATEWAY_SVC=$$(kubectl get svc -n $(NAMESPACE) -o jsonpath='{.items[?(@.metadata.name=="inference-gateway-istio")].metadata.name}' 2>/dev/null); \
	if [ -z "$$GATEWAY_SVC" ]; then \
		GATEWAY_SVC=$$(kubectl get svc -n $(NAMESPACE) -l gateway.networking.k8s.io/gateway-name=inference-gateway -o jsonpath='{.items[0].metadata.name}'); \
	fi; \
	if [ -z "$$GATEWAY_SVC" ]; then \
		echo "[ERROR] Could not find Gateway service in namespace $(NAMESPACE)"; exit 1; \
	fi; \
	echo "[INFO] Forwarding service/$$GATEWAY_SVC 8001:80..."; \
	kubectl port-forward -n $(NAMESPACE) service/$$GATEWAY_SVC 8001:80

.PHONY: test-request
test-request:
	@echo "[INFO] Sending test request..."
	@echo '{"model":"$(MODEL)","messages":[{"role":"user","content":"你好"}],"max_tokens":100,"temperature":0.7,"stream":true}' > /tmp/test.json
	@curl -v -X POST http://localhost:8001/v1/chat/completions \
	  -H "Content-Type: application/json" \
	  -d @/tmp/test.json

.PHONY: benchmark-evalscope
benchmark-evalscope:
	@echo "[INFO] Running Evalscope benchmark..."
	source /usr/local/Ascend/ascend-toolkit/set_env.sh && \
	/usr/bin/python3 -m python /root/lileqi/datasets/evalscope.py

.PHONY: clean
clean:
	@echo "[WARN] Cleaning up all resources..."
	-helm uninstall -n $(NAMESPACE) inference-gateway
	-helm uninstall -n $(NAMESPACE) vllm-$(MODEL_NAME)
	-helm uninstall -n $(NAMESPACE) vllm
	-helm uninstall -n $(NAMESPACE) $(RELEASE_NAME)
	-$(ISTIOCTL) uninstall -y --purge
	-kubectl delete ns $(NAMESPACE)
	-kubectl delete ns $(ISTIO_NAMESPACE)