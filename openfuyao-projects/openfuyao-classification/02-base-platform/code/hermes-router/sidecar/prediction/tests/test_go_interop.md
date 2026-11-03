# Go/Python Prediction Interop Smoke

The Go client and Python sidecar share `api/prediction/v1/prediction.proto`.

```bash
cd sidecar/prediction
uv sync --extra test

# The sidecar serves predictions from a trained bundle, so build one from the
# fixture first (small-sample overrides so the tiny fixture still yields a model).
uv run prediction-train \
  --input tests/fixtures/completed_requests.jsonl \
  --artifact-root /tmp/hermes-prediction-artifacts \
  --target-model test-model \
  --model-version smoke \
  --min-samples-per-slot 2 \
  --min-holdout-samples-per-slot 1 \
  --bucket-cap 0

uv run prediction-sidecar \
  --socket-path /tmp/hermes-prediction-test.sock \
  --artifact-root /tmp/hermes-prediction-artifacts \
  --target-model test-model \
  --model-version smoke
```

In another shell:

```bash
go test ./pkg/epp/framework/plugins/scheduling/scorer/prediction -run TestGRPCPredictorClientPredictsOverUDS
```

The checked-in automated coverage is split between:

- `pkg/epp/framework/plugins/scheduling/scorer/prediction/grpc_client_test.go` for the Go UDS gRPC client contract.
- `sidecar/prediction/tests/test_app_smoke.py` for the Python UDS gRPC sidecar contract.
