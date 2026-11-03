module huawei.com/xpu-exporter

go 1.24.0

replace (
	google.golang.org/grpc => google.golang.org/grpc v1.57.2
	huawei.com/xpu-exporter/v6 => gitee.com/ascend/ascend-npu-exporter/v6 v6.0.0-RC2.b001
	huawei.com/vxpu-device-plugin => ./../xpu-device-plugin
	k8s.io/fy-ext => gitcode.com/openFuyao/kubernetes.git/staging/src/k8s.io/fy-ext v0.0.0-20250930090625-c9b922cb1bc3
)

require (
	github.com/agiledragon/gomonkey/v2 v2.8.0
	github.com/prometheus/client_golang v1.16.0
	github.com/stretchr/testify v1.10.0
	google.golang.org/grpc v1.68.1
	google.golang.org/protobuf v1.36.5
	huawei.com/vxpu-device-plugin v0.0.0-00010101000000-000000000000
)

require (
	github.com/beorn7/perks v1.0.1 // indirect
	github.com/cespare/xxhash/v2 v2.2.0 // indirect
	github.com/davecgh/go-spew v1.1.1 // indirect
	github.com/golang/protobuf v1.5.3 // indirect
	github.com/kr/text v0.2.0 // indirect
	github.com/matttproud/golang_protobuf_extensions v1.0.4 // indirect
	github.com/pmezard/go-difflib v1.0.0 // indirect
	github.com/prometheus/client_model v0.3.0 // indirect
	github.com/prometheus/common v0.42.0 // indirect
	github.com/prometheus/procfs v0.10.1 // indirect
	github.com/rogpeppe/go-internal v1.14.1 // indirect
	github.com/sirupsen/logrus v1.8.2 // indirect
	golang.org/x/net v0.38.0 // indirect
	golang.org/x/sys v0.31.0 // indirect
	golang.org/x/text v0.23.0 // indirect
	google.golang.org/genproto/googleapis/rpc v0.0.0-20241209162323-e6fa225c2576 // indirect
	gopkg.in/yaml.v3 v3.0.1 // indirect
)
