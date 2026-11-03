module gopkg.openfuyao.cn/cluster-api-provider-bke

go 1.24.5

require (
	bou.ke/monkey v1.0.2
	github.com/Masterminds/semver/v3 v3.4.0
	github.com/agiledragon/gomonkey/v2 v2.13.0
	github.com/beevik/ntp v1.4.3
	github.com/blang/semver v3.5.1+incompatible
	github.com/blang/semver/v4 v4.0.0
	github.com/btfak/sntp v0.0.0-20170909025617-bbde69cbb681
	github.com/containerd/console v1.0.5
	github.com/containerd/containerd v1.7.33
	github.com/coreos/go-semver v0.3.1
	github.com/coreos/go-systemd/v22 v22.6.0
	github.com/docker/docker v28.0.0+incompatible
	github.com/evanphx/json-patch v5.9.11+incompatible
	github.com/go-logr/logr v1.4.3
	github.com/google/go-containerregistry v0.14.0
	github.com/google/uuid v1.6.0
	github.com/hashicorp/go-version v1.8.0
	github.com/kevinburke/hostsfile v0.0.0-20220522040509-e5e984885321
	github.com/moby/sys/signal v0.7.1
	github.com/opencontainers/go-digest v1.0.0
	github.com/opencontainers/image-spec v1.1.1
	github.com/opencontainers/runtime-spec v1.2.1
	github.com/pkg/errors v0.9.1
	github.com/pkg/sftp v1.13.6
	github.com/prometheus/client_golang v1.23.2
	github.com/prometheus/client_model v0.6.2
	github.com/robfig/cron/v3 v3.0.1
	github.com/shirou/gopsutil/v3 v3.23.9
	github.com/stretchr/testify v1.11.1
	github.com/vishvananda/netlink v1.3.1
	github.com/wsva/lib_go v1.0.6
	go.etcd.io/etcd/client/v3 v3.6.4
	go.uber.org/zap v1.27.0
	golang.org/x/crypto v0.46.0
	golang.org/x/sync v0.19.0
	gomodules.xyz/jsonpatch/v2 v2.4.0
	google.golang.org/grpc v1.79.3
	gopkg.in/yaml.v3 v3.0.1
	gopkg.openfuyao.cn/common-modules/ologger/v26 v26.6.0
	helm.sh/helm/v3 v3.19.0
	k8s.io/api v0.34.0
	k8s.io/apiextensions-apiserver v0.34.0
	k8s.io/apimachinery v0.34.1
	k8s.io/cli-runtime v0.34.0
	k8s.io/client-go v12.0.0+incompatible
	k8s.io/cri-api v0.28.3
	k8s.io/klog/v2 v2.130.1
	k8s.io/kubectl v0.34.0
	k8s.io/kubernetes v1.26.4
	k8s.io/utils v0.0.0-20250604170112-4c0f3b243397
	sigs.k8s.io/cluster-api v1.5.0
	sigs.k8s.io/controller-runtime v0.22.4
	sigs.k8s.io/yaml v1.6.0
)

// bkeagent containerd/docker 兼容性 replace
replace github.com/opencontainers/runtime-spec => github.com/opencontainers/runtime-spec v1.1.0-rc.2

replace (
	github.com/Microsoft/hcsshim => github.com/Microsoft/hcsshim v0.11.7
	github.com/containerd/cgroups/v3 => github.com/containerd/cgroups/v3 v3.0.2
)

replace (
	// 更新 docker distribution
	github.com/docker/distribution => github.com/docker/distribution v2.8.2+incompatible
	github.com/google/gnostic => github.com/google/gnostic v0.6.9
	github.com/google/gnostic-models => github.com/google/gnostic-models v0.6.8

	go.etcd.io/etcd/client/v3 => go.etcd.io/etcd/client/v3 v3.5.9
	helm.sh/helm/v3 => helm.sh/helm/v3 v3.18.6
	k8s.io/api => k8s.io/api v0.28.0
	k8s.io/apiextensions-apiserver => k8s.io/apiextensions-apiserver v0.28.0
	k8s.io/apimachinery => k8s.io/apimachinery v0.28.0
	k8s.io/apiserver => k8s.io/apiserver v0.28.0
	k8s.io/cli-runtime => k8s.io/cli-runtime v0.28.0
	k8s.io/client-go => k8s.io/client-go v0.28.0
	k8s.io/cloud-provider => k8s.io/cloud-provider v0.28.0
	k8s.io/cluster-bootstrap => k8s.io/cluster-bootstrap v0.28.0
	k8s.io/code-generator => k8s.io/code-generator v0.28.0
	k8s.io/component-base => k8s.io/component-base v0.28.0
	k8s.io/component-helpers => k8s.io/component-helpers v0.28.0
	k8s.io/controller-manager => k8s.io/controller-manager v0.28.0
	k8s.io/cri-api => k8s.io/cri-api v0.28.0
	k8s.io/csi-translation-lib => k8s.io/csi-translation-lib v0.28.0
	k8s.io/dynamic-resource-allocation => k8s.io/dynamic-resource-allocation v0.28.0
	k8s.io/kube-aggregator => k8s.io/kube-aggregator v0.28.0
	k8s.io/kube-controller-manager => k8s.io/kube-controller-manager v0.28.0
	k8s.io/kube-openapi => k8s.io/kube-openapi v0.0.0-20250626183228-af0a60a813f8
	k8s.io/kube-proxy => k8s.io/kube-proxy v0.28.0
	k8s.io/kube-scheduler => k8s.io/kube-scheduler v0.28.0
	k8s.io/kubectl => k8s.io/kubectl v0.28.0
	k8s.io/kubelet => k8s.io/kubelet v0.28.0
	k8s.io/kubernetes => k8s.io/kubernetes v1.27.0-alpha.2
	k8s.io/legacy-cloud-providers => k8s.io/legacy-cloud-providers v0.28.0
	k8s.io/metrics => k8s.io/metrics v0.28.0
	k8s.io/mount-utils => k8s.io/mount-utils v0.28.0
	k8s.io/pod-security-admission => k8s.io/pod-security-admission v0.28.0
	k8s.io/sample-apiserver => k8s.io/sample-apiserver v0.28.0

	sigs.k8s.io/controller-runtime => sigs.k8s.io/controller-runtime v0.15.3
	sigs.k8s.io/kustomize/kyaml => sigs.k8s.io/kustomize/kyaml v0.14.3
)

require (
	cyphar.com/go-pathrs v0.2.1 // indirect
	dario.cat/mergo v1.0.1 // indirect
	github.com/AdaLogics/go-fuzz-headers v0.0.0-20240806141605-e8a1dd7889d6 // indirect
	github.com/AdamKorcz/go-118-fuzz-build v0.0.0-20231105174938-2b5cbb29f3e2 // indirect
	github.com/Azure/go-ansiterm v0.0.0-20250102033503-faa5f7b0171c // indirect
	github.com/BurntSushi/toml v1.6.0 // indirect
	github.com/MakeNowJust/heredoc v1.0.0 // indirect
	github.com/Masterminds/goutils v1.1.1 // indirect
	github.com/Masterminds/sprig/v3 v3.3.0 // indirect
	github.com/Masterminds/squirrel v1.5.4 // indirect
	github.com/Microsoft/go-winio v0.6.2 // indirect
	github.com/Microsoft/hcsshim v0.14.0-rc.1 // indirect
	github.com/asaskevich/govalidator v0.0.0-20230301143203-a9d515a09cc2 // indirect
	github.com/beorn7/perks v1.0.1 // indirect
	github.com/cespare/xxhash/v2 v2.3.0 // indirect
	github.com/chai2010/gettext-go v1.0.2 // indirect
	github.com/cloudflare/cfssl v1.6.4 // indirect
	github.com/containerd/cgroups v1.1.0 // indirect
	github.com/containerd/cgroups/v3 v3.1.0 // indirect
	github.com/containerd/containerd/api v1.10.0 // indirect
	github.com/containerd/continuity v0.4.5 // indirect
	github.com/containerd/errdefs v1.0.0 // indirect
	github.com/containerd/fifo v1.1.0 // indirect
	github.com/containerd/go-cni v1.1.13 // indirect
	github.com/containerd/log v0.1.0 // indirect
	github.com/containerd/platforms v1.0.0-rc.2 // indirect
	github.com/containerd/stargz-snapshotter/estargz v0.16.3 // indirect
	github.com/containerd/ttrpc v1.2.7 // indirect
	github.com/containerd/typeurl/v2 v2.2.3 // indirect
	github.com/containernetworking/cni v1.3.0 // indirect
	github.com/coredns/caddy v1.1.0 // indirect
	github.com/coredns/corefile-migration v1.0.20 // indirect
	github.com/cpuguy83/go-md2man/v2 v2.0.7 // indirect
	github.com/cyphar/filepath-securejoin v0.6.1 // indirect
	github.com/davecgh/go-spew v1.1.2-0.20180830191138-d8f796af33cc // indirect
	github.com/distribution/reference v0.6.0 // indirect
	github.com/docker/cli v28.4.0+incompatible // indirect
	github.com/docker/distribution v2.8.3+incompatible // indirect
	github.com/docker/docker-credential-helpers v0.9.3 // indirect
	github.com/docker/go-connections v0.5.0 // indirect
	github.com/docker/go-events v0.0.0-20190806004212-e31b211e4f1c // indirect
	github.com/docker/go-metrics v0.0.1 // indirect
	github.com/docker/go-units v0.5.0 // indirect
	github.com/docker/libtrust v0.0.0-20160708172513-aabc10ec26b7 // indirect
	github.com/dustin/go-humanize v1.0.1 // indirect
	github.com/emicklei/go-restful/v3 v3.12.2 // indirect
	github.com/evanphx/json-patch/v5 v5.9.0 // indirect
	github.com/exponent-io/jsonpath v0.0.0-20210407135951-1de76d718b3f // indirect
	github.com/fatih/color v1.15.0 // indirect
	github.com/felixge/httpsnoop v1.0.4 // indirect
	github.com/fsnotify/fsnotify v1.9.0 // indirect
	github.com/go-errors/errors v1.4.2 // indirect
	github.com/go-gorp/gorp/v3 v3.1.0 // indirect
	github.com/go-logr/stdr v1.2.2 // indirect
	github.com/go-logr/zapr v1.3.0 // indirect
	github.com/go-ole/go-ole v1.2.6 // indirect
	github.com/go-openapi/jsonpointer v0.21.0 // indirect
	github.com/go-openapi/jsonreference v0.21.0 // indirect
	github.com/go-openapi/swag v0.23.0 // indirect
	github.com/gobuffalo/flect v1.0.2 // indirect
	github.com/gobwas/glob v0.2.3 // indirect
	github.com/godbus/dbus/v5 v5.1.0 // indirect
	github.com/gogo/protobuf v1.3.2 // indirect
	github.com/golang/groupcache v0.0.0-20241129210726-2c02b8208cf8 // indirect
	github.com/golang/protobuf v1.5.4 // indirect
	github.com/golang/snappy v1.0.0 // indirect
	github.com/google/btree v1.1.3 // indirect
	github.com/google/certificate-transparency-go v1.1.4 // indirect
	github.com/google/gnostic-models v0.7.0 // indirect
	github.com/google/go-cmp v0.7.0 // indirect
	github.com/google/gofuzz v1.2.0 // indirect
	github.com/google/pprof v0.0.0-20250820193118-f64d9cf942d6 // indirect
	github.com/gorilla/mux v1.8.1 // indirect
	github.com/gosuri/uitable v0.0.4 // indirect
	github.com/gregjones/httpcache v0.0.0-20190611155906-901d90724c79 // indirect
	github.com/grpc-ecosystem/go-grpc-prometheus v1.2.0 // indirect
	github.com/grpc-ecosystem/grpc-gateway/v2 v2.27.2 // indirect
	github.com/hashicorp/errwrap v1.1.0 // indirect
	github.com/hashicorp/go-immutable-radix v1.3.1 // indirect
	github.com/hashicorp/go-memdb v1.3.4 // indirect
	github.com/hashicorp/go-multierror v1.1.1 // indirect
	github.com/hashicorp/golang-lru v0.5.4 // indirect
	github.com/huandu/xstrings v1.5.0 // indirect
	github.com/imdario/mergo v0.3.16 // indirect
	github.com/inconshreveable/mousetrap v1.1.0 // indirect
	github.com/ishidawataru/sctp v0.0.0-20230406120618-7ff4192f6ff2 // indirect
	github.com/jmoiron/sqlx v1.4.0 // indirect
	github.com/josharian/intern v1.0.0 // indirect
	github.com/json-iterator/go v1.1.12 // indirect
	github.com/klauspost/compress v1.18.1 // indirect
	github.com/kr/fs v0.1.0 // indirect
	github.com/lann/builder v0.0.0-20180802200727-47ae307949d0 // indirect
	github.com/lann/ps v0.0.0-20150810152359-62de8c46ede0 // indirect
	github.com/lib/pq v1.10.9 // indirect
	github.com/liggitt/tabwriter v0.0.0-20181228230101-89fcab3d43de // indirect
	github.com/lufia/plan9stats v0.0.0-20211012122336-39d0f177ccd0 // indirect
	github.com/mailru/easyjson v0.9.0 // indirect
	github.com/mattn/go-colorable v0.1.13 // indirect
	github.com/mattn/go-isatty v0.0.17 // indirect
	github.com/mattn/go-runewidth v0.0.14 // indirect
	github.com/mitchellh/copystructure v1.2.0 // indirect
	github.com/mitchellh/go-homedir v1.1.0 // indirect
	github.com/mitchellh/go-wordwrap v1.0.1 // indirect
	github.com/mitchellh/reflectwalk v1.0.2 // indirect
	github.com/moby/buildkit v0.25.2 // indirect
	github.com/moby/docker-image-spec v1.3.1 // indirect
	github.com/moby/locker v1.0.1 // indirect
	github.com/moby/spdystream v0.5.1 // indirect
	github.com/moby/swarmkit/v2 v2.1.1 // indirect
	github.com/moby/sys/mountinfo v0.7.2 // indirect
	github.com/moby/sys/sequential v0.6.0 // indirect
	github.com/moby/sys/user v0.4.0 // indirect
	github.com/moby/sys/userns v0.1.0 // indirect
	github.com/moby/term v0.5.2 // indirect
	github.com/modern-go/concurrent v0.0.0-20180306012644-bacd9c7ef1dd // indirect
	github.com/modern-go/reflect2 v1.0.3-0.20250322232337-35a7c28c31ee // indirect
	github.com/monochromegane/go-gitignore v0.0.0-20200626010858-205db1a8cc00 // indirect
	github.com/morikuni/aec v1.1.0 // indirect
	github.com/munnerz/goautoneg v0.0.0-20191010083416-a7dc8b61c822 // indirect
	github.com/onsi/ginkgo/v2 v2.23.4 // indirect
	github.com/onsi/gomega v1.37.0 // indirect
	github.com/opencontainers/selinux v1.13.1 // indirect
	github.com/pelletier/go-toml v1.9.5 // indirect
	github.com/peterbourgon/diskv v2.0.1+incompatible // indirect
	github.com/petermattis/goid v0.0.0-20240813172612-4fcff4a6cae7 // indirect
	github.com/pmezard/go-difflib v1.0.1-0.20181226105442-5d4384ee4fb2 // indirect
	github.com/power-devops/perfstat v0.0.0-20210106213030-5aafc221ea8c // indirect
	github.com/prometheus/common v0.66.1 // indirect
	github.com/prometheus/procfs v0.16.1 // indirect
	github.com/rivo/uniseg v0.4.2 // indirect
	github.com/rootless-containers/rootlesskit/v2 v2.3.5 // indirect
	github.com/rubenv/sql-migrate v1.8.1 // indirect
	github.com/russross/blackfriday/v2 v2.1.0 // indirect
	github.com/santhosh-tekuri/jsonschema/v6 v6.0.2 // indirect
	github.com/sasha-s/go-deadlock v0.3.5 // indirect
	github.com/shoenig/go-m1cpu v0.1.6 // indirect
	github.com/shopspring/decimal v1.4.0 // indirect
	github.com/sirupsen/logrus v1.9.3 // indirect
	github.com/spf13/cast v1.7.0 // indirect
	github.com/spf13/cobra v1.10.2 // indirect
	github.com/spf13/pflag v1.0.10 // indirect
	github.com/tklauser/go-sysconf v0.3.12 // indirect
	github.com/tklauser/numcpus v0.6.1 // indirect
	github.com/urfave/cli v1.22.17 // indirect
	github.com/vbatts/tar-split v0.12.1 // indirect
	github.com/vishvananda/netns v0.0.5 // indirect
	github.com/weppos/publicsuffix-go v0.15.1-0.20210511084619-b1f36a2d6c0b // indirect
	github.com/xlab/treeprint v1.2.0 // indirect
	github.com/yusufpapurcu/wmi v1.2.4 // indirect
	github.com/zmap/zcrypto v0.0.0-20210511125630-18f1e0152cfc // indirect
	github.com/zmap/zlint/v3 v3.1.0 // indirect
	go.etcd.io/bbolt v1.4.3 // indirect
	go.etcd.io/etcd/api/v3 v3.6.4 // indirect
	go.etcd.io/etcd/client/pkg/v3 v3.6.4 // indirect
	go.etcd.io/etcd/raft/v3 v3.5.9 // indirect
	go.opencensus.io v0.24.0 // indirect
	go.opentelemetry.io/auto/sdk v1.2.1 // indirect
	go.opentelemetry.io/contrib/instrumentation/net/http/otelhttp v0.62.0 // indirect
	go.opentelemetry.io/otel v1.39.0 // indirect
	go.opentelemetry.io/otel/metric v1.39.0 // indirect
	go.opentelemetry.io/otel/trace v1.39.0 // indirect
	go.starlark.net v0.0.0-20230525235612-a134d8f9ddca // indirect
	go.uber.org/multierr v1.11.0 // indirect
	go.yaml.in/yaml/v2 v2.4.3 // indirect
	go.yaml.in/yaml/v3 v3.0.4 // indirect
	golang.org/x/net v0.48.0 // indirect
	golang.org/x/oauth2 v0.34.0 // indirect
	golang.org/x/sys v0.40.0 // indirect
	golang.org/x/term v0.39.0 // indirect
	golang.org/x/text v0.33.0 // indirect
	golang.org/x/time v0.14.0 // indirect
	google.golang.org/genproto v0.0.0-20240227224415-6ceb2ff114de // indirect
	google.golang.org/genproto/googleapis/api v0.0.0-20251202230838-ff82c1b0f217 // indirect
	google.golang.org/genproto/googleapis/rpc v0.0.0-20251202230838-ff82c1b0f217 // indirect
	google.golang.org/protobuf v1.36.10 // indirect
	gopkg.in/evanphx/json-patch.v4 v4.13.0 // indirect
	gopkg.in/inf.v0 v0.9.1 // indirect
	gopkg.in/natefinch/lumberjack.v2 v2.2.1 // indirect
	gopkg.in/yaml.v2 v2.4.0 // indirect
	k8s.io/apiserver v0.35.1 // indirect
	k8s.io/cluster-bootstrap v0.28.0 // indirect
	k8s.io/component-base v0.35.1 // indirect
	k8s.io/kube-openapi v0.0.0-20250910181357-589584f1c912 // indirect
	oras.land/oras-go/v2 v2.6.0 // indirect
	sigs.k8s.io/json v0.0.0-20250730193827-2d320260d730 // indirect
	sigs.k8s.io/kustomize/api v0.20.1 // indirect
	sigs.k8s.io/kustomize/kyaml v0.20.1 // indirect
	sigs.k8s.io/randfill v1.0.0 // indirect
	sigs.k8s.io/structured-merge-diff/v4 v4.6.0 // indirect
)
