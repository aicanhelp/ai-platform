/******************************************************************
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * cluster-api-provider-bke is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package config

import (
	"flag"
	"os"
	"strconv"

	"gopkg.in/yaml.v3"

	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/constant"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/log"
)

var (
	MetricsAddr            string
	EnableLeaderElection   bool
	ProbeAddr              string
	ProbeScheme            string // "http" or "https", default "http"
	ProbePort              int    // Port for HTTPS health probe, default 9444
	WebhookCertDir         string
	WebhookPort            int
	WebhookHost            string
	BkeClusterConcurrency  int
	BkeMachineConcurrency  int
	EnableInternalUpdate   bool
	OCIDigestCheckInterval int
	OCIRegistryUsername    string
	OCIRegistryPassword    string
	OCIRegistryInsecure    bool
	EnableOCIDigestMonitor bool
	// HelmComponentSupport enables yaml/helm ComponentExecutor injection for declarative DAG.
	// Per-cluster annotation cvo.openfuyao.cn/helm-component overrides this flag when present.
	HelmComponentSupport bool
	ReleaseCacheDir      string
	ClientQPS            float32
	ClientBurst          int
	ClientConfigFile     string
)

const (
	DefaultReleaseCacheDir  = "/var/lib/bke/release-cache"
	DefaultClientQPS        = 50
	DefaultClientBurst      = 100
	DefaultClientConfigFile = "/etc/bke/client-config.yaml"
)

// yaml的key
type ClientConfig struct {
	QPS   float32 `yaml:"qps"`
	Burst int     `yaml:"burst"`
}

func ConfigurationFlag() {
	flag.StringVar(&MetricsAddr, "metrics-bind-address", "0", "The address the metric endpoint binds to. eg. :8080")
	flag.StringVar(&ProbeAddr, "health-probe-bind-address", ":8081", "The address the probe endpoint binds to.")
	flag.StringVar(&ProbeScheme, "health-probe-scheme", "http", "The scheme for health probe: http or https. Default: http")
	flag.IntVar(&ProbePort, "health-probe-port", 9444, "The port for HTTPS health probe server. Default: 9444 (webhook uses 9443)")
	flag.BoolVar(&EnableLeaderElection, "leader-elect", false,
		"Enable leader election for controller manager. "+
			"Enabling this will ensure there is only one active controller manager.")
	flag.StringVar(&WebhookCertDir, "webhook-cert-dir", "/tmp/k8s-webhook-server/serving-certs/",
		"Webhook cert dir, only used when webhook-port is specified.")
	flag.IntVar(&WebhookPort, "webhook-port", 9443, "Webhook Server port")
	flag.StringVar(&WebhookHost, "webhook-host", "", "Webhook Server host")
	flag.IntVar(&BkeClusterConcurrency, "bke-cluster-concurrency", constant.DefaultConcurrency, "Number of BKECluster to process simultaneously")
	flag.IntVar(&BkeMachineConcurrency, "bke-machine-concurrency", constant.DefaultConcurrency, "Number of BKEMachine to process simultaneously")
	flag.BoolVar(&EnableInternalUpdate, "enable-internal-update", false, "Enable internal update")
	flag.IntVar(&OCIDigestCheckInterval, "oci-digest-check-interval", 300, "OCI digest check interval in seconds for UpgradePath monitor. Default: 300 (5 minutes)")
	flag.StringVar(&OCIRegistryUsername, "oci-registry-username", "", "Username for OCI registry authentication (used by UpgradePath digest monitor)")
	flag.StringVar(&OCIRegistryPassword, "oci-registry-password", "", "Password for OCI registry authentication (used by UpgradePath digest monitor)")
	flag.BoolVar(&OCIRegistryInsecure, "oci-registry-insecure-skip-verify", true, "Skip TLS cert verification when pulling OCI artifacts (INSECURE, use only in trusted env)")
	flag.BoolVar(&EnableOCIDigestMonitor, "enable-oci-digest-monitor", true, "Enable OCI digest monitor for UpgradePath controller")
	flag.BoolVar(&HelmComponentSupport, "helm-component-support", false,
		"Enable yaml/helm component executor path for declarative upgrade (annotation cvo.openfuyao.cn/helm-component overrides)")
	flag.StringVar(&ReleaseCacheDir, "release-cache-dir", DefaultReleaseCacheDir,
		"Directory for validated release image bundle disk cache (mount hostPath here in production)")
	flag.Var((*float32Value)(&ClientQPS), "client-qps", "Kubernetes client QPS. Priority: flag > env > config file > default")
	flag.IntVar(&ClientBurst, "client-burst", 0, "Kubernetes client burst. Priority: flag > env > config file > default")
	flag.StringVar(&ClientConfigFile, "client-config-file", DefaultClientConfigFile, "Kubernetes client throttling config file")
}

// ResolveClientConfig resolves client throttling settings after flag.Parse.
func ResolveClientConfig() {
	resolvedQPS := float32(DefaultClientQPS)
	resolvedBurst := DefaultClientBurst
	qpsSource := "default"
	burstSource := "default"

	if cfg, ok := loadClientConfigFile(ClientConfigFile); ok {
		if cfg.QPS > 0 {
			resolvedQPS = cfg.QPS
			qpsSource = "config-file"
		}
		if cfg.Burst > 0 {
			resolvedBurst = cfg.Burst
			burstSource = "config-file"
		}
	}

	if qps := os.Getenv("KUBE_CLIENT_QPS"); qps != "" {
		if v, err := strconv.ParseFloat(qps, 32); err == nil && v > 0 {
			resolvedQPS = float32(v)
			qpsSource = "env"
		}
	}
	if burst := os.Getenv("KUBE_CLIENT_BURST"); burst != "" {
		if v, err := strconv.Atoi(burst); err == nil && v > 0 {
			resolvedBurst = v
			burstSource = "env"
		}
	}

	if isFlagSet("client-qps") && ClientQPS > 0 {
		resolvedQPS = ClientQPS
		qpsSource = "flag"
	}
	if isFlagSet("client-burst") && ClientBurst > 0 {
		resolvedBurst = ClientBurst
		burstSource = "flag"
	}

	ClientQPS = resolvedQPS
	ClientBurst = resolvedBurst
	log.Infof("resolved Kubernetes client throttling config: qps=%v(source=%s), burst=%d(source=%s), configFile=%s", ClientQPS, qpsSource, ClientBurst, burstSource, ClientConfigFile)
}

func loadClientConfigFile(path string) (ClientConfig, bool) {
	var cfg ClientConfig
	if path == "" {
		path = DefaultClientConfigFile
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return cfg, false
	}
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		return ClientConfig{}, false
	}
	return cfg, true
}

func isFlagSet(name string) bool {
	found := false
	flag.CommandLine.Visit(func(f *flag.Flag) {
		if f.Name == name {
			found = true
		}
	})
	return found
}

type float32Value float32

func (f *float32Value) String() string {
	return strconv.FormatFloat(float64(*f), 'f', -1, 32)
}

func (f *float32Value) Set(value string) error {
	v, err := strconv.ParseFloat(value, 32)
	if err != nil {
		return err
	}
	*f = float32Value(v)
	return nil
}
