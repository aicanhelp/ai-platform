/*
 *
 *  * Copyright (c) 2024 China Unicom Digital Technology Co., Ltd.
 *  * openFuyao is licensed under Mulan PSL v2.
 *  * You can use this software according to the terms and conditions of the Mulan PSL v2.
 *  * You may obtain a copy of Mulan PSL v2 at:
 *  *          http://license.coscl.org.cn/MulanPSL2
 *  * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 *  * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 *  * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 *  * See the Mulan PSL v2 for more details.
 *
 */

// Package apps configuration
package apps

import (
	"flag"
	"time"

	"k8s.io/client-go/rest"
	"k8s.io/klog/v2"

	"openfuyao.com/colocation-management/pkg/common"
)

const (
	// DefaultNamespace default namespace
	DefaultNamespace = "openfuyao-colocation"
	// defaultRestConfigQPS default rest config qps
	defaultRestConfigQPS = 30
	// defaultRestConfigBurst default rest config burst
	defaultRestConfigBurst = 50
	// defaultCacheSyncPeriod default cache sync period
	defaultCacheSyncPeriod = 120 * time.Second

	// defaultUsageBufCapacity default usage buf capacity
	defaultUsageBufCapacity = 2000
	// defaultRefreshNodesInterval default refresh nodes interval
	defaultRefreshNodesInterval = 5 * time.Minute
	// defaultRefreshPodsInterval default refresh pods interval
	defaultRefreshPodsInterval = 2 * time.Minute
	// defaultAggregateStateGCInterval default aggregate state gc interval
	defaultAggregateStateGCInterval = 1 * time.Hour
	// defaultRecommendToggleEnabled default recommend toggle enabled
	defaultRecommendToggleEnabled = true
	// defaultEstimateCPUPercentile default estimate cpu percentile
	defaultEstimateCPUPercentile = 0.9
	// defaultEstimateMemoryPercentile default estimate memory percentile
	defaultEstimateMemoryPercentile = 0.9
	// defaultBERecommendPeriod default be recommend period
	defaultBERecommendMinCPUMillicores = 100
	// defaultBERecommendMinMemoryMb default be recommend min memory mb
	defaultBERecommendMinMemoryMb = 500
	// defaultBERecommendPeriod default be recommend period
	defaultBERecommendPeriod = 30 * time.Second
	// defaultSaveCheckpointPeriod default save checkpoint period
	defaultSaveCheckpointPeriod = 5 * time.Minute
	// defaultUseCheckpoint default use checkpoint
	defaultUseCheckpoint = true
	// defaultParallelSaveNodeCount default parallel save node count
	defaultParallelSaveNodeCount = 10
	// defaultCheckpointGCInterval default checkpoint gc interval
	defaultCheckpointGCInterval = 15 * time.Minute
	// defaultWebhookPort default webhook port
	defaultWebhookPort = 443
)

// Configuration for colocation-manager
type Configuration struct {
	Namespace string

	RestConfigQPS   int
	RestConfigBurst int

	MetricsAddr          string
	CacheSyncPeriod      time.Duration
	EnableLeaderElection bool
	HealthProbeAddr      string
	PprofAddr            string
	EnablePprof          bool

	UsageReportAddr  string
	UsageBufCapacity int

	RefreshNodesInterval     time.Duration
	RefreshPodsInterval      time.Duration
	AggregateStateGCInterval time.Duration

	RecommendToggleEnabled      bool
	EstimateCPUPercentile       float64
	EstimateMemoryPercentile    float64
	BERecommendMinCPUMillicores int64
	BERecommendMinMemoryMb      int64
	BERecommendPeriod           time.Duration

	UseCheckpoint         bool
	SaveCheckpointPeriod  time.Duration
	ParallelSaveNodeCount int
	CheckpointGCInterval  time.Duration

	// WebhookPort webhook server port
	WebhookPort int
	// WebhookTlsCertFile path to the x509 certificate for webhook https
	WebhookTlsCertFile string
	// WebhookTlsKeyFile path to the x509 private key matching `CertFile`
	WebhookTlsKeyFile string
	// WebhookSvcNamespace The namespace of the webhook service
	WebhookSvcNamespace string
	// WebhookSvcName The name of the webhook service
	WebhookSvcName string

	FeatureGates map[string]bool
}

// NewConfiguration creates a new Configuration with default values
func NewConfiguration() *Configuration {
	return &Configuration{}
}

// InitFlags initializes the configuration flags
func (c *Configuration) InitFlags(fs *flag.FlagSet) {
	fs.StringVar(&c.Namespace, "namespace", DefaultNamespace, "determines the namespace.")

	fs.IntVar(&c.RestConfigQPS, "rest-config-qps", defaultRestConfigQPS, "QPS of rest config.")
	fs.IntVar(&c.RestConfigBurst, "rest-config-burst", defaultRestConfigBurst, "Burst of rest config.")

	fs.StringVar(&c.MetricsAddr, "metrics-bind-address", "0", "The address the metrics endpoint binds to. "+
		"Use :8443 for HTTPS or :8080 for HTTP, or leave as 0 to disable the metrics service.")
	fs.StringVar(&c.HealthProbeAddr, "health-probe-addr", ":8081", "The address the healthz/readyz endpoint binds to.")
	fs.BoolVar(&c.EnableLeaderElection, "leader-elect", false,
		"Enable leader election for controller manager. "+
			"Enabling this will ensure there is only one active controller manager.")
	fs.DurationVar(&c.CacheSyncPeriod, "cache-sync-period", defaultCacheSyncPeriod,
		"Determines the minimum frequency at which watched resources are reconciled.")

	fs.BoolVar(&c.EnablePprof, "enable-pprof", true, "Enable pprof for controller manager.")
	fs.StringVar(&c.PprofAddr, "pprof-addr", ":8090", "The address the pprof binds to.")
	fs.StringVar(&c.UsageReportAddr, "usage-report-addr",
		":8091", "address to report usage of containers and node on agent")
	fs.IntVar(&c.UsageBufCapacity, "total-usage-buf-capacity", defaultUsageBufCapacity, "total buffer size of usage data")
	fs.DurationVar(&c.RefreshNodesInterval, "refresh-nodes-interval",
		defaultRefreshNodesInterval, "interval for list all nodes from apiserver")
	fs.DurationVar(&c.RefreshPodsInterval, "refresh-pods-interval",
		defaultRefreshPodsInterval, "interval for list all pods from apiserver")
	fs.DurationVar(&c.AggregateStateGCInterval, "aggregate-state-gc-interval", defaultAggregateStateGCInterval, "")

	fs.BoolVar(&c.RecommendToggleEnabled, "recommend-toggle-enabled", defaultRecommendToggleEnabled,
		"true represents update recommendation; false represents clean recommendation.")
	fs.Float64Var(&c.EstimateCPUPercentile, "estimate-cpu-percentile", defaultEstimateCPUPercentile,
		"CPU usage percentile that will be used as a base for CPU target recommendation.")
	fs.Float64Var(&c.EstimateMemoryPercentile, "estimate-memory-percentile", defaultEstimateMemoryPercentile,
		"Memory usage percentile that will be used as a base for memory target recommendation.")
	fs.Int64Var(&c.BERecommendMinCPUMillicores, "be-recommend-min-cpu-millicores", defaultBERecommendMinCPUMillicores,
		`Minimum CPU recommendation for a node`)
	fs.Int64Var(&c.BERecommendMinMemoryMb, "be-recommend-min-memory-mb", defaultBERecommendMinMemoryMb,
		`Minimum memory recommendation for a node`)
	fs.DurationVar(&c.BERecommendPeriod, "be-resource-recommend-period", defaultBERecommendPeriod,
		"recommendation period of best-effort resource for a not")

	fs.DurationVar(&c.SaveCheckpointPeriod, "save-checkpoint-period", defaultSaveCheckpointPeriod,
		"Period for save container checkpoint")
	fs.BoolVar(&c.UseCheckpoint, "use-checkpoint", defaultUseCheckpoint,
		"whether system save container checkpoint at period")
	fs.IntVar(&c.ParallelSaveNodeCount, "parallel-save-node-count",
		defaultParallelSaveNodeCount, "parallel save count of node about checkpoints at period")
	fs.DurationVar(&c.CheckpointGCInterval, "checkpoint-gc-interval", defaultCheckpointGCInterval,
		"delete container state checkpoint CR from apiserver if corresponding pod no longer exist")
	fs.IntVar(&c.WebhookPort, "webhookPort", defaultWebhookPort, "Webhook server port")
	fs.StringVar(&c.WebhookTlsCertFile, "webhookTlsCertFile", "/etc/webhook/certs/server.pem",
		"File containing the x509 Certificate for webhook HTTPS.")
	fs.StringVar(&c.WebhookTlsKeyFile, "webhookTlsKeyFile", "/etc/webhook/certs/key.pem",
		"File containing the x509 private key to webhook --tlsCertFile.")
	fs.StringVar(&c.WebhookSvcNamespace, "webhookSvcNamespace", "openfuyao-colocation",
		"The namespace of the webhook service")
	fs.StringVar(&c.WebhookSvcName, "webhookSvcName", "colocation-manager", "The name of the webhook service")

}

// SetRestConfig set rest config
func (c *Configuration) SetRestConfig(cfg *rest.Config) {
	if c.RestConfigQPS > 0 {
		cfg.QPS = float32(c.RestConfigQPS)
	}
	if c.RestConfigBurst > 0 {
		cfg.Burst = c.RestConfigBurst
	}
}

// PrintAndExitIfRequested print and exit if requested
func (c *Configuration) PrintAndExitIfRequested() {
	if len(c.Namespace) == 0 {
		klog.Fatal("namespace is empty")
	}
	if c.RestConfigQPS < 0 {
		klog.Fatal("rest-config-qps is invalid")
	}
	if c.RestConfigBurst < 0 {
		klog.Fatal("rest-config-burst is invalid")
	}
	if c.EstimateCPUPercentile < 0 || c.EstimateCPUPercentile > 1 {
		klog.Fatal("estimate-cpu-percentile is invalid")
	}
	if c.EstimateMemoryPercentile < 0 || c.EstimateMemoryPercentile > 1 {
		klog.Fatal("estimate-memory-percentile is invalid")
	}

	klog.V(common.GeneralDebugLog).Infof("namespace is %s", c.Namespace)
}
