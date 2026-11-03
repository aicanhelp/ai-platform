/*
* Copyright (c) 2024 China Unicom Digital Technology Co., Ltd.
* openFuyao is licensed under Mulan PSL v2.
* You can use this software according to the terms and conditions of the Mulan PSL v2.
* You may obtain a copy of Mulan PSL v2 at:
*          http://license.coscl.org.cn/MulanPSL2
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
* EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
* MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
* See the Mulan PSL v2 for more details.
* Author: YuXiang Guo
* Date: 2025-09-05
 */

package apps

import (
	"flag"
	"testing"
	"time"

	"k8s.io/client-go/rest"
)

func TestNewConfiguration(t *testing.T) {
	config := NewConfiguration()

	if config == nil {
		t.Fatal("Expected non-nil Configuration, got nil")
	}

	if config.Namespace != "" {
		t.Errorf("Expected empty namespace, got %s", config.Namespace)
	}
	if config.RestConfigQPS != 0 {
		t.Errorf("Expected RestConfigQPS 0, got %d", config.RestConfigQPS)
	}
	if config.RestConfigBurst != 0 {
		t.Errorf("Expected RestConfigBurst 0, got %d", config.RestConfigBurst)
	}
}

// TestInitFlags tests the InitFlags function.
func TestInitFlags(t *testing.T) {
	tests := createTestCases()

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			config := parseFlags(t, tt.args)
			verifyBasicConfig(t, config, tt.expectedConfig)
		})
	}
}

func createTestCases() []struct {
	name           string
	args           []string
	expectedConfig Configuration
} {
	return []struct {
		name           string
		args           []string
		expectedConfig Configuration
	}{
		{
			name:           "default values",
			args:           []string{},
			expectedConfig: createDefaultConfig(),
		},
		{
			name:           "custom values",
			args:           createCustomArgs(),
			expectedConfig: createCustomConfig(),
		},
	}
}

func createDefaultConfig() Configuration {
	return Configuration{
		Namespace:                   DefaultNamespace,
		RestConfigQPS:               defaultRestConfigQPS,
		RestConfigBurst:             defaultRestConfigBurst,
		MetricsAddr:                 "0",
		HealthProbeAddr:             ":8081",
		EnableLeaderElection:        false,
		CacheSyncPeriod:             defaultCacheSyncPeriod,
		EnablePprof:                 true,
		PprofAddr:                   ":8090",
		UsageReportAddr:             ":8091",
		UsageBufCapacity:            defaultUsageBufCapacity,
		RefreshNodesInterval:        defaultRefreshNodesInterval,
		RefreshPodsInterval:         defaultRefreshPodsInterval,
		AggregateStateGCInterval:    defaultAggregateStateGCInterval,
		RecommendToggleEnabled:      defaultRecommendToggleEnabled,
		EstimateCPUPercentile:       defaultEstimateCPUPercentile,
		EstimateMemoryPercentile:    defaultEstimateMemoryPercentile,
		BERecommendMinCPUMillicores: defaultBERecommendMinCPUMillicores,
		BERecommendMinMemoryMb:      defaultBERecommendMinMemoryMb,
		BERecommendPeriod:           defaultBERecommendPeriod,
		SaveCheckpointPeriod:        defaultSaveCheckpointPeriod,
		UseCheckpoint:               defaultUseCheckpoint,
		ParallelSaveNodeCount:       defaultParallelSaveNodeCount,
		CheckpointGCInterval:        defaultCheckpointGCInterval,
		WebhookPort:                 defaultWebhookPort,
		WebhookTlsCertFile:          "/etc/webhook/certs/server.pem",
		WebhookTlsKeyFile:           "/etc/webhook/certs/key.pem",
		WebhookSvcNamespace:         "openfuyao-colocation",
		WebhookSvcName:              "colocation-manager",
	}
}

func createCustomArgs() []string {
	return []string{
		"-namespace=test-namespace", "-rest-config-qps=200", "-rest-config-burst=200",
		"-metrics-bind-address=:8080", "-health-probe-addr=:8082", "-leader-elect=true",
		"-cache-sync-period=5m", "-enable-pprof=false", "-pprof-addr=:8091",
		"-usage-report-addr=:8092", "-total-usage-buf-capacity=2000", "-refresh-nodes-interval=2m",
		"-refresh-pods-interval=1m", "-aggregate-state-gc-interval=10m", "-recommend-toggle-enabled=false",
		"-estimate-cpu-percentile=0.99", "-estimate-memory-percentile=0.99",
		"-be-recommend-min-cpu-millicores=200", "-be-recommend-min-memory-mb=200",
		"-be-resource-recommend-period=2m", "-save-checkpoint-period=10m", "-use-checkpoint=false",
		"-parallel-save-node-count=20", "-checkpoint-gc-interval=1h", "-webhookPort=9444",
		"-webhookTlsCertFile=/custom/cert.pem", "-webhookTlsKeyFile=/custom/key.pem",
		"-webhookSvcNamespace=custom-namespace", "-webhookSvcName=custom-webhook", //
	}
}

func createCustomConfig() Configuration {
	return Configuration{
		Namespace:                   "test-namespace",
		RestConfigQPS:               200,
		RestConfigBurst:             200,
		MetricsAddr:                 ":8080",
		HealthProbeAddr:             ":8082",
		EnableLeaderElection:        true,
		CacheSyncPeriod:             5 * time.Minute,
		EnablePprof:                 false,
		PprofAddr:                   ":8091",
		UsageReportAddr:             ":8092",
		UsageBufCapacity:            2000,
		RefreshNodesInterval:        2 * time.Minute,
		RefreshPodsInterval:         1 * time.Minute,
		AggregateStateGCInterval:    10 * time.Minute,
		RecommendToggleEnabled:      false,
		EstimateCPUPercentile:       0.99,
		EstimateMemoryPercentile:    0.99,
		BERecommendMinCPUMillicores: 200,
		BERecommendMinMemoryMb:      200,
		BERecommendPeriod:           2 * time.Minute,
		SaveCheckpointPeriod:        10 * time.Minute,
		UseCheckpoint:               false,
		ParallelSaveNodeCount:       20,
		CheckpointGCInterval:        1 * time.Hour,
		WebhookPort:                 9444,
		WebhookTlsCertFile:          "/custom/cert.pem",
		WebhookTlsKeyFile:           "/custom/key.pem",
		WebhookSvcNamespace:         "custom-namespace",
		WebhookSvcName:              "custom-webhook",
	}
}

func parseFlags(t *testing.T, args []string) *Configuration {
	config := NewConfiguration()
	fs := flag.NewFlagSet("test", flag.ContinueOnError)
	config.InitFlags(fs)
	if err := fs.Parse(args); err != nil {
		t.Fatalf("Failed to parse flags: %v", err)
	}
	return config
}

func verifyBasicConfig(t *testing.T, config *Configuration, expected Configuration) {
	if config.Namespace != expected.Namespace {
		t.Errorf("Namespace: expected %s, got %s", expected.Namespace, config.Namespace)
	}
	if config.RestConfigQPS != expected.RestConfigQPS {
		t.Errorf("RestConfigQPS: expected %d, got %d", expected.RestConfigQPS, config.RestConfigQPS)
	}
	if config.RestConfigBurst != expected.RestConfigBurst {
		t.Errorf("RestConfigBurst: expected %d, got %d", expected.RestConfigBurst, config.RestConfigBurst)
	}
	if config.MetricsAddr != expected.MetricsAddr {
		t.Errorf("MetricsAddr: expected %s, got %s", expected.MetricsAddr, config.MetricsAddr)
	}
}

func TestSetRestConfig(t *testing.T) {
	tests := createRestConfigTestCases()

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tt.config.SetRestConfig(tt.initialConfig)
			verifyRestConfig(t, tt.initialConfig, tt.expectedConfig)
		})
	}
}

func createRestConfigTestCases() []struct {
	name           string
	config         *Configuration
	initialConfig  *rest.Config
	expectedConfig *rest.Config
} {
	return []struct {
		name           string
		config         *Configuration
		initialConfig  *rest.Config
		expectedConfig *rest.Config
	}{
		{
			name:   "set QPS and Burst",
			config: createConfigWithQPSBurst(100, 200),
			initialConfig: &rest.Config{
				QPS:   50,
				Burst: 100,
			},
			expectedConfig: &rest.Config{
				QPS:   100,
				Burst: 200,
			},
		},
		{
			name:   "zero values should not change config",
			config: createConfigWithQPSBurst(0, 0),
			initialConfig: &rest.Config{
				QPS:   50,
				Burst: 100,
			},
			expectedConfig: &rest.Config{
				QPS:   50,
				Burst: 100,
			},
		},
		{
			name:   "negative values should not change config",
			config: createConfigWithQPSBurst(-1, -1),
			initialConfig: &rest.Config{
				QPS:   50,
				Burst: 100,
			},
			expectedConfig: &rest.Config{
				QPS:   50,
				Burst: 100,
			},
		},
	}
}

func createConfigWithQPSBurst(qps int, burst int) *Configuration {
	return &Configuration{
		RestConfigQPS:   qps,
		RestConfigBurst: burst,
	}
}

func verifyRestConfig(t *testing.T, actual *rest.Config, expected *rest.Config) {
	if actual.QPS != float32(expected.QPS) {
		t.Errorf("QPS: expected %f, got %f", expected.QPS, actual.QPS)
	}
	if actual.Burst != expected.Burst {
		t.Errorf("Burst: expected %d, got %d", expected.Burst, actual.Burst)
	}
}

func TestPrintAndExitIfRequested(t *testing.T) {
	tests := []struct {
		name        string
		config      *Configuration
		shouldPanic bool
	}{
		{
			name: "valid namespace should not exit",
			config: &Configuration{
				Namespace: "test-namespace",
			},
			shouldPanic: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			defer func() {

				if r := recover(); r != nil {
					if !tt.shouldPanic {
						t.Errorf("Unexpected panic: %v", r)
					}
				} else if tt.shouldPanic {
					t.Error("Expected panic but didn't get one")
				}
			}()

			tt.config.PrintAndExitIfRequested()
		})
	}

}

func TestEdgeCases(t *testing.T) {
	t.Run("multiple flag sets", func(t *testing.T) {
		fs1 := flag.NewFlagSet("set1", flag.ContinueOnError)
		fs2 := flag.NewFlagSet("set2", flag.ContinueOnError)

		config := NewConfiguration()
		config.InitFlags(fs1)
		config.InitFlags(fs2)

		err1 := fs1.Parse([]string{"-namespace=test1"})
		err2 := fs2.Parse([]string{"-namespace=test2"})

		if err1 != nil || err2 != nil {
			t.Errorf("Failed to parse flags: %v, %v", err1, err2)
		}
	})

}
