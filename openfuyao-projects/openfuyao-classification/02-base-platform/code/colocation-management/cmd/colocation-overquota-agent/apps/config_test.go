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
	"os"
	"os/exec"
	"testing"
	"time"

	"github.com/spf13/viper"

	"openfuyao.com/colocation-management/pkg/common"
)

func TestNewConfiguration(t *testing.T) {
	config := NewConfiguration()

	if config == nil {
		t.Fatal("Expected non-nil Configuration, got nil")
	}

	if config.KubeletInsecureMetricsPort != 0 {
		t.Errorf("Expected KubeletInsecureMetricsPort 0, got %d", config.KubeletInsecureMetricsPort)
	}
	if config.KubeletSecureMetricsPort != 0 {
		t.Errorf("Expected KubeletSecureMetricsPort 0, got %d", config.KubeletSecureMetricsPort)
	}
	if config.InsecureKubeletMetricsTLS != false {
		t.Errorf("Expected InsecureKubeletMetricsTLS false, got %t", config.InsecureKubeletMetricsTLS)
	}
	if config.ServerAddress != "" {
		t.Errorf("Expected empty ServerAddress, got %s", config.ServerAddress)
	}
	if config.MetricsScrapePeriod != 0 {
		t.Errorf("Expected MetricsScrapePeriod 0, got %v", config.MetricsScrapePeriod)
	}
	if config.ReportsUploadPeriod != 0 {
		t.Errorf("Expected ReportsUploadPeriod 0, got %v", config.ReportsUploadPeriod)
	}
	if config.KubeletIP != "" {
		t.Errorf("Expected empty KubeletIP, got %s", config.KubeletIP)
	}
	if config.Node != "" {
		t.Errorf("Expected empty Node, got %s", config.Node)
	}
	if config.UsageReportAddr != "" {
		t.Errorf("Expected empty UsageReportAddr, got %s", config.UsageReportAddr)
	}
	if config.ReportsBufCapacity != 0 {
		t.Errorf("Expected ReportsBufCapacity 0, got %d", config.ReportsBufCapacity)
	}
	if config.PprofAddr != "" {
		t.Errorf("Expected empty PprofAddr, got %s", config.PprofAddr)
	}
	if config.EnablePprof != false {
		t.Errorf("Expected EnablePprof false, got %t", config.EnablePprof)
	}
	if config.EnableIncludeExporter != false {
		t.Errorf("Expected EnableIncludeExporter false, got %t", config.EnableIncludeExporter)
	}
	if config.ExporterPort != 0 {
		t.Errorf("Expected ExporterPort 0, got %d", config.ExporterPort)
	}
	if config.ExporterPeriod != 0 {
		t.Errorf("Expected ExporterPeriod 0, got %v", config.ExporterPeriod)
	}
	if config.CgroupDriver != "" {
		t.Errorf("Expected empty CgroupDriver, got %s", config.CgroupDriver)
	}
	if config.CgroupRoot != "" {
		t.Errorf("Expected empty CgroupRoot, got %s", config.CgroupRoot)
	}
	if config.NumaAffinityEnabled != false {
		t.Errorf("Expected NumaAffinityEnabled false, got %t", config.NumaAffinityEnabled)
	}
	if config.FeatureGates != nil {
		t.Errorf("Expected nil FeatureGates, got %v", config.FeatureGates)
	}
}

func TestInitFlags(t *testing.T) {
	tests := []struct {
		name           string
		args           []string
		envVars        map[string]string
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
		{
			name:           "environment variables",
			args:           []string{},
			envVars:        map[string]string{"NODE": "env-node", "KUBELET_IP": "env-kubelet-ip"},
			expectedConfig: createEnvConfig(),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			setupTestEnvironment(t, tt.envVars)
			config := parseFlags(t, tt.args)
			verifyConfig(t, config, tt.expectedConfig)
		})
	}
}

func createDefaultConfig() Configuration {
	return Configuration{
		EnablePprof:                true,
		PprofAddr:                  ":8090",
		KubeletInsecureMetricsPort: common.DefaultInsecureKubeletMetricsPort,
		KubeletSecureMetricsPort:   common.DefaultSecureKubeletMetricsPort,
		InsecureKubeletMetricsTLS:  false,
		MetricsScrapePeriod:        common.DefaultMetricsScrapePeriod,
		ReportsUploadPeriod:        common.DefaultReportsUploadPeriod,
		ReportsBufCapacity:         common.DefaultReportsBufCapacity,
		UsageReportAddr:            "colocation-manager.openfuyao-colocation.svc.cluster.local:8091",
		EnableIncludeExporter:      true,
		ExporterPort:               common.DefaultExporterPort,
		ExporterPeriod:             common.DefaultExporterPeriod,
		CgroupRoot:                 "/sys/fs/cgroup",
		CgroupDriver:               "systemd",
		NumaAffinityEnabled:        true,
	}
}

func createCustomArgs() []string {
	return []string{
		"-enable-pprof=false",
		"-pprof-addr=:9090",
		"-node=test-node",
		"-kubelet-ip=192.168.1.100",
		"-kubelet-insecure-metrics-port=10255",
		"-kubelet-secure-metrics-port=10250",
		"-insecure-kubelet-metrics-tls=true",
		"-metrics-scrape-period=30s",
		"-reports-upload-period=1m",
		"-reports-buf-cap=2000",
		"-usage-report-addr=localhost:9091",
		"-enable-include-exporter=false",
		"-exporter-port=9100",
		"-exporter-period=15s",
		"-cgroup-root=/custom/cgroup",
		"-cgroup-driver=cgroupfs",
		"-numa-affinity-enabled=false",
	}
}

func createCustomConfig() Configuration {
	return Configuration{
		EnablePprof:                false,
		PprofAddr:                  ":9090",
		Node:                       "test-node",
		KubeletIP:                  "192.168.1.100",
		KubeletInsecureMetricsPort: 10255,
		KubeletSecureMetricsPort:   10250,
		InsecureKubeletMetricsTLS:  true,
		MetricsScrapePeriod:        30 * time.Second,
		ReportsUploadPeriod:        1 * time.Minute,
		ReportsBufCapacity:         2000,
		UsageReportAddr:            "localhost:9091",
		EnableIncludeExporter:      false,
		ExporterPort:               9100,
		ExporterPeriod:             15 * time.Second,
		CgroupRoot:                 "/custom/cgroup",
		CgroupDriver:               "cgroupfs",
		NumaAffinityEnabled:        false,
	}
}

func createEnvConfig() Configuration {
	config := createDefaultConfig()
	config.Node = "env-node"
	config.KubeletIP = "env-kubelet-ip"
	return config
}

func setupTestEnvironment(t *testing.T, envVars map[string]string) {
	for key, value := range envVars {
		t.Setenv(key, value)
	}
}

func parseFlags(t *testing.T, args []string) *Configuration {
	fs := flag.NewFlagSet("test", flag.ContinueOnError)
	config := NewConfiguration()
	config.InitFlags(fs)

	if err := fs.Parse(args); err != nil {
		t.Fatalf("Failed to parse flags: %v", err)
	}
	return config
}

func verifyConfig(t *testing.T, actual *Configuration, expected Configuration) {
	if actual.EnablePprof != expected.EnablePprof {
		t.Errorf("EnablePprof: expected %t, got %t", expected.EnablePprof, actual.EnablePprof)
	}
	if actual.PprofAddr != expected.PprofAddr {
		t.Errorf("PprofAddr: expected %s, got %s", expected.PprofAddr, actual.PprofAddr)
	}
}

// TestInvalidConfig 测试无效配置的情况
func TestInvalidConfig(t *testing.T) {
	tests := []struct {
		name        string
		config      *Configuration
		expectPanic bool
	}{
		{
			name: "empty node should cause exit",
			config: &Configuration{
				Node: "",
			},
			expectPanic: true,
		},
		{
			name: "empty usage report addr should cause exit",
			config: &Configuration{
				Node:            "test-node",
				UsageReportAddr: "",
			},
			expectPanic: true,
		},
		{
			name: "metrics scrape period too small should cause exit",
			config: &Configuration{
				Node:                "test-node",
				UsageReportAddr:     "localhost:8091",
				MetricsScrapePeriod: common.FreshContainerMinMetricsResolution - time.Second,
			},
			expectPanic: true,
		},
		{
			name: "exporter period too small should cause exit",
			config: &Configuration{
				Node:                "test-node",
				UsageReportAddr:     "localhost:8091",
				MetricsScrapePeriod: common.DefaultMetricsScrapePeriod,
				ExporterPeriod:      common.FreshContainerMinMetricsResolution - time.Second,
			},
			expectPanic: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if os.Getenv("BE_TEST_PRINT_AND_EXIT") == "1" {
				tt.config.PrintAndExitIfRequested()
				return
			}

			cmd := exec.Command(os.Args[0], "-test.run="+t.Name())
			cmd.Env = append(os.Environ(), "BE_TEST_PRINT_AND_EXIT=1")
			err := cmd.Run()

			if tt.expectPanic {
				if err == nil {
					t.Error("Expected process to exit with error, but it didn't")
				}
				if exitErr, ok := err.(*exec.ExitError); ok {
					if !exitErr.Exited() {
						t.Errorf("Process exited unexpectedly: %v", err)
					}
				}
			} else {
				if err != nil {
					t.Errorf("Unexpected error: %v", err)
				}
			}
		})
	}
}

// ValidConfig 测试验证配置
func ValidConfig(t *testing.T) {
	config := &Configuration{
		Node:                "test-node",
		UsageReportAddr:     "localhost:8091",
		MetricsScrapePeriod: common.DefaultMetricsScrapePeriod,
		ExporterPeriod:      common.DefaultExporterPeriod,
	}

	config.PrintAndExitIfRequested()
}

func TestViperIntegration(t *testing.T) {
	t.Run("environment variable binding", func(t *testing.T) {
		viper.Reset()

		t.Setenv("NODE", "env-test-node")
		t.Setenv("KUBELET_IP", "env-test-ip")

		fs := flag.NewFlagSet("test", flag.ContinueOnError)
		config := NewConfiguration()
		config.InitFlags(fs)

		if viper.GetString("node") != "env-test-node" {
			t.Errorf("Expected node env var 'env-test-node', got '%s'", viper.GetString("node"))
		}
		if viper.GetString("kubelet-ip") != "env-test-ip" {
			t.Errorf("Expected kubelet-ip env var 'env-test-ip', got '%s'", viper.GetString("kubelet-ip"))
		}
	})
}
