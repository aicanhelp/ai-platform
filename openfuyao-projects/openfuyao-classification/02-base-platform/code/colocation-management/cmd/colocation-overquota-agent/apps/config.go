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

	"github.com/spf13/viper"
	"k8s.io/klog/v2"

	"openfuyao.com/colocation-management/pkg/common"
)

// Configuration represents the configuration for the colocation-overquota-agent
type Configuration struct {
	KubeletInsecureMetricsPort int
	KubeletSecureMetricsPort   int
	InsecureKubeletMetricsTLS  bool
	ServerAddress              string
	MetricsScrapePeriod        time.Duration
	ReportsUploadPeriod        time.Duration
	KubeletIP                  string
	Node                       string

	UsageReportAddr    string
	ReportsBufCapacity int

	PprofAddr   string
	EnablePprof bool

	EnableIncludeExporter bool
	ExporterPort          int
	ExporterPeriod        time.Duration

	CgroupDriver        string
	CgroupRoot          string
	NumaAffinityEnabled bool

	FeatureGates map[string]bool
}

// NewConfiguration creates a new Configuration instance
func NewConfiguration() *Configuration {
	return &Configuration{}
}

// InitFlags initializes the configuration flags
func (c *Configuration) InitFlags(fs *flag.FlagSet) {
	viper.BindEnv("node", "NODE")
	viper.BindEnv("kubelet-ip", "KUBELET_IP")

	fs.BoolVar(&c.EnablePprof, "enable-pprof",
		true, "Enable pprof for controller manager.")
	fs.StringVar(&c.PprofAddr, "pprof-addr",
		":8090", "The address the pprof binds to.")

	fs.StringVar(&c.Node, "node",
		viper.GetString("node"), "node name")
	fs.StringVar(&c.KubeletIP, "kubelet-ip",
		viper.GetString("kubelet-ip"), "kubelet ip")
	fs.IntVar(&c.KubeletInsecureMetricsPort, "kubelet-insecure-metrics-port",
		common.DefaultInsecureKubeletMetricsPort, "kubelet insecure metrics port")
	fs.IntVar(&c.KubeletSecureMetricsPort, "kubelet-secure-metrics-port",
		common.DefaultSecureKubeletMetricsPort, "kubelet insecure metrics port")
	fs.BoolVar(&c.InsecureKubeletMetricsTLS, "insecure-kubelet-metrics-tls",
		false, "using insecure kubelet metrics port")

	fs.DurationVar(&c.MetricsScrapePeriod, "metrics-scrape-period",
		common.DefaultMetricsScrapePeriod, "period for collecting metrics from kubelet")
	fs.DurationVar(&c.ReportsUploadPeriod, "reports-upload-period",
		common.DefaultReportsUploadPeriod, "period for uploading reports to master")
	fs.IntVar(&c.ReportsBufCapacity, "reports-buf-cap",
		common.DefaultReportsBufCapacity, "buffer capacity of usage reports")
	fs.StringVar(&c.UsageReportAddr, "usage-report-addr",
		"colocation-manager.openfuyao-colocation.svc.cluster.local:8091", "address of uploading report")

	fs.BoolVar(&c.EnableIncludeExporter, "enable-include-exporter",
		true, "enable include exporter")
	fs.IntVar(&c.ExporterPort, "exporter-port",
		common.DefaultExporterPort, "port of include exporter")
	fs.DurationVar(&c.ExporterPeriod, "exporter-period",
		common.DefaultExporterPeriod, "period for export metrics ")

	fs.StringVar(&c.CgroupRoot, "cgroup-root", "/sys/fs/cgroup",
		"path to the root of the cgroup  (default: /sys/fs/cgroup")
	fs.StringVar(&c.CgroupDriver, "cgroup-driver", "systemd",
		"cgroup driver to use (options: 'systemd' or 'cgroupfs')")
	fs.BoolVar(&c.NumaAffinityEnabled, "numa-affinity-enabled", true,
		"Enable NUMA (Non-Uniform Memory Access) affinity for server")
}

// PrintAndExitIfRequested prints the configuration and exits if requested
func (c *Configuration) PrintAndExitIfRequested() {
	if len(c.Node) == 0 {
		klog.Fatalf("node is empty")
	}
	klog.V(common.GeneralDebugLog).Infof("node is %s", c.Node)
	if len(c.KubeletIP) == 0 {
		klog.Warningf("kubelet-ip is empty. use parameter node")
		c.KubeletIP = c.Node
	}
	klog.V(common.GeneralDebugLog).Infof("kubelet-ip is %s", c.KubeletIP)

	if len(c.UsageReportAddr) == 0 {
		klog.Fatalf("usage-report-addr is empty")
	}
	if c.MetricsScrapePeriod-common.CushionTime < common.FreshContainerMinMetricsResolution {
		klog.Fatalf("metrics-scrape-period too small")
	}
	if c.ExporterPeriod-common.CushionTime < common.FreshContainerMinMetricsResolution {
		klog.Fatalf("include-exporter-period too small")
	}
}
