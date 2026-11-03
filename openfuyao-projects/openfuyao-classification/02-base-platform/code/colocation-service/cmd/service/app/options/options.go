/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

// Package options 设置所有配置选项
package options

import (
	"context"
	"fmt"
	"net"

	"github.com/spf13/pflag"
	apiserveroptions "k8s.io/apiserver/pkg/server/options"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/clientcmd"
	cliflag "k8s.io/component-base/cli/flag"
	"k8s.io/component-base/logs"
	logsapi "k8s.io/component-base/logs/api/v1"
	"k8s.io/component-base/metrics"
	"k8s.io/klog/v2"
	netutils "k8s.io/utils/net"

	"openfuyao.com/colocation-service/cmd/service/app/config"
	colocationclientset "openfuyao.com/colocation-service/pkg/client/clientset/versioned"
)

// OverallOptions 包含创建和运行服务所需的所有配置。
type OverallOptions struct {
	SecureServing *apiserveroptions.SecureServingOptionsWithLoopback
	Metrics       *metrics.Options
	Logs          *logs.Options

	// ConfigFile 是代理服务器配置文件的路径。
	ConfigFile string
	// WriteConfigTo 是默认配置将被写入的路径。
	WriteConfigTo string
	// errCh 是用于发送错误的通道。
	errCh chan error

	// master 用于覆盖 kubeconfig 中 apiserver 的 URL。
	master         string
	kubeConfigPath string
	// healthzPort 是健康检查服务端口。
	healthzPort int32
	// metricsPort 是指标服务端口。
	metricsPort int32

	// Flags 保存解析后的命令行标志。
	Flags *cliflag.NamedFlagSets
}

// initFlags 初始化Flags
func (o *OverallOptions) initFlags() {
	if o.Flags != nil {
		return
	}

	nfs := cliflag.NamedFlagSets{}

	o.AddFlags(nfs.FlagSet("misc"))
	o.SecureServing.AddFlags(nfs.FlagSet("secure serving"))
	o.Metrics.AddFlags(nfs.FlagSet("metrics"))
	logsapi.AddFlags(o.Logs, nfs.FlagSet("logs"))

	o.Flags = &nfs
}

// AddFlags 增加flags到fs，并且绑定到Options
func (o *OverallOptions) AddFlags(fs *pflag.FlagSet) {
	fs.StringVar(&o.ConfigFile, "config", o.ConfigFile, "The path to the configuration file.")
	fs.StringVar(&o.WriteConfigTo, "write-config-to", o.WriteConfigTo,
		"If set, write the configuration values to this file and exit.")

	fs.StringVar(&o.master, "master", o.master, "The url that this proxy will use to connect to the Kubernetes apiserver.")
	fs.StringVar(&o.kubeConfigPath, "kubeconfig", o.kubeConfigPath,
		"The path to the kubeconfig file to use for requests to the apiserver.")
}

// NewOptions 初始化Options
func NewOptions() *OverallOptions {
	o := &OverallOptions{
		SecureServing: apiserveroptions.NewSecureServingOptions().WithLoopback(),
		Metrics:       metrics.NewOptions(),
		Logs:          logs.NewOptions(),
	}
	// 设置PairName，但不设置证书目录，默认在内存中生成
	o.SecureServing.ServerCert.CertDirectory = ""
	o.SecureServing.ServerCert.PairName = "colocation-service"
	o.SecureServing.BindPort = config.DefaultServerPort

	o.initFlags()
	return o
}

// Complete 完成所有必须的Options.
func (o *OverallOptions) Complete(fs *pflag.FlagSet) error {
	return nil
}

// Validate 验证所有的Options
func (o *OverallOptions) Validate() []error {
	var errs []error

	return errs
}

// Config 返回Options的配置
func (o *OverallOptions) Config(ctx context.Context) (*config.OverallConfig, error) {
	logger := klog.FromContext(ctx)
	if o.SecureServing != nil {
		err := o.SecureServing.MaybeDefaultWithSelfSignedCerts("localhost", nil,
			[]net.IP{netutils.ParseIPSloppy("127.0.0.1")})
		if err != nil {
			return nil, fmt.Errorf("error creating self-signed certificates: %v", err)
		}
	}

	c := &config.OverallConfig{}
	if err := o.ApplyTo(logger, c); err != nil {
		return nil, err
	}

	return c, nil
}

// ApplyTo 设置 OverallConfig 的各项配置
func (o *OverallOptions) ApplyTo(logger klog.Logger, c *config.OverallConfig) error {
	if err := o.SecureServing.ApplyTo(&c.SecureServing, &c.LoopbackClientConfig); err != nil {
		return err
	}

	restClientConfig, err := clientcmd.BuildConfigFromFlags(o.master, o.kubeConfigPath)
	if err != nil {
		logger.Error(err, "Failed to build config", o.kubeConfigPath, o.master)
		return err
	}
	c.KubeConfig = restClientConfig

	clientset, err := kubernetes.NewForConfig(restClientConfig)
	if err != nil {
		logger.Error(err, "Failed to create clientset")
		return err
	}

	colocationclient, err := colocationclientset.NewForConfig(restClientConfig)
	if err != nil {
		logger.Error(err, "Failed to create colocationclient")
		return err
	}

	c.Client = clientset
	c.ColocationClient = colocationclient

	return nil
}
