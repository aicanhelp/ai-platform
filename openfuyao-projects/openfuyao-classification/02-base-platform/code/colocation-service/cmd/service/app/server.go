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

package app

import (
	"context"
	"fmt"
	"os"

	"github.com/spf13/cobra"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apiserver/pkg/server"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	cliflag "k8s.io/component-base/cli/flag"
	"k8s.io/component-base/cli/globalflag"
	"k8s.io/component-base/logs"
	logsapi "k8s.io/component-base/logs/api/v1"
	"k8s.io/component-base/metrics/features"
	"k8s.io/component-base/term"
	"k8s.io/klog/v2"

	"openfuyao.com/colocation-service/cmd/service/app/config"
	"openfuyao.com/colocation-service/cmd/service/app/options"
	"openfuyao.com/colocation-service/pkg/service"
	"openfuyao.com/colocation-service/pkg/version"
)

func init() {
	utilruntime.Must(logsapi.AddFeatureGates(utilfeature.DefaultMutableFeatureGate))
	utilruntime.Must(features.AddFeatureGates(utilfeature.DefaultMutableFeatureGate))
}

// NewCommand 创建一个具有默认值的*cobra.Command对应
func NewCommand() *cobra.Command {
	opts := options.NewOptions()
	cmd := &cobra.Command{
		Use:   "colocation-service",
		Short: "colocation-service",
		Long:  `colocation-service`,
		RunE: func(cmd *cobra.Command, args []string) error {
			return runCommand(cmd, opts)
		},
		Args: func(cmd *cobra.Command, args []string) error {
			for _, arg := range args {
				if len(arg) > 0 {
					return fmt.Errorf("%q does not take any arguments, got %q", cmd.CommandPath(), args)
				}
			}
			return nil
		},
	}

	// 设置命令行参数
	nfs := opts.Flags
	fs := cmd.Flags()
	for _, f := range nfs.FlagSets {
		fs.AddFlagSet(f)
	}

	// 设置全局标志
	globalflag.AddGlobalFlags(nfs.FlagSet("global"), cmd.Name(), logs.SkipLoggingConfigurationFlags())

	// 设置帮助信息
	cols, _, _ := term.TerminalSize(cmd.OutOrStdout())
	cliflag.SetUsageAndHelpFunc(cmd, *nfs, cols)

	// 标记config文件的扩展名
	if err := cmd.MarkFlagFilename("config", "yaml", "yml", "json"); err != nil {
		klog.Background().Error(err, "Failed to mark flag filename")
	}

	return cmd
}

// runCommand 运行colocation-service服务器
func runCommand(cmd *cobra.Command, opts *options.OverallOptions) error {
	// 尽早激活日志记录，然后用最终的日志记录配置显示标志.
	if err := logsapi.ValidateAndApply(opts.Logs, utilfeature.DefaultFeatureGate); err != nil {
		fmt.Fprintf(os.Stderr, "%v\n", err)
		klog.Fatal(err)
	}
	cliflag.PrintFlags(cmd.Flags())

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go func() {
		stopCh := server.SetupSignalHandler()
		if stopCh != nil {
			<-stopCh
		}
		cancel()
	}()

	cc, server, err := Setup(ctx, opts)
	if err != nil {
		return err
	}

	return Run(ctx, cc, server)
}

// Setup 基于命令行参数与Options创建CompletedConfig和Server
func Setup(ctx context.Context, opts *options.OverallOptions) (*config.CompletedConfig, *service.Server, error) {
	if errs := opts.Validate(); len(errs) > 0 {
		return nil, nil, utilerrors.NewAggregate(errs)
	}

	c, err := opts.Config(ctx)
	if err != nil {
		return nil, nil, err
	}

	cc := c.Complete()

	server, err := service.New(ctx, cc.Client, cc.ColocationClient)
	if err != nil {
		return nil, nil, err
	}

	return &cc, server, nil
}

// Run 基于指定的配置运行Server
func Run(ctx context.Context, cc *config.CompletedConfig, server *service.Server) error {
	logger := klog.FromContext(ctx)
	logger.Info("Starting server", "version", version.Get())
	logger.Info("Golang settings", "GOGC", os.Getenv("GOGC"), "GOMAXPROCS",
		os.Getenv("GOMAXPROCS"), "GOTRACEBACK", os.Getenv("GOTRACEBACK"))

	// 启动服务器
	if cc.SecureServing != nil {
		logger.Info("Starting secure server", "address", cc.SecureServing.Listener.Addr())
		handler := server.Handle
		if _, _, err := cc.SecureServing.Serve(handler, 0, ctx.Done()); err != nil {
			return fmt.Errorf("failed to start secure server: %v", err)
		}
	}

	server.Run(ctx)
	return nil
}
