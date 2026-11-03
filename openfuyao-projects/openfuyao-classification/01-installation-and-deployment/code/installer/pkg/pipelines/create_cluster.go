/******************************************************************
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package pipelines

import (
	"fmt"
	"os"

	"installer/pkg/common"
	"installer/pkg/constants"
	"installer/pkg/executor"
	"installer/pkg/executor/auth"
	"installer/pkg/precheck"
	"installer/pkg/precheck/params"
	"installer/pkg/prepare/asset"
	"installer/pkg/prepare/binary"
	"installer/pkg/prepare/hub"
)

// NewCreateClusterPipeline create k8s cluster pipeline
func NewCreateClusterPipeline(runtime *common.KubeRuntime) error {
	// base check
	tools := []string{constants.Tar, constants.Socat, constants.Conntrack, constants.Iptables, constants.IpSet,
		constants.IpvsAdm, constants.OpenSSL}
	installCheck := precheck.NewCreateClusterFlightCheck(runtime.Cluster.Spec.Nodes,
		common.IsContainsComponentByCfg(runtime.Cluster, constants.FuyaoSystem))
	installCheck.FlightCheck.NodesOperate.Tools = tools
	if err := installCheck.StartCheck(); err != nil {
		return err
	}

	if err := common.UnCompressPkgToOptPath(runtime.Cluster.Spec.Nodes.Masters[0]); err != nil {
		return err
	}

	// prepare
	if err := hub.InstallHarborCfg(runtime); err != nil {
		return err
	}
	pretreatment := asset.NewPretreatment(runtime.Cluster)
	if err := pretreatment.PretreatmentPackage(); err != nil {
		return err
	}

	// component check
	// 生成组件依赖图
	builder := executor.NewGraphBuilder()
	// 此依赖图也用于安装组件
	graph, err := builder.InstallGraphBuilder(runtime.Cluster.Spec.Components, runtime.Cluster.Spec.Nodes.Masters[0])
	if err != nil {
		return err
	}

	// 执行前置检查
	exec := executor.NewExecutor(constants.PreCheck, runtime.Cluster.Spec.Nodes, graph)
	exec.FlightCheck()

	// 准备制品
	prepareArtifact := binary.NewPrepareArtifact(graph, runtime.Cluster.Spec.Nodes, runtime.Arg.ArtifactPath)
	prepareArtifact.AdvancedParams = runtime.Cluster.Spec.AdvancedParams
	prepareArtifact.EnableCopyBinary = true
	if err := prepareArtifact.CreateClusterPrepareArtifact(runtime.Arg.EnableOffline); err != nil {
		return err
	}

	// install
	exec = executor.NewExecutor(constants.Install, runtime.Cluster.Spec.Nodes, graph)
	exec.InitCfgFile = runtime.Arg.InitCfg
	if err := exec.Install(); err != nil {
		return err
	}

	fmt.Println("You have successfully completed the installation")
	if err := auth.PrintDefaultUserInfo(runtime.Cluster); err != nil {
		return err
	}
	common.PostInstallCheck(runtime.Cluster.Spec.Nodes.Masters[0])
	common.RemoveTmpFiles(runtime.Cluster.Spec.Nodes, runtime.Arg.SkipRmTmpFile)
	return nil
}

// CreateCluster create k8s cluster entry
func CreateCluster(args common.Argument) error {
	if args.InitCfg != "" {
		_, err := os.Stat(args.InitCfg)
		if err != nil && os.IsNotExist(err) {
			return fmt.Errorf(fmt.Sprintf("Init config file is nor exixts: %s", args.InitCfg))
		} else if err != nil {
			return fmt.Errorf(fmt.Sprintf("Unknown state of file %s", args.InitCfg))
		}
	}

	if args.EnableOffline {
		if err := params.VerifyArtifactPath(args.ArtifactPath); err != nil {
			return err
		}
	}

	ok, err := params.VerifyZone(constants.Zone, args.FuyaoZone)
	if !ok {
		return err
	}

	runtime, err := common.NewKubeRuntime(args)
	if err != nil {
		return err
	}

	defer runtime.ClearSensitiveData()

	if err = NewCreateClusterPipeline(runtime); err != nil {
		return err
	}
	return nil
}
