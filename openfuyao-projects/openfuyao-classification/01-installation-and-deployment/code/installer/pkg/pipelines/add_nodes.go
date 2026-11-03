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
	"installer/pkg/common"
	"installer/pkg/constants"
	"installer/pkg/executor"
	"installer/pkg/precheck"
	"installer/pkg/precheck/params"
	"installer/pkg/prepare/asset"
	"installer/pkg/prepare/binary"
	"installer/pkg/prepare/hub"
)

// NewAddNodesPipeline is add node pipeline
func NewAddNodesPipeline(runtime *common.KubeRuntime) error {
	// FlightCheck
	tools := []string{constants.Tar, constants.Socat, constants.Conntrack, constants.Iptables}
	flightCheck := precheck.NewAddNodesFlightCheck(runtime.Cluster.Spec.Nodes)
	flightCheck.FlightCheck.NodesOperate.Tools = tools
	if err := flightCheck.StartCheck(); err != nil {
		return err
	}

	if err := common.UnCompressPkgToOptPath(runtime.Cluster.Spec.Nodes.Masters[0]); err != nil {
		return err
	}

	// prepare
	pretreatment := asset.NewPretreatment(runtime.Cluster)
	if err := pretreatment.PretreatmentPackage(); err != nil {
		return err
	}
	if err := hub.InstallHarborCfg(runtime); err != nil {
		return err
	}

	// 生成组件依赖图
	graphBuilder := executor.NewGraphBuilder()
	// 此依赖图也用于安装组件
	graph, err := graphBuilder.InstallGraphBuilder(runtime.Cluster.Spec.Components,
		runtime.Cluster.Spec.Nodes.Masters[0])
	if err != nil {
		return err
	}
	// 执行前置检查
	exec := executor.NewExecutor(constants.PreCheck, runtime.Cluster.Spec.Nodes, graph)
	exec.FlightCheck()

	// 准备制品
	prepareArtifact := binary.NewPrepareArtifact(graph, runtime.Cluster.Spec.Nodes, runtime.Arg.ArtifactPath)
	prepareArtifact.EnableCopyBinary = true
	prepareArtifact.AdvancedParams = runtime.Cluster.Spec.AdvancedParams
	if err := prepareArtifact.CreateClusterPrepareArtifact(runtime.Arg.EnableOffline); err != nil {
		return err
	}

	// add node
	exec = executor.NewExecutor(constants.Install, runtime.Cluster.Spec.Nodes, graph)
	if err := exec.AddNodes(); err != nil {
		return err
	}

	common.PostAddNodeCheck(runtime.Cluster.Spec.Nodes.Masters[0])
	common.RemoveTmpFiles(runtime.Cluster.Spec.Nodes, runtime.Arg.SkipRmTmpFile)
	return nil
}

// AddNodes 添加节点
func AddNodes(args common.Argument) error {
	ok, err := params.VerifyZone(constants.Zone, args.FuyaoZone)
	if !ok {
		return err
	}
	if args.EnableOffline {
		if err := params.VerifyArtifactPath(args.ArtifactPath); err != nil {
			return err
		}
	}
	runtime, err := common.NewKubeRuntime(args)
	if err != nil {
		return err
	}

	defer runtime.ClearSensitiveData()

	if err = NewAddNodesPipeline(runtime); err != nil {
		return err
	}
	return nil
}
