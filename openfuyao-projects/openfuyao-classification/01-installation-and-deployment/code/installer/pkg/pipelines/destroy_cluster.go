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
	"installer/pkg/fuyao/v1beta1/cluster"
	"installer/pkg/precheck"
	"installer/pkg/precheck/params"
	"installer/pkg/prepare/asset"
	"installer/pkg/prepare/binary"
	"installer/pkg/utils"
)

// removeHarborCert 离线卸载时harbor不在当前节点 删除执行节点harbor证书
func removeHarborCert(runtime *common.KubeRuntime) error {
	if !runtime.Arg.EnableOffline || runtime.Arg.SkipDestroyHub {
		return nil
	}

	localIp, err := utils.GetLocalIP()
	if err != nil {
		return err
	}

	nodes := make([]*cluster.Node, 0)
	nodes = append(nodes, runtime.Cluster.Spec.Nodes.Workers...)
	nodes = append(nodes, runtime.Cluster.Spec.Nodes.Masters[1:]...)

	for _, node := range nodes {
		if node.IP == localIp {
			return nil
		}
	}
	if err := utils.Remove(constants.HubCaBkPath); err != nil {
		return err
	}
	return utils.Remove(constants.HubCaPath)
}

// NewDestroyClusterPipeline is new destroy cluster pipeline
func NewDestroyClusterPipeline(runtime *common.KubeRuntime) error {
	flightCheck := precheck.NewDestroyClusterFlightCheck(runtime.Cluster.Spec.Nodes)
	if err := flightCheck.StartCheck(); err != nil {
		return err
	}
	pretreatment := asset.NewPretreatment(runtime.Cluster)
	if err := pretreatment.PretreatmentPackage(); err != nil {
		return err
	}

	// executor
	builder := executor.NewGraphBuilder()
	graph, err := builder.UninstallGraphBuilder(runtime.Cluster.Spec.Components, runtime.Cluster.Spec.Nodes.Masters[0])
	if err != nil {
		return err
	}

	componentMap, err := common.GetInstallHubComponentMap(runtime.Arg)
	if err != nil {
		return err
	}
	prepareArtifact := binary.NewPrepareArtifact(graph, runtime.Cluster.Spec.Nodes, runtime.Arg.ArtifactPath)
	if err := prepareArtifact.CreateClusterPrepareArtifact(runtime.Arg.EnableOffline); err != nil {
		return err
	}
	exec := executor.NewExecutor(constants.Uninstall, runtime.Cluster.Spec.Nodes, graph)
	exec.SkipDestroyHub = runtime.Arg.SkipDestroyHub
	exec.HubComponentMap = componentMap
	if err := exec.Uninstall(); err != nil {
		return err
	}
	if err := removeHarborCert(runtime); err != nil {
		return err
	}
	common.FinishedDestroyHandler(runtime.Cluster.Spec.Nodes)
	return nil
}

// DestroyCluster is destroy cluster
func DestroyCluster(args common.Argument) error {
	if args.EnableOffline {
		err := params.VerifyArtifactPath(args.ArtifactPath)
		if err != nil {
			return err
		}
	}
	runtime, err := common.NewKubeRuntime(args)
	if err != nil {
		return err
	}

	defer runtime.ClearSensitiveData()

	return NewDestroyClusterPipeline(runtime)
}
