/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

// Package all implements the AllChecker, which runs Hardware, K8s, and Config-Env
// checks in sequence and aggregates the results into a single report.
package all

import (
	"fmt"

	"gitcode.com/openFuyao/InferNex/component/infernex-checker/pkg/checker/configenv"
	"gitcode.com/openFuyao/InferNex/component/infernex-checker/pkg/checker/hardware"
	k8schecker "gitcode.com/openFuyao/InferNex/component/infernex-checker/pkg/checker/k8s"
	"gitcode.com/openFuyao/InferNex/component/infernex-checker/pkg/executor"
	"gitcode.com/openFuyao/InferNex/component/infernex-checker/pkg/log"
	"gitcode.com/openFuyao/InferNex/component/infernex-checker/pkg/types"
)

// CheckAll runs full checks in order: Hardware Layer -> K8s Layer -> Config-Env Layer
func CheckAll(
	nodes []types.NodeInfo, values map[string]interface{},
	k8sClient *executor.K8sClient, dnsCheckImage string, enableConnectivityCheck bool) *types.Report {
	report := &types.Report{}

	// Hardware Layer
	log.Info(fmt.Sprintf("Starting Hardware Layer check, %d node(s) total", len(nodes)))
	hwResult := hardware.CheckHardware(nodes, enableConnectivityCheck)
	report.Hardware = hwResult

	passedNodes, passedNodeNames := filterPassedNodes(nodes, hwResult)

	if len(passedNodes) == 0 {
		log.Info("No nodes passed the Hardware Layer check, terminating K8s Layer and Config-Env Layer checks")
		report.HardwareTerminated = true
		return report
	}

	log.Info(fmt.Sprintf("Nodes passed Hardware Layer: %v, starting K8s Layer check", passedNodeNames))

	// K8s Layer
	report.K8s = k8schecker.CheckK8s(k8sClient, passedNodeNames, dnsCheckImage)

	// Config-Env Layer (runs as long as at least one node passed the Hardware Layer, regardless of K8s Layer result)
	log.Info("Starting Config-Env Layer check")
	report.ConfigEnv = configenv.CheckConfigEnv(passedNodes, values)

	return report
}

func filterPassedNodes(nodes []types.NodeInfo, hwResult *types.HardwareResult) ([]types.NodeInfo, []string) {
	var passedNodes []types.NodeInfo
	var passedNodeNames []string
	for _, nodeResult := range hwResult.Nodes {
		if nodeResult.Status != types.StatusPassed {
			continue
		}
		for _, n := range nodes {
			if n.Name == nodeResult.Name {
				passedNodes = append(passedNodes, n)
				passedNodeNames = append(passedNodeNames, n.Name)
				break
			}
		}
	}
	return passedNodes, passedNodeNames
}
