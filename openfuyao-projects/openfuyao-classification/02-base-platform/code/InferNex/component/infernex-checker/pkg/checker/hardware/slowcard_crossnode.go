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

package hardware

import (
	"fmt"

	"gitcode.com/openFuyao/InferNex/component/infernex-checker/pkg/log"
	"gitcode.com/openFuyao/InferNex/component/infernex-checker/pkg/types"
)

// checkH09 collects and displays cross-node communication metrics from H-08 results
func checkH09(nodeResults []types.NodeResult) types.CheckResult {
	log.Info(fmt.Sprintf("H-09: Collecting cross-node metrics across %d nodes", len(nodeResults)))

	// Collect all metrics from H-08 results
	allMetrics, nodesWithMetrics := collectH08Metrics(nodeResults)

	if len(nodesWithMetrics) == 0 {
		return types.CheckResult{
			ID:      "H-09",
			Status:  types.StatusSkipped,
			Message: "No nodes have slow card metrics available, skipping cross-node summary",
		}
	}

	// Group metrics by node for display
	nodeMetrics := make(map[string][]rankMetrics)
	for _, m := range allMetrics {
		nodeMetrics[m.NodeName] = append(nodeMetrics[m.NodeName], m)
	}

	// Build per-node Transit Time summary for terminal display
	var summary string
	for node, metrics := range nodeMetrics {
		var totalTransit float64
		for _, m := range metrics {
			totalTransit += m.TransitTimeMS
		}
		avgTransit := totalTransit / float64(len(metrics))
		summary += fmt.Sprintf(" %s(avg Transit: %.2fms, %d ranks);",
			node, avgTransit, len(metrics))
	}

	log.Info(fmt.Sprintf("H-09: Collected metrics from %d nodes", len(nodesWithMetrics)))

	return types.CheckResult{
		ID:      "H-09",
		Status:  types.StatusInfo,
		Message: fmt.Sprintf("Cross-node communication metrics (%d nodes):%s", len(nodesWithMetrics), summary),
		Detail: map[string]interface{}{
			"nodes_with_metrics": len(nodesWithMetrics),
			"node_metrics":       nodeMetrics,
		},
	}
}

// collectH08Metrics collects rank metrics from H-08 results across all nodes
func collectH08Metrics(nodeResults []types.NodeResult) ([]rankMetrics, map[string]bool) {
	var allMetrics []rankMetrics
	nodesWithMetrics := make(map[string]bool)
	for _, nodeResult := range nodeResults {
		if metrics := extractH08Metrics(nodeResult); len(metrics) > 0 {
			allMetrics = append(allMetrics, metrics...)
			nodesWithMetrics[nodeResult.Name] = true
		}
	}
	return allMetrics, nodesWithMetrics
}

// extractH08Metrics extracts rank metrics from a single node result
func extractH08Metrics(nodeResult types.NodeResult) []rankMetrics {
	for _, check := range nodeResult.Checks {
		if check.ID == "H-08" {
			if metrics, ok := check.Detail["metrics"].([]rankMetrics); ok {
				return metrics
			}
			break
		}
	}
	return nil
}
