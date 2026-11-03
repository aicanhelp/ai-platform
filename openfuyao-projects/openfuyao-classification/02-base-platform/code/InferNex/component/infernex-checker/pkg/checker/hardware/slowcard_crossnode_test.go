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
	"strings"
	"testing"

	"gitcode.com/openFuyao/InferNex/component/infernex-checker/pkg/types"
)

// ---- checkH09 ----

func TestCheckH09_InsufficientNodesWithMetrics(t *testing.T) {
	// Only 1 node has metrics - still shows info
	nodeResults := []types.NodeResult{
		{
			Name:   "node1",
			Status: types.StatusPassed,
			Checks: []types.CheckResult{
				{
					ID:     "H-08",
					Status: types.StatusPassed,
					Detail: map[string]interface{}{
						"metrics": []rankMetrics{
							{RankID: 0, PhysicalDeviceID: 0, NodeName: "node1", TransitTimeMS: 10.0},
						},
					},
				},
			},
		},
	}

	result := checkH09(nodeResults)
	if result.Status != types.StatusInfo {
		t.Errorf("expected StatusInfo, got %v: %s", result.Status, result.Message)
	}
}

func TestCheckH09_NoMetricsAvailable(t *testing.T) {
	// 2 nodes but neither has metrics (H-08 skipped)
	nodeResults := []types.NodeResult{
		{
			Name:   "node1",
			Status: types.StatusPassed,
			Checks: []types.CheckResult{
				{ID: "H-08", Status: types.StatusSkipped},
			},
		},
		{
			Name:   "node2",
			Status: types.StatusPassed,
			Checks: []types.CheckResult{
				{ID: "H-08", Status: types.StatusSkipped},
			},
		},
	}

	result := checkH09(nodeResults)
	if result.Status != types.StatusSkipped {
		t.Errorf("expected StatusSkipped when no metrics, got %v: %s", result.Status, result.Message)
	}
}

func TestCheckH09_MetricsCollected(t *testing.T) {
	// 2 nodes with metrics → should return Info with all metrics
	nodeResults := []types.NodeResult{
		{
			Name:   "node1",
			Status: types.StatusPassed,
			Checks: []types.CheckResult{
				{
					ID:     "H-08",
					Status: types.StatusPassed,
					Detail: map[string]interface{}{
						"metrics": []rankMetrics{
							{RankID: 0, PhysicalDeviceID: 0, NodeName: "node1", TransitTimeMS: 10.0},
							{RankID: 1, PhysicalDeviceID: 1, NodeName: "node1", TransitTimeMS: 10.5},
						},
					},
				},
			},
		},
		{
			Name:   "node2",
			Status: types.StatusPassed,
			Checks: []types.CheckResult{
				{
					ID:     "H-08",
					Status: types.StatusPassed,
					Detail: map[string]interface{}{
						"metrics": []rankMetrics{
							{RankID: 0, PhysicalDeviceID: 0, NodeName: "node2", TransitTimeMS: 100.0},
							{RankID: 1, PhysicalDeviceID: 4, NodeName: "node2", TransitTimeMS: 105.0},
						},
					},
				},
			},
		},
	}

	result := checkH09(nodeResults)
	if result.Status != types.StatusInfo {
		t.Errorf("expected StatusInfo, got %v: %s", result.Status, result.Message)
	}
	if result.Detail == nil {
		t.Fatal("expected non-nil Detail")
	}
	nodeMetrics, ok := result.Detail["node_metrics"].(map[string][]rankMetrics)
	if !ok {
		t.Fatal("expected node_metrics in Detail")
	}
	if len(nodeMetrics) != 2 {
		t.Errorf("expected 2 nodes in node_metrics, got %d", len(nodeMetrics))
	}
	if !strings.Contains(result.Message, "node1") || !strings.Contains(result.Message, "node2") {
		t.Errorf("expected node names in message, got: %s", result.Message)
	}
}

func TestCheckH09_MixedNodeStatus(t *testing.T) {
	// node1 has metrics, node2 H-08 returned warning (no metrics), node3 has metrics
	nodeResults := []types.NodeResult{
		{
			Name:   "node1",
			Status: types.StatusPassed,
			Checks: []types.CheckResult{
				{
					ID:     "H-08",
					Status: types.StatusPassed,
					Detail: map[string]interface{}{
						"metrics": []rankMetrics{
							{RankID: 0, PhysicalDeviceID: 0, NodeName: "node1", TransitTimeMS: 10.0},
						},
					},
				},
			},
		},
		{
			Name:   "node2",
			Status: types.StatusPassed,
			Checks: []types.CheckResult{
				{ID: "H-08", Status: types.StatusWarning, Detail: map[string]interface{}{}},
			},
		},
		{
			Name:   "node3",
			Status: types.StatusPassed,
			Checks: []types.CheckResult{
				{
					ID:     "H-08",
					Status: types.StatusPassed,
					Detail: map[string]interface{}{
						"metrics": []rankMetrics{
							{RankID: 0, PhysicalDeviceID: 0, NodeName: "node3", TransitTimeMS: 10.5},
						},
					},
				},
			},
		},
	}

	result := checkH09(nodeResults)
	// Should proceed with node1 and node3 (2 nodes with metrics)
	if result.Status != types.StatusInfo {
		t.Errorf("expected StatusInfo with 2 nodes having metrics, got %v: %s", result.Status, result.Message)
	}
}
