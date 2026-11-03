// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// openFuyao is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan PSL v2.
// You may obtain a copy of Mulan PSL v2 at:
//          http://license.coscl.org.cn/MulanPSL2
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
// MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
// See the Mulan PSL v2 for more details.

package controller

import (
	"context"
	"fmt"
	"sort"
	"strings"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/labels"
	"sigs.k8s.io/controller-runtime/pkg/client"

	v1 "openfuyao.com/npu-operator/api/v1"
)

const allNodesSelectorName = "<all nodes>"

type componentConflict struct {
	Left  string
	Right string
	Nodes []string
}

type conflictWorkload struct {
	name     string
	managed  bool
	selector map[string]string
}

type conflictRule struct {
	left  string
	right string
}

func (r *NPUClusterPolicyReconciler) detectComponentConflicts(
	ctx context.Context,
) ([]componentConflict, error) {
	if r == nil || r.instance == nil {
		return nil, nil
	}

	spec := &r.instance.Spec
	workloads := map[string]conflictWorkload{
		"mindcluster-device-plugin": {
			name:     "mindcluster-device-plugin",
			managed:  spec.DevicePlugin.Managed,
			selector: nodeSelectorFromPlacement(devicePluginPlacementGetter(spec)),
		},
		"vnpu-device-plugin": {
			name:     "vnpu-device-plugin",
			managed:  spec.VNPU.Managed && spec.VNPU.DevicePlugin.Managed,
			selector: nodeSelectorFromPlacement(vnpuDevicePluginPlacementGetter(spec)),
		},
		"dra-kubelet-plugin": {
			name:     "dra-kubelet-plugin",
			managed:  spec.DRA.Managed && spec.DRA.KubeletPlugin.Managed,
			selector: nodeSelectorFromPlacement(draKubeletPluginPlacementGetter(spec)),
		},
		"vnpu-client-update": {
			name:     "vnpu-client-update",
			managed:  spec.VNPU.Managed && spec.VNPU.ClientUpdate.Managed,
			selector: nodeSelectorFromPlacement(vnpuClientUpdatePlacementGetter(spec)),
		},
		"dra-vcannrt-installer": {
			name: "dra-vcannrt-installer",
			managed: spec.DRA.Managed &&
				spec.DRA.SoftVNPU.Managed &&
				spec.DRA.SoftVNPU.VCANNRTInstaller.Managed,
			selector: nodeSelectorFromPlacement(draVCANNRTInstallerPlacementGetter(spec)),
		},
	}

	rules := []conflictRule{
		{left: "mindcluster-device-plugin", right: "vnpu-device-plugin"},
		{left: "mindcluster-device-plugin", right: "dra-kubelet-plugin"},
		{left: "vnpu-device-plugin", right: "dra-kubelet-plugin"},
		{left: "vnpu-client-update", right: "dra-vcannrt-installer"},
	}

	conflicts := make([]componentConflict, 0)
	for _, rule := range rules {
		left := workloads[rule.left]
		right := workloads[rule.right]
		if !left.managed || !right.managed {
			continue
		}

		nodes, err := findOverlappingNodes(ctx, r.Client, left.selector, right.selector)
		if err != nil {
			return nil, err
		}
		if len(nodes) == 0 {
			continue
		}
		conflicts = append(conflicts, componentConflict{
			Left:  left.name,
			Right: right.name,
			Nodes: nodes,
		})
	}
	return conflicts, nil
}

func nodeSelectorFromPlacement(placement *v1.PlacementSpec) map[string]string {
	if placement == nil || len(placement.NodeSelector) == 0 {
		return nil
	}

	selector := make(map[string]string, len(placement.NodeSelector))
	for key, value := range placement.NodeSelector {
		selector[key] = value
	}
	return selector
}

func selectorsDefinitelyDisjoint(a, b map[string]string) bool {
	for key, leftValue := range a {
		if rightValue, ok := b[key]; ok && leftValue != rightValue {
			return true
		}
	}
	return false
}

func findOverlappingNodes(
	ctx context.Context,
	c client.Client,
	a map[string]string,
	b map[string]string,
) ([]string, error) {
	if selectorsDefinitelyDisjoint(a, b) {
		return nil, nil
	}
	if len(a) == 0 && len(b) == 0 {
		return []string{allNodesSelectorName}, nil
	}

	listSelector, filterSelector := selectorsForDynamicCheck(a, b)
	nodes := &corev1.NodeList{}
	if err := c.List(ctx, nodes, client.MatchingLabels(listSelector)); err != nil {
		return nil, fmt.Errorf("list nodes by selector %v: %w", listSelector, err)
	}

	filter := labels.SelectorFromSet(filterSelector)
	overlapped := make([]string, 0)
	for _, node := range nodes.Items {
		if filter.Matches(labels.Set(node.Labels)) {
			overlapped = append(overlapped, node.Name)
		}
	}
	if len(overlapped) == 0 {
		return nil, nil
	}
	sort.Strings(overlapped)
	return overlapped, nil
}

func selectorsForDynamicCheck(a, b map[string]string) (map[string]string, map[string]string) {
	switch {
	case len(a) == 0:
		return b, a
	case len(b) == 0:
		return a, b
	case len(a) >= len(b):
		return a, b
	default:
		return b, a
	}
}

func formatComponentConflicts(conflicts []componentConflict) string {
	messages := make([]string, 0, len(conflicts))
	for _, conflict := range conflicts {
		messages = append(messages, fmt.Sprintf("%s and %s select overlapping nodes: %s",
			conflict.Left, conflict.Right, strings.Join(conflict.Nodes, ",")))
	}
	return strings.Join(messages, "; ")
}
