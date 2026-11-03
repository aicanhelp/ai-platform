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

// Package pod
package pod

import (
	"k8s.io/api/core/v1"
	"k8s.io/klog/v2"

	"openfuyao.com/colocation-management/pkg/common"
)

// NodeAffinityMutator implements PodMutator to add node affinity for oversold pod
type NodeAffinityMutator struct{}

// NewNodeAffinityMutator inits ColocationLabelMutator
func NewNodeAffinityMutator() PodMutator {
	return &NodeAffinityMutator{}
}

// MutatePod add node affinity for oversold pod
func (m *NodeAffinityMutator) MutatePod(pod *v1.Pod) (bool, error) {

	nodeSelectorTerm, nodeSelector := initNodeAffinityObj()

	nodeAffinityChanged := toDesiredColocationNodeAffinityWhenNil(&pod.Spec, nodeSelector)
	if nodeAffinityChanged {
		return true, nil
	}

	toDesiredColocationNodeAffinity(&pod.Spec, nodeSelectorTerm)

	return true, nil
}

func initNodeAffinityObj() (v1.NodeSelectorTerm, *v1.NodeSelector) {
	nodeSelectorTerm := v1.NodeSelectorTerm{
		MatchExpressions: []v1.NodeSelectorRequirement{
			{
				Key:      common.ColocationNodeLabel,
				Operator: v1.NodeSelectorOpExists,
			},
		},
	}

	nodeSelector := &v1.NodeSelector{
		NodeSelectorTerms: []v1.NodeSelectorTerm{
			nodeSelectorTerm,
		},
	}

	return nodeSelectorTerm, nodeSelector
}

func toDesiredColocationNodeAffinityWhenNil(podSpec *v1.PodSpec, nodeSelector *v1.NodeSelector) bool {
	changed := false

	if podSpec.Affinity == nil {
		changed = true
		podSpec.Affinity = &v1.Affinity{
			NodeAffinity: &v1.NodeAffinity{
				RequiredDuringSchedulingIgnoredDuringExecution: nodeSelector,
			},
		}
		klog.Info("add node affinity")
		return changed
	}

	if podSpec.Affinity.NodeAffinity == nil {
		changed = true
		podSpec.Affinity.NodeAffinity = &v1.NodeAffinity{
			RequiredDuringSchedulingIgnoredDuringExecution: nodeSelector,
		}
		klog.Info("add node affinity")
		return changed
	}

	if podSpec.Affinity.NodeAffinity.RequiredDuringSchedulingIgnoredDuringExecution == nil {
		changed = true
		podSpec.Affinity.NodeAffinity.RequiredDuringSchedulingIgnoredDuringExecution = nodeSelector
		klog.Info("add node affinity")
		return changed
	}

	return changed
}

func toDesiredColocationNodeAffinity(podSpec *v1.PodSpec, nodeSelectorTerm v1.NodeSelectorTerm) {
	var nodeSelectorTerms []v1.NodeSelectorTerm
	for _, nst := range podSpec.Affinity.NodeAffinity.RequiredDuringSchedulingIgnoredDuringExecution.NodeSelectorTerms {
		findNodeSelectorRequirement := false
		if len(nst.MatchFields) == 0 && len(nst.MatchExpressions) == 0 {
			klog.Info("find empty nst.......")
			continue
		}

		for _, nsr := range nst.MatchExpressions {
			if nsr.Key == common.ColocationNodeLabel && nsr.Operator == v1.NodeSelectorOpExists {
				findNodeSelectorRequirement = true
				break
			}
		}

		if !findNodeSelectorRequirement {
			nodeSelectorTerms = append(nodeSelectorTerms, nst)
		}
	}

	nodeSelectorTerms = append(nodeSelectorTerms, nodeSelectorTerm)
	podSpec.Affinity.NodeAffinity.RequiredDuringSchedulingIgnoredDuringExecution.NodeSelectorTerms = nodeSelectorTerms

	return
}
