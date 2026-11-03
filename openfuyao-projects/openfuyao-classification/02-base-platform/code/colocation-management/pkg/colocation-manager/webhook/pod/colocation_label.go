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

package pod

import (
	"k8s.io/api/core/v1"

	"openfuyao.com/colocation-management/pkg/common"
)

// ColocationLabelMutator implements PodMutator to add colocation label
type ColocationLabelMutator struct{}

// NewColocationLabelMutator inits ColocationLabelMutator
func NewColocationLabelMutator() PodMutator {
	return &ColocationLabelMutator{}
}

// MutatePod add colocation label for this pod
func (m *ColocationLabelMutator) MutatePod(pod *v1.Pod) (bool, error) {
	if pod.Annotations == nil {
		pod.Annotations = make(map[string]string)
	}
	pod.Annotations[common.OversubNodeLabel] = "true"

	return true, nil
}
