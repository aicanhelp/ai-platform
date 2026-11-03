// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// openFuyao is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan PSL v2.
// You may obtain a copy of Mulan PSL v2 at:
//          http://license.coscl.org.cn/MulanPSL2
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
// MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
// See the Mulan PSL v2 for more details.

// Package common provides shared structures and functionality for handling labels
// in Kubernetes nodes.
package common

// Labels is a type alias for a map that holds key-value pairs representing labels.
type Labels map[string]string

// Labeler is an interface that defines a method for retrieving labels. Any type that implements
// the Labels method can be considered a Labeler.
type Labeler interface {
	Labels() Labels
}

// Labels returns the Labels of the `Labels` type.
func (labels Labels) Labels() Labels {
	if labels == nil {
		return nil
	}
	return labels
}

// Labelers is a slice of Labeler interfaces.
type Labelers []Labeler

// Merge combines multiple Labeler objects into one. It creates a new Labeler that merges
// the labels from all provided Labelers.
func Merge(labelers ...Labeler) Labeler {
	l := Labelers{}

	for _, labeler := range labelers {
		if labeler != nil {
			l = append(l, labeler)
		}
	}

	return l
}

// Labels returns the combined Labels of all the Labelers in the Labelers slice.
// It aggregates labels from all provided Labelers into a single map.
func (labelers Labelers) Labels() Labels {
	allLabels := make(Labels)
	for _, labeler := range labelers {
		labels := labeler.Labels()
		for k, v := range labels {
			allLabels[k] = v
		}
	}

	return allLabels
}
