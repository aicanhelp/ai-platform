// Copyright (c) 2024 Huawei Technologies Co., Ltd.
// openFuyao is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan PSL v2.
// You may obtain a copy of Mulan PSL v2 at:
//          http://license.coscl.org.cn/MulanPSL2
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
// MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
// See the Mulan PSL v2 for more details.

// Package runtime provides functionality to generate labels related to the container runtime
// used by a Kubernetes node. It helps in identifying the container runtime on nodes and ensures
// compatibility with supported runtimes.
package runtime

import (
	"fmt"
	"net/url"

	corev1 "k8s.io/api/core/v1"

	"openfuyao.com/npu-feature-discovery/internal/lm/common"
)

const (
	// RuntimeLabelKey is the key used to label the container runtime on a Kubernetes node.
	RuntimeLabelKey = "openfuyao.com/container.runtime"
)

var (
	supportedRuntimes = map[string]struct{}{
		"containerd": {},
	}
)

type runtimeInfo struct {
	name string
}

// NewRuntimeLabeler generates a label indicating the container runtime used by the provided
// Kubernetes node. It checks the node's container runtime version and returns the appropriate label.
// If the runtime is not supported, an error is returned.
func NewRuntimeLabeler(node *corev1.Node) (common.Labeler, error) {
	result := common.Labels{}

	rt, err := getRuntime(node)
	if err != nil {
		return result, fmt.Errorf("get CRI runtime: %w", err)
	}
	result[RuntimeLabelKey] = rt.name
	return result, nil
}

func getRuntime(node *corev1.Node) (*runtimeInfo, error) {
	uri, err := url.Parse(node.Status.NodeInfo.ContainerRuntimeVersion)
	if err != nil {
		return nil, fmt.Errorf("parse containerRuntimeVersion: %w", err)
	}

	rt := &runtimeInfo{
		name: uri.Scheme,
	}
	if _, ok := supportedRuntimes[rt.name]; !ok {
		return nil, fmt.Errorf("unsupported CRI runtime: %s", rt.name)
	}
	return rt, nil
}
