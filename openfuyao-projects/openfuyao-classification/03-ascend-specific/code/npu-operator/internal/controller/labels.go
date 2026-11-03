// Copyright (c) 2024 Huawei Technologies Co., Ltd.
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
	"encoding/json"
	"fmt"
	"net/url"
	"strings"

	corev1 "k8s.io/api/core/v1"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

const (
	nfdLabelPrefix               = "feature.node.kubernetes.io/"
	npuPresentLabelKey           = "openfuyao.com/npu.present"
	npuDeviceClassId             = "1200" // Processing accelerators
	npuDeviceVendorId            = "19e5" // Huawei
	npuDeviceLabelPrefix         = nfdLabelPrefix + "pci-" + npuDeviceClassId + "_" + npuDeviceVendorId + "_"
	npuDeviceLabelValue          = "true"
	controlPlaneRoleLabelKey     = "node-role.kubernetes.io/control-plane"
	nodeDEnableLabelKey          = "nodeDEnable"
	nodeDEnableLabelValue        = "on"
	runtimeLabelKey              = "openfuyao.com/container.runtime"
	runtimeVersionAnnotationKey  = "openfuyao.com/container.runtime.version"
	runtimeEndpointAnnotationKey = "openfuyao.com/container.runtime.endpoint"
	vnpuReadyLabelKey            = "huawei.com/vnpu"
	vnpuReadyLabelValue          = "ready"
)

var (
	supportedRuntimes = map[string]struct{}{
		"containerd": {},
	}
	npuDeviceTypes = map[string]string{
		npuDeviceLabelPrefix + "d100.present": "huawei-Ascend310",
		npuDeviceLabelPrefix + "d500.present": "huawei-Ascend310P",
		npuDeviceLabelPrefix + "d801.present": "huawei-Ascend910",
		npuDeviceLabelPrefix + "d802.present": "huawei-Ascend910",
		npuDeviceLabelPrefix + "d803.present": "huawei-Ascend910",
		npuDeviceLabelPrefix + "d806.present": "huawei-AscendA5",
	}
)

type runtimeInfo struct {
	name     string
	version  string
	endpoint string
}

func (r *NPUClusterPolicyReconciler) labelNPUNodes(ctx context.Context) error {
	logger := log.FromContext(ctx)

	r.runtimes = map[string]struct{}{}
	r.nodeRuntimeEndpoints = map[string]string{}
	r.hasNFDLabels = false
	r.hasNPUNodes = false

	nodes, err := r.listNodes(ctx)
	if err != nil {
		return fmt.Errorf("check node labels: %w", err)
	}

	for _, node := range nodes {
		node := node.DeepCopy()
		logger := logger.WithValues("nodeName", node.Name)
		ctx := log.IntoContext(ctx, logger)

		r.hasNFDLabels = r.hasNFDLabels || hasNFDLabels(node.Labels)

		supportRuntime, updateAnnotations, err := r.checkRuntime(ctx, node)
		if err != nil {
			return fmt.Errorf("check CRI runtime support: %w", err)
		}

		isNPUNode := supportRuntime && hasNPUDeviceLabels(node.Labels)
		if supportRuntime {
			r.hasNPUNodes = r.hasNPUNodes || isNPUNode
			logger.Info("NPU node check", "node", node.Name, "hasNPUNodes", r.hasNPUNodes, "supportRuntime", supportRuntime)
		}

		updateNode := r.updateNodeDEnableLabel(node) || updateAnnotations
		updateNode = r.updateVNPUReadyLabel(node, isNPUNode) || updateNode

		if updateNode {
			if err = r.Update(ctx, node); err != nil {
				return fmt.Errorf("update node labels: %w", err)
			}
		}
	}
	return nil
}

func hasNPUDeviceLabels(labels map[string]string) bool {
	for key, value := range labels {
		_, ok := npuDeviceTypes[key]
		if ok && value == npuDeviceLabelValue {
			return true
		}
	}
	return false
}

func hasNPUPresentLabel(labels map[string]string) bool {
	_, ok := labels[npuPresentLabelKey]
	return ok
}

func removeLabelIfExists(labels map[string]string, labelKey string) bool {
	if _, ok := labels[labelKey]; ok {
		delete(labels, labelKey)
		return true
	}
	return false
}

func addLabel(labels map[string]string, key, value string) bool {
	if labels == nil {
		labels = make(map[string]string)
	}
	if v, ok := labels[key]; ok && v == value {
		return false
	}
	labels[key] = value
	return true
}

func (r *NPUClusterPolicyReconciler) updateNodeDEnableLabel(node *corev1.Node) bool {
	updated := false
	if node.Labels == nil {
		node.Labels = make(map[string]string)
	}

	if r.instance.Spec.NodeD.Managed {
		updated = addLabel(node.Labels, nodeDEnableLabelKey, nodeDEnableLabelValue)
	} else {
		updated = removeLabelIfExists(node.Labels, nodeDEnableLabelKey)
	}
	return updated
}

func (r *NPUClusterPolicyReconciler) updateVNPUReadyLabel(node *corev1.Node, isNPUNode bool) bool {
	if node.Labels == nil {
		node.Labels = map[string]string{}
	}
	if r.instance.Spec.VNPU.Managed && isNPUNode {
		return addLabel(node.Labels, vnpuReadyLabelKey, vnpuReadyLabelValue)
	}
	return removeLabelIfExists(node.Labels, vnpuReadyLabelKey)
}

func addAnnotation(anno map[string]string, key, value string) bool {
	if anno == nil {
		anno = make(map[string]string)
	}
	if v, ok := anno[key]; ok && v == value {
		return false
	}
	anno[key] = value
	return true
}

func (r *NPUClusterPolicyReconciler) removeRuntimeAnnotations(annotations map[string]string) bool {
	removed := false
	labelKeys := []string{
		runtimeEndpointAnnotationKey,
		runtimeVersionAnnotationKey,
	}

	for _, key := range labelKeys {
		if _, ok := annotations[key]; ok {
			removed = true
			delete(annotations, key)
		}
	}
	return removed
}

func (r *NPUClusterPolicyReconciler) getRuntime(
	ctx context.Context,
	node *corev1.Node,
) (runtime *runtimeInfo, err error) {
	logger := log.FromContext(ctx)

	// get runtime and version
	runtime, err = r.parseRuntimeInfo(ctx, node)
	if err != nil || runtime == nil {
		return
	}

	// get runtime endpoint
	raw, err := r.client.CoreV1().
		RESTClient().
		Get().
		Resource("nodes").
		Name(node.Name).
		SubResource("proxy").
		Suffix("configz").
		DoRaw(ctx)
	if err != nil {
		err = fmt.Errorf("get proxy/configz subresource: %w", err)
		return
	}
	config := map[string]any{}
	if err = json.Unmarshal(raw, &config); err != nil {
		err = fmt.Errorf("unmarshal proxy/configz subresource: %w", err)
		return
	}
	defer func() {
		if recover() != nil {
			logger.Error(nil, "Failed to get containerRuntimeEndpoint from kubelet")
			runtime = nil
		}
	}()
	kubeletconfig, ok := config["kubeletconfig"].(map[string]any)
	if !ok {
		err = fmt.Errorf("kubeletconfig contains no kubeletconfig")
		return
	}
	containerRuntimeEndpoint, ok := kubeletconfig["containerRuntimeEndpoint"].(string)
	if !ok {
		err = fmt.Errorf("kubeletconfig contains no containerRuntimeEndpoint")
		return
	}
	runtime.endpoint = containerRuntimeEndpoint
	return
}

func (r *NPUClusterPolicyReconciler) parseRuntimeInfo(
	ctx context.Context,
	node *corev1.Node,
) (*runtimeInfo, error) {
	logger := log.FromContext(ctx)

	uri, err := url.Parse(node.Status.NodeInfo.ContainerRuntimeVersion)
	if err != nil {
		return nil, fmt.Errorf("parse containerRuntimeVersion: %w", err)
	}

	runtime := &runtimeInfo{
		name:    uri.Scheme,
		version: uri.Host,
	}

	if _, ok := supportedRuntimes[runtime.name]; !ok {
		logger.Error(nil, "Unsupported CRI runtime", "runtime", runtime)
		return nil, fmt.Errorf("unsupported runtime: %s", runtime.name)
	}
	return runtime, nil
}

func (r *NPUClusterPolicyReconciler) checkRuntime(ctx context.Context, node *corev1.Node) (bool, bool, error) {
	runtime, err := r.getRuntime(ctx, node)
	if err != nil {
		return false, false, fmt.Errorf("get CRI runtime: %w", err)
	}
	if runtime == nil {
		return false, r.removeRuntimeAnnotations(node.Annotations), nil
	}
	r.runtimes[runtime.name] = struct{}{}
	nodeChanged := addAnnotation(node.Annotations, runtimeVersionAnnotationKey, runtime.version)
	nodeChanged = addAnnotation(node.Annotations, runtimeEndpointAnnotationKey, runtime.endpoint) || nodeChanged
	r.nodeRuntimeEndpoints[node.Name] = runtime.endpoint
	return true, nodeChanged, nil
}

func hasNFDLabels(labels map[string]string) bool {
	for key := range labels {
		if strings.HasPrefix(key, nfdLabelPrefix) {
			return true
		}
	}
	return false
}

func (r *NPUClusterPolicyReconciler) listNodes(ctx context.Context) ([]corev1.Node, error) {
	list := &corev1.NodeList{}
	if err := r.List(ctx, list); err != nil {
		return nil, fmt.Errorf("list nodes: %w", err)
	}
	return list.Items, nil
}
