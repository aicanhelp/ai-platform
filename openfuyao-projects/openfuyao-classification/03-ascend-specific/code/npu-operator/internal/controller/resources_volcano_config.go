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
	"errors"
	"fmt"
	"os"
	"regexp"
	"strings"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/version"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"

	v1 "openfuyao.com/npu-operator/api/v1"
)

const volcanoUnifiedPluginVersion = "26.1.1"

// Match only plugin name entries, preserving YAML indentation, quotes and comments.
var volcanoPluginNamePattern = regexp.MustCompile(`(?m)^[ \t]*-[ \t]+name:[ \t]*["']?volcano-npu_[^ \t\r\n"'#]+`)

var (
	volcanoPluginVersionPattern         = regexp.MustCompile(`^v?[0-9]+\.[0-9]+\.([0-9]+|RC[0-9]+)$`)
	errUnrecognizedVolcanoPluginVersion = errors.New("cannot identify MindCluster plugin version in image tag")
)

func volcanoPluginVersion(tag string) string {
	parts := strings.FieldsFunc(tag, func(char rune) bool { return char == '-' || char == '_' })
	var candidates []string
	for _, part := range parts {
		if volcanoPluginVersionPattern.MatchString(part) {
			candidates = append(candidates, part)
		}
	}
	if len(candidates) > 1 && parts[0] == candidates[0] {
		candidates = candidates[1:]
	}
	if len(candidates) != 1 {
		return ""
	}
	return candidates[0]
}

func updateVolcanoSchedulerConfigMap(ctx context.Context, r *NPUClusterPolicyReconciler, obj *corev1.ConfigMap) error {
	updateConfig(r, obj, ctx)
	data := obj.Data["volcano-scheduler.conf"]
	if volcanoFlavor(&r.instance.Spec) != v1.VolcanoFlavorMindCluster || !volcanoPluginNamePattern.MatchString(data) {
		return nil
	}
	pluginVersion, err := ObtainVolcanoVersion(ctx, r)
	if errors.Is(err, errUnrecognizedVolcanoPluginVersion) {
		log.FromContext(ctx).Info("preserving configured volcano plugin name", "reason", err.Error())
		return nil
	}
	if err != nil {
		return fmt.Errorf("obtain volcano version: %w", err)
	}
	if volcanoPluginHasUnifiedName(pluginVersion) {
		obj.Data["volcano-scheduler.conf"] = updateVolcanoPluginName(data, "volcano-npu_"+pluginVersion)
		return nil
	}
	// With no image tag override, retain the configured version and naming convention.
	if pluginVersion == "" && !hasLegacyVolcanoPlugin(data) {
		return nil
	}
	pluginArch, err := resolveVolcanoPluginArch(ctx, r)
	if err != nil {
		return err
	}
	if pluginVersion == "" {
		obj.Data["volcano-scheduler.conf"] = updateVolcanoPluginArch(data, pluginArch)
		return nil
	}
	name := fmt.Sprintf("volcano-npu_%s_%s", pluginVersion, pluginArch)
	obj.Data["volcano-scheduler.conf"] = updateVolcanoPluginName(data, name)
	log.FromContext(ctx).Info("updated volcano scheduler plugin", "pluginName", name)
	return nil
}

func volcanoPluginHasUnifiedName(pluginVersion string) bool {
	parsed, err := version.ParseSemantic(pluginVersion)
	return err == nil && parsed.AtLeast(version.MustParseSemantic(volcanoUnifiedPluginVersion))
}

func hasLegacyVolcanoPlugin(data string) bool {
	for _, entry := range volcanoPluginNamePattern.FindAllString(data, -1) {
		if strings.HasSuffix(entry, "_linux-x86_64") || strings.HasSuffix(entry, "_linux-aarch64") {
			return true
		}
	}
	return false
}

func resolveVolcanoPluginArch(ctx context.Context, r *NPUClusterPolicyReconciler) (string, error) {
	nodeName := os.Getenv("OPERATOR_NODE_NAME")
	if nodeName == "" {
		return "", fmt.Errorf("OPERATOR_NODE_NAME environment variable is not set")
	}
	node := &corev1.Node{}
	if err := r.Get(ctx, client.ObjectKey{Name: nodeName}, node); err != nil {
		return "", fmt.Errorf("get node %s: %w", nodeName, err)
	}
	return volcanoPluginArch(node.Status.NodeInfo.Architecture)
}

func volcanoPluginArch(arch string) (string, error) {
	switch arch {
	case "amd64", "x86_64":
		return "linux-x86_64", nil
	case "arm64", "aarch64":
		return "linux-aarch64", nil
	default:
		return "", fmt.Errorf("unsupported architecture: %s", arch)
	}
}

func updateVolcanoPluginName(data, name string) string {
	return volcanoPluginNamePattern.ReplaceAllStringFunc(data, func(entry string) string {
		return entry[:strings.Index(entry, "volcano-npu_")] + name
	})
}

func updateVolcanoPluginArch(data, pluginArch string) string {
	return volcanoPluginNamePattern.ReplaceAllStringFunc(data, func(entry string) string {
		entry = strings.ReplaceAll(entry, "linux-x86_64", pluginArch)
		return strings.ReplaceAll(entry, "linux-aarch64", pluginArch)
	})
}
