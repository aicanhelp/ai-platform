/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package installer

import (
	"context"
	"fmt"
	"slices"
	"strings"

	"github.com/pkg/errors"
	"gopkg.in/yaml.v3"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/client-go/kubernetes"

	"installer-service/pkg/constant"
	"installer-service/pkg/utils/k8sutil"
	"installer-service/pkg/zlog"
)

const (
	// MinSemverSegments version format  major.minor.patch
	MinSemverSegments = 3
	// PatchVersionIndex patch index
	PatchVersionIndex = 2
)

func getPatchInfo(c *installerClient, obj *unstructured.Unstructured) (string, error) {
	openFuyaoVersion, err := getOpenFuyaoVersionFromUnstructured(obj)
	zlog.Infoln("openFuyaoVersion in unstructured is", openFuyaoVersion)
	if err != nil {
		return "", errors.Wrap(err, "failed to get openFuyao version from input")
	}

	versionList, err := c.GetUpgradeVersions()
	if err != nil {
		return "", errors.Wrap(err, "failed to get available openFuyao versions")
	}
	versionFound := slices.Contains(versionList, openFuyaoVersion)
	if !versionFound {
		return "", fmt.Errorf("openFuyaoVersion '%s' is not supported. Available versions: %v",
			openFuyaoVersion, versionList)
	}

	configMap, err := k8sutil.GetConfigMap(c.clientset, BKEConfigCmKey().Name, BKEConfigCmKey().Namespace)
	if err != nil {
		return "", errors.Wrapf(err, "failed to get ConfigMap %s/%s",
			BKEConfigCmKey().Namespace, BKEConfigCmKey().Name)
	}

	patchConfigMapKey := constant.PatchKeyPrefix + openFuyaoVersion
	patchConfigMapName, exists := configMap.Data[patchConfigMapKey]
	if !exists {
		return "", fmt.Errorf("patch file path not found in ConfigMap for key '%s'", patchConfigMapKey)
	}
	if strings.TrimSpace(patchConfigMapName) == "" {
		return "", fmt.Errorf("patch config map is empty for version '%s'", openFuyaoVersion)
	}

	return patchConfigMapName, nil
}

func updateCRDAddonsWithPatch(addonsRaw []interface{}, patchAddons []AddonInfo) {
	patchMap := buildPatchMap(patchAddons)

	for i, raw := range addonsRaw {
		addon, ok := raw.(map[string]interface{})
		if !ok {
			continue
		}

		name, ok := getString(addon, "name")
		if !ok || name == "" {
			continue
		}

		patchAddon, exists := patchMap[name]
		if !exists {
			continue
		}

		applyPatchToAddon(addon, patchAddon)
		addonsRaw[i] = addon
	}
}

func buildPatchMap(addons []AddonInfo) map[string]AddonInfo {
	m := make(map[string]AddonInfo, len(addons))
	for _, a := range addons {
		m[a.Name] = a
	}
	return m
}

func getString(m map[string]interface{}, key string) (string, bool) {
	v, ok := m[key]
	if !ok {
		return "", false
	}
	s, ok := v.(string)
	return s, ok
}

func applyPatchToAddon(addon map[string]interface{}, patch AddonInfo) {
	if addon == nil {
		return
	}
	if patch.Version != "" {
		addon["version"] = patch.Version
	}
	addon["block"] = patch.Block

	if len(patch.Param) == 0 {
		return
	}

	currentParam := getOrCreateParamMap(addon)
	for k, v := range patch.Param {
		currentParam[k] = v
	}
	addon["param"] = currentParam
}

func getOrCreateParamMap(addon map[string]interface{}) map[string]interface{} {
	if raw, ok := addon["param"]; ok {
		if m, ok := raw.(map[string]interface{}); ok {
			return m
		}
		if oldMap, ok := raw.(map[string]string); ok {
			newMap := make(map[string]interface{}, len(oldMap))
			for k, v := range oldMap {
				newMap[k] = v
			}
			return newMap
		}
	}
	return make(map[string]interface{})
}

func patchAddonsInfo(c *installerClient, obj *unstructured.Unstructured) error {
	patchConfigMapName, err := getPatchInfo(c, obj)
	if err != nil {
		return errors.Wrap(err, "get patch config map info failed")
	}

	patchAddons, err := getAddonsFromPatchCM(c.clientset, patchConfigMapName)
	if err != nil {
		return errors.Wrap(err, "get addons for config map failed")
	}
	zlog.Infoln("addon is", patchAddons)

	addonsRaw, found, err := unstructured.NestedSlice(obj.Object, "spec", "clusterConfig", "addons")
	if err != nil || !found {
		return fmt.Errorf("addons is not found in configmap %s", patchConfigMapName)
	}

	updateCRDAddonsWithPatch(addonsRaw, patchAddons)

	err = unstructured.SetNestedSlice(obj.Object, addonsRaw, "spec", "clusterConfig", "addons")
	if err != nil {
		return errors.Wrap(err, "failed to update cluster configuration with patch version info")
	}

	return nil
}

func getAddonsFromPatchCM(clientSet kubernetes.Interface, patchConfigMapName string) ([]AddonInfo, error) {
	ns := constant.PatchNameSpace
	cm, err := clientSet.CoreV1().ConfigMaps(ns).Get(context.TODO(), patchConfigMapName, metav1.GetOptions{})
	if err != nil {
		return nil, errors.Wrapf(err, "failed to get config map '%s'", patchConfigMapName)
	}

	version := strings.TrimPrefix(patchConfigMapName, constant.PatchValuePrefix)
	patchYAML, exists := cm.Data[version]
	if !exists {
		return nil, fmt.Errorf("key '%s' not found in configmap data", version)
	}

	var config PatchConfig
	if err = yaml.Unmarshal([]byte(patchYAML), &config); err != nil {
		return nil, errors.Wrap(err, "failed to unmarshal patch config YAML")
	}

	return config.Addons, nil
}

func patchCoreDNSAntiAffinity(obj *unstructured.Unstructured) error {
	nodes, found, err := unstructured.NestedSlice(obj.Object, "spec", "clusterConfig", "nodes")
	if err != nil {
		return errors.Wrap(err, "failed to get nodes from cluster config")
	}
	if !found || len(nodes) == 0 {
		zlog.Warn("no nodes found in cluster config, skipping CoreDNS anti-affinity patch")
		return nil
	}

	nodeCount := len(nodes)
	enableAntiAffinity := "false"
	if nodeCount > 1 {
		enableAntiAffinity = "true"
	}
	zlog.Infof("Cluster has %d nodes, setting CoreDNS EnableAntiAffinity to %s", nodeCount, enableAntiAffinity)

	addons, found, err := unstructured.NestedSlice(obj.Object, "spec", "clusterConfig", "addons")
	if err != nil {
		return errors.Wrap(err, "failed to get addons from cluster config")
	}
	if !found {
		zlog.Warn("no addons found in cluster config, skipping CoreDNS anti-affinity patch")
		return nil
	}

	updated := false
	for i, addon := range addons {
		addonMap, ok := addon.(map[string]interface{})
		if !ok {
			continue
		}

		name, ok := getString(addonMap, "name")
		if !ok || name != "coredns" {
			continue
		}

		param := getOrCreateParamMap(addonMap)
		param["EnableAntiAffinity"] = enableAntiAffinity
		addonMap["param"] = param

		addons[i] = addonMap
		updated = true
		zlog.Infof("Updated CoreDNS addon EnableAntiAffinity to %s", enableAntiAffinity)
		break
	}

	if !updated {
		zlog.Warn("coredns addon not found in addons list")
		return nil
	}

	if err := unstructured.SetNestedSlice(obj.Object, addons, "spec", "clusterConfig", "addons"); err != nil {
		return errors.Wrap(err, "failed to update addons in cluster config")
	}

	return nil
}

func IsOnlineMode(cm *corev1.ConfigMap) (bool, error) {
	if len(cm.Data) == 0 {
		return false, errors.New("configMap.Data is nil or empty")
	}
	repoValue, repoOk := cm.Data["otherRepo"]
	imageValue, imageOk := cm.Data["onlineImage"]
	if !repoOk || !imageOk {
		return false, fmt.Errorf("configMap key `otherRepo` or `onlineImage` missing")
	} else if repoValue == "" && imageValue == "" {
		return false, nil
	}
	return true, nil
}
