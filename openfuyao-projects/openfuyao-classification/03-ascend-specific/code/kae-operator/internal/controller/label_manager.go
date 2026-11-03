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

package controller

import (
	"fmt"
	"strings"

	corev1 "k8s.io/api/core/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"

	kaev1 "gopkg.openfuyao.cn/kae-operator/api/v1"
	"gopkg.openfuyao.cn/kae-operator/internal/utils"
	"gopkg.openfuyao.cn/kae-operator/internal/zlog"
)

const (
	nfdLabelPrefix = "feature.node.kubernetes.io/"

	oSReleaseIDLabelKey = "feature.node.kubernetes.io/system-os_release.ID"
	oSVersionIDLabelKey = "feature.node.kubernetes.io/system-os_release.VERSION_ID"

	hprePFLabelKey = "feature.node.kubernetes.io/pci-1000_19e5_a258.present"
	hpreVFLabelKey = "feature.node.kubernetes.io/pci-1000_19e5_a259.present"
	secPFLabelKey  = "feature.node.kubernetes.io/pci-1000_19e5_a255.present"
	sceVFLabelKey  = "feature.node.kubernetes.io/pci-1000_19e5_a256.present"
	zipPFLabelKey  = "feature.node.kubernetes.io/pci-1200_19e5_a250.present"
	zipVFLabelKey  = "feature.node.kubernetes.io/pci-1200_19e5_a251.present"

	driverDeployLabelKey     = "openfuyao.com/kae.deploy.driver"
	hpreDevicePluginLabelKey = "openfuyao.com/kae.hpre.deploy.device-plugin"
	secDevicePluginLabelKey  = "openfuyao.com/kae.sec.deploy.device-plugin"
	zipDevicePluginLabelKey  = "openfuyao.com/kae.zip.deploy.device-plugin"

	psaLabelPrefix = "pod-security.kubernetes.io/"
	privileged     = "privileged"

	commonKAELabelKey = "openfuyao.com/kae.present"
)

var hpreNodeLabels = map[string]string{
	hprePFLabelKey: "true",
	hpreVFLabelKey: "true",
}

var secNodeLabels = map[string]string{
	secPFLabelKey: "true",
	sceVFLabelKey: "true",
}

var zipNodeLabels = map[string]string{
	zipPFLabelKey: "true",
	zipVFLabelKey: "true",
}

var hpreDeployLabels = map[string]string{
	driverDeployLabelKey:     "true",
	hpreDevicePluginLabelKey: "true",
}

var secDeployLabels = map[string]string{
	driverDeployLabelKey:    "true",
	secDevicePluginLabelKey: "true",
}

var zipDeployLabels = map[string]string{
	driverDeployLabelKey:    "true",
	zipDevicePluginLabelKey: "true",
}

var kaeDeployLabels = map[string]string{
	driverDeployLabelKey:     "true",
	hpreDevicePluginLabelKey: "true",
	secDevicePluginLabelKey:  "true",
	zipDevicePluginLabelKey:  "true",
}

var watchLabels = []string{
	hprePFLabelKey,
	hpreVFLabelKey,
	secPFLabelKey,
	sceVFLabelKey,
	zipPFLabelKey,
	zipVFLabelKey,
}

var supportOsVersions = map[string]map[string]bool{
	"openEuler": {
		"22.03": true,
	},
}

// hasKAEHPRELabels return true if node labels contain KAE labels
func hasKAEHPRELabels(labels map[string]string) bool {
	for key, value := range labels {
		if _, ok := hpreNodeLabels[key]; ok {
			if hpreNodeLabels[key] == value {
				return true
			}
		}
	}
	return false
}

// hasKAESECLabels return true if node labels contain KAE SEC labels
func hasKAESECLabels(labels map[string]string) bool {
	for key, value := range labels {
		if _, ok := secNodeLabels[key]; ok {
			if secNodeLabels[key] == value {
				return true
			}
		}
	}
	return false
}

// hasKAEZIPLabels return true if node labels contain KAE ZIP labels
func hasKAEZIPLabels(labels map[string]string) bool {
	for key, value := range labels {
		if _, ok := zipNodeLabels[key]; ok {
			if zipNodeLabels[key] == value {
				return true
			}
		}
	}
	return false
}

func canDeployHPRE(labels map[string]string) bool {
	return hasKAEHPRELabels(labels) && isOSSupportKAE(labels)
}

func canDeploySEC(labels map[string]string) bool {
	return hasKAESECLabels(labels) && isOSSupportKAE(labels)
}

func canDeployZIP(labels map[string]string) bool {
	return hasKAEZIPLabels(labels) && isOSSupportKAE(labels)
}

// hasKAELabels 根据使能情况，判断是否具有KAE硬件
func hasKAELabels(labels map[string]string, enableHPRE, enableSEC, enableZIP bool) bool {
	if enableHPRE && canDeployHPRE(labels) {
		return true
	}
	if enableSEC && canDeploySEC(labels) {
		return true
	}
	if enableZIP && canDeployZIP(labels) {
		return true
	}
	return false
}

// getNodeOSInfo return osName and os version by nfd labels
func getNodeOSInfo(labels map[string]string) (string, string) {
	osName, ok := labels[oSReleaseIDLabelKey]
	if !ok {
		return "unknown", ""
	}
	osVersion, ok := labels[oSVersionIDLabelKey]
	if !ok {
		return osName, ""
	}
	return osName, osVersion
}

// isOSSupportKAE return true if host/VM OS support KAE Driver
func isOSSupportKAE(labels map[string]string) bool {
	osName, osVersion := getNodeOSInfo(labels)
	return supportOsVersions[osName][osVersion]
}

// hasNFDLabels return true if node labels contain NFD labels
func hasNFDLabels(labels map[string]string) bool {
	for key := range labels {
		if strings.HasPrefix(key, nfdLabelPrefix) {
			return true
		}
	}
	return false
}

// hasCommonKAELabel returns true if common KAE label exists among provided node labels
func hasCommonKAELabel(labels map[string]string) bool {
	if _, ok := labels[commonKAELabelKey]; ok {
		if labels[commonKAELabelKey] == "true" {
			return true
		}
	}
	return false
}

// addDeployLabels 根据stateLabels添加labels
func addDeployLabels(labels map[string]string, stateLabels map[string]string) bool {
	modified := false
	if labels == nil {
		return false
	}
	for key, value := range stateLabels {
		if _, ok := labels[key]; !ok {
			zlog.Infof("Adding node label: %s", key)
			labels[key] = value
			modified = true
		}
	}
	return modified
}

// removeDeployLabels 根据stateLabels删除labels
func removeDeployLabels(labels map[string]string, stateLabels map[string]string) bool {
	modified := false
	if labels == nil {
		return false
	}
	for key := range stateLabels {
		if _, ok := labels[key]; ok {
			if labels[key] != "true" {
				continue
			}
			zlog.Infof("Deleting node label: %s", key)
			delete(labels, key)
			modified = true
		}
	}
	return modified
}

// removeAllDeployLabels removes all kaeDeployLabels from the provided map of node labels.
// returns true if the labels map has been modified.
func removeAllDeployLabels(labels map[string]string) bool {
	modified := false
	for key := range kaeDeployLabels {
		if _, ok := labels[key]; ok {
			zlog.Infof("Deleting node label: %s", key)
			delete(labels, key)
			modified = true
		}
	}
	return modified
}

// labelKAENodes labels nodes with KAE's(HPRE、SEC、ZIP) with KAE common label
func (m *KAEPolicyManager) labelKAENodes() error {
	list := &corev1.NodeList{}
	err := m.client.List(m.ctx, list)
	if err != nil {
		return fmt.Errorf("unable to list nodes to check labels, err %s", err.Error())
	}

	clusterHasNFDLabels := false
	updateLabels := false
	operatorSpec := m.kaeInstance.Spec.Operator
	for _, nodeItem := range list.Items {
		node := nodeItem
		labels := node.GetLabels()
		if !clusterHasNFDLabels {
			clusterHasNFDLabels = hasNFDLabels(labels)
		}

		if !hasCommonKAELabel(labels) && hasKAELabels(labels, utils.IsOperatorEnabledHPRE(operatorSpec),
			utils.IsOperatorEnabledSEC(operatorSpec), utils.IsOperatorEnabledZIP(operatorSpec)) {
			zlog.Infof("node: %s has KAE device. setting node CommonKAE label: %s=true", node.ObjectMeta.Name,
				commonKAELabelKey)
			labels[commonKAELabelKey] = "true"
			node.SetLabels(labels)
			updateLabels = true
		} else if hasCommonKAELabel(labels) && !hasKAELabels(labels, utils.IsOperatorEnabledHPRE(operatorSpec),
			utils.IsOperatorEnabledSEC(operatorSpec), utils.IsOperatorEnabledZIP(operatorSpec)) {
			zlog.Infof("node: %s no longer has KAE device. setting node CommonKAE label: %s=false",
				node.ObjectMeta.Name, commonKAELabelKey)
			labels[commonKAELabelKey] = "false"
			removeAllDeployLabels(labels)
			node.SetLabels(labels)
			updateLabels = true
		} else {
			zlog.Info("the hasCommonKAELabel and hasKAELabels states are identical")
		}

		if m.updateNodeDeployLabels(labels, operatorSpec, node.Name) {
			updateLabels = true
			node.SetLabels(labels)
		}

		if updateLabels {
			err = m.client.Update(m.ctx, &node)
			if err != nil {
				return fmt.Errorf("setting node: %s labels by KAE Operator err: %s",
					node.ObjectMeta.Name, err.Error())
			}
		}
	}
	m.hasNFDLabels = clusterHasNFDLabels
	m.hasKAENodes = m.kaeNodes != 0
	zlog.Infof("number of KAE nodes with KAE label: %d, hpre：%d, sec: %d, zip: %d", m.kaeNodes, m.hpreNodes,
		m.secNodes, m.zipNodes)
	return nil
}

// addNodeDeployLabels 更新节点部署组件标签
func (m *KAEPolicyManager) addNodeDeployLabels(labels map[string]string,
	operatorSpec kaev1.OperatorSpec, nodeName string) bool {
	updateLabels := false
	if utils.IsOperatorEnabledHPRE(operatorSpec) && canDeployHPRE(labels) {
		if addDeployLabels(labels, hpreDeployLabels) {
			zlog.Infof("add node: %s hpre state labels", nodeName)
			updateLabels = true
		}
		zlog.Infof("node: %s has hpre device", nodeName)
		m.hpreNodes++
	}
	if utils.IsOperatorEnabledSEC(operatorSpec) && canDeploySEC(labels) {
		if addDeployLabels(labels, secDeployLabels) {
			zlog.Infof("add node: %s sec state labels", nodeName)
			updateLabels = true
		}
		zlog.Infof("node: %s has sec device", nodeName)
		m.secNodes++
	}
	if utils.IsOperatorEnabledSEC(operatorSpec) && canDeployZIP(labels) {
		if addDeployLabels(labels, zipDeployLabels) {
			zlog.Infof("add node: %s zip state labels", nodeName)
			updateLabels = true
		}
		zlog.Infof("node: %s has zip device", nodeName)
		m.zipNodes++
	}
	m.kaeNodes++

	return updateLabels
}

// removeNodeDeployLabels 更新节点部署组件标签
func (m *KAEPolicyManager) removeNodeDeployLabels(labels map[string]string,
	operatorSpec kaev1.OperatorSpec, nodeName string) bool {
	updateLabels := false
	if !utils.IsOperatorEnabledHPRE(operatorSpec) || !canDeployHPRE(labels) {
		if removeDeployLabels(labels, hpreDeployLabels) {
			zlog.Infof("remove node: %s hpre deploy labels", nodeName)
			updateLabels = true
		}
		zlog.Infof("node: %s can not deploy hpre components", nodeName)
	}
	if !utils.IsOperatorEnabledSEC(operatorSpec) || !canDeploySEC(labels) {
		if removeDeployLabels(labels, secDeployLabels) {
			zlog.Infof("remove node: %s sec state labels", nodeName)
			updateLabels = true
		}
		zlog.Infof("node: %s can not deploy sec components", nodeName)
	}
	if !utils.IsOperatorEnabledSEC(operatorSpec) || !canDeployZIP(labels) {
		if removeDeployLabels(labels, zipDeployLabels) {
			zlog.Infof("remove node: %s zip state labels", nodeName)
			updateLabels = true
		}
		zlog.Infof("node: %s can not deploy zip components", nodeName)
	}

	return updateLabels
}

// updateNodeDeployLabels 更新节点deploy标签，如果有kae common标签则先处理删除再处理添加，可防止删除driver公共标签
func (m *KAEPolicyManager) updateNodeDeployLabels(labels map[string]string, operatorSpec kaev1.OperatorSpec,
	nodeName string) bool {
	if hasCommonKAELabel(labels) {
		removed := m.removeNodeDeployLabels(labels, operatorSpec, nodeName)
		added := m.addNodeDeployLabels(labels, operatorSpec, nodeName)
		return removed || added
	}
	return false
}

// setPSALabelForNamespace 给命名空间设置PSA标签
func (m *KAEPolicyManager) setPSALabelForNamespace() error {
	namespaceName := kaePolicyManager.namespace
	namespaceObj := &corev1.Namespace{}
	err := m.client.Get(m.ctx, client.ObjectKey{Name: namespaceName}, namespaceObj)
	if err != nil {
		return fmt.Errorf("ERROR: could not get Namespace %s from client: %v", namespaceName, err)
	}

	labels := namespaceObj.ObjectMeta.Labels
	modified := false
	if labels == nil {
		labels = make(map[string]string)
		modified = true
	}
	for _, mode := range securityModes {
		key := psaLabelPrefix + mode
		if value, ok := labels[key]; !ok || (value != privileged) {
			labels[key] = privileged
			modified = true
		}
	}
	if !modified {
		return nil
	}

	namespaceObj.ObjectMeta.Labels = labels
	err = m.client.Update(m.ctx, namespaceObj)
	if err != nil {
		return fmt.Errorf("unable to label namespace %s with pod security levels: %v", namespaceName, err)
	}
	return nil
}

// hasLabelChanged 检查两个map中指定的key是否发生变化
func hasLabelChanged(oldLabels, newLabels map[string]string) bool {
	for _, key := range watchLabels {
		oldValue, oldExists := oldLabels[key]
		newValue, newExists := newLabels[key]

		if !oldExists && newExists {
			// 新增的键
			zlog.Infof("key: %s added in new map，value: %s", key, newValue)
			return true
		}

		if oldExists && !newExists {
			// 删除的键
			zlog.Infof("key: %s existed in old map, but deleted in new map", key)
			return true
		}

		if oldExists && newExists && (newValue != oldValue) {
			// 值发生了变化的键
			zlog.Infof("key: %s has changed from %s to the value of %s", key, oldValue, newValue)
			return true
		}
	}
	return false
}
