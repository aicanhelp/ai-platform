/*
 *
 *  * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 *  * openFuyao is licensed under Mulan PSL v2.
 *  * You can use this software according to the terms and conditions of the Mulan PSL v2.
 *  * You may obtain a copy of Mulan PSL v2 at:
 *  *          http://license.coscl.org.cn/MulanPSL2
 *  * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 *  * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 *  * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 *  * See the Mulan PSL v2 for more details.
 *
 */

// Package volcano config
package volcano

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"

	"github.com/blang/semver/v4"
	v1 "k8s.io/api/core/v1"
	"k8s.io/client-go/kubernetes"

	"volcano-config-service/pkg/constant"
	"volcano-config-service/pkg/controller"
	"volcano-config-service/pkg/utils/httputil"
	"volcano-config-service/pkg/utils/k8sutil"
	"volcano-config-service/pkg/zlog"
)

const (
	policyStatusOpen    = "open"
	policyStatusClose   = "close"
	policyStatusPartial = "partial"
)

func summarizeNodePolicyStatus(nodes []controller.NodeConfiguration,
	isEnabled func(node controller.NodeConfiguration) bool) string {
	if len(nodes) == 0 {
		return policyStatusClose
	}

	enabledCount := 0
	for _, node := range nodes {
		if isEnabled(node) {
			enabledCount++
		}
	}

	if enabledCount == len(nodes) {
		return policyStatusOpen
	}
	if enabledCount == 0 {
		return policyStatusClose
	}
	return policyStatusPartial
}

func (v volcanoClient) getNodeConfigWithEmptyCheck() (*controller.NodeConfig, *httputil.ResponseJson, int) {
	nodeConfig, err := k8sutil.GetNodeConfig(v.clientset)
	if err != nil {
		return nil, &httputil.ResponseJson{
			Code: constant.Success,
			Msg:  fmt.Sprintf("failed to get NodeConfig: %v", err),
		}, http.StatusInternalServerError
	}

	if len(nodeConfig.Spec.Nodes) == 0 {
		return nil, &httputil.ResponseJson{
			Code: constant.Success,
			Msg:  "no nodes configured",
			Data: policyStatusClose,
		}, http.StatusOK
	}

	return nodeConfig, nil, 0
}

func (v volcanoClient) MethodNumaAwareGet() (*httputil.ResponseJson, int) {
	nodeConfig, errResp, statusCode := v.getNodeConfigWithEmptyCheck()
	if errResp != nil {
		return errResp, statusCode
	}

	status := summarizeNodePolicyStatus(nodeConfig.Spec.Nodes, func(node controller.NodeConfiguration) bool {
		return node.EnableStaticCPUManager
	})

	if status == policyStatusOpen {
		return methodGet(v.clientset, "numa-aware")
	}

	return &httputil.ResponseJson{
		Code: constant.Success,
		Msg:  "success",
		Data: status,
	}, http.StatusOK
}

// setNodeFlagForAllNodes 返回一个 PutNodeConfigWithRetry 的回调，
// 将所有节点的指定开关统一设置为 status 对应的值。
// fieldSelector 指明要修改的字段（EnableStaticCPUManager 或 EnableNUMADistance）。
func setNodeFlagForAllNodes(status string,
	fieldSelector func(node *controller.NodeConfiguration) *bool) func(latest *controller.NodeConfig) (*controller.NodeConfig, error) {

	return func(latest *controller.NodeConfig) (*controller.NodeConfig, error) {
		enabled := status == "open"
		for i := range latest.Spec.Nodes {
			*fieldSelector(&latest.Spec.Nodes[i]) = enabled
		}
		return latest, nil
	}
}

func (v volcanoClient) MethodNumaAwarePut(status string) (*httputil.ResponseJson, int) {
	// 使用冲突重试方式更新 NodeConfig，避免覆盖 numaadj 并发写入的节点数据
	err := k8sutil.PutNodeConfigWithRetry(v.clientset,
		setNodeFlagForAllNodes(status, func(node *controller.NodeConfiguration) *bool {
			return &node.EnableStaticCPUManager
		}))
	if err != nil {
		return &httputil.ResponseJson{
			Code: constant.Success,
			Msg:  fmt.Sprintf("failed to update NodeConfig: %v", err),
		}, http.StatusInternalServerError
	}

	return methodPut(v.clientset, "numa-aware", status)
}

func (v volcanoClient) MethodNumaDistanceGet() (*httputil.ResponseJson, int) {
	nodeConfig, errResp, statusCode := v.getNodeConfigWithEmptyCheck()
	if errResp != nil {
		return errResp, statusCode
	}

	status := summarizeNodePolicyStatus(nodeConfig.Spec.Nodes, func(node controller.NodeConfiguration) bool {
		return node.EnableNUMADistance
	})

	return &httputil.ResponseJson{
		Code: constant.Success,
		Msg:  "success",
		Data: status,
	}, http.StatusOK
}

// GetKubeletVersionFromNode 从节点信息中提取kubelet版本
func GetKubeletVersionFromNode(node *v1.Node) (string, error) {
	if node == nil {
		return "", fmt.Errorf("node is nil")
	}

	// 从节点状态中获取kubelet版本
	kubeletVersion := node.Status.NodeInfo.KubeletVersion
	if kubeletVersion == "" {
		return "", fmt.Errorf("kubelet version not found in node info")
	}

	return kubeletVersion, nil
}

// CheckKubeletVersionSupport 检查kubelet版本是否支持特定功能
func CheckKubeletVersionSupport(version string) (bool, error) {
	isCustomBuild := strings.Contains(version, "NUMADistance")
	if isCustomBuild {
		zlog.FormatInfo("Failed to parse version but custom build detected: %s", version)
		return true, nil
	}

	// 清理版本字符串，去除前缀 'v'
	versionStr := strings.TrimPrefix(version, "v")

	// 清理版本字符串，只保留合法的 semver 部分
	versionStr = cleanVersionString(versionStr)

	currentVersion, err := semver.Parse(versionStr)
	if err != nil {
		return false, fmt.Errorf("failed to parse kubelet version: %v", err)
	}

	// 目标版本 1.28.0
	targetVersion, _ := semver.Parse("1.28.0")

	// 如果版本号大于等于 1.28.0，则支持该特性
	isSupported := currentVersion.GE(targetVersion)

	return isSupported, nil
}

// cleanVersionString 清理版本字符串，确保符合semver格式
func cleanVersionString(version string) string {
	// 如果版本包含额外信息（如 1.28.8-gke.1000），只保留主要版本部分
	parts := strings.Split(version, "-")
	mainVersion := parts[0]

	// 确保版本格式为 x.y.z
	const length = 2
	vParts := strings.Split(mainVersion, ".")
	if len(vParts) == length {
		// 如果格式为 x.y，补充为 x.y.0
		mainVersion = mainVersion + ".0"
	}

	return mainVersion
}

// checkKubeletSupport 获取第一个节点的kubelet版本并检查支持情况
func checkKubeletSupport(clientset kubernetes.Interface) (string, bool, error) {
	// 获取节点列表
	nodeList, err := k8sutil.GetNodeList(clientset)
	if err != nil {
		return "", false, err
	}

	// 检查是否有节点
	if len(nodeList.Items) == 0 {
		return "", false, fmt.Errorf("no nodes found in cluster")
	}

	// 获取第一个节点
	firstNode := &nodeList.Items[0]

	// 获取kubelet版本
	kubeletVersion, err := GetKubeletVersionFromNode(firstNode)
	if err != nil {
		return "", false, err
	}

	// 检查版本支持情况
	isSupported, err := CheckKubeletVersionSupport(kubeletVersion)
	if err != nil {
		return kubeletVersion, false, err
	}

	return kubeletVersion, isSupported, nil
}

func (v volcanoClient) MethodNumaDistancePut(status string) (*httputil.ResponseJson, int) {
	if status == "open" {
		version, isSupport, err := checkKubeletSupport(v.clientset)
		if err != nil {
			return &httputil.ResponseJson{
				Code: constant.Success,
				Msg:  fmt.Sprintf("failed to check Kubelet version: %v", err),
			}, http.StatusInternalServerError
		}

		if !isSupport {
			return &httputil.ResponseJson{
				Code: constant.NotAcceptable,
				Msg:  fmt.Sprintf("Kubelet version: %s does not meet the requirements", version),
			}, http.StatusNotAcceptable
		}
	}

	// 获取并更新NodeConfig
	nodeConfig, err := k8sutil.GetNodeConfig(v.clientset)
	if err != nil {
		return &httputil.ResponseJson{
			Code: constant.Success,
			Msg:  fmt.Sprintf("failed to get NodeConfig: %v", err),
		}, http.StatusInternalServerError
	}

	// 更新NodeConfig并提交更改
	return v.updateAndSubmitNodeConfig(nodeConfig, status)
}

// updateAndSubmitNodeConfig 更新并提交NodeConfig
func (v volcanoClient) updateAndSubmitNodeConfig(nodeConfig *controller.NodeConfig,
	status string) (*httputil.ResponseJson, int) {

	// 使用冲突重试方式更新 NodeConfig，避免覆盖 numaadj 并发写入的节点数据
	err := k8sutil.PutNodeConfigWithRetry(v.clientset,
		setNodeFlagForAllNodes(status, func(node *controller.NodeConfiguration) *bool {
			return &node.EnableNUMADistance
		}))
	if err != nil {
		return &httputil.ResponseJson{
			Code: constant.Success,
			Msg:  fmt.Sprintf("failed to update NodeConfig: %v", err),
		}, http.StatusInternalServerError
	}

	return &httputil.ResponseJson{
		Code: constant.Success,
		Msg:  "success",
	}, http.StatusOK
}

func (v volcanoClient) MethodNumaFastGet() (*httputil.ResponseJson, int) {
	oenuma, err := k8sutil.GetOenuma(v.clientset)
	if err != nil {
		return &httputil.ResponseJson{
			Code: constant.Success,
			Msg:  fmt.Sprintf("failed to get CR Oenuma: %v", err),
		}, http.StatusInternalServerError
	}

	var data string
	if oenuma.Spec.Updateenable == "enable" {
		data = "open"
	} else {
		data = "close"
	}

	return &httputil.ResponseJson{
		Code: constant.Success,
		Msg:  "success",
		Data: data,
	}, http.StatusOK
}

func (v volcanoClient) MethodNumaFastPut(status string) (*httputil.ResponseJson, int) {
	oenuma, err := k8sutil.GetOenuma(v.clientset)
	if err != nil {
		return &httputil.ResponseJson{
			Code: constant.Success,
			Msg:  fmt.Sprintf("failed to get CR Oenuma: %v", err),
		}, http.StatusInternalServerError
	}

	// 更新字段
	if status == "open" {
		oenuma.Spec.Updateenable = "enable"
	} else {
		oenuma.Spec.Updateenable = "disable"
	}
	oenuma.APIVersion = "resource.sop.huawei.com/v1alpha1"
	oenuma.Kind = "Oenuma"

	data, err := json.Marshal(oenuma)
	if err != nil {
		return &httputil.ResponseJson{
			Code: constant.Success,
			Msg:  fmt.Sprintf("failed to marshal Oenuma: %v", err),
		}, http.StatusInternalServerError
	}

	err = k8sutil.PutOenuma(v.clientset, data)
	if err != nil {
		return &httputil.ResponseJson{
			Code: constant.Success,
			Msg:  fmt.Sprintf("failed to update Oenuma: %v", err),
		}, http.StatusInternalServerError
	}

	return &httputil.ResponseJson{
		Code: constant.Success,
		Msg:  "success",
	}, http.StatusOK
}

func (v volcanoClient) MethodBinpackGet() (*httputil.ResponseJson, int) {
	return methodGet(v.clientset, "binpack")
}

func (v volcanoClient) MethodBinpackPut(status string) (*httputil.ResponseJson, int) {
	return methodPut(v.clientset, "binpack", status)
}

func (v volcanoClient) MethodDrfGet() (*httputil.ResponseJson, int) {
	return methodGet(v.clientset, "drf")
}

func (v volcanoClient) MethodDrfPut(status string) (*httputil.ResponseJson, int) {
	return methodPut(v.clientset, "drf", status)
}

func (v volcanoClient) MethodBalanceGet() (*httputil.ResponseJson, int) {
	return methodGet(v.clientset, "balance")
}

func (v volcanoClient) MethodBalancePut(status string) (*httputil.ResponseJson, int) {
	return methodPut(v.clientset, "balance", status)
}

func (v volcanoClient) MethodGangGet() (*httputil.ResponseJson, int) {
	return methodGet(v.clientset, "gang")
}

func (v volcanoClient) MethodGangPut(status string) (*httputil.ResponseJson, int) {
	return gangPut(v.clientset, status)
}

func (v volcanoClient) MethodPriorityGet() (*httputil.ResponseJson, int) {
	return priorityGet(v.clientset)
}

func (v volcanoClient) MethodPriorityPut(status string) (*httputil.ResponseJson, int) {
	return priorityPut(v.clientset, status)
}

func (v volcanoClient) MethodDefault() (*httputil.ResponseJson, int) {
	configmap, err := k8sutil.GetConfigMap(v.clientset, constant.VolcanoConfigServiceConfigmap,
		constant.VolcanoConfigServiceDefaultNamespace)
	if err != nil {
		zlog.Error("GetConfigMap failed, %v", err)
		return httputil.GetDefaultServerFailureResponseJson(), http.StatusInternalServerError
	}

	configContent := `actions: "enqueue, allocate, backfill"
tiers:
- plugins:
  - name: priority
  - name: gang
    enablePreemptable: false
  - name: conformance
- plugins:
  - name: overcommit
  - name: drf
    enablePreemptable: false
  - name: predicates
  - name: proportion
  - name: nodeorder
  - name: binpack`

	configmap.Data[constant.VolcanoConfigServiceConfigmapName] = configContent
	configmap, err = k8sutil.UpdateConfigMap(v.clientset, configmap)
	if err != nil {
		zlog.Error("UpdateConfigMap failed, %v", err)
		return httputil.GetDefaultServerFailureResponseJson(), http.StatusInternalServerError
	}

	err = k8sutil.DeleteVolcanoPod(v.clientset, constant.VolcanoConfigServiceDefaultNamespace)
	if err != nil {
		zlog.Error("delete pod volcano-scheduler failed, %v", err)
		return httputil.GetDefaultServerFailureResponseJson(), http.StatusInternalServerError
	}

	return &httputil.ResponseJson{
		Code: constant.Success,
		Msg:  "success",
	}, http.StatusOK
}

func (v volcanoClient) MethodNodeList() (*httputil.ResponseJson, int) {
	nodeList, err := k8sutil.GetNodeList(v.clientset)
	if err != nil {
		zlog.Error("Get Node List failed, %v", err)
		return httputil.GetDefaultServerFailureResponseJson(), http.StatusInternalServerError
	}

	return &httputil.ResponseJson{
		Code: constant.Success,
		Msg:  "success",
		Data: nodeList,
	}, http.StatusOK
}

func (v volcanoClient) MethodNodeListByName(name string) (*httputil.ResponseJson, int) {
	nodeList, err := k8sutil.GetNodeListByName(v.clientset, name)
	if err != nil {
		zlog.Error("Get Node List ByName failed, %v", err)
		return httputil.GetDefaultServerFailureResponseJson(), http.StatusInternalServerError
	}

	return &httputil.ResponseJson{
		Code: constant.Success,
		Msg:  "success",
		Data: nodeList,
	}, http.StatusOK
}

func (v volcanoClient) MethodCRInfo(name string) (*httputil.ResponseJson, int) {
	CR, err := k8sutil.GetCR(v.dynamicClient, name)
	if err != nil {
		zlog.Error("Get CR failed, %v", err)
		return httputil.GetDefaultServerFailureResponseJson(), http.StatusInternalServerError
	}

	return &httputil.ResponseJson{
		Code: constant.Success,
		Msg:  "success",
		Data: CR,
	}, http.StatusOK
}

func (v volcanoClient) MethodUuidToName(uuid string) (*httputil.ResponseJson, int) {
	podName, err := k8sutil.PodUuidToName(v.clientset, uuid)
	if err != nil {
		zlog.Error("Get pod name failed, %v", err)
		return httputil.GetDefaultServerFailureResponseJson(), http.StatusInternalServerError
	}

	return &httputil.ResponseJson{
		Code: constant.Success,
		Msg:  "success",
		Data: podName,
	}, http.StatusOK
}

func (v volcanoClient) MethodSchedulerGet() (*httputil.ResponseJson, int) {
	schedulerList, err := k8sutil.GetScheduler(v.clientset)
	if err != nil {
		zlog.Error("Get pod name failed, %v", err)
		return httputil.GetDefaultServerFailureResponseJson(), http.StatusInternalServerError
	}

	return &httputil.ResponseJson{
		Code: constant.Success,
		Msg:  "success",
		Data: schedulerList,
	}, http.StatusOK
}

func methodGet(clientset kubernetes.Interface, plugin string) (*httputil.ResponseJson, int) {
	var status string

	configmap, err := k8sutil.GetConfigMap(clientset, constant.VolcanoConfigServiceConfigmap,
		constant.VolcanoConfigServiceDefaultNamespace)
	if err != nil {
		zlog.Error("methodGet GetConfigMap failed, %v", err)
		return httputil.GetDefaultServerFailureResponseJson(), http.StatusInternalServerError
	}

	configContent, exists := configmap.Data[constant.VolcanoConfigServiceConfigmapName]
	if !exists {
		zlog.Error("methodGet GetConfigMap failed, content is empty")
		return httputil.GetDefaultServerFailureResponseJson(), http.StatusInternalServerError
	}

	str := "  - name: " + plugin
	if strings.Contains(configContent, str) {
		zlog.Info("%v plugin is enabled", plugin)
		status = "open"
	} else {
		zlog.Info("%v plugin is disabled", plugin)
		status = "close"
	}

	return &httputil.ResponseJson{
		Code: constant.Success,
		Msg:  "success",
		Data: status,
	}, http.StatusOK
}

func methodPut(clientset kubernetes.Interface, plugin string, status string) (*httputil.ResponseJson, int) {
	configmap, err := k8sutil.GetConfigMap(clientset, constant.VolcanoConfigServiceConfigmap,
		constant.VolcanoConfigServiceDefaultNamespace)
	if err != nil {
		zlog.Error("methodPut GetConfigMap failed, %v", err)
		return httputil.GetDefaultServerFailureResponseJson(), http.StatusInternalServerError
	}

	configContent, exists := configmap.Data[constant.VolcanoConfigServiceConfigmapName]
	if !exists {
		zlog.Error("methodPut GetConfigMap failed, content is empty")
		return httputil.GetDefaultServerFailureResponseJson(), http.StatusInternalServerError
	}

	str := "  - name: " + plugin
	if status == "open" {
		if !strings.Contains(configContent, str) {
			configContent = insertPlugin(configContent, str)
		}
	} else if status == "close" {
		if strings.Contains(configContent, str) {
			configContent = removePlugin(configContent, str)
		}
	} else {
		zlog.Error("status is not open or close, %v", err)
		return httputil.GetDefaultServerFailureResponseJson(), http.StatusInternalServerError
	}

	configmap.Data[constant.VolcanoConfigServiceConfigmapName] = configContent
	configmap, err = k8sutil.UpdateConfigMap(clientset, configmap)
	if err != nil {
		zlog.Error("UpdateConfigMap failed, %v", err)
		return httputil.GetDefaultServerFailureResponseJson(), http.StatusInternalServerError
	}

	err = k8sutil.DeleteVolcanoPod(clientset, constant.VolcanoConfigServiceDefaultNamespace)
	if err != nil {
		zlog.Error("delete pod volcano-scheduler failed, %v", err)
		return httputil.GetDefaultServerFailureResponseJson(), http.StatusInternalServerError
	}

	return &httputil.ResponseJson{
		Code: constant.Success,
		Msg:  "success",
	}, http.StatusOK
}

func priorityGet(clientset kubernetes.Interface) (*httputil.ResponseJson, int) {
	status := "open"

	configmap, err := k8sutil.GetConfigMap(clientset, constant.VolcanoConfigServiceConfigmap,
		constant.VolcanoConfigServiceDefaultNamespace)
	if err != nil {
		zlog.Error("priorityGet GetConfigMap failed, %v", err)
		return httputil.GetDefaultServerFailureResponseJson(), http.StatusInternalServerError
	}

	configContent, exists := configmap.Data[constant.VolcanoConfigServiceConfigmapName]
	if !exists {
		zlog.Error("priorityGet GetConfigMap failed, content is empty")
		return httputil.GetDefaultServerFailureResponseJson(), http.StatusInternalServerError
	}

	str1 := "  - name: priority"
	str2 := "  - name: gang"
	str3 := "enablePreemptable: false"
	str4 := "enableJobStarving: false"
	str5 := "preempt"
	if !strings.Contains(configContent, str1) {
		zlog.Info("priority is disabled")
		status = "close"
	}

	if strings.Contains(configContent, str2) {
		if !strings.Contains(configContent, str3) || !strings.Contains(configContent, str4) {
			zlog.Info("priority is disabled")
			status = "close"
		}
	}

	lines := strings.Split(configContent, "\n")
	for _, line := range lines {
		if strings.Contains(line, "actions:") {
			actions := strings.TrimSpace(strings.Split(line, "actions:")[1])
			if strings.Contains(actions, str5) {
				zlog.Info("priority is enabled")
			} else {
				zlog.Info("priority is disabled")
				status = "close"
			}
			break
		}
	}

	return &httputil.ResponseJson{
		Code: constant.Success,
		Msg:  "success",
		Data: status,
	}, http.StatusOK
}

func priorityPut(clientset kubernetes.Interface, status string) (*httputil.ResponseJson, int) {
	configmap, err := k8sutil.GetConfigMap(clientset, constant.VolcanoConfigServiceConfigmap,
		constant.VolcanoConfigServiceDefaultNamespace)
	if err != nil {
		zlog.Error("priorityPut GetConfigMap failed, %v", err)
		return httputil.GetDefaultServerFailureResponseJson(), http.StatusInternalServerError
	}

	configContent, exists := configmap.Data[constant.VolcanoConfigServiceConfigmapName]
	if !exists {
		zlog.Error("priorityPut GetConfigMap failed, content is empty")
		return httputil.GetDefaultServerFailureResponseJson(), http.StatusInternalServerError
	}

	str := "  - name: priority"
	if status == "open" {
		configContent = replaceActions(configContent, `actions: "allocate, backfill, preempt"`)
		if !strings.Contains(configContent, str) {
			configContent = insertPlugin(configContent, str)
		}
		if strings.Contains(configContent, "  - name: gang") {
			configContent = removePlugin(configContent, "  - name: gang")
			configContent = insertGangPlugin(configContent)
		}
	} else if status == "close" {
		if strings.Contains(configContent, str) {
			configContent = removePlugin(configContent, str)
		}
		configContent = replaceActions(configContent, `actions: "enqueue, allocate, backfill"`)
	} else {
		zlog.Error("methodPriority status is not open or close, %v", err)
		return httputil.GetDefaultServerFailureResponseJson(), http.StatusInternalServerError
	}

	configmap.Data[constant.VolcanoConfigServiceConfigmapName] = configContent
	configmap, err = k8sutil.UpdateConfigMap(clientset, configmap)
	if err != nil {
		zlog.Error("priorityPut UpdateConfigMap failed, %v", err)
		return httputil.GetDefaultServerFailureResponseJson(), http.StatusInternalServerError
	}

	err = k8sutil.DeleteVolcanoPod(clientset, constant.VolcanoConfigServiceDefaultNamespace)
	if err != nil {
		zlog.Error("priorityPut delete pod volcano-scheduler failed, %v", err)
		return httputil.GetDefaultServerFailureResponseJson(), http.StatusInternalServerError
	}

	return &httputil.ResponseJson{
		Code: constant.Success,
		Msg:  "success",
	}, http.StatusOK
}

func gangPut(clientset kubernetes.Interface, status string) (*httputil.ResponseJson, int) {
	configmap, err := k8sutil.GetConfigMap(clientset, constant.VolcanoConfigServiceConfigmap,
		constant.VolcanoConfigServiceDefaultNamespace)
	if err != nil {
		zlog.Error("gangPut GetConfigMap failed, %v", err)
		return httputil.GetDefaultServerFailureResponseJson(), http.StatusInternalServerError
	}

	configContent, exists := configmap.Data[constant.VolcanoConfigServiceConfigmapName]
	if !exists {
		zlog.Error("gangPut GetConfigMap failed, content is empty")
		return httputil.GetDefaultServerFailureResponseJson(), http.StatusInternalServerError
	}

	str := "  - name: gang"
	if status == "open" {
		if !strings.Contains(configContent, str) {
			configContent = insertGangPlugin(configContent)
		}
	} else if status == "close" {
		if strings.Contains(configContent, str) {
			configContent = removePlugin(configContent, str)
		}
	} else {
		zlog.Error("gangPut status is not open or close, %v", err)
		return httputil.GetDefaultServerFailureResponseJson(), http.StatusInternalServerError
	}

	configmap.Data[constant.VolcanoConfigServiceConfigmapName] = configContent
	configmap, err = k8sutil.UpdateConfigMap(clientset, configmap)
	if err != nil {
		zlog.Error("gangPut UpdateConfigMap failed, %v", err)
		return httputil.GetDefaultServerFailureResponseJson(), http.StatusInternalServerError
	}

	err = k8sutil.DeleteVolcanoPod(clientset, constant.VolcanoConfigServiceDefaultNamespace)
	if err != nil {
		zlog.Error("gangPut delete pod volcano-scheduler failed, %v", err)
		return httputil.GetDefaultServerFailureResponseJson(), http.StatusInternalServerError
	}

	return &httputil.ResponseJson{
		Code: constant.Success,
		Msg:  "success",
	}, http.StatusOK
}

func insertPlugin(content string, plugin string) string {
	lines := strings.Split(content, "\n")
	var newLines []string
	pluginInserted := false

	for _, line := range lines {
		newLines = append(newLines, line)
		if strings.TrimSpace(line) == "- plugins:" && !pluginInserted {
			newLines = append(newLines, plugin)
			pluginInserted = true
		}
	}
	return strings.Join(newLines, "\n")
}

func insertGangPlugin(content string) string {
	lines := strings.Split(content, "\n")
	var newLines []string
	pluginInserted := false

	for _, line := range lines {
		newLines = append(newLines, line)
		if strings.TrimSpace(line) == "- plugins:" && !pluginInserted {
			newLines = append(newLines, "  - name: gang")
			newLines = append(newLines, "    enablePreemptable: false")
			newLines = append(newLines, "    enableJobStarving: false")
			pluginInserted = true
		}
	}

	return strings.Join(newLines, "\n")
}

func removePlugin(content string, plugin string) string {
	lines := strings.Split(content, "\n")
	var newLines []string
	skip := false
	for _, line := range lines {
		if strings.Contains(line, plugin) {
			skip = true
			continue
		}
		if skip && (strings.HasPrefix(line, "  - name:") || strings.HasPrefix(line,
			"- plugins:") || strings.HasPrefix(line, "BinaryData")) {
			skip = false
		}
		if !skip {
			newLines = append(newLines, line)
		}
	}
	return strings.Join(newLines, "\n")
}

func replaceActions(configContent string, actions string) string {
	lines := strings.Split(configContent, "\n")
	var updatedLines []string
	for _, line := range lines {
		if strings.HasPrefix(strings.TrimSpace(line), "actions") {
			updatedLines = append(updatedLines, actions)
		} else {
			updatedLines = append(updatedLines, line)
		}
	}
	return strings.Join(updatedLines, "\n")
}
