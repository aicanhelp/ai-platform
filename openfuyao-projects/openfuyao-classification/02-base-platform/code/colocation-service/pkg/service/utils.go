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

package service

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/thoas/go-funk"
	"golang.org/x/mod/semver"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/util/retry"
	"k8s.io/klog/v2"

	colocationv1 "openfuyao.com/colocation-service/pkg/apis/colocation/v1"
)

const (
	// 校验阈值常量
	RubikThresholdMin   = 1
	RubikThresholdMax   = 99
	PercentageMin       = 0
	PercentageMax       = 100
	VolcanoThresholdMax = 100.0

	// DynCache 相关常量
	CachePercentMin = 10
	CachePercentMax = 100

	// QuotaTurbo 相关常量
	QuotaTurboMin = 0
	QuotaTurboMax = 100

	// PSI 相关常量
	PSIThresholdMin = 5.0
	PSIThresholdMax = 100.0
)

// 修改colocation options中的混部节点列表， 把colonode 列表里的节点按operation操作更新（实际上目前列表里每次只有一个节点）
func (server *Server) modifyColocationConfig(coloNode []string, operation string) error {
	// 获取混部配置config
	coloConfig, err := server.client.CoreV1().ConfigMaps(colocationConfigMapNamespace).Get(
		context.TODO(), ColocationConfigMapName, metav1.GetOptions{})
	if err != nil {
		klog.Error(err)
		return err
	}

	var newColocation ColocationOptions
	// 解析colocation-config配置
	if err := json.Unmarshal([]byte(coloConfig.Data["colocation-options"]), &newColocation); err != nil {
		klog.Error(err, "failed to parse configmap colocation options")
		return err
	}

	switch operation {
	case "add":
		newColocation = handleAddOperation(coloNode, newColocation)
	case "delete":
		newColocation = handleDeleteOperation(coloNode, newColocation)
	default:
		return fmt.Errorf("operation must be either 'add' or 'delete'")
	}

	updatedColocationConfig, err1 := json.Marshal(newColocation)
	if err1 != nil {
		klog.Error(err1)
		return err1
	}

	coloConfig.Data["colocation-options"] = string(updatedColocationConfig)

	if err := updateConfigMapwithRetry(server.client, coloConfig, colocationConfigMapNamespace); err != nil {
		return err
	}
	klog.Info("updated configmap: colocation-config")

	return nil
}

func handleAddOperation(coloNode []string, options ColocationOptions) ColocationOptions {
	options.Enable = true
	for _, node := range coloNode {
		if !funk.Contains(options.NodesConfig, NodeConfig{Name: node}) {
			options.NodesConfig = append(options.NodesConfig, NodeConfig{Name: node})
		}
	}
	klog.Infof("new colonodes: %v", options)
	return options
}

func handleDeleteOperation(coloNode []string, options ColocationOptions) ColocationOptions {
	var updatedNodes []NodeConfig
	for _, node := range options.NodesConfig {
		if !funk.Contains(coloNode, node.Name) {
			klog.Infof("keeping node %s in colo", node.Name)
			updatedNodes = append(updatedNodes, node)
		}
	}

	options.NodesConfig = updatedNodes
	if len(updatedNodes) == 0 {
		options.Enable = false
	}

	klog.Infof("new colonodes: %v", options)
	return options
}

// 全量式colocation config - nodeinfo, rubik, volcanoScheduler
func (server *Server) coverColocationConfig(newConfig ColocationConfigRequest) error {
	klog.Info("IN POST: handling cover full configmap")

	// 更新节点混部状态
	if err := server.updateNodeColocationStatus(newConfig.NodeInfo); err != nil {
		return err
	}

	// 更新 Rubik 配置
	if err := server.coverColoConfigRubik(newConfig); err != nil {
		return err
	}

	if err := server.coverColoConfigVolcanoScheduler(newConfig); err != nil {
		return err
	}

	return nil
}

// 更新节点混部状态
func (server *Server) updateNodeColocationStatus(nodeInfos []NodeAttr) error {
	var updateErrors []error

	for _, nodeInfo := range nodeInfos {
		node, err := server.client.CoreV1().Nodes().Get(context.TODO(), nodeInfo.Name, metav1.GetOptions{})
		if err != nil {
			klog.Warningf("Failed to get node %s: %v", nodeInfo.Name, err)
			continue
		}

		// 检查当前状态是否需要更新
		currentIsColocated := isColocationNode(node)
		if currentIsColocated == nodeInfo.IsColocated {
			continue
		}

		// 更新节点标签
		if nodeInfo.IsColocated {
			if err := addNodelabels(server.client, node); err != nil {
				klog.Errorf("Failed to add colocation label to node %s: %v", nodeInfo.Name, err)
				updateErrors = append(updateErrors, fmt.Errorf("node %s: %w", nodeInfo.Name, err))
			}
		} else {
			if err := deleteNodelabels(server.client, node); err != nil {
				klog.Errorf("Failed to remove colocation label from node %s: %v", nodeInfo.Name, err)
				updateErrors = append(updateErrors, fmt.Errorf("node %s: %w", nodeInfo.Name, err))
			}
		}
	}

	// 汇总错误
	if len(updateErrors) > 0 {
		return fmt.Errorf("failed to update %d nodes: %v", len(updateErrors), updateErrors)
	}

	return nil
}

func (server *Server) coverColoConfigRubik(newConfig ColocationConfigRequest) error {

	// 校验 CPU 和 Memory 阈值
	if newConfig.RubikConfig.Eviction.Enabled {
		if newConfig.RubikConfig.Eviction.CPUEvict.Threshold < RubikThresholdMin ||
			newConfig.RubikConfig.Eviction.CPUEvict.Threshold > RubikThresholdMax {
			return fmt.Errorf("rubik CPU threshold must be between %d and %d", RubikThresholdMin, RubikThresholdMax)
		}
		if newConfig.RubikConfig.Eviction.MemoryEvict.Threshold < RubikThresholdMin ||
			newConfig.RubikConfig.Eviction.MemoryEvict.Threshold > RubikThresholdMax {
			return fmt.Errorf("rubik Memory threshold must be between %d and %d", RubikThresholdMin, RubikThresholdMax)
		}
	}

	// 校验高级特性配置
	if err := server.validateRubikHighFeatures(newConfig.RubikConfig); err != nil {
		return err
	}

	// 获取混部配置config
	coloConfig, err := server.client.CoreV1().ConfigMaps(colocationConfigMapNamespace).Get(
		context.TODO(), ColocationConfigMapName, metav1.GetOptions{})
	if err != nil {
		klog.Error(err)
		return err
	}

	updatedRubikConfig, err1 := json.MarshalIndent(newConfig.RubikConfig, "", "    ")
	if err1 != nil {
		klog.Error(err1)
		return err1
	}

	coloConfig.Data["rubik-options"] = string(updatedRubikConfig)

	if err := updateConfigMapwithRetry(server.client, coloConfig, colocationConfigMapNamespace); err != nil {
		return err
	}

	return nil
}

// validateRubikHighFeatures 校验 Rubik 高级特性配置
func (server *Server) validateRubikHighFeatures(config RubikConfig) error {
	// 校验 QuotaTurbo 配置
	if config.QuotaTurbo != nil && config.QuotaTurbo.Enable {
		if config.QuotaTurbo.HighWaterMark < QuotaTurboMin || config.QuotaTurbo.HighWaterMark >= config.QuotaTurbo.AlarmWaterMark {
			return fmt.Errorf("quotaTurbo.highWaterMark must be between %d and %d (exclusive)", QuotaTurboMin, config.QuotaTurbo.AlarmWaterMark)
		}
		if config.QuotaTurbo.AlarmWaterMark <= config.QuotaTurbo.HighWaterMark || config.QuotaTurbo.AlarmWaterMark > QuotaTurboMax {
			return fmt.Errorf("quotaTurbo.alarmWaterMark must be between %d (exclusive) and %d", config.QuotaTurbo.HighWaterMark, QuotaTurboMax)
		}
	}

	// 校验 DynCache 配置
	if config.DynCache != nil && config.DynCache.Enable {
		if config.DynCache.L3Percent != nil {
			if err := validateCachePercent(*config.DynCache.L3Percent, "l3Percent"); err != nil {
				return err
			}
		}
		if config.DynCache.MemBandPercent != nil {
			if err := validateCachePercent(*config.DynCache.MemBandPercent, "memBandPercent"); err != nil {
				return err
			}
		}
	}

	// 校验 PSI 配置
	if config.PSI != nil && config.PSI.Enable {
		if config.PSI.Avg10Threshold < PSIThresholdMin || config.PSI.Avg10Threshold > PSIThresholdMax {
			return fmt.Errorf("psi.avg10Threshold must be between %.1f and %.1f", PSIThresholdMin, PSIThresholdMax)
		}
		validResources := map[string]bool{"cpu": true, "memory": true, "io": true}
		for _, resource := range config.PSI.Resource {
			if !validResources[resource] {
				return fmt.Errorf("invalid psi resource type: %s", resource)
			}
		}
		if len(config.PSI.Resource) == 0 {
			return fmt.Errorf("psi.resource cannot be empty when PSI is enabled")
		}
	}

	return nil
}

// validateCachePercent 校验缓存百分比配置
func validateCachePercent(config CachePercentConfig, name string) error {
	// 校验单个值的范围
	if config.Low < CachePercentMin || config.Low > CachePercentMax {
		return fmt.Errorf("%s.low must be between %d and %d", name, CachePercentMin, CachePercentMax)
	}
	if config.Mid < config.Low || config.Mid > CachePercentMax {
		return fmt.Errorf("%s.mid must be between %d and %d", name, config.Low, CachePercentMax)
	}
	if config.High < config.Mid || config.High > CachePercentMax {
		return fmt.Errorf("%s.high must be between %d and %d", name, config.Mid, CachePercentMax)
	}
	return nil
}

func (server *Server) coverColoConfigVolcanoScheduler(newConfig ColocationConfigRequest) error {
	// 校验 CPU 和 Memory 阈值
	if newConfig.VolcanoSchedulerConfig.UsagePlugin.Enabled {
		if newConfig.VolcanoSchedulerConfig.UsagePlugin.UsageThreshold.Enabled {
			if newConfig.VolcanoSchedulerConfig.UsagePlugin.UsageThreshold.CPUThreshold < PercentageMin ||
				newConfig.VolcanoSchedulerConfig.UsagePlugin.UsageThreshold.CPUThreshold > VolcanoThresholdMax {
				return fmt.Errorf("volcano scheduler CPU threshold must be between %d and %.0f", PercentageMin, VolcanoThresholdMax)
			}
			if newConfig.VolcanoSchedulerConfig.UsagePlugin.UsageThreshold.MemoryThreshold < PercentageMin ||
				newConfig.VolcanoSchedulerConfig.UsagePlugin.UsageThreshold.MemoryThreshold > VolcanoThresholdMax {
				return fmt.Errorf("volcano scheduler memory threshold must be between %d and %.0f", PercentageMin, VolcanoThresholdMax)
			}
		}

	}

	// 获取混部配置config
	coloConfig, err := server.client.CoreV1().ConfigMaps(colocationConfigMapNamespace).Get(
		context.TODO(), ColocationConfigMapName, metav1.GetOptions{})
	if err != nil {
		klog.Error(err)
		return err
	}

	updatedVolcanoConfig, err1 := json.MarshalIndent(newConfig.VolcanoSchedulerConfig, "", "    ")
	if err1 != nil {
		klog.Error(err1)
		return err1
	}

	coloConfig.Data["volcano-scheduler-options"] = string(updatedVolcanoConfig)

	if err := updateConfigMapwithRetry(server.client, coloConfig, colocationConfigMapNamespace); err != nil {
		return err
	}

	return nil
}

func updateConfigMapwithRetry(client clientset.Interface, cm *corev1.ConfigMap, configMapNamespace string) error {
	err := retry.RetryOnConflict(retry.DefaultRetry, func() error {
		_, err := client.CoreV1().ConfigMaps(configMapNamespace).Update(
			context.TODO(), cm, metav1.UpdateOptions{})
		if err != nil {
			return err
		}
		return nil
	})

	if err != nil {
		klog.Error(err, "failed to update configmap")
		return err
	}
	return nil
}

func getColoNodesInColocationConfig(cm *corev1.ConfigMap) ([]string, error) {
	var newColocation ColocationOptions

	// 解析colocation-config配置
	if err := json.Unmarshal([]byte(cm.Data["colocation-options"]), &newColocation); err != nil {
		klog.Error(err, "failed to parse configmap colocation options")
		return nil, err
	}

	if !newColocation.Enable {
		klog.Info("colocation is disabled")
		return nil, nil
	}
	var coloNode []string
	for _, node := range newColocation.NodesConfig {
		coloNode = append(coloNode, node.Name)
	}

	return coloNode, nil
}

func validateNode(n *corev1.Node) bool {
	isOsSupported := true
	// to do:  判断OSImage: openEuler 20.03 (LTS)、openEuler 22.03 (LTS)、 openEuler 24.03 (LTS)
	osImage := n.Status.NodeInfo.OSImage
	// 判断内核版本kernelVersion: 5.10.0-109.0.136.oe2203.x86_64
	baseKernelVersion := "v4.19.90"
	originalKernelVersion := n.Status.NodeInfo.KernelVersion
	kernelVersion := strings.Split(originalKernelVersion, "-")[0]
	kernelVersion = "v" + kernelVersion
	if semver.Compare(kernelVersion, baseKernelVersion) < 0 {
		isOsSupported = false
	}

	if !isOsSupported {
		klog.Infof("node check: can't set to colocation node or overcommit node:\n"+
			"node name: %s, node kernelVersion: %s, node os image: %s, minimal kernelVersion is v4.19.90",
			n.Name, kernelVersion, osImage)
		return false
	}

	return true
}

func addNodelabels(c clientset.Interface, node *corev1.Node) error {
	// 清理旧标签（仅混部标签）
	delete(node.Labels, colocationv1.ColocationNodeLabel)

	// 验证节点是否支持混部
	if ok := validateNode(node); !ok {
		return fmt.Errorf("node %s does not support colocation", node.Name)
	}

	// 添加混部标签
	node.Labels[colocationv1.ColocationNodeLabel] = "true"

	// 更新节点
	_, err := c.CoreV1().Nodes().Update(context.TODO(), node, metav1.UpdateOptions{})
	return err
}

func deleteNodelabels(c clientset.Interface, node *corev1.Node) error {
	// 清理旧标签（仅混部标签）
	delete(node.Labels, colocationv1.ColocationNodeLabel)

	// 更新节点
	_, err := c.CoreV1().Nodes().Update(context.TODO(), node, metav1.UpdateOptions{})
	if err != nil {
		klog.Errorf("error when updating node %s", node.Name)
		return err
	}
	// 删除超卖资源
	return deleteOversoldResources(c, node)
}

func deleteOversoldResources(c clientset.Interface, node *corev1.Node) error {
	delete(node.Status.Allocatable, colocationv1.OversubscriptionCPUResourceAnn)
	delete(node.Status.Allocatable, colocationv1.OversubscriptionMemResourceAnn)
	delete(node.Status.Capacity, colocationv1.OversubscriptionCPUResourceAnn)
	delete(node.Status.Capacity, colocationv1.OversubscriptionMemResourceAnn)
	_, err := c.CoreV1().Nodes().UpdateStatus(context.TODO(), node, metav1.UpdateOptions{})
	if err != nil {
		return err
	}
	klog.Infof("updated node allocatable: %v", node.Status.Allocatable)
	return nil
}

func getNodeAddresses(node corev1.Node) string {
	for _, address := range node.Status.Addresses {
		if address.Type == corev1.NodeInternalIP {
			return address.Address
		}
	}
	return ""
}

func getNodeStatus(node corev1.Node) string {
	nodeStatus := "inactive"
	for _, condition := range node.Status.Conditions {
		if condition.Type == "Ready" && condition.Status == "True" {
			nodeStatus = "active"
			break
		}
	}
	return nodeStatus
}
