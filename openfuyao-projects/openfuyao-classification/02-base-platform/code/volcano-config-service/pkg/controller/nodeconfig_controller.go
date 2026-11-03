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

// Package controller
package controller

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"strings"
	"time"

	"gopkg.in/yaml.v2"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/util/retry"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller"

	"volcano-config-service/pkg/zlog"
)

// NodeConfigController 结构体定义
type NodeConfigController struct {
	client.Client
	Scheme *runtime.Scheme

	// APIReader 直连 apiserver（绕过 informer cache）。DaemonSet 模式下
	// 每个节点一个实例，各实例 cache 的 resourceVersion 滞后，更新 CR 前
	// 必须读 apiserver 最新版本，否则冲突重试无效（issue 92）。
	APIReader client.Reader
}

// ConfigRequest 定义了发送给 agent 的配置请求结构
type ConfigRequest struct {
	EnableStaticCPUManager bool   `json:"enableStaticCPUManager"`
	EnableNUMADistance     bool   `json:"enableNUMADistance"`
	KubeletConfigPath      string `json:"kubeletConfigPath,omitempty"`
}

// KubeletConfig 定义了要修改的k8s配置内容
type KubeletConfig struct {
	Kind                         string            `yaml:"kind"`
	ApiVersion                   string            `yaml:"apiVersion"`
	CpuManagerPolicy             string            `yaml:"cpuManagerPolicy"`
	FeatureGates                 map[string]bool   `yaml:"featureGates"`
	TopologyManagerPolicyOptions map[string]string `yaml:"topologyManagerPolicyOptions"`
	KubeReserved                 map[string]string `yaml:"kubeReserved"`
	SystemReserved               map[string]string `yaml:"systemReserved"`
}

const (
	// RequeueTimeMinutes 定义了重新入队的时间间隔（分钟）
	RequeueTimeMinutes = 5

	// Kubernetes Feature Gate 名称前缀
	featurePrefix = "TopologyManagerPolicy"

	// FeatureTopologyManagerPolicyOptions 定义了拓扑管理器策略选项的特性门控名称
	FeatureTopologyManagerPolicyOptions = featurePrefix + "Options"
	// FeatureTopologyManagerPolicyBetaOptions 定义了拓扑管理器策略Beta选项的特性门控名称
	FeatureTopologyManagerPolicyBetaOptions = featurePrefix + "BetaOptions"
	// FeatureTopologyManagerPolicyAlphaOptions 定义了拓扑管理器策略Alpha选项的特性门控名称
	FeatureTopologyManagerPolicyAlphaOptions = featurePrefix + "AlphaOptions"
)

// SetupWithManager 设置 controller 与 manager
func (r *NodeConfigController) SetupWithManager(mgr ctrl.Manager) error {
	// 设置控制器监听 NodeConfig 资源
	return ctrl.NewControllerManagedBy(mgr).
		For(&NodeConfig{}).
		WithOptions(controller.Options{
			MaxConcurrentReconciles: 1,
		}).
		Complete(r)
}

// Reconcile 处理NodeConfig资源的调谐
func (r *NodeConfigController) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	// 获取当前节点名称
	nodeName := os.Getenv("NODE_NAME")
	if nodeName == "" {
		return ctrl.Result{}, fmt.Errorf("NODE_NAME environment variable not set")
	}

	// 获取 NodeConfig 资源
	nodeConfig := &NodeConfig{}
	if err := r.Get(ctx, req.NamespacedName, nodeConfig); err != nil {
		zlog.FormatInfo("get nodeConfig")
		if !errors.IsNotFound(err) {
			return ctrl.Result{}, err
		}
		return ctrl.Result{Requeue: true}, nil
	}

	// 查找当前节点的配置
	nodeConf := r.findNodeConfiguration(nodeConfig, nodeName)

	// 对存量数据按 nodeName 去重（保留首次出现项）。
	// 历史版本可能已写入重复节点（issue 91），此处自愈清理，
	// 避免 findNodeConfiguration 只命中首条、API 汇总读到冲突状态。
	if deduplicated := deduplicateNodeConfigurations(nodeConfig.Spec.Nodes); deduplicated != nil {
		// 必须将去重结果回写到对象再提交，否则 Update 提交的
		// 仍然是未修改的旧数据，去重等同于空操作
		nodeConfig.Spec.Nodes = deduplicated
		zlog.FormatInfo("检测到 NodeConfig 中存在重复节点配置，已去重")
		if err := r.updateSpecWithRetry(ctx, nodeConfig); err != nil {
			zlog.Error(err, "Failed to deduplicate NodeConfig nodes")
			return ctrl.Result{}, err
		}
		// 去重后重新查找当前节点配置
		nodeConf = r.findNodeConfiguration(nodeConfig, nodeName)
	}

	// 如果找不到当前节点的配置，添加新的配置
	if nodeConf == nil {
		return r.handleMissingNodeConfig(ctx, nodeConfig, nodeName)
	}

	// 检查并处理配置更新
	return r.handleNodeConfigUpdate(ctx, nodeConfig, nodeName, nodeConf)
}

// findNodeConfiguration 在NodeConfig中查找特定节点的配置
func (r *NodeConfigController) findNodeConfiguration(nodeConfig *NodeConfig, nodeName string) *NodeConfiguration {
	for _, nc := range nodeConfig.Spec.Nodes {
		if nc.NodeName == nodeName {
			return &nc
		}
	}
	return nil
}

// findNodeConfigurationByName 在节点配置列表中按 nodeName 查找配置（纯函数版本）
func findNodeConfigurationByName(nodes []NodeConfiguration, nodeName string) *NodeConfiguration {
	for i := range nodes {
		if nodes[i].NodeName == nodeName {
			return &nodes[i]
		}
	}
	return nil
}

// resolveDesiredNodeFlags 新增节点时计算目标开关值：
// 若现有节点配置一致，则继承一致值；否则回退到当前节点实测值。
func resolveDesiredNodeFlags(existingNodes []NodeConfiguration,
	fallbackStaticCPUEnabled, fallbackNUMAEnabled bool) (bool, bool) {
	desiredStaticCPUEnabled := fallbackStaticCPUEnabled
	desiredNUMAEnabled := fallbackNUMAEnabled

	if len(existingNodes) == 0 {
		return desiredStaticCPUEnabled, desiredNUMAEnabled
	}

	uniformStatic := true
	staticValue := existingNodes[0].EnableStaticCPUManager
	for i := 1; i < len(existingNodes); i++ {
		if existingNodes[i].EnableStaticCPUManager != staticValue {
			uniformStatic = false
			break
		}
	}
	if uniformStatic {
		desiredStaticCPUEnabled = staticValue
	}

	uniformNUMA := true
	numaValue := existingNodes[0].EnableNUMADistance
	for i := 1; i < len(existingNodes); i++ {
		if existingNodes[i].EnableNUMADistance != numaValue {
			uniformNUMA = false
			break
		}
	}
	if uniformNUMA {
		desiredNUMAEnabled = numaValue
	}

	return desiredStaticCPUEnabled, desiredNUMAEnabled
}

// handleMissingNodeConfig 处理节点配置不存在的情况
func (r *NodeConfigController) handleMissingNodeConfig(ctx context.Context,
	nodeConfig *NodeConfig, nodeName string) (ctrl.Result, error) {

	currentConfig, err := r.readKubeletConfig(KubeletConfigPath)
	if err != nil {
		zlog.Error(err, "Failed to read kubelet config")
		return ctrl.Result{}, err
	}

	// 解析当前配置
	currentStaticCPUEnabled, currentNUMAEnabled := r.parseCurrentConfiguration(currentConfig)

	// 使用 RetryOnConflict 避免多节点并发写入同一 NodeConfig 时冲突
	if err := retry.RetryOnConflict(retry.DefaultRetry, func() error {
		latest, err := r.getLatestNodeConfig(ctx, nodeConfig)
		if err != nil {
			return err
		}

		// 检查节点是否已存在（可能被其他实例或 InitializeNodeConfig 添加）
		for _, nc := range latest.Spec.Nodes {
			if nc.NodeName == nodeName {
				zlog.FormatInfo("节点 %s 已存在于 NodeConfig 中", nodeName)
				return nil
			}
		}

		desiredStaticCPUEnabled, desiredNUMAEnabled := resolveDesiredNodeFlags(
			latest.Spec.Nodes, currentStaticCPUEnabled, currentNUMAEnabled)

		// 添加新的节点配置
		latest.Spec.Nodes = append(latest.Spec.Nodes, NodeConfiguration{
			NodeName:               nodeName,
			EnableStaticCPUManager: desiredStaticCPUEnabled,
			EnableNUMADistance:     desiredNUMAEnabled,
			KubeletConfigPath:      KubeletConfigPath,
		})

		// 更新 NodeConfig CR
		if err := r.Update(ctx, latest); err != nil {
			zlog.FormatError("更新 NodeConfig 失败: %v", err)
			return err
		}

		return nil
	}); err != nil {
		zlog.Error(err, "Failed to update NodeConfig")
		return ctrl.Result{}, err
	}

	return ctrl.Result{Requeue: true}, nil
}

// handleNodeConfigUpdate 处理节点配置更新
func (r *NodeConfigController) handleNodeConfigUpdate(ctx context.Context,
	nodeConfig *NodeConfig, nodeName string, nodeConf *NodeConfiguration) (ctrl.Result, error) {

	// 读取当前配置
	currentConfig, err := r.readKubeletConfig(KubeletConfigPath)
	if err != nil {
		zlog.Error(err, "Failed to read kubelet config")
		return ctrl.Result{}, err
	}

	// 解析当前配置
	currentStaticCPUEnabled, currentNUMAEnabled := r.parseCurrentConfiguration(currentConfig)

	// 检查是否需要更新配置
	if nodeConf.EnableStaticCPUManager != currentStaticCPUEnabled ||
		nodeConf.EnableNUMADistance != currentNUMAEnabled {

		return r.updateConfiguration(ctx, nodeConfig, nodeName, nodeConf, currentConfig)
	}

	// 配置已生效但 status 可能因并发写丢失而不一致，自愈补写
	return r.healNodeStatus(ctx, nodeConfig, nodeName, nodeConf)
}

// getLatestNodeConfig 从 apiserver 读取最新的 NodeConfig（冲突重试专用）
func (r *NodeConfigController) getLatestNodeConfig(ctx context.Context,
	nodeConfig *NodeConfig) (*NodeConfig, error) {

	latest := &NodeConfig{}
	key := client.ObjectKey{
		Namespace: nodeConfig.Namespace,
		Name:      nodeConfig.Name,
	}
	if r.APIReader != nil {
		return latest, r.APIReader.Get(ctx, key, latest)
	}
	// 未设置 APIReader 时（测试）退回 cache client
	return latest, r.Get(ctx, key, latest)
}

// findNodeStatus 按 nodeName 查找状态条目
func findNodeStatus(nodeConfig *NodeConfig, nodeName string) *NodeStatus {
	for i := range nodeConfig.Status.NodeStatuses {
		if nodeConfig.Status.NodeStatuses[i].NodeName == nodeName {
			return &nodeConfig.Status.NodeStatuses[i]
		}
	}
	return nil
}

// healNodeStatus 自愈：kubelet 配置已生效但 status 未反映时补写 Configured。
// 状态已正确时跳过写操作，避免周期性 Requeue 造成写放大。
func (r *NodeConfigController) healNodeStatus(ctx context.Context,
	nodeConfig *NodeConfig, nodeName string, nodeConf *NodeConfiguration) (ctrl.Result, error) {

	requeue := ctrl.Result{RequeueAfter: time.Minute * RequeueTimeMinutes}

	if status := findNodeStatus(nodeConfig, nodeName); status != nil &&
		status.State == NodeConfigStateConfigured &&
		status.StaticCPUManagerEnabled == nodeConf.EnableStaticCPUManager &&
		status.NUMADistanceEnabled == nodeConf.EnableNUMADistance {
		return requeue, nil
	}

	zlog.FormatInfo("node %s config applied but status inconsistent, healing", nodeName)
	if err := r.updateNodeStatusWithRetry(ctx, nodeConfig, nodeName,
		NodeConfigStateConfigured, ""); err != nil {
		zlog.Error(err, "Failed to heal node status")
		return ctrl.Result{}, err
	}

	return requeue, nil
}

// parseCurrentConfiguration 解析kubelet配置中的策略设置
func (r *NodeConfigController) parseCurrentConfiguration(
	currentConfig map[string]interface{}) (bool, bool) {

	// 解析CPU管理策略
	currentStaticCPUEnabled := false
	if cpuManagerPolicy, ok := currentConfig["cpuManagerPolicy"]; ok {
		if policyStr, isString := cpuManagerPolicy.(string); isString {
			currentStaticCPUEnabled = policyStr == "static"
		}
	}

	// 解析NUMA距离策略
	currentNUMAEnabled := false
	if topologyOptsRaw, exists := currentConfig["topologyManagerPolicyOptions"]; exists {
		currentNUMAEnabled = r.parseNUMADistanceOption(topologyOptsRaw)
	}

	return currentStaticCPUEnabled, currentNUMAEnabled
}

// parseNUMADistanceOption 解析NUMA距离配置选项
func (r *NodeConfigController) parseNUMADistanceOption(topologyOptsRaw interface{}) bool {
	switch opts := topologyOptsRaw.(type) {
	case map[interface{}]interface{}:
		if numaValue, ok := opts["prefer-closest-numa-nodes"]; ok {
			return r.parseNUMABoolValue(numaValue)
		}
	case map[string]interface{}:
		if numaValue, ok := opts["prefer-closest-numa-nodes"]; ok {
			return r.parseNUMABoolValue(numaValue)
		}
	default:
		return false
	}
	return false
}

// parseNUMABoolValue 解析NUMA布尔值配置
func (r *NodeConfigController) parseNUMABoolValue(value interface{}) bool {
	switch v := value.(type) {
	case string:
		return strings.EqualFold(v, "true")
	case bool:
		return v
	default:
		return false
	}
}

// updateConfiguration 执行配置更新流程
func (r *NodeConfigController) updateConfiguration(ctx context.Context,
	nodeConfig *NodeConfig, nodeName string, nodeConf *NodeConfiguration,
	currentConfig map[string]interface{}) (ctrl.Result, error) {

	// 更新状态为配置中
	if err := r.updateNodeStatusWithRetry(ctx, nodeConfig, nodeName,
		NodeConfigStateConfiguring, ""); err != nil {
		zlog.Error(err, "Failed to update node status")
		return ctrl.Result{}, err
	}

	// 更新配置
	if err := r.updateKubeletConfig(currentConfig, nodeConf); err != nil {
		zlog.Error(err, "Failed to update kubelet config")
		if updateErr := r.updateNodeStatusWithRetry(ctx, nodeConfig, nodeName,
			NodeConfigStateFailed, err.Error()); updateErr != nil {
			zlog.Error(updateErr, "Failed to update failure status")
		}
		return ctrl.Result{}, err
	}

	// 更新成功状态
	if err := r.updateNodeStatusWithRetry(ctx, nodeConfig, nodeName,
		NodeConfigStateConfigured, ""); err != nil {
		zlog.Error(err, "Failed to update success status")
		return ctrl.Result{}, err
	}

	return ctrl.Result{RequeueAfter: time.Minute * RequeueTimeMinutes}, nil
}

func (r *NodeConfigController) readKubeletConfig(path string) (map[string]interface{}, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read kubelet config file: %v", err)
	}

	config := make(map[string]interface{})
	if err := yaml.Unmarshal(data, &config); err != nil {
		return nil, fmt.Errorf("failed to unmarshal kubelet config: %v", err)
	}

	return config, nil
}

// updateKubeletConfig 更新kubelet配置
func (r *NodeConfigController) updateKubeletConfig(config map[string]interface{}, nodeConf *NodeConfiguration) error {
	// 跟踪配置是否被修改
	modified := false

	// 获取当前CPU管理策略
	currentPolicy, _ := config["cpuManagerPolicy"].(string)

	// 更新CPU管理策略和NUMA距离设置
	modifiedCPU := r.updateCPUManagerPolicy(config, nodeConf, currentPolicy)
	modifiedNUMA := r.updateNUMADistanceSettings(config, nodeConf, currentPolicy)
	modified = modifiedCPU || modifiedNUMA

	// 确保基本配置字段存在
	r.ensureBasicConfigFields(config)

	// 如果配置有修改，应用更改
	if modified {
		return r.applyConfigChanges(nodeConf, config)
	}

	return nil
}

// updateCPUManagerPolicy 更新CPU管理策略
func (r *NodeConfigController) updateCPUManagerPolicy(
	config map[string]interface{}, nodeConf *NodeConfiguration, currentPolicy string) bool {
	if config == nil {
		return false
	}

	modified := false

	// 启用静态CPU管理
	if nodeConf.EnableStaticCPUManager && currentPolicy != "static" {
		config["cpuManagerPolicy"] = "static"
		r.ensureResourceReservations(config)
		modified = true
	} else if !nodeConf.EnableStaticCPUManager && currentPolicy == "static" {
		// 禁用静态CPU管理
		config["cpuManagerPolicy"] = "none"
		modified = true
	} else {
		return modified
	}

	return modified
}

// ensureResourceReservations 确保资源预留配置存在
func (r *NodeConfigController) ensureResourceReservations(config map[string]interface{}) bool {
	if config == nil {
		return false
	}

	modified := false

	// 检查并添加系统预留资源配置
	if _, ok := config["systemReserved"].(map[string]interface{}); !ok {
		config["systemReserved"] = map[string]interface{}{
			"cpu": "0.5",
		}
		modified = true
	}

	// 检查并添加kube预留资源配置
	if _, ok := config["kubeReserved"].(map[string]interface{}); !ok {
		config["kubeReserved"] = map[string]interface{}{
			"cpu": "0.5",
		}
		modified = true
	}

	return modified
}

// updateNUMADistanceSettings 更新NUMA距离设置
func (r *NodeConfigController) updateNUMADistanceSettings(
	config map[string]interface{}, nodeConf *NodeConfiguration, currentPolicy string) bool {

	modified := false

	if nodeConf.EnableNUMADistance {
		modified = r.enableNUMADistance(config, currentPolicy)
	} else {
		modified = r.disableNUMADistance(config)
	}

	return modified
}

// enableNUMADistance 启用NUMA距离功能
func (r *NodeConfigController) enableNUMADistance(config map[string]interface{}, currentPolicy string) bool {
	if config == nil {
		return false
	}

	modified := false

	// 设置CPU Manager为static
	if currentPolicy != "static" {
		config["cpuManagerPolicy"] = "static"
		r.ensureResourceReservations(config)
		modified = true
	}

	// 更新特性门控
	modifiedFeatures := r.updateFeatureGates(config)

	// 更新拓扑管理器策略选项
	modifiedTopology := r.updateTopologyOptions(config, true)

	return modified || modifiedFeatures || modifiedTopology
}

// disableNUMADistance 禁用NUMA距离功能
func (r *NodeConfigController) disableNUMADistance(config map[string]interface{}) bool {
	modified := false

	if topologyOptsRaw, exists := config["topologyManagerPolicyOptions"]; exists {
		switch topologyOpts := topologyOptsRaw.(type) {
		case map[interface{}]interface{}:
			if _, exists := topologyOpts["prefer-closest-numa-nodes"]; exists {
				topologyOpts["prefer-closest-numa-nodes"] = "false"
				modified = true
			}
		case map[string]interface{}:
			if _, exists := topologyOpts["prefer-closest-numa-nodes"]; exists {
				topologyOpts["prefer-closest-numa-nodes"] = "false"
				modified = true
			}
		default:
			return modified
		}
	}

	return modified
}

// updateFeatureGates 更新特性门控设置
func (r *NodeConfigController) updateFeatureGates(config map[string]interface{}) bool {
	if config == nil {
		return false
	}

	featureGates, ok := config["featureGates"].(map[string]interface{})
	if !ok {
		featureGates = make(map[string]interface{})
		config["featureGates"] = featureGates
	}

	featureGates[FeatureTopologyManagerPolicyOptions] = true
	featureGates[FeatureTopologyManagerPolicyBetaOptions] = true
	featureGates[FeatureTopologyManagerPolicyAlphaOptions] = true

	return true
}

// updateTopologyOptions 更新拓扑管理器选项
func (r *NodeConfigController) updateTopologyOptions(config map[string]interface{}, enableNUMA bool) bool {
	if config == nil {
		return false
	}

	topologyOpts, ok := config["topologyManagerPolicyOptions"].(map[string]interface{})
	if !ok {
		topologyOpts = make(map[string]interface{})
		config["topologyManagerPolicyOptions"] = topologyOpts
	}

	currentNUMAValue, _ := topologyOpts["prefer-closest-numa-nodes"].(string)
	numaValue := "true"
	if !enableNUMA {
		numaValue = "false"
	}

	if currentNUMAValue != numaValue {
		topologyOpts["prefer-closest-numa-nodes"] = numaValue
		return true
	}

	return false
}

// ensureBasicConfigFields 确保基本配置字段存在
func (r *NodeConfigController) ensureBasicConfigFields(config map[string]interface{}) {
	// 首先检查config是否为nil
	if config == nil {
		return
	}

	if _, ok := config["kind"].(string); !ok {
		config["kind"] = "KubeletConfiguration"
	}

	if _, ok := config["apiVersion"].(string); !ok {
		config["apiVersion"] = "kubelet.config.k8s.io/v1beta1"
	}
}

// applyConfigChanges 应用配置更改：删除状态文件、保存配置并重启kubelet
func (r *NodeConfigController) applyConfigChanges(nodeConf *NodeConfiguration, config map[string]interface{}) error {
	// 删除 cpu_manager_state 文件
	if err := r.deleteCPUManagerStateFile(); err != nil {
		return fmt.Errorf("failed to delete cpu_manager_state file: %v", err)
	}

	// 保存kubelet配置
	if err := r.saveKubeletConfig(nodeConf.KubeletConfigPath, config); err != nil {
		return fmt.Errorf("failed to save kubelet config: %v", err)
	}

	// 重启kubelet服务
	if KubeletConfigPath == "/var/lib/kubelet/config.yaml" {
		if err := r.restartKubelet(); err != nil {
			return fmt.Errorf("failed to restart kubelet: %v", err)
		}
	} else {
		if err := r.restartBkeKubelet(); err != nil {
			return fmt.Errorf("failed to restart kubelet: %v", err)
		}
	}

	return nil
}

// 添加删除 cpu_manager_state 文件的方法
func (r *NodeConfigController) deleteCPUManagerStateFile() error {
	//  cpu_manager_state 文件的路径为 /var/lib/kubelet/cpu_manager_state
	statePath := "/var/lib/kubelet/cpu_manager_state"
	if err := os.Remove(statePath); err != nil && !os.IsNotExist(err) {
		return err
	}
	return nil
}

func (r *NodeConfigController) saveKubeletConfig(path string, config map[string]interface{}) error {
	data, err := yaml.Marshal(config)
	if err != nil {
		return fmt.Errorf("failed to marshal kubelet config: %v", err)
	}

	// 创建临时文件
	const FilePermission = 0644
	tmpfile := path + ".tmp"
	if err := os.WriteFile(tmpfile, data, FilePermission); err != nil {
		return fmt.Errorf("failed to write temporary config file: %v", err)
	}

	// 重命名临时文件到目标文件
	if err := os.Rename(tmpfile, path); err != nil {
		// 清理临时文件
		if err := os.Remove(tmpfile); err != nil {
			return fmt.Errorf("failed to remove temporary file: %v", err)
		}
		return fmt.Errorf("failed to replace config file: %v", err)
	}

	return nil
}

func (r *NodeConfigController) restartKubelet() error {
	// 使用 systemctl 重启 kubelet
	cmd := exec.Command("nsenter",
		"-i/mnt/proc/1/ns/ipc",
		"-m/mnt/proc/1/ns/mnt",
		"-n/mnt/proc/1/ns/net",
		"systemctl", "restart", "kubelet")
	return cmd.Run()
}

func (r *NodeConfigController) restartBkeKubelet() error {
	// 使用 systemctl 重启 kubelet
	cmd := exec.Command("nsenter",
		"-i/mnt/proc/1/ns/ipc",
		"-m/mnt/proc/1/ns/mnt",
		"-n/mnt/proc/1/ns/net",
		"nerdctl", "-n", "k8s.io", "restart", "kubelet")
	return cmd.Run()
}

// updateSpecWithRetry 以冲突重试方式提交去重后的 spec。
// 每次重试均基于 apiserver 最新版本重新去重，避免覆盖并发写入。
func (r *NodeConfigController) updateSpecWithRetry(ctx context.Context,
	nodeConfig *NodeConfig) error {

	return retry.RetryOnConflict(retry.DefaultRetry, func() error {
		latest, err := r.getLatestNodeConfig(ctx, nodeConfig)
		if err != nil {
			return err
		}

		deduplicated := deduplicateNodeConfigurations(latest.Spec.Nodes)
		if deduplicated == nil {
			// 最新数据已无重复，无需提交
			nodeConfig.Spec.Nodes = latest.Spec.Nodes
			return nil
		}
		latest.Spec.Nodes = deduplicated
		if err := r.Update(ctx, latest); err != nil {
			return err
		}

		// 回写最新内容供后续查找使用
		nodeConfig.Spec.Nodes = latest.Spec.Nodes
		return nil
	})
}

// updateNodeStatusWithRetry 更新本节点状态条目。
// 始终基于 apiserver 最新版本提交，且只覆盖本节点条目，
// 避免多节点并发写 status 时互相覆盖。
func (r *NodeConfigController) updateNodeStatusWithRetry(ctx context.Context,
	nodeConfig *NodeConfig, nodeName, state, errMsg string) error {

	return retry.RetryOnConflict(retry.DefaultRetry, func() error {
		latest, err := r.getLatestNodeConfig(ctx, nodeConfig)
		if err != nil {
			return err
		}

		updateNodeStatuses(latest, nodeConfig, nodeName, state, errMsg)
		latest.Status.LastUpdateTime = metav1.Now()

		return r.Status().Update(ctx, latest)
	})
}

// updateNodeStatuses 在 NodeStatuses 中更新指定节点的状态条目，
// 不存在时追加新条目。
func updateNodeStatuses(latest *NodeConfig, nodeConfig *NodeConfig,
	nodeName, state, errMsg string) {

	for i, status := range latest.Status.NodeStatuses {
		if status.NodeName == nodeName {
			latest.Status.NodeStatuses[i].State = state
			latest.Status.NodeStatuses[i].Error = errMsg
			latest.Status.NodeStatuses[i].LastUpdateTime = metav1.Now()
			if state == NodeConfigStateConfigured {
				applyNodeConfiguredFlags(&latest.Status.NodeStatuses[i],
					latest.Spec.Nodes, nodeConfig.Spec.Nodes, nodeName)
			}
			return
		}
	}

	// 没有找到现有状态，添加新的
	newStatus := NodeStatus{
		NodeName:       nodeName,
		State:          state,
		Error:          errMsg,
		LastUpdateTime: metav1.Now(),
	}
	if state == NodeConfigStateConfigured {
		applyNodeConfiguredFlags(&newStatus, latest.Spec.Nodes, nodeConfig.Spec.Nodes, nodeName)
	}
	latest.Status.NodeStatuses = append(latest.Status.NodeStatuses, newStatus)
}

// applyNodeConfiguredFlags 将节点配置开关写入状态条目。
// 注意：不能直接用 Status 列表下标索引 Spec.Nodes，
// 两者长度可能不一致（并发丢更新时 Status 条目数
// 可能超过 Spec.Nodes），否则会 index out of range panic；
// 即使下标合法，两个列表顺序不一致时也会把其他节点
// 的开关写进当前节点。因此始终按 nodeName 查找。
func applyNodeConfiguredFlags(status *NodeStatus,
	primaryNodes, fallbackNodes []NodeConfiguration, nodeName string) {

	nodeConf := findNodeConfigurationByName(primaryNodes, nodeName)
	if nodeConf == nil {
		// 兜底：latest 中找不到该节点配置时，退回调用方
		// 传入的 nodeConfig，同样按 name 查找避免位置错配
		nodeConf = findNodeConfigurationByName(fallbackNodes, nodeName)
	}

	if nodeConf != nil {
		status.StaticCPUManagerEnabled = nodeConf.EnableStaticCPUManager
		status.NUMADistanceEnabled = nodeConf.EnableNUMADistance
	}
}

// InitializeNodeConfig 初始化 NodeConfig CR 或将当前节点添加到现有 CR 中
func InitializeNodeConfig(client client.Client, apiReader client.Reader) error {
	ctx := context.Background()

	// 获取当前节点名称
	nodeName, err := getNodeName()
	if err != nil {
		return err
	}

	// 尝试获取 NodeConfig 资源
	nodeConfig := &NodeConfig{}
	zlog.FormatInfo("正在检查 nodeConfig 是否存在")

	err = client.Get(ctx, types.NamespacedName{Name: "kubelet-config", Namespace: "default"}, nodeConfig)
	if err == nil {
		zlog.FormatInfo("nodeConfig 已存在，检查是否需要添加当前节点")

		// 检查当前节点是否已在配置中
		nodeFound := false
		for _, nc := range nodeConfig.Spec.Nodes {
			if nc.NodeName == nodeName {
				nodeFound = true
				zlog.FormatInfo("节点 %s 已存在于 NodeConfig 中", nodeName)
				break
			}
		}

		// 如果节点不在配置中，添加它
		if !nodeFound {
			zlog.FormatInfo("正在将节点 %s 添加到现有 NodeConfig 中", nodeName)
			return addNodeToExistingConfig(ctx, client, apiReader, nodeConfig, nodeName)
		}

		return nil
	}

	if !errors.IsNotFound(err) {
		return err
	}

	// NodeConfig 不存在，创建一个新的
	return createNewNodeConfig(ctx, client, nodeName)
}

// nodeConfigurationExists 检查指定节点是否已存在于节点配置列表中
func nodeConfigurationExists(nodes []NodeConfiguration, nodeName string) bool {
	for _, nc := range nodes {
		if nc.NodeName == nodeName {
			return true
		}
	}
	return false
}

// deduplicateNodeConfigurations 按 nodeName 去重，保留首次出现的配置项。
// 无重复时返回 nil（调用方据此跳过更新，避免无意义的 API 调用）。
func deduplicateNodeConfigurations(nodes []NodeConfiguration) []NodeConfiguration {
	seen := make(map[string]struct{}, len(nodes))
	deduplicated := make([]NodeConfiguration, 0, len(nodes))
	for _, nc := range nodes {
		if _, ok := seen[nc.NodeName]; ok {
			continue
		}
		seen[nc.NodeName] = struct{}{}
		deduplicated = append(deduplicated, nc)
	}

	if len(deduplicated) == len(nodes) {
		return nil
	}
	return deduplicated
}

// addNodeToExistingConfig 将当前节点添加到现有的 NodeConfig CR 中。
// apiReader 非 nil 时冲突重试读取走 apiserver（多实例场景）。
func addNodeToExistingConfig(ctx context.Context, client client.Client,
	apiReader client.Reader, nodeConfig *NodeConfig, nodeName string) error {

	// 读取当前 kubelet 配置
	currentConfig, err := readCurrentKubeletConfig()
	if err != nil {
		return err
	}

	// 分析当前配置状态
	currentStaticCPUEnabled, currentNUMAEnabled := analyzeCurrentConfig(currentConfig)

	getLatest := func() (*NodeConfig, error) {
		latest := &NodeConfig{}
		key := types.NamespacedName{
			Name:      nodeConfig.Name,
			Namespace: nodeConfig.Namespace,
		}
		if apiReader != nil {
			return latest, apiReader.Get(ctx, key, latest)
		}
		return latest, client.Get(ctx, key, latest)
	}

	// 将节点添加到配置中
	return retry.RetryOnConflict(retry.DefaultRetry, func() error {
		latest, err := getLatest()
		if err != nil {
			return err
		}

		// 入口检查与写入之间存在并发窗口，可能已被其他实例添加，
		// 已存在则幂等返回
		if nodeConfigurationExists(latest.Spec.Nodes, nodeName) {
			zlog.FormatInfo("节点 %s 已存在于 NodeConfig 中", nodeName)
			return nil
		}

		desiredStaticCPUEnabled, desiredNUMAEnabled := resolveDesiredNodeFlags(
			latest.Spec.Nodes, currentStaticCPUEnabled, currentNUMAEnabled)

		newNodeConf := NodeConfiguration{
			NodeName:               nodeName,
			EnableStaticCPUManager: desiredStaticCPUEnabled,
			EnableNUMADistance:     desiredNUMAEnabled,
			KubeletConfigPath:      KubeletConfigPath,
		}

		// 添加新的节点配置
		latest.Spec.Nodes = append(latest.Spec.Nodes, newNodeConf)

		// 更新 NodeConfig CR
		if err := client.Update(ctx, latest); err != nil {
			zlog.FormatError("更新 NodeConfig 失败: %v", err)
			return err
		}

		zlog.FormatInfo("成功将节点 %s 添加到 NodeConfig", nodeName)
		return nil
	})
}

// getNodeName 获取当前节点名称
func getNodeName() (string, error) {
	nodeName := os.Getenv("NODE_NAME")
	if nodeName == "" {
		return "", fmt.Errorf("NODE_NAME environment variable not set")
	}
	return nodeName, nil
}

// createNewNodeConfig 创建新的节点配置
func createNewNodeConfig(ctx context.Context, client client.Client, nodeName string) error {
	// 读取当前kubelet配置
	currentConfig, err := readCurrentKubeletConfig()
	if err != nil {
		return err
	}

	// 分析当前配置状态
	currentStaticCPUEnabled, currentNUMAEnabled := analyzeCurrentConfig(currentConfig)

	// 创建并保存新的NodeConfig CR
	return createNodeConfigCR(ctx, client, nodeName, currentStaticCPUEnabled, currentNUMAEnabled)
}

// readCurrentKubeletConfig 读取当前kubelet配置
func readCurrentKubeletConfig() (map[string]interface{}, error) {
	zlog.FormatInfo("reading current kubelet config")
	currentConfig := make(map[string]interface{})

	data, err := os.ReadFile(KubeletConfigPath)
	if err != nil {
		zlog.FormatError("Failed to read kubelet config: %v", err)
		return nil, fmt.Errorf("failed to read kubelet config: %v", err)
	}

	if err := yaml.Unmarshal(data, &currentConfig); err != nil {
		zlog.FormatError("Failed to unmarshal kubelet config: %v", err)
		return nil, fmt.Errorf("failed to unmarshal kubelet config: %v", err)
	}

	return currentConfig, nil
}

// analyzeCurrentConfig 分析当前配置状态
func analyzeCurrentConfig(currentConfig map[string]interface{}) (bool, bool) {
	// 检查CPU管理器策略
	cpuManagerPolicy, ok := currentConfig["cpuManagerPolicy"].(string)
	currentStaticCPUEnabled := ok && cpuManagerPolicy == "static"

	// 检查NUMA距离设置
	var currentNUMAEnabled bool
	if topologyOpts, ok := currentConfig["topologyManagerPolicyOptions"].(map[string]interface{}); ok {
		numaValue, exists := topologyOpts["prefer-closest-numa-nodes"]
		currentNUMAEnabled = exists && numaValue == "true"
	}

	return currentStaticCPUEnabled, currentNUMAEnabled
}

// createNodeConfigCR 创建NodeConfig自定义资源
func createNodeConfigCR(ctx context.Context, client client.Client,
	nodeName string, staticCPUEnabled, numaEnabled bool) error {

	zlog.FormatInfo("creating new nodeConfig CR")
	nodeConfig := &NodeConfig{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "kubelet-config",
			Namespace: "default",
		},
		Spec: NodeConfigSpec{
			Nodes: []NodeConfiguration{
				{
					NodeName:               nodeName,
					EnableStaticCPUManager: staticCPUEnabled,
					EnableNUMADistance:     numaEnabled,
					KubeletConfigPath:      KubeletConfigPath,
				},
			},
		},
	}

	if err := client.Create(ctx, nodeConfig); err != nil {
		// CR 已被其他节点创建（多节点同时启动），改为添加当前节点
		if errors.IsAlreadyExists(err) {
			zlog.FormatInfo("NodeConfig already exists (likely created by another node), adding current node instead")
			existing := &NodeConfig{}
			if getErr := client.Get(ctx, types.NamespacedName{
				Name:      "kubelet-config",
				Namespace: "default",
			}, existing); getErr != nil {
				return fmt.Errorf("failed to get existing NodeConfig: %v", getErr)
			}
			return addNodeToExistingConfig(ctx, client, nil, existing, nodeName)
		}
		zlog.FormatError("Failed to create NodeConfig: %v", err)
		return fmt.Errorf("failed to create NodeConfig: %v", err)
	}

	zlog.FormatInfo("successfully created nodeConfig CR")
	return nil
}
