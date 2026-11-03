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
 * Author: Junlin Liu
 * Date: 2025-05-20
 */

// Package numaaffinity 提供NUMA亲和性相关的功能和实现
package numaaffinity

import (
	"context"
	"encoding/json"
	"fmt"
	"strconv"
	"sync"
	"time"

	"k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/klog/v2"

	"openfuyao.com/colocation-management/pkg/colocation-overquota-agent/nriserver/config"
	"openfuyao.com/colocation-management/pkg/colocation-overquota-agent/nriserver/core"
	"openfuyao.com/colocation-management/pkg/colocation-overquota-agent/nriserver/hooks"
	"openfuyao.com/colocation-management/pkg/common"
)

const (
	// ActionName 是NUMA亲和性Hook的名称
	ActionName = "numaAffinity"
	// ActionPreCreateContainer 是NUMA亲和性Hook的创建动作名称
	ActionPreCreateContainer = "PreCreateContainer"
	// ActionPreUpdateContainer 是NUMA亲和性Hook的更新动作名称
	ActionPreUpdateContainer = "PreUpdateContainerResources"
	// ConfigMapName colocation-config ConfigMap名称
	ConfigMapName = "colocation-config"
	// NumaAffinityOptionsKey ConfigMap中NUMA亲和性配置的key
	NumaAffinityOptionsKey = "numa-affinity-options"
	// DefaultNamespace 默认命名空间
	DefaultNamespace = "openfuyao-colocation"
	// DefaultWatchTime 默认的watch时间
	DefaultWatchTime = 30 * time.Second
)

// NUMA亲和性配置结构
type NumaAffinityConfig struct {
	Enable bool `json:"enable"`
}

// Runner 是NUMA亲和性Runner的结构体
type Runner struct {
	config         *config.NumaAffinityConfig
	numaTopology   *core.NumaTopology
	k8sClient      kubernetes.Interface
	namespace      string
	watcherStarted bool
	mutex          sync.RWMutex
}

var numaAffinityRunner *Runner

// GetNUMAAffinityRunner 返回单例的NUMA亲和性Runner实例
func GetNUMAAffinityRunner() *Runner {
	if numaAffinityRunner == nil {
		numaAffinityRunner = NewNumaAffinityRunner()
		klog.V(common.AdvanceDebugLog).Infof("Created new NUMA affinity runner instance")
	} else {
		klog.V(common.AdvanceDebugLog).Infof("Returning existing NUMA affinity runner instance")
	}
	return numaAffinityRunner
}

// NewNumaAffinityRunner 创建一个新的NUMA亲和性Runner实例
func NewNumaAffinityRunner() *Runner {
	runner := &Runner{
		namespace: DefaultNamespace,
	}

	if err := runner.initK8sClient(); err != nil {
		klog.Errorf("Failed to initialize Kubernetes client: %v", err)
	}
	return runner
}

// initK8sClient 初始化Kubernetes客户端
func (r *Runner) initK8sClient() error {
	config, err := rest.InClusterConfig()
	if err != nil {
		klog.Errorf("Failed to create in-cluster config: %v", err)
		return err
	}

	clientset, err := kubernetes.NewForConfig(config)
	if err != nil {
		klog.Errorf("Failed to create Kubernetes clientset: %v", err)
		return err
	}

	r.k8sClient = clientset
	klog.Infof("Kubernetes client initialized successfully, namespace: %s", r.namespace)
	return nil
}

// readNumaAffinityConfigFromConfigMap 从colocation-config ConfigMap读取NUMA亲和性配置
func (r *Runner) readNumaAffinityConfigFromConfigMap(ctx context.Context) (bool, error) {
	if r.k8sClient == nil {
		return false, fmt.Errorf("kubernetes client not initialized")
	}

	configMap, err := r.k8sClient.CoreV1().ConfigMaps(r.namespace).Get(ctx, ConfigMapName, v1.GetOptions{})
	if err != nil {
		return false, fmt.Errorf("failed to get %s ConfigMap in namespace %s: %v", ConfigMapName, r.namespace, err)
	}

	numaOptionsStr, exists := configMap.Data[NumaAffinityOptionsKey]
	if !exists {
		klog.Warningf("%s not found in %s ConfigMap, using default (disabled)", NumaAffinityOptionsKey, ConfigMapName)
		return false, nil
	}

	var config NumaAffinityConfig
	if err := json.Unmarshal([]byte(numaOptionsStr), &config); err != nil {
		return false, fmt.Errorf("failed to unmarshal %s: %v", NumaAffinityOptionsKey, err)
	}

	klog.Infof("NUMA affinity configuration loaded from ConfigMap: enable=%t", config.Enable)
	return config.Enable, nil
}

// RegisterFunc 注册NUMA亲和性Hook函数
func (r *Runner) RegisterFunc() {
	klog.Infof("Registering NUMA affinity hook functions using hooks package")
	_, err := hooks.RegisterFunc("", ActionPreCreateContainer, ActionName, r.ArgsCreateContainerWrapper)
	if err != nil {
		klog.Errorf("Failed to register %s hook for %s: %v", ActionName, ActionPreCreateContainer, err)
	}
	_, err = hooks.RegisterFunc("", ActionPreUpdateContainer, ActionName, r.ArgsUpdateContainerWrapper)
	if err != nil {
		klog.Errorf("Failed to register %s hook for %s: %v", ActionName, ActionPreUpdateContainer, err)
	}
}

// Init 初始化NUMA亲和性Runner
func (r *Runner) Init(cfg config.Config) error {
	r.config = &cfg.NumaAffinity

	ctx := context.Background()
	enabled, err := r.readNumaAffinityConfigFromConfigMap(ctx)
	if err != nil {
		klog.Warningf("Failed to read NUMA affinity config from ConfigMap, using original config: %v", err)
	} else {
		r.config.Enabled = enabled
	}

	if !r.watcherStarted && r.k8sClient != nil {
		go r.startConfigMapWatcher()
		r.watcherStarted = true
		klog.Infof("Colocation configMap watcher started")
	}

	return r.initializeTopology()
}

// ArgsCreateContainerWrapper 是处理创建容器时的钩子函数
func (r *Runner) ArgsCreateContainerWrapper(info core.Info) error {
	containerInfo, ok := info.(*core.ContainerInfo)
	if !ok {
		klog.Errorf("ArgsCreateContainerWrapper (%s): received info of unexpected type %T", ActionName, info)
		return fmt.Errorf("unexpected info type for %s hook, expected *core.ContainerInfo", ActionName)
	}
	if containerInfo != nil && containerInfo.ContainerRequest.ContainerMeta.Name != "" {
		klog.V(common.AdvanceDebugLog).Infof("ArgsCreateContainerWrapper (%s)", ActionName)
	} else if containerInfo != nil {
		klog.V(common.AdvanceDebugLog).Infof(
			"ArgsCreateContainerWrapper (%s): Processing container with no name set", ActionName)
	} else {
		klog.Errorf("ArgsCreateContainerWrapper (%s): received nil containerInfo", ActionName)
	}

	return r.processCreateContainer(context.Background(), containerInfo)
}

// processCreateContainer 处理创建容器的逻辑
func (r *Runner) processCreateContainer(ctx context.Context, containerInfo *core.ContainerInfo) error {
	// 检查全局配置是否启用以及Runner是否正确初始化 (numaTopology 是否有效)
	if r == nil || r.config == nil || !r.config.Enabled || r.numaTopology == nil {
		if r != nil && r.config != nil && !r.config.Enabled {
			klog.V(common.AdvanceDebugLog).Infof(
				"processCreateContainer (%s): NUMA affinity is disabled via configuration. Skipping.", ActionName)
		} else if r != nil && r.config != nil && r.config.Enabled && r.numaTopology == nil {
			klog.V(common.AdvanceDebugLog).Infof(
				"processCreateContainer (%s): NUMA affinity enabled, NUMA topology is unavailable. Skipping.", ActionName)
		} else {
			klog.V(common.AdvanceDebugLog).Infof(
				"processCreateContainer (%s): Runner or config is not properly initialized. Skipping.", ActionName)
		}
		return nil
	}

	if containerInfo == nil {
		klog.Errorf("processCreateContainer (%s): received nil containerInfo", ActionName)
		return fmt.Errorf("nil containerInfo in %s", ActionName)
	}

	r.mutex.RLock()
	enabled := r.config != nil && r.config.Enabled
	topology := r.numaTopology
	r.mutex.RUnlock()

	if !enabled || topology == nil {
		klog.V(common.AdvanceDebugLog).Infof("processCreateContainer: NUMA affinity disabled or topology unavailable. Skipping.")
		return nil
	}

	containerID := getContainerID(containerInfo)
	podAnnotations := containerInfo.ContainerRequest.PodSandboxConfig.Annotations
	qosLevel := core.GetPodQosLevel(podAnnotations)

	klog.Infof("Container (%s): container %s. QoS: %s. NUMA: %t", ActionName, containerID, qosLevel, r.config.Enabled)

	switch qosLevel {
	case common.QosHLS:
		klog.Infof("Container (%s): HLS pod (%s) detected. native CPU management.", ActionName, containerID)
		return nil
	case common.QosLs:
		klog.Infof("Container (%s): LS pod (%s) detected. Applying NUMA affinity.", ActionName, containerID)
		err := applyNUMAAffinityForContainer(topology, containerInfo)
		if err != nil {
			klog.Errorf("Container (%s): Failed NUMA affinity, LS pod (%s): %v.", ActionName, containerID, err)
			return nil // 允许容器继续启动
		}
		klog.Infof("Container (%s): Successfully applied NUMA affinity for LS pod (%s).", ActionName, containerID)
		return nil
	case common.QosBe:
		klog.Infof("Container (%s): BE pod (%s) detected. Skipping NUMA affinity.", ActionName, containerID)
		return nil
	default:
		klog.Infof("Container (%s): Unknown QoS ('%s') for pod (%s). Skipping.", ActionName, qosLevel, containerID)
		return nil
	}
}

// applyNUMAAffinityForContainer 对容器应用NUMA亲和性
func applyNUMAAffinityForContainer(topology *core.NumaTopology, containerInfo *core.ContainerInfo) error {
	if topology == nil || len(topology.Nodes) == 0 {
		return fmt.Errorf("NUMA topology is not available")
	}
	if containerInfo == nil {
		return fmt.Errorf("containerInfo is nil")
	}

	containerID := getContainerID(containerInfo)
	minCPUCount, err := calculateMinCPUCount(containerInfo, containerID)
	if err != nil {
		return err
	}
	if minCPUCount == 0 {
		return nil
	}
	bestNodeID, err := topology.GetBestNumaNodeForAffinity(minCPUCount)
	if err != nil {
		return fmt.Errorf("failed for container %s (requesting %d cores): %v", containerID, minCPUCount, err)
	}
	allNodeCPUs, err := topology.GetNumaNodeCPUs(bestNodeID)
	if err != nil {
		return fmt.Errorf("failed to get CPU list for NUMA node %d for container %s: %v", bestNodeID, containerID, err)
	}
	klog.Infof("applyNUMA: Container %s: Available CPUs NUMA %d: %v.", containerID, bestNodeID, allNodeCPUs)

	if len(allNodeCPUs) == 0 {
		klog.Warningf("applyNUMA: Container %s: NUMA node %d has no CPUs available.", containerID, bestNodeID)
		return nil
	}
	setCPUResourcesForContainer(containerInfo, containerID, bestNodeID, allNodeCPUs)
	return nil
}

// getContainerID 获取容器ID
func getContainerID(containerInfo *core.ContainerInfo) string {
	containerID := containerInfo.ContainerRequest.ContainerMeta.ID
	if containerID == "" {
		containerID = containerInfo.ContainerRequest.ContainerMeta.Name
	}
	return containerID
}

// calculateMinCPUCount 计算容器的最小CPU核心数
func calculateMinCPUCount(containerInfo *core.ContainerInfo, containerID string) (int, error) {
	cpuMilliRequest := containerInfo.ContainerRequest.CPUMilliRequest
	if cpuMilliRequest <= 0 {
		klog.Warningf(
			"applyNUMA: Container %s non-positive CPU request %d milliCPUs. Skipping.", containerID, cpuMilliRequest)
		return 0, nil
	}

	cpuRequestCores := float64(cpuMilliRequest) / float64(core.DefaultCPURequestMilliCores)
	minCPUCount := int(cpuRequestCores)
	if cpuRequestCores > float64(minCPUCount) {
		minCPUCount++
	}
	if minCPUCount <= 0 {
		minCPUCount = 1
	}

	klog.Infof("applyNUMA: Container %s: Minimum CPU cores needed: %d.", containerID, minCPUCount)
	return minCPUCount, nil
}

// setCPUResourcesForContainer 设置容器的CPU资源
func setCPUResourcesForContainer(containerInfo *core.ContainerInfo,
	containerID string, bestNodeID int, allNodeCPUs []int) {
	// 设置容器的CPUSet
	if cpuSetString := core.CPUListToString(allNodeCPUs); cpuSetString != "" {
		klog.Infof("applyNUMA: CPUSet for container %s to '%s' (NUMA node %d affinity, not pinning).",
			containerID, cpuSetString, bestNodeID)
		if containerInfo.ContainerResponese.Resources.CPUSet == nil {
			containerInfo.ContainerResponese.Resources.CPUSet = new(string)
		}
		*containerInfo.ContainerResponese.Resources.CPUSet = cpuSetString
	} else {
		klog.Warningf("applyNUMA: Container %s: cpuset empty for NUMA node %d.",
			containerID, bestNodeID)
	}
	// 设置容器的CPUSetMem
	memNodeString := strconv.Itoa(bestNodeID)
	klog.Infof("applyNUMA: CPUSetMem for container %s to '%s' (NUMA node %d affinity, not pinning).",
		containerID, memNodeString, bestNodeID)
	if containerInfo.ContainerResponese.Resources.CPUSetMem == nil {
		containerInfo.ContainerResponese.Resources.CPUSetMem = new(string)
	}
	*containerInfo.ContainerResponese.Resources.CPUSetMem = memNodeString
}

// ArgsUpdateContainerWrapper 是处理更新容器时的钩子函数
func (r *Runner) ArgsUpdateContainerWrapper(info core.Info) error {
	return nil
}

// processUpdateContainer 处理更新容器的逻辑
func (r *Runner) processUpdateContainer(ctx context.Context, containerInfo *core.ContainerInfo) error {
	return nil
}

// startConfigMapWatcher 启动ConfigMap监听器
func (r *Runner) startConfigMapWatcher() {
	klog.Infof("Starting ConfigMap watcher for %s", ConfigMapName)

	ctx := context.Background()

	for {
		select {
		case <-ctx.Done():
			klog.Infof("ConfigMap watcher stopped due to context cancellation")
			return
		default:
			if err := r.watchConfigMap(); err != nil {
				klog.Errorf("ConfigMap watcher error: %v, retrying in %v", err, DefaultWatchTime)
				time.Sleep(DefaultWatchTime)
			}
		}
	}
}

// watchConfigMap 监听ConfigMap变化
func (r *Runner) watchConfigMap() error {
	if r.k8sClient == nil {
		return fmt.Errorf("kubernetes client not initialized")
	}

	watcher, err := r.k8sClient.CoreV1().ConfigMaps(r.namespace).Watch(context.Background(), v1.ListOptions{
		FieldSelector: fmt.Sprintf("metadata.name=%s", ConfigMapName),
	})
	if err != nil {
		return fmt.Errorf("failed to create ConfigMap watcher: %v", err)
	}
	defer watcher.Stop()

	klog.Infof("ConfigMap watcher started for %s/%s", r.namespace, ConfigMapName)

	for event := range watcher.ResultChan() {
		switch event.Type {
		case watch.Modified, watch.Added:
			r.handleConfigMapUpdate()
		case watch.Deleted:
			klog.Warningf("ConfigMap %s was deleted, disabling NUMA affinity", ConfigMapName)
			r.mutex.Lock()
			r.config.Enabled = false
			r.numaTopology = nil
			r.mutex.Unlock()
		case watch.Error:
			klog.Warningf("ConfigMap watcher received an error event: %v", event.Object)
		}
	}
	return fmt.Errorf("watcher channel closed unexpectedly")
}

// handleConfigMapUpdate 处理ConfigMap更新
func (r *Runner) handleConfigMapUpdate() {
	ctx := context.Background()
	enabled, err := r.readNumaAffinityConfigFromConfigMap(ctx)
	if err != nil {
		klog.Errorf("Failed to read updated ConfigMap: %v", err)
		return
	}

	r.mutex.RLock()
	oldEnabled := r.config.Enabled
	r.mutex.RUnlock()

	if oldEnabled != enabled {
		klog.Infof("NUMA affinity configuration changed: %t -> %t", oldEnabled, enabled)
		r.mutex.Lock()
		r.config.Enabled = enabled
		r.mutex.Unlock()

		if err := r.initializeTopology(); err != nil {
			klog.Errorf("Failed to re-initialize NUMA topology after config change: %v", err)
		}
	}
}

// initializeTopology
func (r *Runner) initializeTopology() error {
	r.mutex.RLock()
	enabled := r.config.Enabled
	r.mutex.RUnlock()

	var newTopology *core.NumaTopology
	var err error

	if enabled {
		newTopology, err = core.GetNumaTopology()
		if err != nil {
			klog.Errorf("Failed to get NUMA topology: %v. NUMA affinity will be effectively disabled.", err)
			newTopology = nil
		} else if newTopology == nil || len(newTopology.Nodes) == 0 {
			klog.Warningf("NUMA topology information is empty or not detected. NUMA affinity will be effectively disabled.")
			newTopology = nil
		} else {
			klog.Infof("NUMA affinity feature enabled: detected %d NUMA nodes.", len(newTopology.Nodes))
		}
	}

	r.mutex.Lock()
	r.numaTopology = newTopology
	r.mutex.Unlock()

	return nil
}
