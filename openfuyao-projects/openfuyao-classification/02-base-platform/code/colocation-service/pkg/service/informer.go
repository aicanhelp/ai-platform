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
	"os"
	"reflect"
	"strings"
	"sync"

	"github.com/thoas/go-funk"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/util/retry"
	"k8s.io/klog/v2"
)

const (
	// ColocationConfigMapName 默认 configmap 名称
	ColocationConfigMapName = "colocation-config"
	// RubikConfigMapName 默认 rubik configmap
	RubikConfigMapName = "colocation-rubik-config"
	// VolcanoSchedulerConfigMapNamespace 默认 volcano scheduler configmap namespace
	VolcanoSchedulerConfigMapNamespace = "volcano-system"
	// VolcanoSchedulerConfigMapName 默认volcano scheduler configmap name
	VolcanoSchedulerConfigMapName = "volcano-scheduler-configmap"

	// Rubik 配置默认值常量
	DefaultSyncInterval      = 100
	DefaultAdjustInterval    = 1000
	DefaultPerfDuration      = 1000
	DefaultPSIInterval       = 10
	DefaultPSIAvg10Threshold = 5.0
)

var (
	// colocationConfigMapNamespace 默认 configmap namespace
	colocationConfigMapNamespace = os.Getenv("POD_NAMESPACE")
	// enableRubikHotReload 是否启用 Rubik 热重载
	enableRubikHotReload = os.Getenv("ENABLE_RUBIK_HOT_RELOAD") == "true"
)

// ColocationOptions 定义混部配置结构体
type ColocationOptions struct {
	// Enable 混部开启标识
	Enable bool `json:"enable"`
	// NodesConfig 节点配置
	NodesConfig []NodeConfig `json:"nodesConfig"`
}

// NodeConfig 定义混部节点配置结构体
type NodeConfig struct {
	// Name 配置名称
	Name string `json:"name"`
}

// RubikOptions 定义Rubik配置结构体
type RubikOptions struct {
	// CPUEvict cpu驱逐配置
	CPUEvict CPUEvictConfig `json:"cpuevict"`
	// MemoryEvict 内存驱逐配置
	MemoryEvict MemoryEvictConfig `json:"memoryevict"`
}

// CPUEvictConfig rubik-cpu 配置
type CPUEvictConfig struct {
	// Threshold 窗口期间平均cpu利用率阈值
	Threshold int `json:"threshold,omitempty"`
	// Interval 节点cpu利用率采集间隔 - s
	Interval int `json:"interval,omitempty"`
	// Windows 节点cpu利用率采集窗口 - s
	Windows int `json:"windows,omitempty"`
	// Cooldown 冷却时间，两次驱逐之间的间隔冷却时间 - s
	Cooldown int `json:"cooldown,omitempty"`
}

// MemoryEvictConfig rubik-memory 配置
type MemoryEvictConfig struct {
	// Threshold 窗口期间平均cpu利用率阈值
	Threshold int `json:"threshold,omitempty"`
	// Interval 节点cpu利用率采集间隔 - s
	Interval int `json:"interval,omitempty"`
	// Cooldown 冷却时间，两次驱逐之间的间隔冷却时间 - s
	Cooldown int `json:"cooldown,omitempty"`
}

// ConfigMapEvenHandler cm处理结构体
type ConfigMapEvenHandler struct {
	client clientset.Interface
	mu     sync.Mutex
}

// FullRubikConfig rubik configmap结构体定义
type FullRubikConfig struct {
	Agent struct {
		CgroupDriver    string   `json:"cgroupDriver"`
		LogDriver       string   `json:"logDriver"`
		LogDir          string   `json:"logDir"`
		LogSize         int      `json:"logSize"`
		LogLevel        string   `json:"logLevel"`
		CgroupRoot      string   `json:"cgroupRoot"`
		EnabledFeatures []string `json:"enabledFeatures"`
	} `json:"agent"`
	Preemption struct {
		Resource []string `json:"resource"`
	} `json:"preemption"`
	CPUEvict    CPUEvictConfig        `json:"cpuevict,omitempty"`
	MemoryEvict MemoryEvictConfig     `json:"memoryevict,omitempty"`
	QuotaTurbo  *QuotaTurboFullConfig `json:"quotaTurbo,omitempty"`
	DynMemory   *DynMemoryFullConfig  `json:"dynMemory,omitempty"`
	DynCache    *DynCacheFullConfig   `json:"dynCache,omitempty"`
	PSI         *PSIFullConfig        `json:"psi,omitempty"`
}
type QuotaTurboFullConfig struct {
	HighWaterMark  int `json:"highWaterMark"`
	AlarmWaterMark int `json:"alarmWaterMark"`
	SyncInterval   int `json:"syncInterval"`
}

type DynCacheFullConfig struct {
	DefaultLimitMode string              `json:"defaultLimitMode"`
	AdjustInterval   int                 `json:"adjustInterval"`
	PerfDuration     int                 `json:"perfDuration"`
	L3Percent        *CachePercentConfig `json:"l3Percent"`
	MemBandPercent   *CachePercentConfig `json:"memBandPercent"`
}

type DynMemoryFullConfig struct {
	Policy string `json:"policy"`
}

type PSIFullConfig struct {
	Interval       int      `json:"interval"`
	Resource       []string `json:"resource"`
	Avg10Threshold float64  `json:"avg10Threshold"`
}

// NewConfigMapEvenHandler new cm handler
func NewConfigMapEvenHandler(client clientset.Interface) *ConfigMapEvenHandler {
	return &ConfigMapEvenHandler{
		client: client,
	}
}

// handleConfigMapEvent 处理ConfigMap事件的公共逻辑
func (e *ConfigMapEvenHandler) handleConfigMapEvent(oldCM, newCM *corev1.ConfigMap, eventType string) {
	logger := klog.Background()
	if newCM.Name != ColocationConfigMapName {
		return
	}

	logger.Info("receive "+eventType+" event",
		"configmap", newCM.Name,
		"namespace", newCM.Namespace)

	// 统一处理三个传递逻辑
	handlers := []func(*corev1.ConfigMap, *corev1.ConfigMap) error{
		e.passModifiToNodes,
		e.passModifiToRubik,
		e.passModifiToVolcano,
	}

	for _, handler := range handlers {
		if err := handler(oldCM, newCM); err != nil {
			logger.Error(err, "failed to handle configmap event")
			return
		}
	}
}

// OnAdd 实现eventhandler方法
func (e *ConfigMapEvenHandler) OnAdd(obj interface{}, is bool) {
	oldCM, err := getDefaultConfigmap()
	if err != nil {
		return
	}

	newCM, ok := obj.(*corev1.ConfigMap)
	if !ok {
		klog.Error("OnAdd: expected ConfigMap, got something else")
		return
	}

	e.handleConfigMapEvent(oldCM, newCM, "add")
}

// OnUpdate 实现eventhandler方法
func (e *ConfigMapEvenHandler) OnUpdate(oldObj, newObj interface{}) {
	oldCM, ok1 := oldObj.(*corev1.ConfigMap)
	newCM, ok2 := newObj.(*corev1.ConfigMap)
	if !ok1 || !ok2 {
		klog.Error("OnUpdate: expected ConfigMap, got something else")
		return
	}
	klog.Info("start to solve configmap update")
	klog.Info("old cm config: ", oldCM.Data["colocation-options"])
	klog.Info("new cm config: ", newCM.Data["colocation-options"])
	e.handleConfigMapEvent(oldCM, newCM, "update")
}

// OnDelete 实现evenhandler 方法
func (e *ConfigMapEvenHandler) OnDelete(event interface{}) {
	logger := klog.Background()
	cm, ok := event.(*corev1.ConfigMap)
	if !ok {
		klog.Error("OnDelete: expected ConfigMap, got something else")
		return
	}

	if cm.Name == ColocationConfigMapName {
		logger.Info("receive deleted event", "configmap:", cm.Name, "namespace:", cm.Namespace)
		go e.handleConfigMapDelete(cm)
	}
}

func (e *ConfigMapEvenHandler) passModifiToVolcano(oldCM, newCM *corev1.ConfigMap) error {
	oldVolcano, err := e.getVolcanoConfig(oldCM)
	if err != nil {
		return err
	}

	newVolcano, err := e.getVolcanoConfig(newCM)
	if err != nil {
		return err
	}

	if reflect.DeepEqual(oldVolcano, newVolcano) {
		klog.Infof("No modification on Volcano Scheduler configs")
		return nil
	}

	e.mu.Lock()
	defer e.mu.Unlock()

	// 获取volcano scheduler configmap
	volSchedulerConfigJson, err := e.client.CoreV1().ConfigMaps(VolcanoSchedulerConfigMapNamespace).Get(
		context.TODO(), VolcanoSchedulerConfigMapName, metav1.GetOptions{})
	if err != nil {
		klog.Error(err)
		return err
	}

	if err := e.editVolcanoSchedulerUsagePlugin(oldVolcano, newVolcano, volSchedulerConfigJson); err != nil {
		return err
	}

	if err := updateConfigMapwithRetry(e.client, volSchedulerConfigJson, VolcanoSchedulerConfigMapNamespace); err != nil {
		return fmt.Errorf("update ConfigMap failed: %v", err)
	}

	if err := e.deleteDesiredPods("volcano-scheduler", VolcanoSchedulerConfigMapNamespace); err != nil {
		return err
	}

	return nil
}

func insertUsagePlugin(yamlStr, defaultPlugin, metricsStr string) string {
	// 先insert usagePlugin配置
	start := strings.Index(yamlStr, "- name: usage")
	if start != -1 {
		klog.InfoS("usage plugin already exists.")
		return yamlStr
	}

	//  找到第一个 plugins 块的结束位置
	pluginsBlockEnd := strings.Index(yamlStr, "- plugins:\n") + len("- plugins:\n")
	if pluginsBlockEnd < len("- plugins:\n") {
		return yamlStr
	}

	yamlStr = yamlStr[:pluginsBlockEnd] + defaultPlugin + yamlStr[pluginsBlockEnd:]

	// 再insert metrics，只在第一次开启usage调度时insert一次
	start = strings.Index(yamlStr, "metrics:\n")
	if start != -1 {
		klog.InfoS("metrics configuration already exists")
		return yamlStr
	}

	prometheusService := os.Getenv("PROMETHEUS_SERVICE")
	if prometheusService == "" {
		klog.Error("prometheus endpoint is not provided")
		return yamlStr
	}
	metricsStr = strings.Replace(metricsStr, "{PROMETHEUS_ENDPOINT}", prometheusService, 1)

	return yamlStr + metricsStr
}

func removeUsagePlugin(yamlStr string) string {
	// 定位目标插件起始行
	start := strings.Index(yamlStr, "- name: usage")

	if start == -1 {
		klog.InfoS("failed to find usage plugin")
		return yamlStr
	}

	next := strings.Index(yamlStr[start+1:], "- name:") + 1

	for yamlStr[start] != '\n' {
		start--
		next++
	}
	// 跳过回车
	start++
	next--

	// 找到块结束位置
	end := start + next
	for yamlStr[end] != '\n' {
		end--
	}
	return yamlStr[:start] + strings.TrimLeft(yamlStr[end:], "\n")
}

func updateUsageParams(yamlStr string, enable bool, cpuThreshold, memThreshold float64) (string, error) {
	// 检查是否存在 usage 插件
	usageIndex := strings.Index(yamlStr, "name: usage")
	if usageIndex == -1 {
		return yamlStr, fmt.Errorf("usage plugin not found")
	}
	thresholdsStart := usageIndex

	// 定位 cpu 和 mem 行
	lines := strings.Split(yamlStr[thresholdsStart:], "\n")
	var newLines []string
	foundCPU, foundMem := false, false

	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		switch {
		case strings.HasPrefix(trimmed, "cpu:"):
			newLines = append(newLines, fmt.Sprintf("        cpu: %v", cpuThreshold))
			foundCPU = true
		case strings.HasPrefix(trimmed, "mem:"):
			newLines = append(newLines, fmt.Sprintf("        mem: %v", memThreshold))
			foundMem = true
		case strings.HasPrefix(trimmed, "enablePredicate:"):
			newLines = append(newLines, fmt.Sprintf("    enablePredicate: %v", enable))
			foundMem = true
		default:
			newLines = append(newLines, line)
		}
	}

	// 验证是否找到两个字段
	if !foundCPU || !foundMem {
		return yamlStr, fmt.Errorf("failed to find both cpu and mem thresholds")
	}

	// 重建 YAML
	updated := yamlStr[:thresholdsStart] + strings.Join(newLines, "\n")
	return updated, nil
}

var (
	defaultUsagePlugin = "  - name: usage\n" +
		"    enablePredicate: false\n" +
		"    arguments:\n" +
		"      usage.weight: 5\n" +
		"      cpu.weight: 1\n" +
		"      memory.weight: 1\n" +
		"      thresholds:\n" +
		"        cpu: 0\n" +
		"        mem: 0\n"
	usageMetricsTemplate = "metrics:\n" +
		"  type: prometheus\n" +
		"  address: {PROMETHEUS_ENDPOINT}\n" +
		"  interval: 30s"
)

func (e *ConfigMapEvenHandler) editVolcanoSchedulerUsagePlugin(
	oldCM, newCM VolcanoSchedulerConfig,
	realVolScheduler *corev1.ConfigMap) error {
	plugin := realVolScheduler.Data["volcano-scheduler.conf"]

	if newCM.UsagePlugin.Enabled == true {
		if oldCM.UsagePlugin.Enabled == false {
			plugin = insertUsagePlugin(plugin, defaultUsagePlugin, usageMetricsTemplate)
		}
	} else {
		if oldCM.UsagePlugin.Enabled == true {
			plugin = removeUsagePlugin(plugin)
		}
	}

	// 如果开启负载均衡调度，就要设置负载均衡调度时各资源停止调度水位线
	var err error
	if newCM.UsagePlugin.Enabled == true {
		if newCM.UsagePlugin.UsageThreshold.Enabled == true {
			plugin, err = updateUsageParams(plugin, true,
				newCM.UsagePlugin.UsageThreshold.CPUThreshold,
				newCM.UsagePlugin.UsageThreshold.MemoryThreshold)
			if err != nil {
				return err
			}
			klog.InfoS("update usage plugin thresholds",
				"cpu", newCM.UsagePlugin.UsageThreshold.CPUThreshold,
				"mem", newCM.UsagePlugin.UsageThreshold.MemoryThreshold)
		} else {
			plugin, err = updateUsageParams(plugin, false, 0.0, 0.0)
			if err != nil {
				return err
			}
			klog.InfoS("update usage plugin thresholds",
				"cpu", newCM.UsagePlugin.UsageThreshold.CPUThreshold,
				"mem", newCM.UsagePlugin.UsageThreshold.MemoryThreshold)
		}
	}

	realVolScheduler.Data["volcano-scheduler.conf"] = plugin

	return nil
}

func (e *ConfigMapEvenHandler) passModifiToRubik(oldCM, newCM *corev1.ConfigMap) error {
	oldRubik, err := e.getRubikConfig(oldCM)
	if err != nil {
		return err
	}

	newRubik, err := e.getRubikConfig(newCM)
	if err != nil {
		return err
	}

	if reflect.DeepEqual(oldRubik, newRubik) {
		klog.Infof("No modification on Rubik configs")
		return nil
	}

	e.mu.Lock()
	defer e.mu.Unlock()

	// 获取rubik configmap
	rubikConfigJson, err := e.client.CoreV1().ConfigMaps(colocationConfigMapNamespace).Get(
		context.TODO(), RubikConfigMapName, metav1.GetOptions{})
	if err != nil {
		klog.Error(err)
		return err
	}

	// 更新 rubik 完整配置
	err = e.updateFullRubikConfig(rubikConfigJson, newRubik)
	if err != nil {
		return err
	}

	if err := updateConfigMapwithRetry(e.client, rubikConfigJson, colocationConfigMapNamespace); err != nil {
		return fmt.Errorf("update ConfigMap failed: %v", err)
	}

	if enableRubikHotReload {
		klog.InfoS("Rubik hot reload enabled, rubik will reload configuration automatically")
	} else {
		klog.InfoS("Rubik hot reload disabled, deleting rubik pods for configuration reload")
		if err := e.deleteDesiredPods("colocation-rubik", colocationConfigMapNamespace); err != nil {
			return err
		}
	}

	return nil
}

// updateFullRubikConfig 更新完整的 rubik 配置
func (e *ConfigMapEvenHandler) updateFullRubikConfig(rubikConfigJson *corev1.ConfigMap, newRubik RubikConfig) error {
	var config FullRubikConfig
	if err := json.Unmarshal([]byte(rubikConfigJson.Data["config.json"]), &config); err != nil {
		return fmt.Errorf("unmarshal failed: %v", err)
	}

	// 更新基础驱逐配置
	e.updateEvictionConfig(&config, newRubik)

	// 更新高级特性配置
	e.updateHighFeaturesConfig(&config, newRubik)

	// 序列化
	updatedConfig, err := json.MarshalIndent(config, "", "    ")
	if err != nil {
		return err
	}

	rubikConfigJson.Data["config.json"] = string(updatedConfig)

	klog.InfoS("updated rubik ConfigMap with new features")
	return nil
}

func (e *ConfigMapEvenHandler) updateEvictionConfig(config *FullRubikConfig, newRubik RubikConfig) {
	// 更新 EnabledFeatures 中的驱逐特性
	targetFeatures := []string{"cpuevict", "memoryevict"}

	if newRubik.Eviction.Enabled {
		// 添加驱逐特性
		for _, feature := range targetFeatures {
			if !funk.ContainsString(config.Agent.EnabledFeatures, feature) {
				config.Agent.EnabledFeatures = append(config.Agent.EnabledFeatures, feature)
			}
		}
		// 更新配置值
		config.CPUEvict.Threshold = newRubik.Eviction.CPUEvict.Threshold
		config.MemoryEvict.Threshold = newRubik.Eviction.MemoryEvict.Threshold
	} else {
		// 移除驱逐特性
		config.Agent.EnabledFeatures = funk.FilterString(config.Agent.EnabledFeatures,
			func(s string) bool {
				return !funk.ContainsString(targetFeatures, s)
			})
		// 清空配置
		config.CPUEvict = CPUEvictConfig{}
		config.MemoryEvict = MemoryEvictConfig{}
	}
}

// updateHighFeaturesConfig 更新高级特性配置
func (e *ConfigMapEvenHandler) updateHighFeaturesConfig(config *FullRubikConfig, newRubik RubikConfig) {
	// 更新 QuotaTurbo
	if newRubik.QuotaTurbo != nil && newRubik.QuotaTurbo.Enable {
		e.addFeatureToEnabledList(&config.Agent.EnabledFeatures, "quotaTurbo")
		if config.QuotaTurbo == nil {
			config.QuotaTurbo = &QuotaTurboFullConfig{}
		}
		config.QuotaTurbo.HighWaterMark = newRubik.QuotaTurbo.HighWaterMark
		config.QuotaTurbo.AlarmWaterMark = newRubik.QuotaTurbo.AlarmWaterMark
		config.QuotaTurbo.SyncInterval = DefaultSyncInterval
	} else {
		e.removeFeatureFromEnabledList(&config.Agent.EnabledFeatures, "quotaTurbo")
		config.QuotaTurbo = nil
	}

	// 更新 DynMemory
	if newRubik.DynMemory != nil && newRubik.DynMemory.Enable {
		e.addFeatureToEnabledList(&config.Agent.EnabledFeatures, "dynMemory")
		if config.DynMemory == nil {
			config.DynMemory = &DynMemoryFullConfig{}
		}
		config.DynMemory.Policy = "fssr"
	} else {
		e.removeFeatureFromEnabledList(&config.Agent.EnabledFeatures, "dynMemory")
		config.DynMemory = nil
	}

	// 更新 DynCache
	if newRubik.DynCache != nil && newRubik.DynCache.Enable {
		e.addFeatureToEnabledList(&config.Agent.EnabledFeatures, "dynCache")
		if config.DynCache == nil {
			config.DynCache = &DynCacheFullConfig{}
		}
		config.DynCache.DefaultLimitMode = "dynamic"
		config.DynCache.AdjustInterval = DefaultAdjustInterval
		config.DynCache.PerfDuration = DefaultPerfDuration
		config.DynCache.L3Percent = newRubik.DynCache.L3Percent
		config.DynCache.MemBandPercent = newRubik.DynCache.MemBandPercent
	} else {
		e.removeFeatureFromEnabledList(&config.Agent.EnabledFeatures, "dynCache")
		config.DynCache = nil
	}

	// 更新 PSI
	if newRubik.PSI != nil && newRubik.PSI.Enable {
		e.addFeatureToEnabledList(&config.Agent.EnabledFeatures, "psi")
		if config.PSI == nil {
			config.PSI = &PSIFullConfig{}
		}
		config.PSI.Interval = DefaultPSIInterval
		config.PSI.Resource = newRubik.PSI.Resource
		config.PSI.Avg10Threshold = newRubik.PSI.Avg10Threshold
	} else {
		e.removeFeatureFromEnabledList(&config.Agent.EnabledFeatures, "psi")
		config.PSI = nil
	}
}

func (e *ConfigMapEvenHandler) addFeatureToEnabledList(features *[]string, feature string) {
	if !funk.ContainsString(*features, feature) {
		*features = append(*features, feature)
	}
}

func (e *ConfigMapEvenHandler) removeFeatureFromEnabledList(features *[]string, feature string) {
	*features = funk.FilterString(*features, func(s string) bool {
		return s != feature
	})
}

// 会删除rubik pod或者volcano-scheduler pod
// 1. 把所有rubik pod删除一次，之后会自动拉起，因为pod无法监听到rubik configmap的变动
// 2. delete volcano-scheduler pod since when usage scheduler is initiated for the first time, prometheus endpoint
// is not loaded automatically
func (e *ConfigMapEvenHandler) deleteDesiredPods(podPrefix, namespace string) error {
	// 获取匹配的Pod列表
	pods, err := e.client.CoreV1().Pods(namespace).List(context.TODO(), metav1.ListOptions{
		LabelSelector: labels.Everything().String(),
	})
	if err != nil {
		return fmt.Errorf("failed to list pods: %v", err)
	}

	// 过滤并删除Pod
	var deleteErrors []error
	for _, pod := range pods.Items {
		if len(pod.Name) >= len(podPrefix) && pod.Name[:len(podPrefix)] == podPrefix {
			klog.InfoS("Deleting pod", "pod", pod.Name)
			if err := e.client.CoreV1().Pods(namespace).Delete(
				context.TODO(),
				pod.Name,
				metav1.DeleteOptions{},
			); err != nil {
				deleteErrors = append(deleteErrors, fmt.Errorf("failed to delete pod %s: %v", pod.Name, err))
				continue
			}
			klog.InfoS("Successfully deleted pod", "pod", pod.Name)
		}
	}

	// 汇总错误
	if len(deleteErrors) > 0 {
		return fmt.Errorf("encountered %d deletion errors: %v", len(deleteErrors), deleteErrors)
	}

	klog.Info("All matching pods deleted successfully")
	return nil
}

func (e *ConfigMapEvenHandler) getVolcanoConfig(cm *corev1.ConfigMap) (VolcanoSchedulerConfig, error) {
	var result VolcanoSchedulerConfig
	if err := json.Unmarshal([]byte(cm.Data["volcano-scheduler-options"]), &result); err != nil {
		klog.Error(err, "failed to parse volcano-scheduler-options in colocation config")
		return VolcanoSchedulerConfig{}, err
	}

	return result, nil
}

func (e *ConfigMapEvenHandler) getRubikConfig(cm *corev1.ConfigMap) (RubikConfig, error) {
	var result RubikConfig
	if err := json.Unmarshal([]byte(cm.Data["rubik-options"]), &result); err != nil {
		klog.Error(err, "failed to parse rubik-options in colocation config")
		return RubikConfig{}, err
	}

	return result, nil
}

func (e *ConfigMapEvenHandler) passModifiToNodes(oldCM, newCM *corev1.ConfigMap) error {
	newColoNodes, err := getColoNodesInColocationConfig(newCM)
	if err != nil {
		return err
	}
	oldColoNodes, err := getColoNodesInColocationConfig(oldCM)
	if err != nil {
		return err
	}

	if checkNodesModification(newColoNodes, oldColoNodes) {
		klog.Infof("No modification on nodes")
		return err
	}

	klog.Info("old nodes", oldColoNodes)
	klog.Info("new nodes", newColoNodes)

	// lock and unlock
	e.mu.Lock()
	defer e.mu.Unlock()

	nodes, err := e.client.CoreV1().Nodes().List(context.TODO(), metav1.ListOptions{})
	if err != nil {
		klog.Error(err)
		return err
	}

	for _, node := range nodes.Items {
		// 从普通转为混部节点
		if funk.Contains(newColoNodes, node.Name) && !funk.Contains(oldColoNodes, node.Name) {
			klog.Infof("node %s is set to colocation node", node.Name)
			e.addNodelabelsWithRetry(&node)
		}

		// 从混部转为普通节点
		if !funk.Contains(newColoNodes, node.Name) && funk.Contains(oldColoNodes, node.Name) {
			klog.Infof("node %s is set to normal node", node.Name)
			e.deleteNodelabelsWithRetry(&node)
		}
	}
	return nil
}

func checkNodesModification(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}

	// 使用map来统计元素出现次数
	countMap := make(map[string]int)

	for _, item := range a {
		countMap[item]++
	}

	for _, item := range b {
		if count, exists := countMap[item]; !exists || count == 0 {
			return false
		}
		countMap[item]--
	}

	for _, count := range countMap {
		if count != 0 {
			return false
		}
	}

	return true
}

func (e *ConfigMapEvenHandler) handleConfigMapDelete(cm *corev1.ConfigMap) {
	// 旧config需要完全删除才能重建
	waitUntilConfigmapCleaned(e.client, cm)
	oldCM := cm.DeepCopy()
	newCM, err := getDefaultConfigmap()
	if err != nil {
		return
	}

	// 重建默认的config
	e.handleConfigMapEvent(oldCM, newCM, "delete")

	if err := e.createDefaultConfigmap(); err != nil {
		klog.Error(err, "failed to recreate default configmap")
		return
	}
}

func waitUntilConfigmapCleaned(client clientset.Interface, cm *corev1.ConfigMap) {
	logger := klog.Background()
	for {
		_, err := client.CoreV1().ConfigMaps(colocationConfigMapNamespace).Get(
			context.TODO(), cm.Name, metav1.GetOptions{},
		)
		if err != nil && errors.IsNotFound(err) {
			logger.Info("configmap has been fully cleaned up, default configmap should ce created",
				"default configmap:", cm.Name, "deafult namespace:", cm.Namespace)
			break
		}

		logger.Info("onDelete: configmap has marked deleted, but finalizer is not finished",
			"default configmap:", cm.Name, "default namespace:", cm.Namespace)
	}
	return
}

func (e *ConfigMapEvenHandler) addNodelabelsWithRetry(n *corev1.Node) {
	err := retry.RetryOnConflict(retry.DefaultRetry, func() error {
		// 每次重试都重新获取最新的 Node 对象
		freshNode, err := e.client.CoreV1().Nodes().Get(context.TODO(), n.Name, metav1.GetOptions{})
		if err != nil {
			return err
		}

		return addNodelabels(e.client, freshNode)
	})
	if err != nil {
		klog.Error(err, "node name:", n.Name, "failed to modify node labels")
		return
	}
}

func (e *ConfigMapEvenHandler) deleteNodelabelsWithRetry(n *corev1.Node) {
	err := retry.RetryOnConflict(retry.DefaultRetry, func() error {
		// 每次重试都重新获取最新的 Node 对象
		freshNode, err := e.client.CoreV1().Nodes().Get(context.TODO(), n.Name, metav1.GetOptions{})
		if err != nil {
			return err
		}
		return deleteNodelabels(e.client, freshNode)
	})
	if err != nil {
		klog.Error(err, "node name:", n.Name, "failed to modify node labels")
		return
	}
}

func getDefaultColocationOptions() *ColocationOptions {
	return &ColocationOptions{
		Enable:      false,
		NodesConfig: []NodeConfig{},
	}
}

func getDefaultRubikConfig() *RubikConfig {
	return &RubikConfig{
		Eviction: RubikEviction{
			Enabled: false,
		},
		QuotaTurbo: &QuotaTurboConfig{
			Enable: false,
		},
		DynMemory: &DynMemoryConfig{
			Enable: false,
		},
		DynCache: &DynCacheConfig{
			Enable: false,
		},
		PSI: &PSIConfig{
			Enable:   false,
			Resource: []string{"cpu", "memory"},
		},
	}
}

func getDefaultVolcanoSchedulerConfig() *VolcanoSchedulerConfig {
	return &VolcanoSchedulerConfig{
		UsagePlugin: UsagePluginConfig{
			Enabled: false,
			UsageThreshold: UsageThresholdConfig{
				Enabled: false,
			},
		},
	}
}

func getDefaultConfigmap() (*corev1.ConfigMap, error) {
	co := getDefaultColocationOptions()
	ro := getDefaultRubikConfig()
	vo := getDefaultVolcanoSchedulerConfig()

	coData, err1 := json.Marshal(co)
	roData, err2 := json.Marshal(ro)
	voData, err3 := json.Marshal(vo)
	if err1 != nil || err2 != nil || err3 != nil {
		return nil, fmt.Errorf("failed to marshal default config")
	}

	cm := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      ColocationConfigMapName,
			Namespace: colocationConfigMapNamespace,
		},
		Data: map[string]string{
			"colocation-options":        string(coData),
			"rubik-options":             string(roData),
			"volcano-scheduler-options": string(voData),
		},
	}
	return cm, nil
}

func (e *ConfigMapEvenHandler) createDefaultConfigmap() error {

	defaultCM, err := getDefaultConfigmap()
	if err != nil {
		return err
	}

	_, err2 := e.client.CoreV1().ConfigMaps(colocationConfigMapNamespace).Create(
		context.TODO(), defaultCM, metav1.CreateOptions{})

	return err2
}
