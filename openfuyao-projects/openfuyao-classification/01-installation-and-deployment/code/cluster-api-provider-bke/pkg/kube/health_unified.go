/******************************************************************
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain n copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package kube

import (
	stderrors "errors"
	"fmt"
	"strings"
	"sync"
	"time"

	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	kerrors "k8s.io/apimachinery/pkg/util/errors"
	"sigs.k8s.io/yaml"

	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	healthconfig "gopkg.openfuyao.cn/cluster-api-provider-bke/config/health"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/log"
)

type HealthCheckPriority int

const (
	PriorityCritical HealthCheckPriority = iota
	PriorityImportant
	PriorityOptional
)

func (p HealthCheckPriority) String() string {
	switch p {
	case PriorityCritical:
		return "critical"
	case PriorityImportant:
		return "important"
	case PriorityOptional:
		return "optional"
	default:
		return "unknown"
	}
}

type ComponentName string

const (
	NameEtcd                  ComponentName = "etcd"
	NameKubeAPIServer         ComponentName = "kube-apiserver"
	NameKubeControllerManager ComponentName = "kube-controller-manager"
	NameKubeScheduler         ComponentName = "kube-scheduler"
	NameCalicoNode            ComponentName = "calico-node"
	NameCalicoKubeControllers ComponentName = "calico-kube-controllers"
	NameKubeProxy             ComponentName = "kube-proxy"
	NameCoreDNS               ComponentName = "coredns"
	NameNode                  ComponentName = "node"
)

type ComponentInfo struct {
	Name      ComponentName
	Namespace string
	Prefix    string
	PodName   string
	Priority  HealthCheckPriority
}

func (c ComponentInfo) String() string {
	if c.Namespace == "" {
		return string(c.Name)
	}
	if c.PodName != "" {
		return fmt.Sprintf("%s/%s", c.Namespace, c.PodName)
	}
	if c.Prefix != "" {
		return fmt.Sprintf("%s/%s(%s)", c.Namespace, c.Prefix, c.Name)
	}
	return fmt.Sprintf("%s/%s", c.Namespace, c.Name)
}

type HealthCheckError struct {
	Component ComponentInfo
	Reason    string
	Err       error
}

func (e *HealthCheckError) Error() string {
	return fmt.Sprintf("[%s] %s (%s): %v", e.Component.Priority, e.Component, e.Reason, e.Err)
}

func (e *HealthCheckError) Unwrap() error {
	return e.Err
}

type IntervalConfig struct {
	Critical  time.Duration
	Important time.Duration
	Optional  time.Duration
	Normal    time.Duration
}

type HealthCheckConfig struct {
	Intervals  IntervalConfig
	Components []ComponentCheck
}

type HealthCheckResult struct {
	NodeErrors               []error
	CriticalComponentErrors  []error
	ImportantComponentErrors []error
	OptionalComponentErrors  []error
}

type UnifiedHealthChecker struct {
	client *Client
	log    *log.Logger
	cache  *healthCheckCache
	config HealthCheckConfig
}

// DefaultHealthCheckConfig 从默认 YAML 加载健康检查配置。
// 8 个核心组件只维护在 config/health/config.yaml 中，代码里仅保留解析和兜底逻辑。
func DefaultHealthCheckConfig() HealthCheckConfig {
	config, err := parseHealthCheckConfigYAML(healthconfig.DefaultConfig)
	if err != nil {
		log.Warnf("failed to parse embedded health check config, using fallback: %v", err)
		return fallbackHealthCheckConfig()
	}
	return config
}

// LoadHealthCheckConfig 优先从目标集群的 ConfigMap 加载健康检查配置，失败时回退到默认 YAML。
func (c *Client) LoadHealthCheckConfig() HealthCheckConfig {
	config, err := c.loadHealthCheckConfig()
	if err != nil {
		if c != nil && c.Log != nil {
			c.Log.Warnf("failed to load health check config from ConfigMap %s/%s, using default: %v", healthconfig.Namespace, healthconfig.Name, err)
		} else {
			log.Warnf("failed to load health check config from ConfigMap %s/%s, using default: %v", healthconfig.Namespace, healthconfig.Name, err)
		}
		return DefaultHealthCheckConfig()
	}
	return config
}

func (c *Client) loadHealthCheckConfig() (HealthCheckConfig, error) {
	if c == nil || c.ClientSet == nil {
		return HealthCheckConfig{}, fmt.Errorf("kubernetes clientset is nil")
	}
	cm, err := c.ClientSet.CoreV1().ConfigMaps(healthconfig.Namespace).Get(c.context(), healthconfig.Name, metav1.GetOptions{})
	if err != nil {
		if apierrors.IsNotFound(err) {
			return HealthCheckConfig{}, fmt.Errorf("ConfigMap not found")
		}
		return HealthCheckConfig{}, err
	}
	data := strings.TrimSpace(cm.Data[healthconfig.DataKey])
	if data == "" {
		return HealthCheckConfig{}, fmt.Errorf("%s is empty", healthconfig.DataKey)
	}
	return parseHealthCheckConfigYAML(data)
}

type healthCheckConfigFile struct {
	Intervals  healthCheckIntervalFile    `json:"intervals" yaml:"intervals"`
	Components []healthCheckComponentFile `json:"components" yaml:"components"`
}

type healthCheckIntervalFile struct {
	Critical  string `json:"critical" yaml:"critical"`
	Important string `json:"important" yaml:"important"`
	Optional  string `json:"optional" yaml:"optional"`
	Normal    string `json:"normal" yaml:"normal"`
}

type healthCheckComponentFile struct {
	Name      string   `json:"name" yaml:"name"`
	Namespace string   `json:"namespace" yaml:"namespace"`
	Prefixes  []string `json:"prefixes" yaml:"prefixes"`
	Priority  string   `json:"priority" yaml:"priority"`
}

func parseHealthCheckConfigYAML(data string) (HealthCheckConfig, error) {
	var raw healthCheckConfigFile
	if err := yaml.Unmarshal([]byte(data), &raw); err != nil {
		return HealthCheckConfig{}, err
	}
	intervals, err := parseIntervalConfig(raw.Intervals)
	if err != nil {
		return HealthCheckConfig{}, err
	}
	if len(raw.Components) == 0 {
		return HealthCheckConfig{}, fmt.Errorf("components is empty")
	}
	components := make([]ComponentCheck, 0, len(raw.Components))
	for _, rawComponent := range raw.Components {
		component, err := parseComponentCheck(rawComponent)
		if err != nil {
			return HealthCheckConfig{}, err
		}
		components = append(components, component)
	}
	return HealthCheckConfig{Intervals: intervals, Components: components}, nil
}

func parseIntervalConfig(raw healthCheckIntervalFile) (IntervalConfig, error) {
	defaults := fallbackHealthCheckConfig().Intervals
	critical, err := parseDurationOrDefault(raw.Critical, defaults.Critical)
	if err != nil {
		return IntervalConfig{}, fmt.Errorf("invalid critical interval: %w", err)
	}
	important, err := parseDurationOrDefault(raw.Important, defaults.Important)
	if err != nil {
		return IntervalConfig{}, fmt.Errorf("invalid important interval: %w", err)
	}
	optional, err := parseDurationOrDefault(raw.Optional, defaults.Optional)
	if err != nil {
		return IntervalConfig{}, fmt.Errorf("invalid optional interval: %w", err)
	}
	normal, err := parseDurationOrDefault(raw.Normal, defaults.Normal)
	if err != nil {
		return IntervalConfig{}, fmt.Errorf("invalid normal interval: %w", err)
	}
	return IntervalConfig{Critical: critical, Important: important, Optional: optional, Normal: normal}, nil
}

func parseDurationOrDefault(value string, defaultValue time.Duration) (time.Duration, error) {
	if strings.TrimSpace(value) == "" {
		return defaultValue, nil
	}
	return time.ParseDuration(value)
}

func parseComponentCheck(raw healthCheckComponentFile) (ComponentCheck, error) {
	name := strings.TrimSpace(raw.Name)
	if name == "" {
		return ComponentCheck{}, fmt.Errorf("component name is empty")
	}
	namespace := strings.TrimSpace(raw.Namespace)
	if namespace == "" {
		return ComponentCheck{}, fmt.Errorf("component %s namespace is empty", name)
	}
	if len(raw.Prefixes) == 0 {
		return ComponentCheck{}, fmt.Errorf("component %s prefixes is empty", name)
	}
	priority, err := parseHealthCheckPriority(raw.Priority)
	if err != nil {
		return ComponentCheck{}, fmt.Errorf("component %s priority invalid: %w", name, err)
	}
	prefixes := make([]string, 0, len(raw.Prefixes))
	for _, prefix := range raw.Prefixes {
		prefix = strings.TrimSpace(prefix)
		if prefix == "" {
			return ComponentCheck{}, fmt.Errorf("component %s contains empty prefix", name)
		}
		prefixes = append(prefixes, prefix)
	}
	return ComponentCheck{Name: ComponentName(name), Namespace: namespace, Prefixes: prefixes, Priority: priority}, nil
}

func healthCheckConfigSummary(config HealthCheckConfig) string {
	parts := make([]string, 0, len(config.Components))
	for _, component := range config.Components {
		parts = append(parts, fmt.Sprintf("%s:%s:%s:%s", component.Name, component.Namespace, strings.Join(component.Prefixes, ","), component.Priority))
	}
	return strings.Join(parts, ";")
}

func parseHealthCheckPriority(value string) (HealthCheckPriority, error) {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case PriorityCritical.String():
		return PriorityCritical, nil
	case PriorityImportant.String():
		return PriorityImportant, nil
	case PriorityOptional.String():
		return PriorityOptional, nil
	default:
		return PriorityOptional, fmt.Errorf("unknown priority %q", value)
	}
}

func fallbackHealthCheckConfig() HealthCheckConfig {
	return HealthCheckConfig{
		Intervals: IntervalConfig{
			Critical:  5 * time.Second,
			Important: 15 * time.Second,
			Optional:  30 * time.Second,
			Normal:    5 * time.Minute,
		},
	}
}

func NewUnifiedHealthChecker(client *Client, config HealthCheckConfig) *UnifiedHealthChecker {
	return &UnifiedHealthChecker{
		client: client,
		log:    client.Log,
		cache:  newHealthCheckCache(client),
		config: config,
	}
}

// Check 按节点、关键组件、重要组件、可选组件的顺序执行健康检查。
// 节点和同一优先级内的组件并行检查；关键组件失败会立即返回。
func (h *UnifiedHealthChecker) Check(cluster *bkev1beta1.BKECluster, currentVersion string, bkeNodes bkev1beta1.BKENodes) error {
	startedAt := time.Now()
	if currentVersion == "" {
		currentVersion = cluster.Spec.ClusterConfig.Cluster.KubernetesVersion
	}

	h.log.Infof("core health check start, cluster=%s/%s, version=%s, components=%d", cluster.Namespace, cluster.Name, currentVersion, len(h.config.Components))

	result := &HealthCheckResult{}
	if err := h.checkNodesParallel(currentVersion, bkeNodes, result); err != nil {
		return err
	}

	critical, important, optional := h.groupByPriority()
	if err := h.checkComponentsParallel(critical, result); err != nil {
		return err
	}
	if err := h.checkComponentsParallel(important, result); err != nil {
		h.log.Warnf("important component health check returned error: %v", err)
	}
	if err := h.checkComponentsParallel(optional, result); err != nil {
		h.log.Debugf("optional component health check returned error: %v", err)
	}

	if err := h.aggregateResult(result); err != nil {
		return err
	}

	h.log.Infof("core health check finished, cluster=%s/%s, duration=%s", cluster.Namespace, cluster.Name, time.Since(startedAt))
	return nil
}

func (h *UnifiedHealthChecker) groupByPriority() (critical, important, optional []ComponentCheck) {
	for _, component := range h.config.Components {
		switch component.Priority {
		case PriorityCritical:
			critical = append(critical, component)
		case PriorityImportant:
			important = append(important, component)
		default:
			optional = append(optional, component)
		}
	}
	return critical, important, optional
}

func (h *UnifiedHealthChecker) checkNodesParallel(currentVersion string, bkeNodes bkev1beta1.BKENodes, result *HealthCheckResult) error {
	startedAt := time.Now()
	nodeLi, err := h.cache.GetNodes()
	if err != nil {
		return newNodeError("", "ListNodesFailed", err)
	}
	h.log.Infof("node health check start, total=%d, version=%s", len(nodeLi.Items), currentVersion)

	var errs []error
	skippedNodes := 0
	var mu sync.Mutex
	var statusMu sync.Mutex
	var wg sync.WaitGroup
	for _, node := range nodeLi.Items {
		node := node
		if bkeNodes.GetNodeStateNeedSkip(GetNodeIP(&node)) {
			skippedNodes++
		}
		wg.Add(1)
		go func() {
			defer wg.Done()

			if err := h.checkNode(&node, currentVersion, bkeNodes, &statusMu); err != nil {
				mu.Lock()
				errs = append(errs, err)
				mu.Unlock()
			}
		}()
	}
	wg.Wait()
	h.log.Infof("node health check finished, total=%d, skipped=%d, failed=%d, duration=%s", len(nodeLi.Items), skippedNodes, len(errs), time.Since(startedAt))
	if len(errs) > 0 {
		result.NodeErrors = append(result.NodeErrors, errs...)
		return kerrors.NewAggregate(errs)
	}
	return nil
}

func (h *UnifiedHealthChecker) checkNode(node *corev1.Node, currentVersion string, bkeNodes bkev1beta1.BKENodes, statusMu *sync.Mutex) error {
	if node == nil {
		return newNodeError("", "NodeNil", fmt.Errorf("node is nil"))
	}

	nodeIP := GetNodeIP(node)
	if bkeNodes.GetNodeStateNeedSkip(nodeIP) {
		h.log.Debugf("node %q (IP: %s) health check skipped due to needskip=true", node.Name, nodeIP)
		return nil
	}

	if NodeReady(node) {
		statusMu.Lock()
		bkeNodes.SetNodeStateWithMessage(nodeIP, confv1beta1.NodeReady, "")
		statusMu.Unlock()
	}

	if err := h.client.NodeHealthCheck(node, currentVersion, h.log); err != nil {
		statusMu.Lock()
		bkeNodes.SetNodeStateWithMessage(nodeIP, confv1beta1.NodeNotReady, err.Error())
		statusMu.Unlock()
		h.log.Debugf("node %q health check failed: %v", node.Name, err)
		return newNodeError(node.Name, "NodeNotReady", err)
	}

	return nil
}

func (h *UnifiedHealthChecker) checkComponentsParallel(components []ComponentCheck, result *HealthCheckResult) error {
	if len(components) == 0 {
		return nil
	}

	priority := components[0].Priority
	startedAt := time.Now()
	h.log.Infof("%s component health check start, total=%d", priority, len(components))

	var errs []error
	var mu sync.Mutex
	var wg sync.WaitGroup
	for _, check := range components {
		check := check
		wg.Add(1)
		go func() {
			defer wg.Done()
			if err := h.checkComponent(check); err != nil {
				mu.Lock()
				errs = append(errs, err)
				mu.Unlock()
			}
		}()
	}
	wg.Wait()
	h.log.Infof("%s component health check finished, total=%d, failed=%d, duration=%s", priority, len(components), len(errs), time.Since(startedAt))

	switch priority {
	case PriorityCritical:
		result.CriticalComponentErrors = append(result.CriticalComponentErrors, errs...)
	case PriorityImportant:
		result.ImportantComponentErrors = append(result.ImportantComponentErrors, errs...)
	default:
		result.OptionalComponentErrors = append(result.OptionalComponentErrors, errs...)
	}

	if len(errs) > 0 && priority == PriorityCritical {
		return kerrors.NewAggregate(errs)
	}
	return nil
}

func (h *UnifiedHealthChecker) checkComponent(check ComponentCheck) error {
	pods, err := h.cache.GetPods(check.Namespace)
	if err != nil {
		return newComponentError(check, "", "", "ListPodsFailed", err)
	}

	var errs []error
	for _, prefix := range check.Prefixes {
		if err := h.verifyComponentPods(check, pods, prefix); err != nil {
			errs = append(errs, err)
		}
	}
	return kerrors.NewAggregate(errs)
}

func (h *UnifiedHealthChecker) verifyComponentPods(check ComponentCheck, pods []corev1.Pod, prefix string) error {
	matched := filterPodsWithPrefix(pods, prefix)
	if len(matched) == 0 {
		return newComponentError(check, prefix, "", "PodNotFound",
			fmt.Errorf("no pods with prefix %q in %s", prefix, check.Namespace))
	}

	if prefix == "coredns" {
		var errs []error
		for _, pod := range matched {
			if err := isPodHealthy(pod); err == nil {
				return nil
			} else {
				errs = append(errs, newComponentError(check, prefix, pod.Name, getPodUnhealthyReason(pod), err))
			}
		}
		return kerrors.NewAggregate(errs)
	}

	var errs []error
	for _, pod := range matched {
		if err := isPodHealthy(pod); err != nil {
			errs = append(errs, newComponentError(check, prefix, pod.Name, getPodUnhealthyReason(pod), err))
		}
	}
	return kerrors.NewAggregate(errs)
}

func (h *UnifiedHealthChecker) aggregateResult(result *HealthCheckResult) error {
	var errs []error
	errs = append(errs, result.NodeErrors...)
	errs = append(errs, result.CriticalComponentErrors...)
	errs = append(errs, result.ImportantComponentErrors...)
	errs = append(errs, result.OptionalComponentErrors...)
	if len(errs) == 0 {
		return nil
	}

	if len(result.ImportantComponentErrors) > 0 {
		h.log.Warnf("important components health check failed: %v", kerrors.NewAggregate(result.ImportantComponentErrors))
	}
	if len(result.OptionalComponentErrors) > 0 {
		h.log.Debugf("optional components health check failed: %v", kerrors.NewAggregate(result.OptionalComponentErrors))
	}
	return kerrors.NewAggregate(errs)
}

func newComponentError(check ComponentCheck, prefix, podName, reason string, err error) *HealthCheckError {
	return &HealthCheckError{
		Component: ComponentInfo{
			Name:      check.Name,
			Namespace: check.Namespace,
			Prefix:    prefix,
			PodName:   podName,
			Priority:  check.Priority,
		},
		Reason: reason,
		Err:    err,
	}
}

func newNodeError(nodeName, reason string, err error) *HealthCheckError {
	name := NameNode
	if nodeName != "" {
		name = ComponentName(nodeName)
	}
	return &HealthCheckError{
		Component: ComponentInfo{
			Name:     name,
			Priority: PriorityCritical,
		},
		Reason: reason,
		Err:    err,
	}
}

func getPodUnhealthyReason(pod corev1.Pod) string {
	if pod.Status.Phase != corev1.PodRunning {
		return "PodNotRunning"
	}
	if !isPodReadyConditionTrue(pod) {
		return "PodNotReady"
	}
	for _, cs := range pod.Status.InitContainerStatuses {
		if cs.State.Waiting != nil && isFatalWaitingReason(cs.State.Waiting.Reason) {
			return cs.State.Waiting.Reason
		}
	}
	for _, cs := range pod.Status.ContainerStatuses {
		if cs.State.Waiting != nil && isFatalWaitingReason(cs.State.Waiting.Reason) {
			return cs.State.Waiting.Reason
		}
	}
	return "PodUnhealthy"
}

func IsCriticalHealthCheckError(err error) bool {
	return hasPriority(err, PriorityCritical)
}

func IsImportantHealthCheckError(err error) bool {
	return hasPriority(err, PriorityImportant)
}

func hasPriority(err error, target HealthCheckPriority) bool {
	if err == nil {
		return false
	}
	var hcErr *HealthCheckError
	if stderrors.As(err, &hcErr) && hcErr != nil && hcErr.Component.Priority == target {
		return true
	}
	if agg, ok := err.(kerrors.Aggregate); ok {
		for _, e := range agg.Errors() {
			if hasPriority(e, target) {
				return true
			}
		}
	}
	type multiUnwrapper interface {
		Unwrap() []error
	}
	if multi, ok := err.(multiUnwrapper); ok {
		for _, e := range multi.Unwrap() {
			if hasPriority(e, target) {
				return true
			}
		}
	}
	type unwrapper interface {
		Unwrap() error
	}
	if wrapped, ok := err.(unwrapper); ok {
		return hasPriority(wrapped.Unwrap(), target)
	}
	return false
}

func ComponentErrorsByPriority(err error, priority HealthCheckPriority) []*HealthCheckError {
	if err == nil {
		return nil
	}
	var result []*HealthCheckError
	var hcErr *HealthCheckError
	if stderrors.As(err, &hcErr) && hcErr != nil && hcErr.Component.Priority == priority {
		result = append(result, hcErr)
	}
	if agg, ok := err.(kerrors.Aggregate); ok {
		for _, e := range agg.Errors() {
			result = append(result, ComponentErrorsByPriority(e, priority)...)
		}
	}
	return result
}

// GetHealthCheckRequeueInterval 根据健康检查错误中的最高优先级决定下次重试间隔。
type healthCheckRequeueError struct {
	err       error
	intervals IntervalConfig
}

func (e *healthCheckRequeueError) Error() string {
	return e.err.Error()
}

func (e *healthCheckRequeueError) Unwrap() error {
	return e.err
}

func (e *healthCheckRequeueError) RequeueInterval() time.Duration {
	return getHealthCheckRequeueInterval(e.err, e.intervals)
}

func newHealthCheckRequeueError(err error, intervals IntervalConfig) error {
	if err == nil {
		return nil
	}
	return &healthCheckRequeueError{err: err, intervals: intervals}
}

// GetHealthCheckRequeueInterval 根据健康检查错误中的最高优先级决定下次重试间隔。
func GetHealthCheckRequeueInterval(err error) time.Duration {
	var requeueErr *healthCheckRequeueError
	if stderrors.As(err, &requeueErr) && requeueErr != nil {
		return requeueErr.RequeueInterval()
	}
	return getHealthCheckRequeueInterval(err, DefaultHealthCheckConfig().Intervals)
}

func getHealthCheckRequeueInterval(err error, intervals IntervalConfig) time.Duration {
	if err == nil {
		return intervals.Normal
	}
	if IsCriticalHealthCheckError(err) {
		return intervals.Critical
	}
	if IsImportantHealthCheckError(err) {
		return intervals.Important
	}
	return intervals.Optional
}
