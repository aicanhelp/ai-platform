/******************************************************************
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * cluster-api-provider-bke is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package kube

import (
	"fmt"
	"strings"

	"github.com/pkg/errors"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	kerrors "k8s.io/apimachinery/pkg/util/errors"

	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/bkeagent/mfutil"
	labelhelper "gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/label"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/log"
)

// CheckClusterHealth 先执行优化后的核心健康检查，再继续执行原有 addon 健康检查。
// 核心健康检查覆盖节点和规格中的 8 个基础组件；addon 续查用于保留旧行为。
func (c *Client) CheckClusterHealth(cluster *bkev1beta1.BKECluster, currentVersion string, bkeNodes bkev1beta1.BKENodes) error {
	c.Log.Infof("cluster %s/%s health check start", cluster.Namespace, cluster.Name)
	config := c.LoadHealthCheckConfig()
	c.Log.Infof("cluster %s/%s health check config loaded, intervals critical=%s important=%s optional=%s normal=%s, components=%s",
		cluster.Namespace, cluster.Name, config.Intervals.Critical, config.Intervals.Important, config.Intervals.Optional, config.Intervals.Normal, healthCheckConfigSummary(config))
	checker := NewUnifiedHealthChecker(c, config)
	if err := checker.Check(cluster, currentVersion, bkeNodes); err != nil {
		return newHealthCheckRequeueError(err, config.Intervals)
	}
	c.Log.Infof("cluster %s/%s core health check pass, continue addon health check", cluster.Namespace, cluster.Name)
	if err := c.CheckAddonComponentsHealth(cluster, c.Log); err != nil {
		return err
	}
	c.Log.Infof("cluster %s/%s health check pass", cluster.Namespace, cluster.Name)
	return nil
}

func (c *Client) CheckComponentHealth(node *corev1.Node) error {
	var errs []error
	for _, component := range mfutil.GetControlPlaneComponents() {
		pod, err := c.GetPod(metav1.NamespaceSystem, StaticPodName(component, node.Name))
		if err != nil {
			errs = append(errs, errors.Errorf("get pod %s/%s failed: %v", metav1.NamespaceSystem, StaticPodName(component, node.Name), err))
			continue
		}
		if pod.Status.Phase != corev1.PodRunning {
			errs = append(errs, errors.Errorf("pod %s/%s is not in running phase, current phase is %q", metav1.NamespaceSystem, StaticPodName(component, node.Name), pod.Status.Phase))
		}
	}
	return kerrors.NewAggregate(errs)
}

// ComponentCheck 定义需要检查的命名空间和对应的 Pod 前缀
type ComponentCheck struct {
	Name      ComponentName
	Namespace string
	Prefixes  []string
	Priority  HealthCheckPriority
}

type AddonCheck struct {
	Addon      string
	Components []ComponentCheck
}

// 可选安装的扩展件：仅当集群 Spec.Addons 中包含对应 addon 时才执行健康检查
var extraAddonComponents = []AddonCheck{
	{
		Addon: "coredns",
		Components: []ComponentCheck{
			{
				Namespace: "kube-system",
				Prefixes:  []string{"coredns"},
			},
		},
	},
	{
		Addon: "kubeproxy",
		Components: []ComponentCheck{
			{
				Namespace: "kube-system",
				Prefixes:  []string{"kube-proxy-"},
			},
		},
	},
	{
		Addon: "calico",
		Components: []ComponentCheck{
			{
				Namespace: "kube-system",
				Prefixes: []string{
					"calico-kube-controllers-",
					"calico-node-",
				},
			},
		},
	},
	{
		Addon: "cluster-api",
		Components: []ComponentCheck{
			{
				Namespace: "cluster-system",
				Prefixes: []string{
					"capi-controller-manager",
					"bke-controller-manager"},
			},
		},
	},
	{
		Addon: "openfuyao-system-controller",
		Components: []ComponentCheck{
			{
				Namespace: "kube-system",
				Prefixes:  []string{"metrics-server-"},
			},
			{
				Namespace: "ingress-nginx",
				Prefixes:  []string{"ingress-nginx-controller"},
			},
			{
				Namespace: "monitoring",
				Prefixes: []string{
					"alertmanager-main-",
					"blackbox-exporter-",
					"kube-state-metrics-",
					"node-exporter-",
					"prometheus-k8s-",
					"prometheus-operator-",
				},
			},
			{
				Namespace: "openfuyao-system",
				Prefixes: []string{
					"application-management-service-",
					"console-service-",
					"console-website-",
					"local-harbor-", // 匹配所有 local-harbor- 开头的 Pod
					"marketplace-service-",
					"monitoring-service-",
					"oauth-server-",
					"oauth-webhook-",
					"plugin-management-service-",
					"user-management-operator-",
					"web-terminal-service-",
				},
			},
			{
				Namespace: "openfuyao-system-controller",
				Prefixes:  []string{"openfuyao-system-controller-"},
			},
		},
	},
}

// 全局配置：定义集群基础控制面组件（不含可选 addon 如 coredns、kube-proxy、calico）
var neededComponentChecks = []ComponentCheck{
	{
		Namespace: "kube-system",
		Prefixes: []string{
			"etcd-",
			"kube-apiserver-",
			"kube-controller-manager-",
			"kube-scheduler-",
		},
	},
}

// CheckAllComponentsHealth check all components health
func (c *Client) CheckAllComponentsHealth(cluster *bkev1beta1.BKECluster, log *log.Logger) error {
	var errs []error
	for _, check := range neededComponentChecks {
		if err := c.processComponentCheck(check); err != nil {
			errs = append(errs, err)
		}
	}
	if len(errs) > 0 {
		return kerrors.NewAggregate(errs)
	}

	addons := cluster.Spec.ClusterConfig.Addons

	for _, addon := range addons {
		_, needCheck := findAddonComponent(addon.Name)
		if !needCheck {
			log.Debugf("addon %q is not in extraAddonComponents, skip health check", addon.Name)
			continue
		}
		if err := c.processAddonComponentCheck(addon.Name); err != nil {
			errs = append(errs, err)
		}
	}
	return kerrors.NewAggregate(errs)
}

// coreHealthAddons 已经由统一健康检查中的 8 个核心组件覆盖。
// CheckClusterHealth 后续执行 addon 检查时跳过这些 addon，避免重复检查同一批 Pod；
// CheckAllComponentsHealth 保持旧语义，不使用该去重表。
var coreHealthAddons = map[string]struct{}{
	"calico":    {},
	"coredns":   {},
	"kubeproxy": {},
}

// CheckAddonComponentsHealth 保留原 addon 检查路径，但跳过核心健康检查已覆盖的 addon。
func (c *Client) CheckAddonComponentsHealth(cluster *bkev1beta1.BKECluster, log *log.Logger) error {
	var errs []error
	checkedAddons := 0
	skippedCoreAddons := 0
	for _, addon := range cluster.Spec.ClusterConfig.Addons {
		if _, covered := coreHealthAddons[addon.Name]; covered {
			skippedCoreAddons++
			log.Debugf("addon %q is covered by core health check, skip duplicate addon health check", addon.Name)
			continue
		}
		_, needCheck := findAddonComponent(addon.Name)
		if !needCheck {
			log.Debugf("addon %q is not in extraAddonComponents, skip health check", addon.Name)
			continue
		}
		checkedAddons++
		if err := c.processAddonComponentCheck(addon.Name); err != nil {
			errs = append(errs, err)
		}
	}
	log.Infof("addon health check finished, checked=%d, skippedCore=%d, failed=%d", checkedAddons, skippedCoreAddons, len(errs))
	return kerrors.NewAggregate(errs)
}

func (c *Client) processComponentCheck(check ComponentCheck) error {
	pods, err := c.getPods(check.Namespace)
	if err != nil {
		return fmt.Errorf("list pods in %s failed: %w", check.Namespace, err)
	}

	var errs []error
	for _, prefix := range check.Prefixes {
		if err := c.verifyComponentPods(pods, prefix, check.Namespace); err != nil {
			errs = append(errs, err)
		}
	}
	return kerrors.NewAggregate(errs)
}

func findAddonComponent(addon string) (*AddonCheck, bool) {
	for _, addonComponent := range extraAddonComponents {
		if addon == addonComponent.Addon {
			return &addonComponent, true
		}
	}
	return nil, false
}

func (c *Client) processAddonComponentCheck(addon string) error {
	addonComponent, found := findAddonComponent(addon)
	if !found {
		return fmt.Errorf("addon(%v) not in extra addons(%v)", addon, extraAddonComponents)
	}

	var errs []error
	for _, component := range addonComponent.Components {
		if err := c.processComponentCheck(component); err != nil {
			errs = append(errs, err)
		}
	}
	return kerrors.NewAggregate(errs)
}

func (c *Client) getPods(namespace string) ([]corev1.Pod, error) {
	list, err := c.ClientSet.CoreV1().Pods(namespace).List(c.Ctx, metav1.ListOptions{})
	if err != nil {
		return nil, err
	}
	return list.Items, nil
}

func (c *Client) verifyComponentPods(pods []corev1.Pod, prefix string, namespace string) error {
	var errs []error
	matched := filterPodsWithPrefix(pods, prefix)
	if len(matched) == 0 {
		return fmt.Errorf("no pods with prefix '%s' in %s", prefix, namespace)
	}

	// if coredns pod, one coredns pod is ok should be ok.
	if prefix == "coredns" {
		var errs []error
		for _, pod := range matched {
			if err := isPodHealthy(pod); err == nil {
				return nil
			} else {
				errs = append(errs, fmt.Errorf("pod %s/%s unhealthy: %w", pod.Namespace, pod.Name, err))
			}
		}
		return kerrors.NewAggregate(errs)
	}

	for _, pod := range matched {
		if err := isPodHealthy(pod); err != nil {
			errs = append(errs, fmt.Errorf("pod %s/%s unhealthy: %w", pod.Namespace, pod.Name, err))
		}
	}
	return kerrors.NewAggregate(errs)
}

func isPodHealthy(pod corev1.Pod) error {
	if pod.Status.Phase != corev1.PodRunning {
		return fmt.Errorf("status: %s", pod.Status.Phase)
	}

	if !isPodReadyConditionTrue(pod) {
		return fmt.Errorf("pod condition Ready is not True")
	}

	for _, cs := range pod.Status.InitContainerStatuses {
		// init container successfully completed: Terminated with ExitCode=0, skip
		if cs.State.Terminated != nil && cs.State.Terminated.ExitCode == 0 {
			continue
		}
		if !cs.Ready {
			return fmt.Errorf("init container %s not ready", cs.Name)
		}
		if cs.State.Waiting != nil && isFatalWaitingReason(cs.State.Waiting.Reason) {
			return fmt.Errorf("init container %s waiting reason: %s", cs.Name, cs.State.Waiting.Reason)
		}
	}

	for _, cs := range pod.Status.ContainerStatuses {
		if !cs.Ready {
			return fmt.Errorf("container %s not ready", cs.Name)
		}
		if cs.State.Waiting != nil && isFatalWaitingReason(cs.State.Waiting.Reason) {
			return fmt.Errorf("container %s waiting reason: %s", cs.Name, cs.State.Waiting.Reason)
		}
	}

	return nil
}

func isPodReadyConditionTrue(pod corev1.Pod) bool {
	for _, condition := range pod.Status.Conditions {
		if condition.Type == corev1.PodReady {
			return condition.Status == corev1.ConditionTrue
		}
	}
	return false
}

func isFatalWaitingReason(reason string) bool {
	switch reason {
	case "CrashLoopBackOff", "ImagePullBackOff", "ErrImagePull", "CreateContainerConfigError", "CreateContainerError":
		return true
	default:
		return false
	}
}

func filterPodsWithPrefix(pods []corev1.Pod, prefix string) []corev1.Pod {
	filtered := make([]corev1.Pod, 0)
	for _, pod := range pods {
		if strings.HasPrefix(pod.Name, prefix) {
			filtered = append(filtered, pod)
		}
	}
	return filtered
}

// NodeHealthCheck performs a health check on the given node.
func (c *Client) NodeHealthCheck(node *corev1.Node, expectVersion string, log *log.Logger) error {
	return c.nodeHealthCheck(node, expectVersion, log, c.CheckComponentHealth)
}

// 统一的健康检查基础函数
func (c *Client) nodeHealthCheck(
	node *corev1.Node,
	expectVersion string,
	log *log.Logger,
	componentCheckFunc func(*corev1.Node) error,
) error {
	// Step 1: 检查节点就绪状态
	if err := checkNodeReady(node); err != nil {
		return err
	}

	// Step 2: 检查节点版本
	if err := checkNodeVersion(node, expectVersion); err != nil {
		return err
	}

	// Step 3: 主节点组件检查
	if labelhelper.IsMasterNode(node) {
		if err := componentCheckFunc(node); err != nil {
			return err
		}
	}

	log.Debugf("node %q health status pass check", node.Name)
	return nil
}

// 检查节点就绪状态
func checkNodeReady(node *corev1.Node) error {
	for _, condition := range node.Status.Conditions {
		if condition.Type == corev1.NodeReady && condition.Status != corev1.ConditionTrue {
			return fmt.Errorf("node %s is not ready", node.Name)
		}
	}
	return nil
}

// 检查节点版本
func checkNodeVersion(node *corev1.Node, expectVersion string) error {
	if expectVersion == "" {
		return nil
	}
	if node.Status.NodeInfo.KubeletVersion != expectVersion {
		return fmt.Errorf("node %q version %q is not match bkeCluster KubernetesVersion %q",
			node.Name, node.Status.NodeInfo.KubeletVersion, expectVersion)
	}
	return nil
}

func StaticPodName(component, nodeName string) string {
	return fmt.Sprintf("%s-%s", component, nodeName)
}

func NodeReady(node *corev1.Node) bool {
	for _, condition := range node.Status.Conditions {
		if condition.Type == corev1.NodeReady && condition.Status == corev1.ConditionTrue {
			return true
		}
	}
	return false
}
