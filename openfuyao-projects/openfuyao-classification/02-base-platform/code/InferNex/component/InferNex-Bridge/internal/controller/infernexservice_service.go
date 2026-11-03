/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
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
	"context"
	"fmt"
	"strings"

	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"

	infernexv1alpha1 "gitcode.com/openFuyao/InferNex/api/v1alpha1"
)

const (
	defaultComponentServicePort  = 8000
	cacheIndexerServicePort      = 28080
	managedServiceLabelKey       = "infernex.io/managed-service"
	managedServiceLabelValueTrue = "true"
	pdWorkloadServiceComponent   = "pd-workload"
	// hermesEPPContainerName is the required Hermes / gateway-api-inference-extension
	// container name in InferNexService router templates. Sidecars (tokenizer, prediction, …)
	// may appear in any order; Service targetPorts are resolved from this container only.
	hermesEPPContainerName = "main"
)

func (r *InferNexServiceReconciler) reconcileComponentService(
	ctx context.Context,
	owner *infernexv1alpha1.InferNexService,
	component string,
	plan componentPlan,
) error {
	if plan.ServicePort <= 0 || plan.DisableService {
		return nil
	}
	sel := componentServiceSelector(owner, component, plan)
	target := firstContainerTargetPort(component, plan.Template)
	serviceComponent := managedServiceComponentLabel(owner, component, plan)
	singleton := isSingletonComponentService(component)
	for _, name := range componentServiceNamesForPlan(owner, component, plan) {
		svc := &corev1.Service{ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: owner.Namespace}}
		if singleton {
			// A singleton Service is initialized once. After creation, neither this
			// owner nor another owner may rewrite its selector, ports, or labels.
			// External canonical Services are therefore naturally preserved as well.
			existing := &corev1.Service{}
			if err := r.Get(ctx, client.ObjectKeyFromObject(svc), existing); err == nil {
				return nil
			} else if !apierrors.IsNotFound(err) {
				return err
			}
		}
		if _, err := controllerutil.CreateOrUpdate(ctx, r.Client, svc, func() error {
			if singleton {
				svc.Labels = singletonServiceLabels(component)
				svc.OwnerReferences = nil
			} else {
				svc.Labels = serviceManagerLabels(owner.Name, serviceComponent)
			}
			svc.Spec.Type = corev1.ServiceTypeClusterIP
			svc.Spec.Selector = sel
			svc.Spec.Ports = componentServicePorts(component, plan, target)
			if singleton {
				// Mooncake Services are namespace singletons. Their Deployment and
				// selector use the same fixed identity; no InferNexService owns them.
				return nil
			}
			return controllerutil.SetControllerReference(owner, svc, r.Scheme)
		}); err != nil {
			return err
		}
	}
	return nil
}

// isSingletonComponentService reports whether the component uses a
// namespace-scoped singleton Service and Deployment.
func isSingletonComponentService(component string) bool {
	switch component {
	case "mooncake-master", "mooncake-metadata":
		return true
	default:
		return false
	}
}

func isSingletonServiceName(name string) bool {
	switch name {
	case "mooncake-master-service", "redis-service":
		return true
	default:
		return false
	}
}

func mooncakeSingletonServiceName(component string) string {
	switch component {
	case "mooncake-master":
		return "mooncake-master-service"
	case "mooncake-metadata":
		return "redis-service"
	default:
		return ""
	}
}

func isBridgeManagedMooncakeService(svc *corev1.Service) bool {
	if svc == nil {
		return false
	}
	labels := svc.Labels
	if labels["infernex.io/singleton"] == "mooncake" {
		return true
	}
	// Services created by older Bridge versions had the generic managed-service
	// label but no singleton marker. Require the canonical Mooncake component
	// label as well, so an arbitrary external Service is never taken over.
	return labels[managedServiceLabelKey] == managedServiceLabelValueTrue &&
		labels["app.kubernetes.io/name"] == "infernex-component" &&
		(labels["infernex.io/component"] == "mooncake-master" ||
			labels["infernex.io/component"] == "mooncake-metadata")
}

func singletonServiceLabels(component string) map[string]string {
	labels := map[string]string{
		"app.kubernetes.io/managed-by": "infernex-bridge",
		"infernex.io/singleton":        "mooncake",
		"infernex.io/managed-service":  managedServiceLabelValueTrue,
		"infernex.io/component":        component,
	}
	switch component {
	case "mooncake-master":
		labels["app"] = "mooncake-master"
	case "mooncake-metadata":
		labels["app"] = "redis"
	default:
		return nil
	}
	return labels
}

// migrateLegacySingletonServices converts old Bridge-managed canonical Services
// to the owner-independent singleton shape. It is deliberately namespace-scoped:
// a Service can contain owner references for several old InferNexServices, and
// removing only the currently reconciled owner would leave Kubernetes GC able to
// delete the shared Service later.
func (r *InferNexServiceReconciler) migrateLegacySingletonServices(ctx context.Context, namespace string) error {
	for _, component := range []string{"mooncake-master", "mooncake-metadata"} {
		svc := &corev1.Service{ObjectMeta: metav1.ObjectMeta{
			Name:      mooncakeSingletonServiceName(component),
			Namespace: namespace,
		}}
		if err := r.Get(ctx, client.ObjectKeyFromObject(svc), svc); err != nil {
			if apierrors.IsNotFound(err) {
				continue
			}
			return err
		}
		if !isBridgeManagedMooncakeService(svc) {
			continue
		}

		base := svc.DeepCopy()
		changed := false
		if svc.Labels["infernex.io/singleton"] != "mooncake" {
			svc.Labels = singletonServiceLabels(component)
			svc.Spec.Selector = mooncakeSingletonWorkloadSelector(component)
			changed = true
		}
		refs := make([]metav1.OwnerReference, 0, len(svc.OwnerReferences))
		for _, ref := range svc.OwnerReferences {
			if ref.Kind == "InferNexService" && strings.HasPrefix(ref.APIVersion, "infernex.io/") {
				changed = true
				continue
			}
			refs = append(refs, ref)
		}
		svc.OwnerReferences = refs
		if !changed {
			continue
		}
		if err := r.Patch(ctx, svc, client.MergeFrom(base)); err != nil && !apierrors.IsNotFound(err) {
			return err
		}
	}
	return nil
}

func (r *InferNexServiceReconciler) pruneComponentServices(
	ctx context.Context,
	owner *infernexv1alpha1.InferNexService,
	desired map[string]componentPlan,
) error {
	var list corev1.ServiceList
	if err := r.List(
		ctx,
		&list,
		client.InNamespace(owner.Namespace),
		client.MatchingLabels{"infernex.io/owner": owner.Name},
	); err != nil {
		return err
	}
	for i := range list.Items {
		s := &list.Items[i]
		// Older Bridge versions put infernex.io/owner on the namespace-wide
		// Mooncake Services. Never let this per-owner pruner delete a canonical
		// singleton merely because the last-writer owner is stopped or disabled.
		if isSingletonServiceName(s.Name) {
			continue
		}
		if s.Labels[managedServiceLabelKey] != managedServiceLabelValueTrue {
			continue
		}
		component := s.Labels["infernex.io/component"]
		if component == "" {
			continue
		}
		if shouldKeepManagedService(owner, desired, component, s.Name) {
			continue
		}
		if err := r.Delete(ctx, s); err != nil && !apierrors.IsNotFound(err) {
			return err
		}
	}
	return nil
}

func serviceManagerLabels(ownerName, component string) map[string]string {
	return map[string]string{
		"app.kubernetes.io/name":      "infernex-component",
		"infernex.io/owner":           ownerName,
		"infernex.io/component":       component,
		"infernex.io/managed-service": "true",
	}
}

func managedServiceComponentLabel(owner *infernexv1alpha1.InferNexService, component string, plan componentPlan) string {
	if usesSharedWorkloadService(owner, component, plan) {
		return pdWorkloadServiceComponent
	}
	return component
}

func componentServiceNamesForPlan(owner *infernexv1alpha1.InferNexService, component string, plan componentPlan) []string {
	if usesSharedWorkloadService(owner, component, plan) {
		return []string{pdWorkloadServiceName(owner.Name)}
	}
	return componentServiceNames(owner, component)
}

func componentServiceNames(owner *infernexv1alpha1.InferNexService, component string) []string {
	switch component {
	case "mooncake-master", "mooncake-metadata":
		return []string{mooncakeSingletonServiceName(component)}
	case "engine-pd-prefill", "engine-pd-decode":
		if usesSharedPDWorkloadService(owner, component) {
			return []string{pdWorkloadServiceName(owner.Name)}
		}
		return []string{fmt.Sprintf("%s-%s", owner.Name, component)}
	default:
		return []string{fmt.Sprintf("%s-%s", owner.Name, component)}
	}
}

func shouldKeepComponentServiceForPlan(
	owner *infernexv1alpha1.InferNexService,
	component, serviceName string,
	plan componentPlan,
) bool {
	for _, n := range componentServiceNamesForPlan(owner, component, plan) {
		if serviceName == n {
			return true
		}
	}
	return false
}

func shouldKeepManagedService(owner *infernexv1alpha1.InferNexService, desired map[string]componentPlan, component, serviceName string) bool {
	if owner == nil {
		return false
	}
	if isSingletonComponentService(component) || isSingletonServiceName(serviceName) {
		// Mooncake Services are namespace singletons. In particular, preserve
		// legacy owner-labeled Services during upgrade/stop/disable paths until a
		// normal reconcile can migrate their labels and selector.
		return true
	}
	if component == pdWorkloadServiceComponent {
		for _, key := range []string{"engine-pd-prefill", "engine-pd-decode", "engine-aggregate"} {
			plan, ok := desired[key]
			if !ok || plan.ServicePort <= 0 || plan.DisableService {
				continue
			}
			if shouldKeepComponentServiceForPlan(owner, key, serviceName, plan) {
				return true
			}
		}
		return false
	}
	plan, ok := desired[component]
	return ok && plan.ServicePort > 0 && !plan.DisableService &&
		shouldKeepComponentServiceForPlan(owner, component, serviceName, plan)
}

func componentServiceSelector(owner *infernexv1alpha1.InferNexService, component string, plan componentPlan) map[string]string {
	if isSingletonComponentService(component) {
		return mooncakeSingletonWorkloadSelector(component)
	}
	if usesSharedPDWorkloadService(owner, component) {
		workloadName, _ := pdWorkloadIdentity(owner)
		if workloadName == "" {
			workloadName = owner.Name
		}
		return map[string]string{
			labelAppKubernetesIOName:   workloadName,
			labelAppKubernetesIOPartOf: "infernex",
			labelInfernexOwner:         owner.Name,
		}
	}
	return deploymentPodMatchLabels(owner, component, plan)
}

func usesSharedPDWorkloadService(owner *infernexv1alpha1.InferNexService, component string) bool {
	if owner == nil || owner.Spec.SourceRef != nil {
		return false
	}
	return component == "engine-pd-prefill" || component == "engine-pd-decode"
}

// usesSharedWorkloadService routes ClusterIP traffic through <infsvc>-workload-svc so LWS
// leader StatefulSet serviceName (ex-<infsvc>-engine-aggregate) stays headless for LWS_LEADER_ADDRESS DNS.
func usesSharedWorkloadService(owner *infernexv1alpha1.InferNexService, component string, plan componentPlan) bool {
	if usesSharedPDWorkloadService(owner, component) {
		return true
	}
	if owner == nil || owner.Spec.SourceRef != nil {
		return false
	}
	return component == "engine-aggregate" && plan.WorkloadKind == workloadKindLeaderWorkerSet
}

func pdWorkloadServiceName(ownerName string) string {
	return fmt.Sprintf("%s-workload-svc", ownerName)
}

func hermesEPPContainer(tpl *corev1.PodTemplateSpec) (*corev1.Container, bool) {
	if tpl == nil {
		return nil, false
	}
	for i := range tpl.Spec.Containers {
		if tpl.Spec.Containers[i].Name == hermesEPPContainerName {
			return &tpl.Spec.Containers[i], true
		}
	}
	return nil, false
}

func firstContainerTargetPort(component string, tpl *corev1.PodTemplateSpec) intstr.IntOrString {
	defaultPort := defaultComponentServicePort
	if component == "cache-indexer" {
		defaultPort = cacheIndexerServicePort
	}
	if component == hermesRouterComponent {
		return hermesNamedContainerTargetPort(tpl, "grpc", intstr.FromInt(defaultEndpointPickerPort))
	}
	if tpl == nil || len(tpl.Spec.Containers) == 0 {
		return intstr.FromInt(defaultPort)
	}
	ports := tpl.Spec.Containers[0].Ports
	if len(ports) == 0 {
		return intstr.FromInt(defaultPort)
	}
	p := ports[0]
	if p.Name != "" {
		return intstr.FromString(p.Name)
	}
	if p.ContainerPort > 0 {
		return intstr.FromInt(int(p.ContainerPort))
	}
	return intstr.FromInt(defaultPort)
}

func componentServicePorts(component string, plan componentPlan, defaultTarget intstr.IntOrString) []corev1.ServicePort {
	if component != hermesRouterComponent {
		return []corev1.ServicePort{{
			Name:       "http",
			Port:       plan.ServicePort,
			Protocol:   corev1.ProtocolTCP,
			TargetPort: defaultTarget,
		}}
	}

	ports := []corev1.ServicePort{{
		Name:       "grpc",
		Port:       plan.ServicePort,
		Protocol:   corev1.ProtocolTCP,
		TargetPort: hermesNamedContainerTargetPort(plan.Template, "grpc", defaultTarget),
	}}

	if healthPort, ok := hermesNamedContainerPort(plan.Template, "grpc-health"); ok {
		ports = append(ports, corev1.ServicePort{
			Name:       "grpc-health",
			Port:       healthPort,
			Protocol:   corev1.ProtocolTCP,
			TargetPort: intstr.FromString("grpc-health"),
		})
	}

	return ports
}

func hermesNamedContainerTargetPort(tpl *corev1.PodTemplateSpec, portName string, fallback intstr.IntOrString) intstr.IntOrString {
	c, ok := hermesEPPContainer(tpl)
	if !ok {
		return fallback
	}
	return namedPortTargetOnContainer(c, portName, fallback)
}

func hermesNamedContainerPort(tpl *corev1.PodTemplateSpec, portName string) (int32, bool) {
	c, ok := hermesEPPContainer(tpl)
	if !ok {
		return 0, false
	}
	return namedPortOnContainer(c, portName)
}

func namedPortTargetOnContainer(c *corev1.Container, portName string, fallback intstr.IntOrString) intstr.IntOrString {
	if c == nil {
		return fallback
	}
	for _, p := range c.Ports {
		if p.Name != portName {
			continue
		}
		if p.Name != "" {
			return intstr.FromString(p.Name)
		}
		if p.ContainerPort > 0 {
			return intstr.FromInt(int(p.ContainerPort))
		}
	}
	return fallback
}

func namedPortOnContainer(c *corev1.Container, portName string) (int32, bool) {
	if c == nil {
		return 0, false
	}
	for _, p := range c.Ports {
		if p.Name == portName && p.ContainerPort > 0 {
			return p.ContainerPort, true
		}
	}
	return 0, false
}
