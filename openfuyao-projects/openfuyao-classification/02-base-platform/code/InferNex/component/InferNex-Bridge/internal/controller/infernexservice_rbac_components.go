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

	corev1 "k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"

	infernexv1alpha1 "gitcode.com/openFuyao/InferNex/api/v1alpha1"
)

const (
	cacheIndexerComponent               = "cache-indexer"
	hermesRouterComponent               = "hermes-router"
	elasticScalerComponent              = "pd-orchestrator-elastic-scaler"
	tidalComponent                      = "pd-orchestrator-tidal"
	rsgComponent                        = "pd-orchestrator-rsg"
	hardwareMonitorComponent            = "eagle-eye-hardware-monitor"
	networkPerformanceExporterComponent = "eagle-eye-network-performance-exporter"
)

func hermesRouterServiceAccountName(ownerName string) string {
	return fmt.Sprintf("%s-hermes-router-sa", ownerName)
}

func hermesRouterRBACObjectNames(ownerName string) (roleName, roleBindingName string) {
	return fmt.Sprintf("%s-hermes-router-role", ownerName), fmt.Sprintf("%s-hermes-router-rolebinding", ownerName)
}

func componentControllerSAName(component string) string {
	switch component {
	case cacheIndexerComponent:
		return "cache-indexer-sa"
	case elasticScalerComponent:
		return "elastic-scaler-sa"
	case tidalComponent:
		return "tidal-sa"
	case rsgComponent:
		return "rsg-sa"
	case hardwareMonitorComponent:
		return "hardware-monitor-serviceaccount"
	case networkPerformanceExporterComponent:
		return "network-performance-exporter"
	default:
		return ""
	}
}

func componentControllerRBACNames(ownerName, component string) (string, string, string, string) {
	clusterRoleName := fmt.Sprintf("%s-%s-controller-role", ownerName, component)
	clusterRoleBindingName := fmt.Sprintf("%s-%s-controller-rolebinding", ownerName, component)
	leaderRoleName := fmt.Sprintf("%s-%s-leader-election-role", ownerName, component)
	leaderRoleBindingName := fmt.Sprintf("%s-%s-leader-election-rolebinding", ownerName, component)
	return clusterRoleName, clusterRoleBindingName, leaderRoleName, leaderRoleBindingName
}

func setComponentOwnerLabels(labels map[string]string, ownerName, component string) map[string]string {
	if labels == nil {
		labels = map[string]string{}
	}
	labels["infernex.io/owner"] = ownerName
	if component != "" {
		labels["infernex.io/component"] = component
	}
	return labels
}

func (r *InferNexServiceReconciler) ensureOwnedServiceAccount(
	ctx context.Context,
	owner *infernexv1alpha1.InferNexService,
	component string,
	saName string,
) error {
	sa := &corev1.ServiceAccount{ObjectMeta: metav1.ObjectMeta{Name: saName, Namespace: owner.Namespace}}
	_, err := controllerutil.CreateOrUpdate(ctx, r.Client, sa, func() error {
		sa.Labels = setComponentOwnerLabels(sa.Labels, owner.Name, component)
		if isSingletonComponentSA(component) {
			// Component-controller SA name is namespace-scoped singleton (cache-indexer-sa,
			// tidal-sa, etc.); multiple InferNexService owners must share it. Use additive
			// non-controller ownerRef so K8s GC removes the SA only when all owners go.
			return controllerutil.SetOwnerReference(owner, sa, r.Scheme)
		}
		return controllerutil.SetControllerReference(owner, sa, r.Scheme)
	})
	return err
}

// isSingletonComponentSA mirrors componentControllerSAName: if that function maps the
// component to a fixed (non-owner-prefixed) name, the SA is a namespace-shared singleton.
// Per-owner SAs (e.g., hermes-router) return "" from componentControllerSAName and are
// not singletons.
func isSingletonComponentSA(component string) bool {
	return componentControllerSAName(component) != ""
}

func singletonComponentSAComponents() []string {
	return []string{cacheIndexerComponent, elasticScalerComponent, tidalComponent, rsgComponent, hardwareMonitorComponent, networkPerformanceExporterComponent}
}

func (r *InferNexServiceReconciler) syncSingletonComponentSAOwnerRefsCleanup(
	ctx context.Context,
	owner *infernexv1alpha1.InferNexService,
	desired map[string]componentPlan,
) error {
	for _, component := range singletonComponentSAComponents() {
		if _, ok := desired[component]; ok {
			continue
		}
		saName := componentControllerSAName(component)
		if saName == "" {
			continue
		}
		sa := &corev1.ServiceAccount{ObjectMeta: metav1.ObjectMeta{Name: saName, Namespace: owner.Namespace}}
		if err := r.Get(ctx, client.ObjectKeyFromObject(sa), sa); err != nil {
			if apierrors.IsNotFound(err) {
				continue
			}
			return err
		}
		if err := r.pruneOwnerRefByUIDOrDelete(ctx, owner, sa); err != nil {
			return err
		}
	}
	return nil
}

func (r *InferNexServiceReconciler) ensureOwnedRoleBinding(
	ctx context.Context,
	owner *infernexv1alpha1.InferNexService,
	component string,
	bindingName, roleName, saName string,
	labels map[string]string,
) error {
	rb := &rbacv1.RoleBinding{ObjectMeta: metav1.ObjectMeta{Name: bindingName, Namespace: owner.Namespace}}
	_, err := controllerutil.CreateOrUpdate(ctx, r.Client, rb, func() error {
		rb.Labels = setComponentOwnerLabels(rb.Labels, owner.Name, component)
		for k, v := range labels {
			rb.Labels[k] = v
		}
		rb.RoleRef = rbacv1.RoleRef{
			APIGroup: rbacv1.GroupName,
			Kind:     "Role",
			Name:     roleName,
		}
		rb.Subjects = []rbacv1.Subject{{
			Kind:      rbacv1.ServiceAccountKind,
			Name:      saName,
			Namespace: owner.Namespace,
		}}
		return controllerutil.SetControllerReference(owner, rb, r.Scheme)
	})
	return err
}

func (r *InferNexServiceReconciler) ensureComponentControllerRBAC(
	ctx context.Context,
	owner *infernexv1alpha1.InferNexService,
	component string,
) error {
	switch component {
	case cacheIndexerComponent:
		return r.ensureCacheIndexerRBAC(ctx, owner)
	case hermesRouterComponent:
		return r.ensureHermesRouterRBAC(ctx, owner)
	case elasticScalerComponent:
		return r.ensureElasticScalerRBAC(ctx, owner)
	case tidalComponent:
		return r.ensureTidalRBAC(ctx, owner)
	case rsgComponent:
		return r.ensureRSGRBAC(ctx, owner)
	case hardwareMonitorComponent:
		return r.ensureHardwareMonitorRBAC(ctx, owner)
	case networkPerformanceExporterComponent:
		return r.ensureOwnedServiceAccount(ctx, owner, component, componentControllerSAName(component))
	default:
		return nil
	}
}

func (r *InferNexServiceReconciler) syncComponentControllerRBACCleanup(
	ctx context.Context,
	owner *infernexv1alpha1.InferNexService,
	desired map[string]componentPlan,
) error {
	if err := r.syncSingletonComponentSAOwnerRefsCleanup(ctx, owner, desired); err != nil {
		return err
	}
	for _, component := range []string{cacheIndexerComponent, hermesRouterComponent, elasticScalerComponent, tidalComponent, rsgComponent, hardwareMonitorComponent} {
		if _, ok := desired[component]; ok {
			continue
		}
		if component == hermesRouterComponent {
			if err := r.deleteHermesRouterRBAC(ctx, owner); err != nil {
				return err
			}
			continue
		}
		if err := r.deleteComponentControllerRBAC(ctx, owner, component); err != nil {
			return err
		}
		if err := r.deleteComponentWebhookCertSecret(ctx, owner, component); err != nil {
			return err
		}
	}
	return nil
}

// ensureHermesRouterRBAC grants the Hermes / gateway-api-inference-extension EPP in-process
// controller watches (aligned with KServe LLMISVC scheduler EPP Role) within the InferNexService namespace.
func (r *InferNexServiceReconciler) ensureHermesRouterRBAC(
	ctx context.Context,
	owner *infernexv1alpha1.InferNexService,
) error {
	ns := owner.Namespace
	saName := hermesRouterServiceAccountName(owner.Name)
	roleName, roleBindingName := hermesRouterRBACObjectNames(owner.Name)

	if err := r.ensureOwnedServiceAccount(ctx, owner, hermesRouterComponent, saName); err != nil {
		return err
	}

	role := &rbacv1.Role{ObjectMeta: metav1.ObjectMeta{Name: roleName, Namespace: ns}}
	_, err := controllerutil.CreateOrUpdate(ctx, r.Client, role, func() error {
		if role.Labels == nil {
			role.Labels = map[string]string{}
		}
		role.Labels["infernex.io/owner"] = owner.Name
		role.Labels["infernex.io/component"] = hermesRouterComponent
		role.Labels["infernex.io/managed-role"] = "true"
		role.Rules = []rbacv1.PolicyRule{
			{
				APIGroups: []string{""},
				Resources: []string{"pods"},
				Verbs:     []string{"get", "list", "watch"},
			},
			{
				APIGroups: []string{"inference.networking.k8s.io", "inference.networking.x-k8s.io"},
				Resources: []string{"inferencepools", "inferenceobjectives", "inferencemodels"},
				Verbs:     []string{"get", "list", "watch"},
			},
			{
				APIGroups: []string{"inference.networking.x-k8s.io"},
				Resources: []string{"inferencemodelrewrites", "inferencepoolimports"},
				Verbs:     []string{"get", "list", "watch"},
			},
			{
				APIGroups: []string{"discovery.k8s.io"},
				Resources: []string{"endpointslices"},
				Verbs:     []string{"get", "list", "watch"},
			},
			{
				APIGroups: []string{"coordination.k8s.io"},
				Resources: []string{"leases"},
				Verbs:     []string{"get", "list", "watch", "create", "update", "patch", "delete"},
			},
		}
		return controllerutil.SetControllerReference(owner, role, r.Scheme)
	})
	if err != nil {
		return err
	}

	return r.ensureOwnedRoleBinding(ctx, owner, hermesRouterComponent, roleBindingName, roleName, saName, map[string]string{
		"infernex.io/managed-rolebinding": "true",
	})
}

func (r *InferNexServiceReconciler) deleteHermesRouterRBAC(ctx context.Context, owner *infernexv1alpha1.InferNexService) error {
	ns := owner.Namespace
	saName := hermesRouterServiceAccountName(owner.Name)
	roleName, roleBindingName := hermesRouterRBACObjectNames(owner.Name)
	for _, obj := range []client.Object{
		&rbacv1.RoleBinding{ObjectMeta: metav1.ObjectMeta{Name: roleBindingName, Namespace: ns}},
		&rbacv1.Role{ObjectMeta: metav1.ObjectMeta{Name: roleName, Namespace: ns}},
		&corev1.ServiceAccount{ObjectMeta: metav1.ObjectMeta{Name: saName, Namespace: ns}},
	} {
		if err := r.Delete(ctx, obj); err != nil && !apierrors.IsNotFound(err) {
			return err
		}
	}
	return nil
}

func (r *InferNexServiceReconciler) ensureCacheIndexerRBAC(
	ctx context.Context,
	owner *infernexv1alpha1.InferNexService,
) error {
	return r.ensureControllerRBAC(ctx, owner, cacheIndexerComponent, []rbacv1.PolicyRule{
		{
			APIGroups: []string{""},
			Resources: []string{"pods/status"},
			Verbs:     []string{"get"},
		},
		{
			APIGroups: []string{""},
			Resources: []string{"pods"},
			Verbs:     []string{"get", "list", "watch"},
		},
	})
}

func (r *InferNexServiceReconciler) ensureElasticScalerRBAC(
	ctx context.Context,
	owner *infernexv1alpha1.InferNexService,
) error {
	return r.ensureControllerRBAC(ctx, owner, elasticScalerComponent, []rbacv1.PolicyRule{
		{
			APIGroups: []string{"*"},
			Resources: []string{"*"},
			Verbs:     []string{"get", "list", "patch", "update", "watch"},
		},
		{
			APIGroups: []string{"apps"},
			Resources: []string{"deployments/scale", "statefulsets/scale"},
			Verbs:     []string{"get", "patch", "update"},
		},
		{
			APIGroups: []string{"autoscaling"},
			Resources: []string{"horizontalpodautoscalers"},
			Verbs:     []string{"create", "delete", "get", "list", "patch", "update", "watch"},
		},
		{
			APIGroups: []string{"autoscaling.openfuyao.com"},
			Resources: []string{"resourcescalinggroups/scale"},
			Verbs:     []string{"get", "patch", "update"},
		},
		{
			APIGroups: []string{"elasticscaler.io"},
			Resources: []string{"elasticscalers"},
			Verbs:     []string{"create", "delete", "get", "list", "patch", "update", "watch"},
		},
		{
			APIGroups: []string{"elasticscaler.io"},
			Resources: []string{"elasticscalers/finalizers"},
			Verbs:     []string{"update"},
		},
		{
			APIGroups: []string{"elasticscaler.io"},
			Resources: []string{"elasticscalers/status"},
			Verbs:     []string{"get", "patch", "update"},
		},
	})
}

func (r *InferNexServiceReconciler) ensureTidalRBAC(
	ctx context.Context,
	owner *infernexv1alpha1.InferNexService,
) error {
	return r.ensureControllerRBAC(ctx, owner, tidalComponent, []rbacv1.PolicyRule{
		{
			APIGroups: []string{"*"},
			Resources: []string{"*"},
			Verbs:     []string{"get"},
		},
		{
			APIGroups: []string{"tidal.io"},
			Resources: []string{"tidals"},
			Verbs:     []string{"get", "list", "watch"},
		},
		{
			APIGroups: []string{"tidal.io"},
			Resources: []string{"tidals/status"},
			Verbs:     []string{"get", "update"},
		},
		{
			APIGroups: []string{"tidal.io"},
			Resources: []string{"tidals/finalizers"},
			Verbs:     []string{"update"},
		},
	})
}

func (r *InferNexServiceReconciler) ensureRSGRBAC(
	ctx context.Context,
	owner *infernexv1alpha1.InferNexService,
) error {
	return r.ensureControllerRBAC(ctx, owner, rsgComponent, []rbacv1.PolicyRule{
		{
			APIGroups: []string{"autoscaling.openfuyao.com"},
			Resources: []string{"resourcescalinggroups"},
			Verbs:     []string{"create", "delete", "get", "list", "patch", "update", "watch"},
		},
		{
			APIGroups: []string{"autoscaling.openfuyao.com"},
			Resources: []string{"resourcescalinggroups/finalizers"},
			Verbs:     []string{"update"},
		},
		{
			APIGroups: []string{"autoscaling.openfuyao.com"},
			Resources: []string{"resourcescalinggroups/status"},
			Verbs:     []string{"get", "patch", "update"},
		},
		{
			APIGroups: []string{"apps"},
			Resources: []string{"deployments", "statefulsets", "replicasets", "controllerrevisions"},
			Verbs:     []string{"get", "list", "watch", "create", "update", "patch", "delete"},
		},
		{
			APIGroups: []string{""},
			Resources: []string{"services", "configmaps"},
			Verbs:     []string{"get", "list", "watch", "create", "update", "patch", "delete"},
		},
		{
			APIGroups: []string{"leaderworkerset.x-k8s.io"},
			Resources: []string{"leaderworkersets"},
			Verbs:     []string{"get", "list", "watch", "create", "update", "patch", "delete"},
		},
		{
			APIGroups: []string{"workloads.x-k8s.io"},
			Resources: []string{"rolebasedgroups"},
			Verbs:     []string{"get", "list", "watch"},
		},
	})
}

func (r *InferNexServiceReconciler) ensureHardwareMonitorRBAC(
	ctx context.Context,
	owner *infernexv1alpha1.InferNexService,
) error {
	return r.ensureControllerRBAC(ctx, owner, hardwareMonitorComponent, []rbacv1.PolicyRule{
		{
			APIGroups: []string{""},
			Resources: []string{"pods"},
			Verbs:     []string{"get", "list", "watch"},
		},
	})
}

func (r *InferNexServiceReconciler) ensureControllerRBAC(
	ctx context.Context,
	owner *infernexv1alpha1.InferNexService,
	component string,
	clusterRules []rbacv1.PolicyRule,
) error {
	ns := owner.Namespace
	saName := componentControllerSAName(component)
	if saName == "" {
		return nil
	}
	clusterRoleName, clusterRoleBindingName, leaderRoleName, leaderRoleBindingName :=
		componentControllerRBACNames(owner.Name, component)

	if err := r.ensureOwnedServiceAccount(ctx, owner, component, saName); err != nil {
		return err
	}

	clusterRole := &rbacv1.ClusterRole{ObjectMeta: metav1.ObjectMeta{Name: clusterRoleName}}
	_, err := controllerutil.CreateOrUpdate(ctx, r.Client, clusterRole, func() error {
		if clusterRole.Labels == nil {
			clusterRole.Labels = map[string]string{}
		}
		clusterRole.Labels["infernex.io/owner"] = owner.Name
		clusterRole.Labels["infernex.io/component"] = component
		clusterRole.Labels["infernex.io/managed-cluster-rbac"] = "true"
		clusterRole.Rules = clusterRules
		return nil
	})
	if err != nil {
		return err
	}

	clusterRoleBinding := &rbacv1.ClusterRoleBinding{ObjectMeta: metav1.ObjectMeta{Name: clusterRoleBindingName}}
	_, err = controllerutil.CreateOrUpdate(ctx, r.Client, clusterRoleBinding, func() error {
		if clusterRoleBinding.Labels == nil {
			clusterRoleBinding.Labels = map[string]string{}
		}
		clusterRoleBinding.Labels["infernex.io/owner"] = owner.Name
		clusterRoleBinding.Labels["infernex.io/component"] = component
		clusterRoleBinding.Labels["infernex.io/managed-cluster-rbac"] = "true"
		clusterRoleBinding.RoleRef = rbacv1.RoleRef{
			APIGroup: rbacv1.GroupName,
			Kind:     "ClusterRole",
			Name:     clusterRoleName,
		}
		clusterRoleBinding.Subjects = []rbacv1.Subject{{
			Kind:      rbacv1.ServiceAccountKind,
			Name:      saName,
			Namespace: ns,
		}}
		return nil
	})
	if err != nil {
		return err
	}

	leaderRole := &rbacv1.Role{ObjectMeta: metav1.ObjectMeta{Name: leaderRoleName, Namespace: ns}}
	_, err = controllerutil.CreateOrUpdate(ctx, r.Client, leaderRole, func() error {
		if leaderRole.Labels == nil {
			leaderRole.Labels = map[string]string{}
		}
		leaderRole.Labels["infernex.io/owner"] = owner.Name
		leaderRole.Labels["infernex.io/component"] = component
		leaderRole.Rules = []rbacv1.PolicyRule{
			{
				APIGroups: []string{""},
				Resources: []string{"configmaps"},
				Verbs:     []string{"get", "list", "watch", "create", "update", "patch", "delete"},
			},
			{
				APIGroups: []string{"coordination.k8s.io"},
				Resources: []string{"leases"},
				Verbs:     []string{"get", "list", "watch", "create", "update", "patch", "delete"},
			},
			{
				APIGroups: []string{""},
				Resources: []string{"events"},
				Verbs:     []string{"create", "patch"},
			},
		}
		return controllerutil.SetControllerReference(owner, leaderRole, r.Scheme)
	})
	if err != nil {
		return err
	}

	return r.ensureOwnedRoleBinding(ctx, owner, component, leaderRoleBindingName, leaderRoleName, saName, nil)
}

func (r *InferNexServiceReconciler) deleteComponentControllerRBAC(
	ctx context.Context,
	owner *infernexv1alpha1.InferNexService,
	component string,
) error {
	ns := owner.Namespace
	saName := componentControllerSAName(component)
	if saName == "" {
		return nil
	}
	clusterRoleName, clusterRoleBindingName, leaderRoleName, leaderRoleBindingName :=
		componentControllerRBACNames(owner.Name, component)

	// Per-owner RBAC objects (names include owner.Name) can be deleted directly; the
	// singleton SA is shared and must NOT be directly deleted — let K8s OwnerReference
	// GC remove it only when all InferNexService owners are gone.
	for _, obj := range []client.Object{
		&rbacv1.RoleBinding{ObjectMeta: metav1.ObjectMeta{Name: leaderRoleBindingName, Namespace: ns}},
		&rbacv1.Role{ObjectMeta: metav1.ObjectMeta{Name: leaderRoleName, Namespace: ns}},
		&rbacv1.ClusterRoleBinding{ObjectMeta: metav1.ObjectMeta{Name: clusterRoleBindingName}},
		&rbacv1.ClusterRole{ObjectMeta: metav1.ObjectMeta{Name: clusterRoleName}},
	} {
		if err := r.Delete(ctx, obj); err != nil && !apierrors.IsNotFound(err) {
			return err
		}
	}
	if !isSingletonComponentSA(component) {
		sa := &corev1.ServiceAccount{ObjectMeta: metav1.ObjectMeta{Name: saName, Namespace: ns}}
		if err := r.Delete(ctx, sa); err != nil && !apierrors.IsNotFound(err) {
			return err
		}
	} else {
		sa := &corev1.ServiceAccount{ObjectMeta: metav1.ObjectMeta{Name: saName, Namespace: ns}}
		if err := r.Get(ctx, client.ObjectKeyFromObject(sa), sa); err != nil {
			if !apierrors.IsNotFound(err) {
				return err
			}
		} else if err := r.pruneOwnerRefByUIDOrDelete(ctx, owner, sa); err != nil {
			return err
		} else {
			// Shared service account kept or cleaned up successfully: nothing else to do.
		}
	}
	return nil
}
