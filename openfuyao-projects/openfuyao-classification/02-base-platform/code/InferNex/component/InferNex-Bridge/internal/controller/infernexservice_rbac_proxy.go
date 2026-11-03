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

	coreapi "k8s.io/api/core/v1"
	rbacapi "k8s.io/api/rbac/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1api "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"

	infernexv1alpha1 "gitcode.com/openFuyao/InferNex/api/v1alpha1"
)

const (
	proxyServiceAccountSuffix = "infernex-pd-proxy"
	proxyRoleSuffix           = "infernex-pd-proxy-pods"
	proxyRoleBindingSuffix    = "infernex-pd-proxy-binding"
	maxProxySANameBaseLen     = 40
)

// ensureProxyPodRBAC creates namespaced ServiceAccount, Role (pods get/list/watch), RoleBinding
// so an independent PD proxy can discover prefill/decode Pods via the API.
func (r *InferNexServiceReconciler) ensureProxyPodRBAC(
	ctx context.Context,
	owner *infernexv1alpha1.InferNexService,
) error {
	ns := owner.Namespace
	saName := proxySANamespacedName(owner.Name)
	roleName := fmt.Sprintf("%s-%s", owner.Name, proxyRoleSuffix)
	rbName := fmt.Sprintf("%s-%s", owner.Name, proxyRoleBindingSuffix)

	if err := r.ensureOwnedServiceAccount(ctx, owner, "proxy-server", saName); err != nil {
		return err
	}

	role := &rbacapi.Role{ObjectMeta: metav1api.ObjectMeta{Name: roleName, Namespace: ns}}
	_, err := controllerutil.CreateOrUpdate(ctx, r.Client, role, func() error {
		if role.Labels == nil {
			role.Labels = map[string]string{}
		}
		role.Labels["infernex.io/owner"] = owner.Name
		role.Labels["infernex.io/component"] = "proxy-server"
		role.Rules = []rbacapi.PolicyRule{
			{
				APIGroups: []string{""},
				Resources: []string{"pods"},
				Verbs:     []string{"get", "list", "watch"},
			},
			{
				APIGroups: []string{""},
				Resources: []string{"pods/status"},
				Verbs:     []string{"get"},
			},
		}
		return controllerutil.SetControllerReference(owner, role, r.Scheme)
	})
	if err != nil {
		return err
	}

	return r.ensureOwnedRoleBinding(ctx, owner, "proxy-server", rbName, roleName, saName, nil)
}

func proxySANamespacedName(infsvcName string) string {
	base := infsvcName
	if len(base) > maxProxySANameBaseLen {
		base = base[:maxProxySANameBaseLen]
	}
	return fmt.Sprintf("%s-%s", base, proxyServiceAccountSuffix)
}

// syncProxyRBACCleanup removes namespaced proxy RBAC when there is no proxy workload or when PD
// discovery targets another namespace (Role in InferNexService NS cannot list pods elsewhere).
func (r *InferNexServiceReconciler) syncProxyRBACCleanup(
	ctx context.Context,
	owner *infernexv1alpha1.InferNexService,
	desired map[string]componentPlan,
) error {
	var proxy *componentPlan
	for k := range desired {
		if !desired[k].IsProxyServer {
			continue
		}
		p := desired[k]
		proxy = &p
		break
	}
	if proxy != nil && proxy.ProxyPDWorkloadNS == owner.Namespace {
		return nil
	}
	return r.deleteProxyRBAC(ctx, owner)
}

func (r *InferNexServiceReconciler) deleteProxyRBAC(
	ctx context.Context,
	owner *infernexv1alpha1.InferNexService,
) error {
	ns := owner.Namespace
	saName := proxySANamespacedName(owner.Name)
	roleName := fmt.Sprintf("%s-%s", owner.Name, proxyRoleSuffix)
	rbName := fmt.Sprintf("%s-%s", owner.Name, proxyRoleBindingSuffix)
	for _, obj := range []client.Object{
		&rbacapi.RoleBinding{ObjectMeta: metav1api.ObjectMeta{Name: rbName, Namespace: ns}},
		&rbacapi.Role{ObjectMeta: metav1api.ObjectMeta{Name: roleName, Namespace: ns}},
		&coreapi.ServiceAccount{ObjectMeta: metav1api.ObjectMeta{Name: saName, Namespace: ns}},
	} {
		if err := r.Delete(ctx, obj); err != nil && !apierrors.IsNotFound(err) {
			return err
		}
	}
	return nil
}
