package controller

import (
	"context"
	"strings"
	"testing"

	corev1 "k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"

	infernexv1alpha1 "gitcode.com/openFuyao/InferNex/api/v1alpha1"
)

func rbacTestScheme(t *testing.T) *runtime.Scheme {
	t.Helper()
	s := gatewayTestScheme(t)
	if err := rbacv1.AddToScheme(s); err != nil {
		t.Fatal(err)
	}
	return s
}

func TestRBACNameAndLabelHelpers(t *testing.T) {
	t.Parallel()
	if got := hermesRouterServiceAccountName("demo"); got != "demo-hermes-router-sa" {
		t.Fatalf("unexpected hermes router sa name: %q", got)
	}
	roleName, rbName := hermesRouterRBACObjectNames("demo")
	if roleName == "" || rbName == "" {
		t.Fatalf("expected non-empty hermes rbac names: %q %q", roleName, rbName)
	}
	if got := componentControllerSAName(cacheIndexerComponent); got != "cache-indexer-sa" {
		t.Fatalf("unexpected cache indexer sa name: %q", got)
	}
	if got := componentControllerSAName("unknown"); got != "" {
		t.Fatalf("unknown component sa should be empty, got %q", got)
	}
	labels := setComponentOwnerLabels(nil, "demo", "cache-indexer")
	if labels["infernex.io/owner"] != "demo" || labels["infernex.io/component"] != "cache-indexer" {
		t.Fatalf("unexpected owner/component labels: %v", labels)
	}
	longName := strings.Repeat("a", maxProxySANameBaseLen+10)
	saName := proxySANamespacedName(longName)
	if len(saName) >= len(longName)+len(proxyServiceAccountSuffix) {
		t.Fatalf("proxy sa name should be truncated, got %q", saName)
	}
}

func TestEnsureAndDeleteComponentControllerRBAC(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	s := rbacTestScheme(t)
	owner := &infernexv1alpha1.InferNexService{
		ObjectMeta: metav1.ObjectMeta{Name: "demo", Namespace: "ns-a", UID: types.UID("uid-demo")},
	}
	cl := fake.NewClientBuilder().WithScheme(s).WithObjects(owner).Build()
	r := &InferNexServiceReconciler{Client: cl, Scheme: s}

	err := r.ensureControllerRBAC(ctx, owner, cacheIndexerComponent, []rbacv1.PolicyRule{{
		APIGroups: []string{""},
		Resources: []string{"pods"},
		Verbs:     []string{"get"},
	}})
	if err != nil {
		t.Fatalf("ensureControllerRBAC error: %v", err)
	}

	sa := &corev1.ServiceAccount{}
	if err := cl.Get(ctx, types.NamespacedName{Namespace: "ns-a", Name: "cache-indexer-sa"}, sa); err != nil {
		t.Fatalf("expected serviceaccount created: %v", err)
	}
	assertNonControllerOwnerRef(t, sa.OwnerReferences, owner.Name, owner.UID)
	clusterRoleName, clusterRoleBindingName, leaderRoleName, leaderRoleBindingName := componentControllerRBACNames(owner.Name, cacheIndexerComponent)

	clusterRole := &rbacv1.ClusterRole{}
	if err := cl.Get(ctx, types.NamespacedName{Name: clusterRoleName}, clusterRole); err != nil {
		t.Fatalf("expected clusterrole created: %v", err)
	}
	if len(clusterRole.Rules) == 0 {
		t.Fatal("expected clusterrole rules set")
	}

	if err := r.deleteComponentControllerRBAC(ctx, owner, cacheIndexerComponent); err != nil {
		t.Fatalf("deleteComponentControllerRBAC error: %v", err)
	}
	for _, obj := range []struct {
		key types.NamespacedName
		obj clientObject
	}{
		{types.NamespacedName{Name: clusterRoleName}, &rbacv1.ClusterRole{}},
		{types.NamespacedName{Name: clusterRoleBindingName}, &rbacv1.ClusterRoleBinding{}},
		{types.NamespacedName{Namespace: "ns-a", Name: leaderRoleName}, &rbacv1.Role{}},
		{types.NamespacedName{Namespace: "ns-a", Name: leaderRoleBindingName}, &rbacv1.RoleBinding{}},
	} {
		err := cl.Get(ctx, obj.key, obj.obj)
		if !apierrors.IsNotFound(err) {
			t.Fatalf("expected object deleted for %v, got err=%v", obj.key, err)
		}
	}
	if err := cl.Get(ctx, types.NamespacedName{Namespace: "ns-a", Name: "cache-indexer-sa"}, &corev1.ServiceAccount{}); !apierrors.IsNotFound(err) {
		t.Fatalf("expected last-owner singleton serviceaccount deleted, got err=%v", err)
	}
}

func TestSyncProxyRBACCleanup_DeletesWhenProxyMissing(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	s := rbacTestScheme(t)
	owner := &infernexv1alpha1.InferNexService{
		ObjectMeta: metav1.ObjectMeta{Name: "demo", Namespace: "ns-a", UID: types.UID("uid-demo")},
	}
	saName := proxySANamespacedName(owner.Name)
	roleName := owner.Name + "-" + proxyRoleSuffix
	rbName := owner.Name + "-" + proxyRoleBindingSuffix
	cl := fake.NewClientBuilder().WithScheme(s).WithObjects(
		owner,
		&corev1.ServiceAccount{ObjectMeta: metav1.ObjectMeta{Name: saName, Namespace: owner.Namespace}},
		&rbacv1.Role{ObjectMeta: metav1.ObjectMeta{Name: roleName, Namespace: owner.Namespace}},
		&rbacv1.RoleBinding{ObjectMeta: metav1.ObjectMeta{Name: rbName, Namespace: owner.Namespace}},
	).Build()
	r := &InferNexServiceReconciler{Client: cl, Scheme: s}

	if err := r.syncProxyRBACCleanup(ctx, owner, map[string]componentPlan{}); err != nil {
		t.Fatalf("syncProxyRBACCleanup error: %v", err)
	}
	for _, obj := range []struct {
		key types.NamespacedName
		obj clientObject
	}{
		{types.NamespacedName{Namespace: owner.Namespace, Name: saName}, &corev1.ServiceAccount{}},
		{types.NamespacedName{Namespace: owner.Namespace, Name: roleName}, &rbacv1.Role{}},
		{types.NamespacedName{Namespace: owner.Namespace, Name: rbName}, &rbacv1.RoleBinding{}},
	} {
		err := cl.Get(ctx, obj.key, obj.obj)
		if !apierrors.IsNotFound(err) {
			t.Fatalf("expected proxy rbac object deleted for %v, got err=%v", obj.key, err)
		}
	}
}

func TestEnsureComponentControllerRBAC_Dispatch(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	s := rbacTestScheme(t)
	owner := &infernexv1alpha1.InferNexService{
		ObjectMeta: metav1.ObjectMeta{Name: "demo", Namespace: "ns-a", UID: types.UID("uid-demo")},
	}
	cl := fake.NewClientBuilder().WithScheme(s).WithObjects(owner).Build()
	r := &InferNexServiceReconciler{Client: cl, Scheme: s}

	if err := r.ensureComponentControllerRBAC(ctx, owner, cacheIndexerComponent); err != nil {
		t.Fatalf("ensureComponentControllerRBAC cache-indexer error: %v", err)
	}
	if err := r.ensureComponentControllerRBAC(ctx, owner, hermesRouterComponent); err != nil {
		t.Fatalf("ensureComponentControllerRBAC hermes-router error: %v", err)
	}
	if err := r.ensureComponentControllerRBAC(ctx, owner, "unknown"); err != nil {
		t.Fatalf("ensureComponentControllerRBAC unknown should not fail: %v", err)
	}
}

func TestSyncComponentControllerRBACCleanup_PrunesUndesired(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	s := rbacTestScheme(t)
	owner := &infernexv1alpha1.InferNexService{
		ObjectMeta: metav1.ObjectMeta{Name: "demo", Namespace: "ns-a", UID: types.UID("uid-demo")},
	}
	cl := fake.NewClientBuilder().WithScheme(s).WithObjects(owner).Build()
	r := &InferNexServiceReconciler{Client: cl, Scheme: s}

	if err := r.ensureComponentControllerRBAC(ctx, owner, cacheIndexerComponent); err != nil {
		t.Fatalf("setup cache-indexer rbac failed: %v", err)
	}
	if err := r.ensureComponentControllerRBAC(ctx, owner, hermesRouterComponent); err != nil {
		t.Fatalf("setup hermes rbac failed: %v", err)
	}
	if err := r.syncComponentControllerRBACCleanup(ctx, owner, map[string]componentPlan{
		hermesRouterComponent: {ServicePort: 8000},
	}); err != nil {
		t.Fatalf("syncComponentControllerRBACCleanup failed: %v", err)
	}
	// cache-indexer SA has only this owner, so dropping the ownerRef deletes it.
	err := cl.Get(ctx, types.NamespacedName{Namespace: owner.Namespace, Name: componentControllerSAName(cacheIndexerComponent)}, &corev1.ServiceAccount{})
	if !apierrors.IsNotFound(err) {
		t.Fatalf("expected cache-indexer SA deleted after last ownerRef was pruned, got err=%v", err)
	}
	// hermes SA should remain because desired includes router.
	err = cl.Get(ctx, types.NamespacedName{Namespace: owner.Namespace, Name: hermesRouterServiceAccountName(owner.Name)}, &corev1.ServiceAccount{})
	if err != nil {
		t.Fatalf("expected hermes SA still present, got err=%v", err)
	}
}

func TestEnsureComponentControllerRBAC_SharedSingletonServiceAccountOwnerRefs(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	s := rbacTestScheme(t)
	ownerA := &infernexv1alpha1.InferNexService{
		ObjectMeta: metav1.ObjectMeta{Name: "isvc-a", Namespace: "ns-a", UID: types.UID("uid-a")},
	}
	ownerB := &infernexv1alpha1.InferNexService{
		ObjectMeta: metav1.ObjectMeta{Name: "isvc-b", Namespace: "ns-a", UID: types.UID("uid-b")},
	}
	cl := fake.NewClientBuilder().WithScheme(s).WithObjects(ownerA, ownerB).Build()
	r := &InferNexServiceReconciler{Client: cl, Scheme: s}

	for _, owner := range []*infernexv1alpha1.InferNexService{ownerA, ownerB} {
		if err := r.ensureComponentControllerRBAC(ctx, owner, cacheIndexerComponent); err != nil {
			t.Fatalf("ensureComponentControllerRBAC(%s) error: %v", owner.Name, err)
		}
	}

	sa := &corev1.ServiceAccount{}
	if err := cl.Get(ctx, types.NamespacedName{Namespace: "ns-a", Name: componentControllerSAName(cacheIndexerComponent)}, sa); err != nil {
		t.Fatalf("expected shared serviceaccount: %v", err)
	}
	assertNonControllerOwnerRef(t, sa.OwnerReferences, ownerA.Name, ownerA.UID)
	assertNonControllerOwnerRef(t, sa.OwnerReferences, ownerB.Name, ownerB.UID)

	if err := r.deleteComponentControllerRBAC(ctx, ownerA, cacheIndexerComponent); err != nil {
		t.Fatalf("deleteComponentControllerRBAC(%s) error: %v", ownerA.Name, err)
	}
	if err := cl.Get(ctx, types.NamespacedName{Namespace: "ns-a", Name: componentControllerSAName(cacheIndexerComponent)}, sa); err != nil {
		t.Fatalf("expected shared serviceaccount after pruning first owner: %v", err)
	}
	assertMissingOwnerRef(t, sa.OwnerReferences, ownerA.Name, ownerA.UID)
	assertNonControllerOwnerRef(t, sa.OwnerReferences, ownerB.Name, ownerB.UID)

	if err := r.deleteComponentControllerRBAC(ctx, ownerB, cacheIndexerComponent); err != nil {
		t.Fatalf("deleteComponentControllerRBAC(%s) error: %v", ownerB.Name, err)
	}
	if err := cl.Get(ctx, types.NamespacedName{Namespace: "ns-a", Name: componentControllerSAName(cacheIndexerComponent)}, sa); !apierrors.IsNotFound(err) {
		t.Fatalf("expected shared serviceaccount deleted after last ownerRef was pruned, got err=%v", err)
	}
}

func assertNonControllerOwnerRef(t *testing.T, refs []metav1.OwnerReference, name string, uid types.UID) {
	t.Helper()
	for _, ref := range refs {
		if ref.Name != name || ref.UID != uid {
			continue
		}
		if ref.Controller != nil && *ref.Controller {
			t.Fatalf("ownerRef for %s should not be controller: %#v", name, ref)
		}
		return
	}
	t.Fatalf("missing ownerRef for %s/%s in %#v", name, uid, refs)
}

func assertMissingOwnerRef(t *testing.T, refs []metav1.OwnerReference, name string, uid types.UID) {
	t.Helper()
	for _, ref := range refs {
		if ref.Name == name && ref.UID == uid {
			t.Fatalf("unexpected ownerRef for %s/%s in %#v", name, uid, refs)
		}
	}
}

func TestEnsureHermesRouterRBAC_GIEWatchRules(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	s := rbacTestScheme(t)
	owner := &infernexv1alpha1.InferNexService{
		ObjectMeta: metav1.ObjectMeta{Name: "demo", Namespace: "ns-a"},
	}
	cl := fake.NewClientBuilder().WithScheme(s).WithObjects(owner).Build()
	r := &InferNexServiceReconciler{Client: cl, Scheme: s}

	if err := r.ensureHermesRouterRBAC(ctx, owner); err != nil {
		t.Fatalf("ensureHermesRouterRBAC error: %v", err)
	}
	roleName, _ := hermesRouterRBACObjectNames(owner.Name)
	role := &rbacv1.Role{}
	if err := cl.Get(ctx, types.NamespacedName{Namespace: owner.Namespace, Name: roleName}, role); err != nil {
		t.Fatalf("get hermes role: %v", err)
	}
	for _, tc := range []struct {
		group    string
		resource string
	}{
		{"inference.networking.x-k8s.io", "inferencemodelrewrites"},
		{"inference.networking.x-k8s.io", "inferencepoolimports"},
		{"discovery.k8s.io", "endpointslices"},
	} {
		if !rbacPolicyRuleAllows(role.Rules, tc.group, tc.resource, "list") {
			t.Fatalf("hermes role missing %s/%s list: %#v", tc.group, tc.resource, role.Rules)
		}
	}
}

func rbacPolicyRuleAllows(rules []rbacv1.PolicyRule, group, resource, verb string) bool {
	for _, rule := range rules {
		groupOK := false
		for _, g := range rule.APIGroups {
			if g == group {
				groupOK = true
				break
			}
		}
		if !groupOK {
			continue
		}
		resourceOK := false
		for _, res := range rule.Resources {
			if res == resource {
				resourceOK = true
				break
			}
		}
		if !resourceOK {
			continue
		}
		for _, v := range rule.Verbs {
			if v == verb || v == "*" {
				return true
			}
		}
	}
	return false
}

func TestDeleteHermesRouterRBAC(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	s := rbacTestScheme(t)
	owner := &infernexv1alpha1.InferNexService{
		ObjectMeta: metav1.ObjectMeta{Name: "demo", Namespace: "ns-a"},
	}
	cl := fake.NewClientBuilder().WithScheme(s).WithObjects(owner).Build()
	r := &InferNexServiceReconciler{Client: cl, Scheme: s}

	if err := r.ensureHermesRouterRBAC(ctx, owner); err != nil {
		t.Fatalf("setup hermes rbac failed: %v", err)
	}
	if err := r.deleteHermesRouterRBAC(ctx, owner); err != nil {
		t.Fatalf("deleteHermesRouterRBAC error: %v", err)
	}
	saName := hermesRouterServiceAccountName(owner.Name)
	roleName, rbName := hermesRouterRBACObjectNames(owner.Name)
	for _, obj := range []struct {
		key types.NamespacedName
		obj clientObject
	}{
		{types.NamespacedName{Namespace: owner.Namespace, Name: saName}, &corev1.ServiceAccount{}},
		{types.NamespacedName{Namespace: owner.Namespace, Name: roleName}, &rbacv1.Role{}},
		{types.NamespacedName{Namespace: owner.Namespace, Name: rbName}, &rbacv1.RoleBinding{}},
	} {
		err := cl.Get(ctx, obj.key, obj.obj)
		if !apierrors.IsNotFound(err) {
			t.Fatalf("expected hermes rbac deleted for %v, got err=%v", obj.key, err)
		}
	}
}

func TestEnsureComponentRBACVariants(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	s := rbacTestScheme(t)
	owner := &infernexv1alpha1.InferNexService{
		ObjectMeta: metav1.ObjectMeta{Name: "demo", Namespace: "ns-a"},
	}
	cl := fake.NewClientBuilder().WithScheme(s).WithObjects(owner).Build()
	r := &InferNexServiceReconciler{Client: cl, Scheme: s}

	for _, component := range []string{elasticScalerComponent, tidalComponent, rsgComponent, hardwareMonitorComponent} {
		if err := r.ensureComponentControllerRBAC(ctx, owner, component); err != nil {
			t.Fatalf("ensureComponentControllerRBAC(%s) error: %v", component, err)
		}
		saName := componentControllerSAName(component)
		if saName == "" {
			continue
		}
		if err := cl.Get(ctx, types.NamespacedName{Namespace: owner.Namespace, Name: saName}, &corev1.ServiceAccount{}); err != nil {
			t.Fatalf("expected SA for %s: %v", component, err)
		}
	}
}

func TestEnsureProxyPodRBAC_CreatesResources(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	s := rbacTestScheme(t)
	owner := &infernexv1alpha1.InferNexService{
		ObjectMeta: metav1.ObjectMeta{Name: "demo", Namespace: "ns-a"},
	}
	cl := fake.NewClientBuilder().WithScheme(s).WithObjects(owner).Build()
	r := &InferNexServiceReconciler{Client: cl, Scheme: s}

	if err := r.ensureProxyPodRBAC(ctx, owner); err != nil {
		t.Fatalf("ensureProxyPodRBAC error: %v", err)
	}
	if err := cl.Get(ctx, types.NamespacedName{Namespace: owner.Namespace, Name: proxySANamespacedName(owner.Name)}, &corev1.ServiceAccount{}); err != nil {
		t.Fatalf("proxy service account not created: %v", err)
	}
	if err := cl.Get(ctx, types.NamespacedName{Namespace: owner.Namespace, Name: owner.Name + "-" + proxyRoleSuffix}, &rbacv1.Role{}); err != nil {
		t.Fatalf("proxy role not created: %v", err)
	}
	if err := cl.Get(ctx, types.NamespacedName{Namespace: owner.Namespace, Name: owner.Name + "-" + proxyRoleBindingSuffix}, &rbacv1.RoleBinding{}); err != nil {
		t.Fatalf("proxy rolebinding not created: %v", err)
	}
}

type clientObject interface {
	metav1.Object
	runtime.Object
}
