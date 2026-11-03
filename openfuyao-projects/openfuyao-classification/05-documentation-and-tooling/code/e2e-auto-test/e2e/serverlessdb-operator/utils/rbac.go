package utils

import (
	"context"
	"fmt"

	authenticationv1 "k8s.io/api/authentication/v1"
	corev1 "k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/k8s"
)

const (
	tenantSAName   = "sdb-e2e-tenant"
	tenantRoleName = "sdb-e2e-tenant-role"
	norbacSAName   = "sdb-e2e-norbac"
)

// tenantRules mirrors the operator's requiredAccessChecks (token_cache.go)
// so a tenant token passes the SelfSubjectRulesReview pre-check and can
// perform the actual allocate/resize/release operations on the API server.
var tenantRules = []rbacv1.PolicyRule{
	{APIGroups: []string{GroupName}, Resources: []string{ResDBInsts}, Verbs: []string{"get", "list", "patch", "delete"}},
	{APIGroups: []string{GroupName}, Resources: []string{ResDBInsts + "/status"}, Verbs: []string{"patch"}},
	{APIGroups: []string{""}, Resources: []string{"pods"}, Verbs: []string{"get", "patch"}},
	{APIGroups: []string{""}, Resources: []string{"services"}, Verbs: []string{"get", "create", "delete"}},
}

// CreateTenantSA provisions a dedicated ServiceAccount + Role + RoleBinding
// in ns and returns a bearer token for it. All objects are namespace-scoped
// and are cleaned up when the test namespace is deleted.
func CreateTenantSA(ctx context.Context, k8sClient *k8s.K8SClient, ns string) (string, error) {
	saClient := k8sClient.Clientset.CoreV1().ServiceAccounts(ns)
	_, err := saClient.Get(ctx, tenantSAName, metav1.GetOptions{})
	if apierrors.IsNotFound(err) {
		_, err = saClient.Create(ctx, &corev1.ServiceAccount{
			ObjectMeta: metav1.ObjectMeta{Name: tenantSAName, Namespace: ns},
		}, metav1.CreateOptions{})
		if err != nil {
			return "", fmt.Errorf("create tenant SA: %w", err)
		}
	} else if err != nil {
		return "", fmt.Errorf("get tenant SA: %w", err)
	}

	roleClient := k8sClient.Clientset.RbacV1().Roles(ns)
	_, err = roleClient.Get(ctx, tenantRoleName, metav1.GetOptions{})
	if apierrors.IsNotFound(err) {
		_, err = roleClient.Create(ctx, &rbacv1.Role{
			ObjectMeta: metav1.ObjectMeta{Name: tenantRoleName, Namespace: ns},
			Rules:      tenantRules,
		}, metav1.CreateOptions{})
		if err != nil {
			return "", fmt.Errorf("create tenant Role: %w", err)
		}
	} else if err != nil {
		return "", fmt.Errorf("get tenant Role: %w", err)
	}

	rbClient := k8sClient.Clientset.RbacV1().RoleBindings(ns)
	_, err = rbClient.Get(ctx, tenantRoleName, metav1.GetOptions{})
	if apierrors.IsNotFound(err) {
		_, err = rbClient.Create(ctx, &rbacv1.RoleBinding{
			ObjectMeta: metav1.ObjectMeta{Name: tenantRoleName, Namespace: ns},
			Subjects:   []rbacv1.Subject{{Kind: "ServiceAccount", Name: tenantSAName, Namespace: ns}},
			RoleRef:    rbacv1.RoleRef{Kind: "Role", Name: tenantRoleName, APIGroup: "rbac.authorization.k8s.io"},
		}, metav1.CreateOptions{})
		if err != nil {
			return "", fmt.Errorf("create tenant RoleBinding: %w", err)
		}
	} else if err != nil {
		return "", fmt.Errorf("get tenant RoleBinding: %w", err)
	}

	return requestSAToken(ctx, k8sClient, ns, tenantSAName)
}

// CreateNoRBACSA provisions a ServiceAccount with no RoleBinding, so its token
// passes authentication but fails the operator's authorization pre-check (403).
func CreateNoRBACSA(ctx context.Context, k8sClient *k8s.K8SClient, ns string) (string, error) {
	saClient := k8sClient.Clientset.CoreV1().ServiceAccounts(ns)
	_, err := saClient.Get(ctx, norbacSAName, metav1.GetOptions{})
	if apierrors.IsNotFound(err) {
		_, err = saClient.Create(ctx, &corev1.ServiceAccount{
			ObjectMeta: metav1.ObjectMeta{Name: norbacSAName, Namespace: ns},
		}, metav1.CreateOptions{})
		if err != nil {
			return "", fmt.Errorf("create norbac SA: %w", err)
		}
	} else if err != nil {
		return "", fmt.Errorf("get norbac SA: %w", err)
	}

	return requestSAToken(ctx, k8sClient, ns, norbacSAName)
}

// requestSAToken requests a bound service-account token via the
// serviceaccounts/token subresource (K8s >=1.24). The token is valid for 2h,
// enough for the whole suite.
func requestSAToken(ctx context.Context, k8sClient *k8s.K8SClient, ns, saName string) (string, error) {
	tr := &authenticationv1.TokenRequest{
		Spec: authenticationv1.TokenRequestSpec{
			ExpirationSeconds: ptrInt64(7200),
		},
	}
	resp, err := k8sClient.Clientset.CoreV1().ServiceAccounts(ns).CreateToken(ctx, saName, tr, metav1.CreateOptions{})
	if err != nil {
		return "", fmt.Errorf("create token for SA %s: %w", saName, err)
	}
	return resp.Status.Token, nil
}

func ptrInt64(v int64) *int64 { return &v }
