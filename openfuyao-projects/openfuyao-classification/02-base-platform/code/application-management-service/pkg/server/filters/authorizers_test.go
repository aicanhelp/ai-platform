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
package filters

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/emicklei/go-restful/v3"
	rbacv1 "k8s.io/api/rbac/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/client-go/kubernetes/fake"

	"application-management-service/pkg/constant"
)

func TestOpenFuyaoRoleContains(t *testing.T) {
	if !openFuyaoRoleContains([]string{"cluster-admin"}, "openfuyao-cluster-admin") {
		t.Fatalf("Expected role match")
	}
	if openFuyaoRoleContains([]string{"cluster-admin"}, "openfuyao-platform-admin") {
		t.Fatalf("Expected role not match")
	}
	if openFuyaoRoleContains([]string{"cluster-admin"}, "") {
		t.Fatalf("Expected empty query not match")
	}
}

func TestGetOpenFuyaoClusterRoleType(t *testing.T) {
	crb := &rbacv1.ClusterRoleBinding{
		ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{constant.OpenFuyaoRoleLabel: "platform-role"}},
		RoleRef:    rbacv1.RoleRef{Name: "platform-admin"},
	}
	if got := getOpenFuyaoClusterRoleType(crb); got != "platform-admin" {
		t.Fatalf("Expected platform-admin, got %s", got)
	}

	crb = &rbacv1.ClusterRoleBinding{
		ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{constant.OpenFuyaoRoleLabel: "cluster-role"}},
		RoleRef:    rbacv1.RoleRef{Name: "cluster-viewer"},
	}
	if got := getOpenFuyaoClusterRoleType(crb); got != "cluster-viewer" {
		t.Fatalf("Expected cluster-viewer, got %s", got)
	}

	crb = &rbacv1.ClusterRoleBinding{
		ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{constant.OpenFuyaoRoleLabel: "platform-role"}},
		RoleRef:    rbacv1.RoleRef{Name: "unknown-platform-role"},
	}
	if got := getOpenFuyaoClusterRoleType(crb); got != "" {
		t.Fatalf("Expected empty for unsupported platform role, got %s", got)
	}

	crb = &rbacv1.ClusterRoleBinding{
		ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{constant.OpenFuyaoRoleLabel: "cluster-role"}},
		RoleRef:    rbacv1.RoleRef{Name: "unknown-cluster-role"},
	}
	if got := getOpenFuyaoClusterRoleType(crb); got != "" {
		t.Fatalf("Expected empty for unsupported cluster role, got %s", got)
	}

	crb = &rbacv1.ClusterRoleBinding{
		ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{constant.OpenFuyaoRoleLabel: "unknown"}},
		RoleRef:    rbacv1.RoleRef{Name: "whatever"},
	}
	if got := getOpenFuyaoClusterRoleType(crb); got != "" {
		t.Fatalf("Expected empty, got %s", got)
	}
}

func TestGetOpenFuyaoClusterRoleBindings(t *testing.T) {
	clientset := fake.NewSimpleClientset(
		&rbacv1.ClusterRoleBinding{
			ObjectMeta: metav1.ObjectMeta{
				Name:   "crb1",
				Labels: map[string]string{constant.OpenFuyaoRoleLabel: "cluster-role"},
			},
			Subjects: []rbacv1.Subject{{Kind: "User", Name: "alice"}},
			RoleRef:  rbacv1.RoleRef{Name: "cluster-admin"},
		},
		&rbacv1.ClusterRoleBinding{
			ObjectMeta: metav1.ObjectMeta{
				Name:   "crb2",
				Labels: map[string]string{constant.OpenFuyaoRoleLabel: "cluster-role"},
			},
			Subjects: []rbacv1.Subject{{Kind: "User", Name: "bob"}},
			RoleRef:  rbacv1.RoleRef{Name: "cluster-viewer"},
		},
	)

	req, _ := http.NewRequest(http.MethodGet, "http://example", nil)

	crbs, err := getOpenFuyaoClusterRoleBindings("alice", clientset, req)
	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}
	if len(crbs) != 1 {
		t.Fatalf("Expected 1 crb, got %d", len(crbs))
	}
	if crbs[0].Name != "crb1" {
		t.Fatalf("Expected crb1, got %s", crbs[0].Name)
	}
}

func TestGetOpenFuyaoClusterRoleBindings_NotFound(t *testing.T) {
	clientset := fake.NewSimpleClientset(
		&rbacv1.ClusterRoleBinding{
			ObjectMeta: metav1.ObjectMeta{
				Name:   "crb2",
				Labels: map[string]string{constant.OpenFuyaoRoleLabel: "cluster-role"},
			},
			Subjects: []rbacv1.Subject{{Kind: "User", Name: "bob"}},
			RoleRef:  rbacv1.RoleRef{Name: "cluster-viewer"},
		},
	)

	req, _ := http.NewRequest(http.MethodGet, "http://example", nil)

	_, err := getOpenFuyaoClusterRoleBindings("alice", clientset, req)
	if err == nil {
		t.Fatalf("Expected error, got nil")
	}
}

func TestAuthorizeByAdmittedRoles_UserInfoMissing(t *testing.T) {
	httpReq := httptest.NewRequest(http.MethodGet, "http://example", nil)
	req := restful.NewRequest(httpReq)
	clientset := fake.NewSimpleClientset()

	if got := AuthorizeByAdmittedRoles(req, []string{"platform-admin"}, clientset); got {
		t.Fatalf("expected false when userinfo missing")
	}
}

func TestAuthorizeByAdmittedRoles_PlatformAdminMatch(t *testing.T) {
	httpReq := httptest.NewRequest(http.MethodGet, "http://example", nil)
	httpReq = httpReq.WithContext(context.WithValue(httpReq.Context(), constant.UserKey, &user.DefaultInfo{Name: "alice"}))
	req := restful.NewRequest(httpReq)

	clientset := fake.NewSimpleClientset(
		&rbacv1.ClusterRoleBinding{
			ObjectMeta: metav1.ObjectMeta{
				Name:   "crb-platform",
				Labels: map[string]string{constant.OpenFuyaoRoleLabel: "platform-role"},
			},
			RoleRef:  rbacv1.RoleRef{Name: "platform-admin"},
			Subjects: []rbacv1.Subject{{Kind: "User", Name: "alice"}},
			// Subjects.Kind is not used by logic, but set for completeness.
		},
	)

	if got := AuthorizeByAdmittedRoles(req, []string{"platform-admin"}, clientset); !got {
		t.Fatalf("expected true for platform-admin")
	}
}

func TestAuthorizeByAdmittedRoles_ClusterViewerMatch(t *testing.T) {
	httpReq := httptest.NewRequest(http.MethodGet, "http://example", nil)
	httpReq = httpReq.WithContext(context.WithValue(httpReq.Context(), constant.UserKey, &user.DefaultInfo{Name: "alice"}))
	req := restful.NewRequest(httpReq)

	clientset := fake.NewSimpleClientset(
		&rbacv1.ClusterRoleBinding{
			ObjectMeta: metav1.ObjectMeta{
				Name:   "crb-cluster",
				Labels: map[string]string{constant.OpenFuyaoRoleLabel: "cluster-role"},
			},
			RoleRef:  rbacv1.RoleRef{Name: "cluster-viewer"},
			Subjects: []rbacv1.Subject{{Kind: "User", Name: "alice"}},
		},
	)

	if got := AuthorizeByAdmittedRoles(req, []string{"cluster-viewer"}, clientset); !got {
		t.Fatalf("expected true for cluster-viewer")
	}
}

func TestAuthorizeByAdmittedRoles_NoMatchingCRB(t *testing.T) {
	httpReq := httptest.NewRequest(http.MethodGet, "http://example", nil)
	httpReq = httpReq.WithContext(context.WithValue(httpReq.Context(), constant.UserKey, &user.DefaultInfo{Name: "alice"}))
	req := restful.NewRequest(httpReq)

	// CRB exists but subject is not "alice", so Authorize should fail.
	clientset := fake.NewSimpleClientset(
		&rbacv1.ClusterRoleBinding{
			ObjectMeta: metav1.ObjectMeta{
				Name:   "crb-wrong-user",
				Labels: map[string]string{constant.OpenFuyaoRoleLabel: "cluster-role"},
			},
			RoleRef:  rbacv1.RoleRef{Name: "cluster-viewer"},
			Subjects: []rbacv1.Subject{{Kind: "User", Name: "bob"}},
		},
	)

	if got := AuthorizeByAdmittedRoles(req, []string{"cluster-viewer"}, clientset); got {
		t.Fatalf("expected false when no matching CRB for user")
	}
}

func TestAuthorizeByAdmittedRoles_NoAdmittedRoleMatch(t *testing.T) {
	httpReq := httptest.NewRequest(http.MethodGet, "http://example", nil)
	httpReq = httpReq.WithContext(context.WithValue(httpReq.Context(), constant.UserKey, &user.DefaultInfo{Name: "alice"}))
	req := restful.NewRequest(httpReq)

	clientset := fake.NewSimpleClientset(
		&rbacv1.ClusterRoleBinding{
			ObjectMeta: metav1.ObjectMeta{
				Name:   "crb-cluster",
				Labels: map[string]string{constant.OpenFuyaoRoleLabel: "cluster-role"},
			},
			RoleRef:  rbacv1.RoleRef{Name: "cluster-viewer"},
			Subjects: []rbacv1.Subject{{Kind: "User", Name: "alice"}},
		},
	)

	if got := AuthorizeByAdmittedRoles(req, []string{"cluster-admin"}, clientset); got {
		t.Fatalf("expected false when admittedRoles does not include resolved role")
	}
}
