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

package v1

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/emicklei/go-restful/v3"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"

	"openfuyao.com/multi-cluster-service/pkg/cluster"
	"openfuyao.com/multi-cluster-service/pkg/utils"
)

// mockOKInspectClient implements cluster.OperationsInterface without panicking so
// handler type-assertion failure branches can be reached.
type mockOKInspectClient struct {
	MockClusterClient
}

func (m *mockOKInspectClient) Inspect(_ context.Context, _ cluster.InspectOptions) (*cluster.ListOfCluster, error) {
	list := &cluster.ListOfCluster{Info: map[string]*cluster.InformationOfCluster{}}
	list.Info["member1"] = &cluster.InformationOfCluster{ClusterName: "member1"}
	return list, nil
}

// newHandlerRequest builds a restful request with path parameter and user context.
func newHandlerRequest(method, path, body, clusterTag, user string) *restful.Request {
	httpReq := httptest.NewRequest(method, path, strings.NewReader(body))
	// go-restful WriteHeaderAndEntity requires a compatible Accept header
	httpReq.Header.Set("Accept", "application/json")
	req := restful.NewRequest(httpReq)
	if clusterTag != "" {
		req.PathParameters()["clusterTag"] = clusterTag
	}
	if user != "" {
		req.Request = req.Request.WithContext(
			context.WithValue(req.Request.Context(), "user", user))
	}
	return req
}

func runHandler(t *testing.T, req *restful.Request, fn func(*restful.Request, *restful.Response)) int {
	t.Helper()
	rec := httptest.NewRecorder()
	resp := restful.NewResponse(rec)
	// standalone (non-routed) responses need an explicit Accept mime
	resp.SetRequestAccepts(restful.MIME_JSON)
	fn(req, resp)
	return rec.Code
}

func TestCheckUserAccessExtra(t *testing.T) {
	opClient := &cluster.OperationClient{}

	t.Run("missing user info", func(t *testing.T) {
		req := newHandlerRequest("POST", "/", "", "", "")
		code := runHandler(t, req, func(req *restful.Request, resp *restful.Response) {
			if checkUserAccess(req, resp, opClient, "join:cluster") {
				t.Error("checkUserAccess() = true, want false")
			}
		})
		if code != http.StatusBadRequest {
			t.Errorf("code = %d, want 400", code)
		}
	})

	t.Run("client type assertion failed", func(t *testing.T) {
		req := newHandlerRequest("POST", "/", "", "", "alice")
		code := runHandler(t, req, func(req *restful.Request, resp *restful.Response) {
			if checkUserAccess(req, resp, &MockClusterClient{}, "join:cluster") {
				t.Error("checkUserAccess() = true, want false")
			}
		})
		if code != http.StatusInternalServerError {
			t.Errorf("code = %d, want 500", code)
		}
	})

	t.Run("access check failed", func(t *testing.T) {
		patches := gomonkey.ApplyFunc(utils.IsAccessPass,
			func(_ kubernetes.Interface, _ string, _ string) (bool, error) {
				return false, errors.New("boom")
			})
		defer patches.Reset()

		req := newHandlerRequest("POST", "/", "", "", "alice")
		code := runHandler(t, req, func(req *restful.Request, resp *restful.Response) {
			if checkUserAccess(req, resp, opClient, "join:cluster") {
				t.Error("checkUserAccess() = true, want false")
			}
		})
		if code != http.StatusInternalServerError {
			t.Errorf("code = %d, want 500", code)
		}
	})

	t.Run("access forbidden", func(t *testing.T) {
		patches := gomonkey.ApplyFunc(utils.IsAccessPass,
			func(_ kubernetes.Interface, _ string, _ string) (bool, error) {
				return false, nil
			})
		defer patches.Reset()

		req := newHandlerRequest("POST", "/", "", "", "alice")
		code := runHandler(t, req, func(req *restful.Request, resp *restful.Response) {
			if checkUserAccess(req, resp, opClient, "join:cluster") {
				t.Error("checkUserAccess() = true, want false")
			}
		})
		if code != http.StatusForbidden {
			t.Errorf("code = %d, want 403", code)
		}
	})

	t.Run("access passed", func(t *testing.T) {
		patches := gomonkey.ApplyFunc(utils.IsAccessPass,
			func(_ kubernetes.Interface, _ string, _ string) (bool, error) {
				return true, nil
			})
		defer patches.Reset()

		req := newHandlerRequest("POST", "/", "", "", "alice")
		code := runHandler(t, req, func(req *restful.Request, resp *restful.Response) {
			if !checkUserAccess(req, resp, opClient, "join:cluster") {
				t.Error("checkUserAccess() = false, want true")
			}
		})
		if code != http.StatusOK {
			t.Errorf("code = %d, want 200", code)
		}
	})
}

func TestHandlerRegisterClusterExtra(t *testing.T) {
	t.Run("invalid request body", func(t *testing.T) {
		h := NewHandler()
		req := newHandlerRequest("POST", "/clusters", "{invalid", "", "")
		if code := runHandler(t, req, h.registerCluster); code != http.StatusBadRequest {
			t.Errorf("code = %d, want 400", code)
		}
	})

	t.Run("empty cluster name", func(t *testing.T) {
		h := NewHandler()
		req := newHandlerRequest("POST", "/clusters", `{}`, "", "")
		if code := runHandler(t, req, h.registerCluster); code != http.StatusBadRequest {
			t.Errorf("code = %d, want 400", code)
		}
	})

	t.Run("invalid cluster name", func(t *testing.T) {
		h := NewHandler()
		req := newHandlerRequest("POST", "/clusters", `{"name": "Bad_Name"}`, "", "")
		if code := runHandler(t, req, h.registerCluster); code != http.StatusBadRequest {
			t.Errorf("code = %d, want 400", code)
		}
	})

	t.Run("cluster already exists", func(t *testing.T) {
		h := NewHandler()
		h.ClusterClient = &cluster.OperationClient{}
		patches := gomonkey.ApplyFunc(utils.IsAccessPass,
			func(_ kubernetes.Interface, _ string, _ string) (bool, error) {
				return true, nil
			})
		patches.ApplyMethod(reflect.TypeOf(&cluster.OperationClient{}), "Join",
			func(_ *cluster.OperationClient, _ context.Context, _ cluster.JoinOptions) error {
				return errors.New("cluster ID already exists")
			})
		defer patches.Reset()

		req := newHandlerRequest("POST", "/clusters", `{"name": "member1"}`, "", "alice")
		if code := runHandler(t, req, h.registerCluster); code != http.StatusBadRequest {
			t.Errorf("code = %d, want 400", code)
		}
	})

	t.Run("join failed generic error", func(t *testing.T) {
		h := NewHandler()
		h.ClusterClient = &cluster.OperationClient{}
		patches := gomonkey.ApplyFunc(utils.IsAccessPass,
			func(_ kubernetes.Interface, _ string, _ string) (bool, error) {
				return true, nil
			})
		patches.ApplyMethod(reflect.TypeOf(&cluster.OperationClient{}), "Join",
			func(_ *cluster.OperationClient, _ context.Context, _ cluster.JoinOptions) error {
				return errors.New("boom")
			})
		defer patches.Reset()

		req := newHandlerRequest("POST", "/clusters", `{"name": "member1"}`, "", "alice")
		if code := runHandler(t, req, h.registerCluster); code != http.StatusInternalServerError {
			t.Errorf("code = %d, want 500", code)
		}
	})

	t.Run("success", func(t *testing.T) {
		k8sAPI := newFakeMcsK8s("test-cluster-uid-0001", nil)
		kubeconfig := k8sAPI.listKubeconfig(t)
		body := `{"name": "member1", "kubeconfig": ` + string(kubeconfig) + `}`

		karmadaAPI := newFakeMcsKarmada()
		mgr := newMcsMgrClient(t)
		h := NewHandler()
		h.ClusterClient = &cluster.OperationClient{
			KarmadaVersionedClient: karmadaAPI.newClient(t),
			KarmadaClient:          k8sAPI.newClient(t),
			MgrClient:              mgr,
		}

		patches := gomonkey.ApplyFunc(utils.IsAccessPass,
			func(_ kubernetes.Interface, _ string, _ string) (bool, error) {
				return true, nil
			})
		patches.ApplyFunc(utils.GenerateClusterSecret,
			func(_ kubernetes.Interface, tag string) (*corev1.Secret, *corev1.Secret, error) {
				return &corev1.Secret{ObjectMeta: metav1.ObjectMeta{Name: tag}},
					&corev1.Secret{ObjectMeta: metav1.ObjectMeta{Name: tag + "-imp"}}, nil
			})
		patches.ApplyFunc(utils.GenerateKarmadaSecret,
			func(_ kubernetes.Interface, clusterSt, impersonatorSt *corev1.Secret, _ string) (*corev1.Secret, *corev1.Secret, error) {
				return clusterSt, impersonatorSt, nil
			})
		defer patches.Reset()

		req := newHandlerRequest("POST", "/clusters", body, "", "alice")
		if code := runHandler(t, req, h.registerCluster); code != http.StatusOK {
			t.Errorf("code = %d, want 200", code)
		}

		obj := newV1beta1Cluster("member1")
		if err := mgr.Get(context.TODO(), v1beta1ObjectKey("member1"), obj); err != nil {
			t.Fatalf("v1beta1 cluster not created: %v", err)
		}
		registered, err := karmadaAPI.clustersSnapshot("member1")
		if err != nil || registered == nil {
			t.Fatalf("karmada cluster not registered: %v (err=%v)", registered, err)
		}
	})
}

func TestHandlerGetResourceUsageExtra(t *testing.T) {
	t.Run("too many query values", func(t *testing.T) {
		h := NewHandler()
		h.ClusterClient = &mockOKInspectClient{}
		req := newHandlerRequest("GET", "/resources/clusters?clusterTag=c1&other=x", "", "", "alice")
		if code := runHandler(t, req, h.getResourceUsage); code != http.StatusBadRequest {
			t.Errorf("code = %d, want 400", code)
		}
	})

	t.Run("empty cluster tag value", func(t *testing.T) {
		h := NewHandler()
		h.ClusterClient = &mockOKInspectClient{}
		req := newHandlerRequest("GET", "/resources/clusters?clusterTag=", "", "", "alice")
		if code := runHandler(t, req, h.getResourceUsage); code != http.StatusBadRequest {
			t.Errorf("code = %d, want 400", code)
		}
	})

	t.Run("host cluster is being onboarded", func(t *testing.T) {
		karmadaAPI := newFakeMcsKarmada(newMcsKarmadaCluster("member1", true))
		h := NewHandler()
		h.ClusterClient = &cluster.OperationClient{
			KarmadaVersionedClient: karmadaAPI.newClient(t),
			MgrClient:              newMcsMgrClient(t),
		}

		req := newHandlerRequest("GET", "/resources/clusters", "", "", "alice")
		if code := runHandler(t, req, h.getResourceUsage); code != http.StatusBadGateway {
			t.Errorf("code = %d, want 502", code)
		}
	})

	t.Run("cluster ID no exists", func(t *testing.T) {
		// list-all mode: host cluster ready but no matching v1beta1 clusters ->
		// inspect fails per cluster with "cluster ID no exists".
		// NOTE: the single-cluster path of Inspect panics on missing clusters
		// (unchecked nil from inspect()) - recorded as a product defect.
		karmadaAPI := newFakeMcsKarmada(
			newMcsKarmadaCluster("host", true),
			newMcsKarmadaCluster("member1", true),
		)
		h := NewHandler()
		h.ClusterClient = &cluster.OperationClient{
			KarmadaVersionedClient: karmadaAPI.newClient(t),
			MgrClient:              newMcsMgrClient(t),
		}

		req := newHandlerRequest("GET", "/resources/clusters", "", "", "alice")
		if code := runHandler(t, req, h.getResourceUsage); code != http.StatusBadRequest {
			t.Errorf("code = %d, want 400", code)
		}
	})

	t.Run("user information missing", func(t *testing.T) {
		karmadaAPI := newFakeMcsKarmada(
			newMcsKarmadaCluster("host", true),
			newMcsKarmadaCluster("member1", true),
		)
		mgr := newMcsMgrClient(t, newMcsMgrCluster("member1", nil, nil))
		h := NewHandler()
		h.ClusterClient = &cluster.OperationClient{
			KarmadaVersionedClient: karmadaAPI.newClient(t),
			MgrClient:              mgr,
		}

		req := newHandlerRequest("GET", "/resources/clusters?clusterTag=member1", "", "", "")
		if code := runHandler(t, req, h.getResourceUsage); code != http.StatusBadRequest {
			t.Errorf("code = %d, want 400", code)
		}
	})

	t.Run("client type assertion failed", func(t *testing.T) {
		karmadaAPI := newFakeMcsKarmada(
			newMcsKarmadaCluster("host", true),
			newMcsKarmadaCluster("member1", true),
		)
		mgr := newMcsMgrClient(t, newMcsMgrCluster("member1", nil, nil))
		h := NewHandler()
		h.ClusterClient = &mockOKInspectClient{}
		_ = karmadaAPI
		_ = mgr

		req := newHandlerRequest("GET", "/resources/clusters?clusterTag=member1", "", "", "alice")
		if code := runHandler(t, req, h.getResourceUsage); code != http.StatusInternalServerError {
			t.Errorf("code = %d, want 500", code)
		}
	})

	t.Run("access list failed", func(t *testing.T) {
		karmadaAPI := newFakeMcsKarmada(
			newMcsKarmadaCluster("host", true),
			newMcsKarmadaCluster("member1", true),
		)
		mgr := newMcsMgrClient(t, newMcsMgrCluster("member1", nil, nil))
		h := NewHandler()
		h.ClusterClient = &cluster.OperationClient{
			KarmadaVersionedClient: karmadaAPI.newClient(t),
			KarmadaClient:          newFakeMcsK8s("", nil).newClient(t),
			MgrClient:              mgr,
		}

		patches := gomonkey.ApplyFunc(utils.CheckUserAccessList,
			func(_ kubernetes.Interface, _ string) ([]string, error) {
				return nil, errors.New("boom")
			})
		defer patches.Reset()

		req := newHandlerRequest("GET", "/resources/clusters?clusterTag=member1", "", "", "alice")
		if code := runHandler(t, req, h.getResourceUsage); code != http.StatusInternalServerError {
			t.Errorf("code = %d, want 500", code)
		}
	})

	t.Run("success filters clusters for user", func(t *testing.T) {
		karmadaAPI := newFakeMcsKarmada(
			newMcsKarmadaCluster("host", true),
			newMcsKarmadaCluster("member1", true),
			newMcsKarmadaCluster("member2", true),
		)
		mgr := newMcsMgrClient(t,
			newMcsMgrCluster("member1", nil, nil),
			newMcsMgrCluster("member2", nil, nil),
		)
		h := NewHandler()
		h.ClusterClient = &cluster.OperationClient{
			KarmadaVersionedClient: karmadaAPI.newClient(t),
			KarmadaClient:          newFakeMcsK8s("", nil).newClient(t),
			MgrClient:              mgr,
		}

		patches := gomonkey.ApplyFunc(utils.CheckUserAccessList,
			func(_ kubernetes.Interface, _ string) ([]string, error) {
				return []string{"member1"}, nil
			})
		defer patches.Reset()

		req := newHandlerRequest("GET", "/resources/clusters?clusterTag=member1", "", "", "alice")
		if code := runHandler(t, req, h.getResourceUsage); code != http.StatusOK {
			t.Errorf("code = %d, want 200", code)
		}
	})
}

func TestHandlerGetClusterResourceUsageExtra(t *testing.T) {
	t.Run("missing cluster tag", func(t *testing.T) {
		h := NewHandler()
		h.ClusterClient = &MockClusterClient{}
		req := newHandlerRequest("GET", "/clusters//resources", "", "", "alice")
		if code := runHandler(t, req, h.getClusterResourceUsage); code != http.StatusBadRequest {
			t.Errorf("code = %d, want 400", code)
		}
	})

	t.Run("client type assertion failed", func(t *testing.T) {
		h := NewHandler()
		h.ClusterClient = &MockClusterClient{}
		req := newHandlerRequest("GET", "/clusters/member1/resources", "", "member1", "alice")
		if code := runHandler(t, req, h.getClusterResourceUsage); code != http.StatusInternalServerError {
			t.Errorf("code = %d, want 500", code)
		}
	})

	t.Run("cluster ID no exists", func(t *testing.T) {
		karmadaAPI := newFakeMcsKarmada()
		h := NewHandler()
		h.ClusterClient = &cluster.OperationClient{KarmadaVersionedClient: karmadaAPI.newClient(t)}

		req := newHandlerRequest("GET", "/clusters/member1/resources", "", "member1", "alice")
		if code := runHandler(t, req, h.getClusterResourceUsage); code != http.StatusBadRequest {
			t.Errorf("code = %d, want 400", code)
		}
	})

	t.Run("inspection failed", func(t *testing.T) {
		karmadaAPI := newFakeMcsKarmada(newMcsKarmadaCluster("member1", true))
		karmadaAPI.failGet["member1"] = 500
		h := NewHandler()
		h.ClusterClient = &cluster.OperationClient{KarmadaVersionedClient: karmadaAPI.newClient(t)}

		req := newHandlerRequest("GET", "/clusters/member1/resources", "", "member1", "alice")
		if code := runHandler(t, req, h.getClusterResourceUsage); code != http.StatusInternalServerError {
			t.Errorf("code = %d, want 500", code)
		}
	})

	t.Run("success", func(t *testing.T) {
		karmadaAPI := newFakeMcsKarmada(newMcsKarmadaCluster("member1", true))
		h := NewHandler()
		h.ClusterClient = &cluster.OperationClient{KarmadaVersionedClient: karmadaAPI.newClient(t)}

		req := newHandlerRequest("GET", "/clusters/member1/resources", "", "member1", "alice")
		if code := runHandler(t, req, h.getClusterResourceUsage); code != http.StatusOK {
			t.Errorf("code = %d, want 200", code)
		}
	})
}

func TestHandlerKubeconfigRetrievalExtra(t *testing.T) {
	t.Run("missing cluster tag", func(t *testing.T) {
		h := NewHandler()
		h.ClusterClient = &MockClusterClient{}
		req := newHandlerRequest("GET", "/clusters//kubeconfig", "", "", "alice")
		if code := runHandler(t, req, h.kubeconfigRetrieval); code != http.StatusBadRequest {
			t.Errorf("code = %d, want 400", code)
		}
	})

	t.Run("user information missing", func(t *testing.T) {
		h := NewHandler()
		h.ClusterClient = &MockClusterClient{}
		req := newHandlerRequest("GET", "/clusters/member1/kubeconfig", "", "member1", "")
		if code := runHandler(t, req, h.kubeconfigRetrieval); code != http.StatusBadRequest {
			t.Errorf("code = %d, want 400", code)
		}
	})

	t.Run("client type assertion failed", func(t *testing.T) {
		h := NewHandler()
		h.ClusterClient = &MockClusterClient{}
		req := newHandlerRequest("GET", "/clusters/member1/kubeconfig", "", "member1", "alice")
		if code := runHandler(t, req, h.kubeconfigRetrieval); code != http.StatusInternalServerError {
			t.Errorf("code = %d, want 500", code)
		}
	})

	t.Run("access check failed", func(t *testing.T) {
		k8sAPI := newFakeMcsK8s("", nil)
		mgr := newMcsMgrClient(t)
		h := NewHandler()
		h.ClusterClient = &cluster.OperationClient{KarmadaClient: k8sAPI.newClient(t), MgrClient: mgr}

		patches := gomonkey.ApplyMethod(reflect.TypeOf(&cluster.OperationClient{}), "CheckUserKubeconfigAccess",
			func(_ *cluster.OperationClient, _, _ string) (bool, json.RawMessage, error) {
				return false, nil, errors.New("boom")
			})
		defer patches.Reset()

		req := newHandlerRequest("GET", "/clusters/member1/kubeconfig", "", "member1", "alice")
		if code := runHandler(t, req, h.kubeconfigRetrieval); code != http.StatusInternalServerError {
			t.Errorf("code = %d, want 500", code)
		}
	})

	t.Run("access forbidden", func(t *testing.T) {
		k8sAPI := newFakeMcsK8s("test-cluster-uid-0001", nil)
		kubeconfig := k8sAPI.mapKubeconfig(t)
		mgr := newMcsMgrClient(t, newMcsMgrCluster("member1", kubeconfig, nil))
		h := NewHandler()
		h.ClusterClient = &cluster.OperationClient{KarmadaClient: k8sAPI.newClient(t), MgrClient: mgr}

		req := newHandlerRequest("GET", "/clusters/member1/kubeconfig", "", "member1", "bob")
		if code := runHandler(t, req, h.kubeconfigRetrieval); code != http.StatusForbidden {
			t.Errorf("code = %d, want 403", code)
		}
	})

	t.Run("success returns kubeconfig", func(t *testing.T) {
		k8sAPI := newFakeMcsK8s("test-cluster-uid-0001", []string{"alice-platform-admin"})
		kubeconfig := k8sAPI.mapKubeconfig(t)
		mgr := newMcsMgrClient(t, newMcsMgrCluster("member1", kubeconfig, []byte(`{"out":"config"}`)))
		h := NewHandler()
		h.ClusterClient = &cluster.OperationClient{KarmadaClient: k8sAPI.newClient(t), MgrClient: mgr}

		req := newHandlerRequest("GET", "/clusters/member1/kubeconfig", "", "member1", "alice")
		if code := runHandler(t, req, h.kubeconfigRetrieval); code != http.StatusOK {
			t.Errorf("code = %d, want 200", code)
		}
	})
}

func TestHandlerEditClusterExtra(t *testing.T) {
	patchAccess := func() *gomonkey.Patches {
		return gomonkey.ApplyFunc(utils.IsAccessPass,
			func(_ kubernetes.Interface, _ string, _ string) (bool, error) {
				return true, nil
			})
	}

	t.Run("invalid labels rejected", func(t *testing.T) {
		h := NewHandler()
		h.ClusterClient = &cluster.OperationClient{}
		patches := patchAccess()
		defer patches.Reset()

		// 65-char key exceeds MaxLabelLength -> IsLabelsValid fails
		longKey := strings.Repeat("a", 65)
		body := `{"clusterlabels": {"` + longKey + `": "c"}}`
		req := newHandlerRequest("PATCH", "/clusters/member1", body, "member1", "alice")
		if code := runHandler(t, req, h.editCluster); code != http.StatusBadRequest {
			t.Errorf("code = %d, want 400", code)
		}
	})

	t.Run("cluster ID no exists", func(t *testing.T) {
		h := NewHandler()
		h.ClusterClient = &cluster.OperationClient{}
		patches := patchAccess()
		patches.ApplyMethod(reflect.TypeOf(&cluster.OperationClient{}), "Edit",
			func(_ *cluster.OperationClient, _ context.Context, _ cluster.EditOptions) error {
				return errors.New("cluster ID no exists")
			})
		defer patches.Reset()

		req := newHandlerRequest("PATCH", "/clusters/member1", `{"clusterlabels": {"env": "prod"}}`, "member1", "alice")
		if code := runHandler(t, req, h.editCluster); code != http.StatusBadRequest {
			t.Errorf("code = %d, want 400", code)
		}
	})

	t.Run("edit failed generic error", func(t *testing.T) {
		h := NewHandler()
		h.ClusterClient = &cluster.OperationClient{}
		patches := patchAccess()
		patches.ApplyMethod(reflect.TypeOf(&cluster.OperationClient{}), "Edit",
			func(_ *cluster.OperationClient, _ context.Context, _ cluster.EditOptions) error {
				return errors.New("boom")
			})
		defer patches.Reset()

		req := newHandlerRequest("PATCH", "/clusters/member1", `{"clusterlabels": {"env": "prod"}}`, "member1", "alice")
		if code := runHandler(t, req, h.editCluster); code != http.StatusInternalServerError {
			t.Errorf("code = %d, want 500", code)
		}
	})

	t.Run("success", func(t *testing.T) {
		h := NewHandler()
		h.ClusterClient = &cluster.OperationClient{}
		patches := patchAccess()
		patches.ApplyMethod(reflect.TypeOf(&cluster.OperationClient{}), "Edit",
			func(_ *cluster.OperationClient, _ context.Context, _ cluster.EditOptions) error {
				return nil
			})
		defer patches.Reset()

		req := newHandlerRequest("PATCH", "/clusters/member1", `{"clusterlabels": {"env": "prod"}}`, "member1", "alice")
		if code := runHandler(t, req, h.editCluster); code != http.StatusOK {
			t.Errorf("code = %d, want 200", code)
		}
	})
}

func TestHandlerAuthPolicyEnforcerExtra(t *testing.T) {
	t.Run("authorize failed", func(t *testing.T) {
		h := NewHandler()
		h.ClusterClient = &cluster.OperationClient{}
		patches := gomonkey.ApplyMethod(reflect.TypeOf(&cluster.OperationClient{}), "Authorize",
			func(_ *cluster.OperationClient, _ context.Context, _ cluster.AuthorizeOptions) error {
				return errors.New("boom")
			})
		defer patches.Reset()

		req := newHandlerRequest("POST", "/userinfo",
			`{"UserInfo": {"alice": {"IdentityName": "alice"}}}`, "", "")
		if code := runHandler(t, req, h.authPolicyEnforcer); code != http.StatusInternalServerError {
			t.Errorf("code = %d, want 500", code)
		}
	})
}

func TestHandlerUnregisterClusterExtra(t *testing.T) {
	t.Run("unjoin failed", func(t *testing.T) {
		h := NewHandler()
		h.ClusterClient = &cluster.OperationClient{}
		patches := gomonkey.ApplyFunc(utils.IsAccessPass,
			func(_ kubernetes.Interface, _ string, _ string) (bool, error) {
				return true, nil
			})
		patches.ApplyMethod(reflect.TypeOf(&cluster.OperationClient{}), "Unjoin",
			func(_ *cluster.OperationClient, _ context.Context, _ cluster.UnjoinOptions) error {
				return errors.New("boom")
			})
		defer patches.Reset()

		req := newHandlerRequest("DELETE", "/clusters/member1", "", "member1", "alice")
		if code := runHandler(t, req, h.unregisterCluster); code != http.StatusInternalServerError {
			t.Errorf("code = %d, want 500", code)
		}
	})
}
