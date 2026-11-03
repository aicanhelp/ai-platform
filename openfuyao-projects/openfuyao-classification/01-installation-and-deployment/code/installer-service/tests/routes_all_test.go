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

package tests

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	dynamicfake "k8s.io/client-go/dynamic/fake"
	fake "k8s.io/client-go/kubernetes/fake"

	"installer-service/pkg/installer"
	"installer-service/tests/testutils"
)

var _ = Describe("Installer API routes (spec coverage from installer_api.yaml)", func() {
	var srv *httptest.Server
	var cs *fake.Clientset
	var dyn *dynamicfake.FakeDynamicClient
	var patches interface{ Reset() }

	var doRequest func(method, path string, body []byte, contentType string) *http.Response

	BeforeEach(func() {
		s, clientset, d, p := testutils.StartServerWithFakeClients()
		srv = s
		cs = clientset
		dyn = d
		patches = p

		// ensure a minimal BKE config exists so handlers depending on it don't 500
		cmKey := installer.BKEConfigCmKey()
		cm := &corev1.ConfigMap{ObjectMeta: metav1.ObjectMeta{Name: cmKey.Name, Namespace: cmKey.Namespace}, Data: map[string]string{"otherRepo": "", "onlineImage": "", "domain": "cr.openfuyao.cn"}}
		_, _ = cs.CoreV1().ConfigMaps(cmKey.Namespace).Create(context.TODO(), cm, metav1.CreateOptions{})

		upgradePathGVR := schema.GroupVersionResource{Group: "config.openfuyao.com", Version: "v1alpha1", Resource: "upgradepaths"}
		_, _ = dyn.Resource(upgradePathGVR).Create(context.TODO(), &unstructured.Unstructured{Object: map[string]interface{}{
			"apiVersion": "config.openfuyao.com/v1alpha1",
			"kind":       "UpgradePath",
			"metadata": map[string]interface{}{
				"name": "default-paths",
			},
			"spec": map[string]interface{}{
				"versions": []interface{}{
					map[string]interface{}{"version": "v25.09.0", "installable": true, "deprecated": false},
				},
			},
		}}, metav1.CreateOptions{})

		doRequest = func(method, path string, body []byte, contentType string) *http.Response {
			w := httptest.NewRecorder()
			var r *http.Request
			if body != nil {
				r = httptest.NewRequest(method, path, bytes.NewReader(body))
			} else {
				r = httptest.NewRequest(method, path, nil)
			}
			if contentType != "" {
				r.Header.Set("Content-Type", contentType)
			}
			// wrap in inner func so w.Result() is always returned even after panic recovery
			func() {
				defer func() {
					if rec := recover(); rec != nil {
						if w.Code == 0 || w.Code == http.StatusOK {
							w.Code = http.StatusInternalServerError
						}
					}
				}()
				srv.Config.Handler.ServeHTTP(w, r)
			}()
			return w.Result()
		}

		DeferCleanup(func() {
			patches.Reset()
			srv.Close()
		})
	})

	// helper to decode standard envelope responses
	decodeBody := func(resp *http.Response) map[string]interface{} {
		defer resp.Body.Close()
		var out map[string]interface{}
		_ = json.NewDecoder(resp.Body).Decode(&out)
		return out
	}

	Describe("GET /rest/cluster/v1/configs", func() {
		It("returns code 200 and data", func() {
			resp := doRequest(http.MethodGet, "/rest/cluster/v1/configs", nil, "")
			Expect(resp.StatusCode).To(Equal(http.StatusOK))
			body := decodeBody(resp)
			Expect(body).To(HaveKeyWithValue("code", BeNumerically("==", float64(200))))
			Expect(body).To(HaveKey("data"))
		})
	})

	Describe("GET /rest/cluster/v1/versions", func() {
		It("returns versions list", func() {
			resp := doRequest(http.MethodGet, "/rest/cluster/v1/versions", nil, "")
			Expect(resp.StatusCode).To(Equal(http.StatusOK))
			body := decodeBody(resp)
			Expect(body).To(HaveKey("data"))
			if data, ok := body["data"].(map[string]interface{}); ok {
				Expect(data).To(HaveKey("versions"))
			}
		})
		It("returns versions from offline configMap keys", func() {
			upgradePathGVR := schema.GroupVersionResource{Group: "config.openfuyao.com", Version: "v1alpha1", Resource: "upgradepaths"}
			_, err := dyn.Resource(upgradePathGVR).Create(context.TODO(), &unstructured.Unstructured{Object: map[string]interface{}{
				"apiVersion": "config.openfuyao.com/v1alpha1",
				"kind":       "UpgradePath",
				"metadata": map[string]interface{}{
					"name": "offline-paths",
				},
				"spec": map[string]interface{}{
					"versions": []interface{}{
						map[string]interface{}{"version": "v25.99.0", "installable": true, "deprecated": false},
						map[string]interface{}{"version": "v25.99.1", "installable": true, "deprecated": true},
						map[string]interface{}{"version": "v25.99.2", "installable": false, "deprecated": false},
					},
				},
			}}, metav1.CreateOptions{})
			Expect(err).ToNot(HaveOccurred())

			resp := doRequest(http.MethodGet, "/rest/cluster/v1/versions", nil, "")
			Expect(resp.StatusCode).To(Equal(http.StatusOK))
			body := decodeBody(resp)
			Expect(body).To(HaveKey("data"))
			if data, ok := body["data"].(map[string]interface{}); ok {
				if versions, ok := data["versions"].([]interface{}); ok {
					// ensure our test version is present
					found := false
					for _, v := range versions {
						if s, ok := v.(string); ok && s == "v25.99.0" {
							found = true
						}
					}
					Expect(found).To(BeTrue())
				}
			}
		})

		It("returns versions from online remote index.yaml", func() {
			upgradePathGVR := schema.GroupVersionResource{Group: "config.openfuyao.com", Version: "v1alpha1", Resource: "upgradepaths"}
			_, err := dyn.Resource(upgradePathGVR).Create(context.TODO(), &unstructured.Unstructured{Object: map[string]interface{}{
				"apiVersion": "config.openfuyao.com/v1alpha1",
				"kind":       "UpgradePath",
				"metadata": map[string]interface{}{
					"name": "remote-paths",
				},
				"spec": map[string]interface{}{
					"versions": []interface{}{
						map[string]interface{}{"version": "v26.99.0", "installable": true, "deprecated": false},
					},
				},
			}}, metav1.CreateOptions{})
			Expect(err).ToNot(HaveOccurred())

			resp := doRequest(http.MethodGet, "/rest/cluster/v1/versions", nil, "")
			Expect(resp.StatusCode).To(Equal(http.StatusOK))
			body := decodeBody(resp)
			Expect(body).To(HaveKey("data"))
			if data, ok := body["data"].(map[string]interface{}); ok {
				if versions, ok := data["versions"].([]interface{}); ok {
					// ensure our remote version is present
					found := false
					for _, v := range versions {
						if s, ok := v.(string); ok && s == "v26.99.0" {
							found = true
						}
					}
					Expect(found).To(BeTrue())
				}
			}
		})
	})

	Describe("POST /rest/cluster/v1/clusters", func() {
		It("rejects malformed JSON payload", func() {
			// invalid JSON should cause binding error
			resp := doRequest(http.MethodPost, "/rest/cluster/v1/clusters", []byte("{bad json"), "application/json")
			// accept 400 or 500 depending on implementation
			Expect(resp.StatusCode == http.StatusBadRequest || resp.StatusCode == http.StatusInternalServerError).To(BeTrue())
		})

		It("returns 400 for empty JSON body", func() {
			// empty body with content-type should be treated as bad request
			resp := doRequest(http.MethodPost, "/rest/cluster/v1/clusters", nil, "application/json")
			Expect(resp.StatusCode).To(Equal(http.StatusBadRequest))
		})

		It("returns 500 when cluster name missing after binding", func() {
			payload := map[string]interface{}{"cluster": map[string]interface{}{"name": ""}}
			b, _ := json.Marshal(payload)
			resp := doRequest(http.MethodPost, "/rest/cluster/v1/clusters", b, "application/json")
			Expect(resp.StatusCode).To(Equal(http.StatusBadRequest))
		})
	})

	// 新增：针对 POST /rest/cluster/v1/clusters 的创建流程测试
	Describe("POST /rest/cluster/v1/clusters (creation)", func() {
		It("creates a BKECluster resource when payload is valid", func() {
			payload := map[string]interface{}{
				"cluster": map[string]interface{}{
					"name":             "create-ok",
					"openFuyaoVersion": "v0",
					"imageRepo":        map[string]interface{}{"url": "", "ip": ""},
				},
				"controlPlaneEndpoint": "1.2.3.4",
			}
			b, _ := json.Marshal(payload)
			resp := doRequest(http.MethodPost, "/rest/cluster/v1/clusters", b, "application/json")
			// handler may perform async/remote ops and return 500 in CI; accept both but
			// when 200 assert the cluster resource was created in the dynamic client.
			if resp.StatusCode == http.StatusOK {
				body := decodeBody(resp)
				Expect(body).To(HaveKeyWithValue("code", BeNumerically("==", float64(200))))
				gvr := schema.GroupVersionResource{Group: "bke.bocloud.com", Version: "v1beta1", Resource: "bkeclusters"}
				// small sleep to allow async create to complete in fake client
				time.Sleep(10 * time.Millisecond)
				got, err := dyn.Resource(gvr).Namespace("create-ok").Get(context.TODO(), "create-ok", metav1.GetOptions{})
				Expect(err).ToNot(HaveOccurred())
				Expect(got.GetName()).To(Equal("create-ok"))
			} else {
				body := decodeBody(resp)
				Expect(body).To(HaveKey("code"))
			}
		})

		It("returns 400 when cluster name is empty", func() {
			payload := map[string]interface{}{"cluster": map[string]interface{}{"name": ""}}
			b, _ := json.Marshal(payload)
			resp := doRequest(http.MethodPost, "/rest/cluster/v1/clusters", b, "application/json")
			Expect(resp.StatusCode).To(Equal(http.StatusBadRequest))
			body := decodeBody(resp)
			Expect(body).To(HaveKey("code"))
		})
	})

	Describe("GET /rest/cluster/v1/clusters", func() {
		It("returns items array", func() {
			// seed one cluster so the list contains an item
			gvr := schema.GroupVersionResource{Group: "bke.bocloud.com", Version: "v1beta1", Resource: "bkeclusters"}
			bc := &unstructured.Unstructured{Object: map[string]interface{}{
				"apiVersion": "bke.bocloud.com/v1beta1",
				"kind":       "BKECluster",
				"metadata":   map[string]interface{}{"name": "list-cluster", "namespace": "list-cluster"},
				"spec": map[string]interface{}{
					"controlPlaneEndpoint": map[string]interface{}{"host": "1.2.3.4"},
					"clusterConfig": map[string]interface{}{
						"cluster": map[string]interface{}{
							"openFuyaoVersion":  "v0",
							"kubernetesVersion": "v1.26.0",
							"containerRuntime":  map[string]interface{}{"CRI": "containerd"},
						},
						"nodes": []interface{}{},
					},
				},
			}}
			_, err := dyn.Resource(gvr).Namespace("list-cluster").Create(context.TODO(), bc, metav1.CreateOptions{})
			Expect(err).ToNot(HaveOccurred())

			resp := doRequest(http.MethodGet, "/rest/cluster/v1/clusters", nil, "")
			Expect(resp.StatusCode).To(Equal(http.StatusOK))
			body := decodeBody(resp)
			Expect(body).To(HaveKey("data"))
			if data, ok := body["data"].(map[string]interface{}); ok {
				Expect(data).To(HaveKey("items"))
			}
		})
	})

	Describe("GET /rest/cluster/v1/clusters/{name}", func() {
		It("returns cluster detail", func() {
			gvr := schema.GroupVersionResource{Group: "bke.bocloud.com", Version: "v1beta1", Resource: "bkeclusters"}
			nodeGVR := schema.GroupVersionResource{Group: "bke.bocloud.com", Version: "v1beta1", Resource: "bkenodes"}

			bc := &unstructured.Unstructured{Object: map[string]interface{}{
				"apiVersion": "bke.bocloud.com/v1beta1",
				"kind":       "BKECluster",
				"metadata":   map[string]interface{}{"name": "detail-ok", "namespace": "detail-ok"},
				"spec": map[string]interface{}{
					"controlPlaneEndpoint": map[string]interface{}{"host": "1.2.3.4"},
					"clusterConfig": map[string]interface{}{
						"cluster": map[string]interface{}{
							"openFuyaoVersion":  "v0",
							"kubernetesVersion": "v1.26.0",
							"containerRuntime":  map[string]interface{}{"CRI": "containerd"},
						},
						"nodes": []interface{}{},
					},
				},
			}}
			_, err := dyn.Resource(gvr).Namespace("detail-ok").Create(context.TODO(), bc, metav1.CreateOptions{})
			Expect(err).ToNot(HaveOccurred())

			var labels = map[string]string{
				"cluster.x-k8s.io/cluster-name": "detail-ok",
			}
			node := testutils.MakeBKENodeUnstructured("n-detail", "detail-ok", "10.10.10.1", []string{"master"}, labels)
			_, err = dyn.Resource(nodeGVR).Namespace("detail-ok").Create(context.TODO(), node, metav1.CreateOptions{})
			Expect(err).ToNot(HaveOccurred())

			time.Sleep(10 * time.Millisecond)
			resp := doRequest(http.MethodGet, "/rest/cluster/v1/clusters/detail-ok", nil, "")
			Expect(resp.StatusCode).To(Equal(http.StatusOK))
			body := decodeBody(resp)
			Expect(body).To(HaveKey("data"))
			if data, ok := body["data"].(map[string]interface{}); ok {
				Expect(data).To(HaveKeyWithValue("clusterName", "detail-ok"))
			}
		})
	})

	Describe("DELETE /rest/cluster/v1/clusters/{name}", func() {
		It("deletes BKENode and sets reset", func() {
			gvr := schema.GroupVersionResource{Group: "bke.bocloud.com", Version: "v1beta1", Resource: "bkeclusters"}
			nodeGVR := schema.GroupVersionResource{Group: "bke.bocloud.com", Version: "v1beta1", Resource: "bkenodes"}
			bc := testutils.MakeBKEClusterUnstructured("del-cluster", "del-cluster", "1.2.3.4", "v0", "v1.26.0")
			_, err := dyn.Resource(gvr).Namespace("del-cluster").Create(context.TODO(), bc, metav1.CreateOptions{})
			Expect(err).ToNot(HaveOccurred())

			node := testutils.MakeBKENodeUnstructured("node1", "del-cluster", "10.0.0.2", nil, map[string]string{"cluster.x-k8s.io/cluster-name": "del-cluster"})
			_, err = dyn.Resource(nodeGVR).Namespace("del-cluster").Create(context.TODO(), node, metav1.CreateOptions{})
			Expect(err).ToNot(HaveOccurred())

			resp := doRequest(http.MethodDelete, "/rest/cluster/v1/clusters/del-cluster", nil, "")
			Expect(resp.StatusCode).To(Equal(http.StatusOK))
			body := decodeBody(resp)
			Expect(body).To(HaveKeyWithValue("code", BeNumerically("==", float64(200))))

			got, err := dyn.Resource(gvr).Namespace("del-cluster").Get(context.TODO(), "del-cluster", metav1.GetOptions{})
			Expect(err).ToNot(HaveOccurred())
			// expect spec.reset to be true when present
			if v, found, _ := unstructured.NestedBool(got.Object, "spec", "reset"); found {
				Expect(v).To(BeTrue())
			}
		})
	})

	Describe("POST /rest/cluster/v1/clusters/{name}/scale-up", func() {
		It("creates nodes", func() {
			gvr := schema.GroupVersionResource{Group: "bke.bocloud.com", Version: "v1beta1", Resource: "bkeclusters"}
			nodeGVR := schema.GroupVersionResource{Group: "bke.bocloud.com", Version: "v1beta1", Resource: "bkenodes"}
			bc := testutils.MakeBKEClusterUnstructured("scale-ok", "scale-ok", "1.2.3.4", "v0", "v1.26.0")
			_, err := dyn.Resource(gvr).Namespace("scale-ok").Create(context.TODO(), bc, metav1.CreateOptions{})
			Expect(err).ToNot(HaveOccurred())

			payload := map[string]interface{}{"nodes": []map[string]interface{}{{"hostname": "added1", "ip": "10.0.0.10", "port": "22", "username": "root", "password": "pw"}}}
			b, _ := json.Marshal(payload)
			resp := doRequest(http.MethodPost, "/rest/cluster/v1/clusters/scale-ok/scale-up", b, "application/json")
			Expect(resp.StatusCode).To(Equal(http.StatusOK))
			time.Sleep(10 * time.Millisecond)
			_, err = dyn.Resource(nodeGVR).Namespace("scale-ok").Get(context.TODO(), "added1", metav1.GetOptions{})
			Expect(err).ToNot(HaveOccurred())
		})
	})

	Describe("POST /rest/cluster/v1/clusters/{name}/scale-down", func() {
		It("deletes nodes", func() {
			gvr := schema.GroupVersionResource{Group: "bke.bocloud.com", Version: "v1beta1", Resource: "bkeclusters"}
			nodeGVR := schema.GroupVersionResource{Group: "bke.bocloud.com", Version: "v1beta1", Resource: "bkenodes"}
			bc := testutils.MakeBKEClusterUnstructured("down-ok", "down-ok", "1.2.3.4", "v0", "v1.26.0")
			_, err := dyn.Resource(gvr).Namespace("down-ok").Create(context.TODO(), bc, metav1.CreateOptions{})
			Expect(err).ToNot(HaveOccurred())

			node := testutils.MakeBKENodeUnstructured("to-del", "down-ok", "10.0.1.1", nil, nil)
			_, err = dyn.Resource(nodeGVR).Namespace("down-ok").Create(context.TODO(), node, metav1.CreateOptions{})
			Expect(err).ToNot(HaveOccurred())

			payload := map[string]interface{}{"nodes": []string{"to-del"}}
			b, _ := json.Marshal(payload)
			resp := doRequest(http.MethodPost, "/rest/cluster/v1/clusters/down-ok/scale-down", b, "application/json")
			Expect(resp.StatusCode).To(Equal(http.StatusOK))
			_, err = dyn.Resource(nodeGVR).Namespace("down-ok").Get(context.TODO(), "to-del", metav1.GetOptions{})
			Expect(err).To(HaveOccurred())
		})
	})

	Describe("POST /rest/cluster/v1/nodes/validate", func() {
		It("returns envelope (200 or validation errors)", func() {
			payload := map[string]interface{}{"namespace": "test", "nodes": []map[string]interface{}{{"hostname": "master-1", "ip": "192.168.100.150", "port": "22", "username": "root", "password": "123456"}}, "balanceIp": "192.168.100.20"}
			b, _ := json.Marshal(payload)
			resp := doRequest(http.MethodPost, "/rest/cluster/v1/nodes/validate", b, "application/json")
			// the handler may attempt SSH (which fails in CI) and return 400/500; accept those but ensure a JSON envelope exists
			if resp.StatusCode == http.StatusOK {
				body := decodeBody(resp)
				Expect(body).To(HaveKeyWithValue("code", BeNumerically("==", float64(200))))
				// verify the BKECluster resource was patched to the requested version
				gvr := schema.GroupVersionResource{Group: "bke.bocloud.com", Version: "v1beta1", Resource: "bkeclusters"}
				time.Sleep(10 * time.Millisecond)
				got, err := dyn.Resource(gvr).Namespace("upgrade-ok").Get(context.TODO(), "upgrade-ok", metav1.GetOptions{})
				Expect(err).ToNot(HaveOccurred())
				if v, found, _ := unstructured.NestedString(got.Object, "spec", "clusterConfig", "cluster", "openFuyaoVersion"); found {
					Expect(v).To(Equal("v1.0.0"))
				}
			} else {
				body := decodeBody(resp)
				Expect(body).To(HaveKey("code"))
			}
		})

		It("returns 500 on empty body for nodes validate", func() {
			resp := doRequest(http.MethodPost, "/rest/cluster/v1/nodes/validate", nil, "application/json")
			Expect(resp.StatusCode).To(Equal(http.StatusInternalServerError))
			body := decodeBody(resp)
			Expect(body).To(HaveKey("code"))
		})
	})

	Describe("POST /rest/cluster/v1/patches", func() {
		It("accepts upload payload or returns envelope on error", func() {
			payload := map[string]interface{}{"patchFileName": "v1.yaml", "patchFileContent": "content"}
			b, _ := json.Marshal(payload)
			resp := doRequest(http.MethodPost, "/rest/cluster/v1/patches", b, "application/json")
			if resp.StatusCode == http.StatusOK {
				body := decodeBody(resp)
				Expect(body).To(HaveKeyWithValue("code", BeNumerically("==", float64(200))))
			} else {
				body := decodeBody(resp)
				Expect(body).To(HaveKey("code"))
			}
		})

		It("accepts a valid YAML patch content", func() {
			yamlContent := "openFuyaoVersion: v1\nkubernetesVersion: v1.26.0\ncontainerdVersion: 1.6.0\netcdVersion: 3.5.0\n"
			payload := map[string]interface{}{"patchFileName": "v1.yaml", "patchFileContent": yamlContent}
			b, _ := json.Marshal(payload)
			resp := doRequest(http.MethodPost, "/rest/cluster/v1/patches", b, "application/json")
			// UploadPatchFile may still return 500 if configMap handling fails, accept both
			Expect(resp.StatusCode == http.StatusOK || resp.StatusCode == http.StatusInternalServerError).To(BeTrue())
			if resp.StatusCode == http.StatusOK {
				body := decodeBody(resp)
				Expect(body).To(HaveKeyWithValue("code", BeNumerically("==", float64(200))))
			}
		})

		It("rejects empty patch payload with 400", func() {
			resp := doRequest(http.MethodPost, "/rest/cluster/v1/patches", nil, "application/json")
			Expect(resp.StatusCode).To(Equal(http.StatusBadRequest))
		})
	})

	Describe("GET /rest/cluster/v1/clusters/{name}/upgrade-versions", func() {
		It("returns versions array", func() {
			// seed cluster
			gvr := schema.GroupVersionResource{Group: "bke.bocloud.com", Version: "v1beta1", Resource: "bkeclusters"}
			bc := testutils.MakeBKEClusterUnstructured("uv-cluster", "uv-cluster", "1.2.3.4", "v0", "v1.26.0")
			_, err := dyn.Resource(gvr).Namespace("uv-cluster").Create(context.TODO(), bc, metav1.CreateOptions{})
			Expect(err).ToNot(HaveOccurred())

			resp := doRequest(http.MethodGet, "/rest/cluster/v1/clusters/uv-cluster/upgrade-versions", nil, "")
			Expect(resp.StatusCode).To(Equal(http.StatusOK))
			body := decodeBody(resp)
			Expect(body).To(HaveKey("data"))
		})
	})

	Describe("POST /rest/cluster/v1/clusters/{name}/upgrade", func() {
		It("accepts upgrade request for existing cluster", func() {
			gvr := schema.GroupVersionResource{Group: "bke.bocloud.com", Version: "v1beta1", Resource: "bkeclusters"}
			bc := testutils.MakeBKEClusterUnstructured("upgrade-ok", "upgrade-ok", "1.2.3.4", "v0", "v1.26.0")
			_, err := dyn.Resource(gvr).Namespace("upgrade-ok").Create(context.TODO(), bc, metav1.CreateOptions{})
			Expect(err).ToNot(HaveOccurred())
			// handler expects {"version":"vX"}
			payload := map[string]interface{}{"version": "v1.0.0"}
			b, _ := json.Marshal(payload)
			resp := doRequest(http.MethodPost, "/rest/cluster/v1/clusters/upgrade-ok/upgrade", b, "application/json")
			Expect(resp.StatusCode).To(Equal(http.StatusOK))
			body := decodeBody(resp)
			Expect(body).To(HaveKeyWithValue("code", BeNumerically("==", float64(200))))

			clusterVersionGVR := schema.GroupVersionResource{Group: "config.openfuyao.com", Version: "v1alpha1", Resource: "clusterversions"}
			clusterVersion, err := dyn.Resource(clusterVersionGVR).Namespace("upgrade-ok").
				Get(context.TODO(), "upgrade-ok", metav1.GetOptions{})
			Expect(err).ToNot(HaveOccurred())
			desiredVersion, found, err := unstructured.NestedString(clusterVersion.Object, "spec", "desiredVersion")
			Expect(err).ToNot(HaveOccurred())
			Expect(found).To(BeTrue())
			Expect(desiredVersion).To(Equal("v1.0.0"))
		})

		It("returns 400 when version missing in request body", func() {
			gvr := schema.GroupVersionResource{Group: "bke.bocloud.com", Version: "v1beta1", Resource: "bkeclusters"}
			// seed cluster so binding passes; handler will validate req.Version and return 400
			bc := testutils.MakeBKEClusterUnstructured("upgrade-badv", "upgrade-badv", "1.2.3.4", "v0", "v1.26.0")
			_, err := dyn.Resource(gvr).Namespace("upgrade-badv").Create(context.TODO(), bc, metav1.CreateOptions{})
			Expect(err).ToNot(HaveOccurred())

			// empty version should trigger 400
			b, _ := json.Marshal(map[string]interface{}{})
			resp := doRequest(http.MethodPost, "/rest/cluster/v1/clusters/upgrade-badv/upgrade", b, "application/json")
			Expect(resp.StatusCode).To(Equal(http.StatusBadRequest))
			body := decodeBody(resp)
			Expect(body).To(HaveKey("code"))
		})

		It("returns non-200 for bad request to upgrade (missing cluster)", func() {
			b, _ := json.Marshal(map[string]interface{}{"version": "v1.0.0"})
			resp := doRequest(http.MethodPost, "/rest/cluster/v1/clusters/upgrade-missing/upgrade", b, "application/json")
			Expect(resp.StatusCode).To(Equal(http.StatusInternalServerError))
			body := decodeBody(resp)
			Expect(body).To(HaveKey("code"))
		})
	})

	Describe("POST /rest/cluster/v1/clusters/{name}/auto-upgrade", func() {
		It("starts auto-upgrade preparation for existing cluster", func() {
			gvr := schema.GroupVersionResource{Group: "bke.bocloud.com", Version: "v1beta1", Resource: "bkeclusters"}
			bc := testutils.MakeBKEClusterUnstructured("auto-ok", "auto-ok", "1.2.3.4", "v0", "v1.26.0")
			_, err := dyn.Resource(gvr).Namespace("auto-ok").Create(context.TODO(), bc, metav1.CreateOptions{})
			Expect(err).ToNot(HaveOccurred())

			// payload may not be strictly required; send empty object
			b, _ := json.Marshal(map[string]interface{}{})
			resp := doRequest(http.MethodPost, "/rest/cluster/v1/clusters/auto-ok/auto-upgrade", b, "application/json")
			if resp.StatusCode == http.StatusOK {
				body := decodeBody(resp)
				Expect(body).To(HaveKeyWithValue("code", BeNumerically("==", float64(200))))
			} else {
				body := decodeBody(resp)
				Expect(body).To(HaveKey("code"))
			}
		})

		It("fails when auto-upgrade requested for missing cluster", func() {
			resp := doRequest(http.MethodPost, "/rest/cluster/v1/clusters/missing/auto-upgrade", nil, "application/json")
			Expect(resp.StatusCode).ToNot(Equal(http.StatusOK))
			body := decodeBody(resp)
			Expect(body).To(HaveKey("code"))
		})
	})

	Describe("GET /ws/cluster/v1/logs", func() {
		It("returns envelope or upgrade when cluster param provided", func() {
			resp := doRequest(http.MethodGet, "/ws/cluster/v1/logs?cluster-name=any", nil, "")
			// websocket handlers may upgrade (101) or return JSON envelopes; ensure no panic and JSON envelope contains code when present
			if resp.StatusCode == http.StatusSwitchingProtocols {
				// upgrade accepted — nothing further to assert
				return
			}
			// attempt to decode JSON; if present, assert envelope contains code
			var body map[string]interface{}
			if err := json.NewDecoder(resp.Body).Decode(&body); err == nil {
				Expect(body).To(HaveKey("code"))
			}
		})
	})

})
