package console

import (
	"encoding/json"
	"io"
	"net/http"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

var _ = Describe("Pod 资源接口测试", Label("with-workload-cluster"), func() {
	Context("Pod 创建", func() {
		It("创建成功", func() {
			resp, err := createPod("test-pod-success", "default")
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusCreated), string(body))

			defer func() {
				_, err := makeRequest("DELETE", "/api/v1/namespaces/default/pods/test-pod-success", nil)
				Expect(err).NotTo(HaveOccurred())
			}()
		})

		It("创建失败 - 字段有误", func() {
			podYAML := map[string]interface{}{
				"apiVersion": "v1",
				"kind":       "Pod",
				"metadata": map[string]interface{}{
					"name":      "test-pod-invalid-image",
					"namespace": "default",
				},
				"spec": map[string]interface{}{
					"containers": []map[string]interface{}{
						{
							"name": "test-container",
						},
					},
				},
			}

			resp, err := makeRequest("POST", "/api/v1/namespaces/default/pods", podYAML)
			Expect(err).NotTo(HaveOccurred())

			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusUnprocessableEntity), string(body))
		})

		It("创建失败 - 已存在同名", func() {
			resp1, err := createPod("test-existing-pod", "default")
			Expect(err).NotTo(HaveOccurred())
			defer resp1.Body.Close()
			Expect(resp1.StatusCode).To(Equal(http.StatusCreated))

			defer func() {
				_, err := makeRequest("DELETE", "/api/v1/namespaces/default/pods/test-existing-pod", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			resp2, err := createPod("test-existing-pod", "default")
			Expect(err).NotTo(HaveOccurred())
			defer resp2.Body.Close()
			body, err := io.ReadAll(resp2.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp2.StatusCode).To(Equal(http.StatusConflict), string(body))
		})

		It("创建失败 - Namespace 不存在", func() {
			resp, err := createPod("test-pod-nonexistent-ns", "nonexistent-namespace")
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusNotFound), string(body))
		})
	})

	Context("Pod 查询", func() {
		It("查询成功", func() {
			_, err := createPod("test-pod-query", "default")
			Expect(err).NotTo(HaveOccurred())

			defer func() {
				_, err := makeRequest("DELETE", "/api/v1/namespaces/default/pods/test-pod-query", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			resp, err := makeRequest("GET", "/api/v1/namespaces/default/pods/test-pod-query", nil)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusOK), string(body))

			var pod map[string]interface{}
			err = json.Unmarshal(body, &pod)
			Expect(err).NotTo(HaveOccurred())
			Expect(pod["metadata"].(map[string]interface{})["name"]).To(Equal("test-pod-query"))
		})
	})

	Context("Pod 修改", func() {
		It("修改成功", func() {
			_, err := createPod("test-pod-update-success", "default")
			Expect(err).NotTo(HaveOccurred())

			defer func() {
				_, err := makeRequest("DELETE", "/api/v1/namespaces/default/pods/test-pod-update-success", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			patch := createPatch("test-pod-update-success", "default")
			resp, err := makeRequest("PATCH", "/api/v1/namespaces/default/pods/test-pod-update-success", patch)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusOK), string(body))
		})

		It("修改失败 - 字段有误", func() {
			_, err := createPod("test-pod-field-error", "default")
			Expect(err).NotTo(HaveOccurred())

			defer func() {
				_, err := makeRequest("DELETE", "/api/v1/namespaces/default/pods/test-pod-field-error", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			patch := createMalformedPatch("test-pod-field-error", "default")
			resp, err := makeRequest("PATCH", "/api/v1/namespaces/default/pods/test-pod-field-error", patch)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusUnprocessableEntity), string(body))
		})

		It("修改同一命名空间其他同类资源", func() {
			_, err := createPod("test-pod-other-resource", "default")
			Expect(err).NotTo(HaveOccurred())
			_, err = createPod("other-pod", "default")
			Expect(err).NotTo(HaveOccurred())

			defer func() {
				_, err := makeRequest("DELETE", "/api/v1/namespaces/default/pods/test-pod-other-resource", nil)
				Expect(err).NotTo(HaveOccurred())
				_, err = makeRequest("DELETE", "/api/v1/namespaces/default/pods/other-pod", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			patch := createPatch("other-pod", "default")
			resp, err := makeRequest("PATCH", "/api/v1/namespaces/default/pods/test-pod-other-resource", patch)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusBadRequest), string(body))
		})

		It("修改不存在的Namespace", func() {
			_, err := createPod("test-pod-nonexistent-ns-update", "default")
			Expect(err).NotTo(HaveOccurred())

			defer func() {
				_, err := makeRequest("DELETE", "/api/v1/namespaces/default/pods/test-pod-nonexistent-ns-update", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			patch := createPatch("test-pod-nonexistent-ns-update", "nonexistent-namespace")
			resp, err := makeRequest("PATCH", "/api/v1/namespaces/default/pods/test-pod-nonexistent-ns-update", patch)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusBadRequest), string(body))
		})
	})

	Context("Pod 删除", func() {
		It("删除成功", func() {
			_, err := createPod("test-pod-delete", "default")
			Expect(err).NotTo(HaveOccurred())

			resp, err := makeRequest("DELETE", "/api/v1/namespaces/default/pods/test-pod-delete", nil)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusOK), string(body))
		})
	})
})

var _ = Describe("Deployment 资源接口测试", Label("with-workload-cluster"), func() {
	Context("Deployment 创建", func() {
		It("创建成功", func() {
			resp, err := createDeployment("test-deployment-success", "default")
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusCreated), string(body))

			defer func() {
				_, err := makeRequest("DELETE", "/apis/apps/v1/namespaces/default/deployments/test-deployment-success", nil)
				Expect(err).NotTo(HaveOccurred())
			}()
		})

		It("创建失败 - 字段有误", func() {
			deploymentYAML := map[string]interface{}{
				"apiVersion": "apps/v1",
				"kind":       "Deployment",
				"metadata": map[string]interface{}{
					"name":      "test-deployment-invalid-replicas",
					"namespace": "default",
				},
				"spec": map[string]interface{}{
					"replicas": -1,
					"selector": map[string]interface{}{
						"matchLabels": map[string]interface{}{
							"app": "test",
						},
					},
					"template": map[string]interface{}{
						"metadata": map[string]interface{}{
							"labels": map[string]interface{}{
								"app": "test",
							},
						},
						"spec": map[string]interface{}{
							"containers": []map[string]interface{}{
								{
									"name":  "test-container",
									"image": "nginx:latest",
								},
							},
						},
					},
				},
			}

			resp, err := makeRequest("POST", "/apis/apps/v1/namespaces/default/deployments", deploymentYAML)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusUnprocessableEntity), string(body))
		})

		It("创建失败 - 已存在同名", func() {
			resp1, err := createDeployment("test-existing-deployment", "default")
			Expect(err).NotTo(HaveOccurred())
			defer resp1.Body.Close()
			Expect(resp1.StatusCode).To(Equal(http.StatusCreated))

			defer func() {
				_, err := makeRequest("DELETE", "/apis/apps/v1/namespaces/default/deployments/test-existing-deployment", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			resp2, err := createDeployment("test-existing-deployment", "default")
			Expect(err).NotTo(HaveOccurred())
			defer resp2.Body.Close()
			body, err := io.ReadAll(resp2.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp2.StatusCode).To(Equal(http.StatusConflict), string(body))
		})

		It("创建失败 - Namespace 不存在", func() {
			resp, err := createDeployment("test-deployment-nonexistent-ns", "nonexistent-namespace")
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusNotFound), string(body))
		})
	})

	Context("Deployment 查询", func() {
		It("查询成功", func() {
			_, err := createDeployment("test-deployment-query", "default")
			Expect(err).NotTo(HaveOccurred())

			defer func() {
				_, err := makeRequest("DELETE", "/apis/apps/v1/namespaces/default/deployments/test-deployment-query", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			resp, err := makeRequest("GET", "/apis/apps/v1/namespaces/default/deployments/test-deployment-query", nil)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusOK), string(body))

			var deployment map[string]interface{}
			err = json.Unmarshal(body, &deployment)
			Expect(err).NotTo(HaveOccurred())
			Expect(deployment["metadata"].(map[string]interface{})["name"]).To(Equal("test-deployment-query"))
		})

		It("修改成功", func() {
			_, err := createDeployment("test-deployment-update", "default")
			Expect(err).NotTo(HaveOccurred())

			defer func() {
				_, err := makeRequest("DELETE", "/apis/apps/v1/namespaces/default/deployments/test-deployment-update", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			patch := createPatch("test-deployment-update", "default")
			resp, err := makeRequest("PATCH", "/apis/apps/v1/namespaces/default/deployments/test-deployment-update", patch)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusOK), string(body))
		})

		It("修改失败 - 字段有误", func() {
			_, err := createDeployment("test-deployment-field-error", "default")
			Expect(err).NotTo(HaveOccurred())

			defer func() {
				_, err := makeRequest("DELETE", "/apis/apps/v1/namespaces/default/deployments/test-deployment-field-error", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			patch := createMalformedPatch("test-deployment-field-error", "default")
			resp, err := makeRequest("PATCH", "/apis/apps/v1/namespaces/default/deployments/test-deployment-field-error", patch)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusUnprocessableEntity), string(body))
		})

		It("修改同一命名空间其他同类资源", func() {
			_, err := createDeployment("test-deployment-other-resource", "default")
			Expect(err).NotTo(HaveOccurred())
			_, err = createDeployment("other-deployment", "default")
			Expect(err).NotTo(HaveOccurred())

			defer func() {
				_, err := makeRequest("DELETE", "/apis/apps/v1/namespaces/default/deployments/test-deployment-other-resource", nil)
				Expect(err).NotTo(HaveOccurred())
				_, err = makeRequest("DELETE", "/apis/apps/v1/namespaces/default/deployments/other-deployment", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			patch := createPatch("other-deployment", "default")
			resp, err := makeRequest("PATCH", "/apis/apps/v1/namespaces/default/deployments/test-deployment-other-resource", patch)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusBadRequest), string(body))
		})

		It("修改不存在的Namespace", func() {
			_, err := createDeployment("test-deployment-nonexistent-ns-update", "default")
			Expect(err).NotTo(HaveOccurred())

			defer func() {
				_, err := makeRequest("DELETE", "/apis/apps/v1/namespaces/default/deployments/test-deployment-nonexistent-ns-update", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			patch := createPatch("test-deployment-nonexistent-ns-update", "nonexistent-namespace")
			resp, err := makeRequest("PATCH", "/apis/apps/v1/namespaces/default/deployments/test-deployment-nonexistent-ns-update", patch)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusBadRequest), string(body))
		})

		It("删除成功", func() {
			_, err := createDeployment("test-deployment-delete", "default")
			Expect(err).NotTo(HaveOccurred())

			resp, err := makeRequest("DELETE", "/apis/apps/v1/namespaces/default/deployments/test-deployment-delete", nil)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusOK), string(body))
		})
	})
})

var _ = Describe("StatefulSet 资源接口测试", Label("with-workload-cluster"), func() {
	Context("StatefulSet 创建", func() {
		It("创建成功", func() {
			resp, err := createStatefulSet("test-statefulset-success", "default")
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusCreated), string(body))

			defer func() {
				_, err := makeRequest("DELETE", "/apis/apps/v1/namespaces/default/statefulsets/test-statefulset-success", nil)
				Expect(err).NotTo(HaveOccurred())
			}()
		})

		It("创建失败 - 字段有误", func() {
			statefulSetYAML := map[string]interface{}{
				"apiVersion": "apps/v1",
				"kind":       "StatefulSet",
				"metadata": map[string]interface{}{
					"name":      "test-statefulset-invalid-service",
					"namespace": "default",
				},
				"spec": map[string]interface{}{
					"replicas": -1,
					"selector": map[string]interface{}{
						"matchLabels": map[string]interface{}{
							"app": "test",
						},
					},
					"template": map[string]interface{}{
						"metadata": map[string]interface{}{
							"labels": map[string]interface{}{
								"app": "test",
							},
						},
						"spec": map[string]interface{}{
							"containers": []map[string]interface{}{
								{
									"name":  "test-container",
									"image": "nginx:latest",
								},
							},
						},
					},
				},
			}

			resp, err := makeRequest("POST", "/apis/apps/v1/namespaces/default/statefulsets", statefulSetYAML)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusUnprocessableEntity), string(body))
		})

		It("创建失败 - 已存在同名", func() {
			resp1, err := createStatefulSet("test-existing-statefulset", "default")
			Expect(err).NotTo(HaveOccurred())
			defer resp1.Body.Close()
			Expect(resp1.StatusCode).To(Equal(http.StatusCreated))

			defer func() {
				_, err := makeRequest("DELETE", "/apis/apps/v1/namespaces/default/statefulsets/test-existing-statefulset", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			resp2, err := createStatefulSet("test-existing-statefulset", "default")
			Expect(err).NotTo(HaveOccurred())
			defer resp2.Body.Close()
			body, err := io.ReadAll(resp2.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp2.StatusCode).To(Equal(http.StatusConflict), string(body))
		})

		It("创建失败 - Namespace 不存在", func() {
			resp, err := createStatefulSet("test-statefulset-nonexistent-ns", "nonexistent-namespace")
			defer func() {
				_, err := makeRequest("DELETE", "/api/v1/namespaces/default/pods/test-existing-pod", nil)
				Expect(err).NotTo(HaveOccurred())
			}()
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusNotFound), string(body))
		})
	})

	Context("StatefulSet 查询", func() {
		It("查询成功", func() {
			_, err := createStatefulSet("test-statefulset-query", "default")
			Expect(err).NotTo(HaveOccurred())

			defer func() {
				_, err := makeRequest("DELETE", "/apis/apps/v1/namespaces/default/statefulsets/test-statefulset-query", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			resp, err := makeRequest("GET", "/apis/apps/v1/namespaces/default/statefulsets/test-statefulset-query", nil)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusOK), string(body))

			var statefulSet map[string]interface{}
			err = json.Unmarshal(body, &statefulSet)
			Expect(err).NotTo(HaveOccurred())
			Expect(statefulSet["metadata"].(map[string]interface{})["name"]).To(Equal("test-statefulset-query"))
		})
	})

	Context("StatefulSet 修改", func() {
		It("修改成功", func() {
			_, err := createStatefulSet("test-statefulset-update-success", "default")
			Expect(err).NotTo(HaveOccurred())

			defer func() {
				_, err := makeRequest("DELETE", "/apis/apps/v1/namespaces/default/statefulsets/test-statefulset-update-success", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			patch := createPatch("test-statefulset-update-success", "default")
			resp, err := makeRequest("PATCH", "/apis/apps/v1/namespaces/default/statefulsets/test-statefulset-update-success", patch)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusOK), string(body))
		})

		It("修改失败 - 字段有误", func() {
			_, err := createStatefulSet("test-statefulset-field-error", "default")
			Expect(err).NotTo(HaveOccurred())

			defer func() {
				_, err := makeRequest("DELETE", "/apis/apps/v1/namespaces/default/statefulsets/test-statefulset-field-error", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			patch := createMalformedPatch("test-statefulset-field-error", "default")
			resp, err := makeRequest("PATCH", "/apis/apps/v1/namespaces/default/statefulsets/test-statefulset-field-error", patch)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusUnprocessableEntity), string(body))
		})

		It("修改同一命名空间其他同类资源", func() {
			_, err := createStatefulSet("test-statefulset-other-resource", "default")
			Expect(err).NotTo(HaveOccurred())
			_, err = createStatefulSet("other-statefulset", "default")
			Expect(err).NotTo(HaveOccurred())

			defer func() {
				_, err := makeRequest("DELETE", "/apis/apps/v1/namespaces/default/statefulsets/test-statefulset-other-resource", nil)
				Expect(err).NotTo(HaveOccurred())
				_, err = makeRequest("DELETE", "/apis/apps/v1/namespaces/default/statefulsets/other-statefulset", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			patch := createPatch("other-statefulset", "default")
			resp, err := makeRequest("PATCH", "/apis/apps/v1/namespaces/default/statefulsets/test-statefulset-other-resource", patch)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusBadRequest), string(body))
		})

		It("修改不存在的Namespace", func() {
			_, err := createStatefulSet("test-statefulset-nonexistent-ns-update", "default")
			Expect(err).NotTo(HaveOccurred())

			defer func() {
				_, err := makeRequest("DELETE", "/apis/apps/v1/namespaces/default/statefulsets/test-statefulset-nonexistent-ns-update", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			patch := createPatch("test-statefulset-nonexistent-ns-update", "nonexistent-namespace")
			resp, err := makeRequest("PATCH", "/apis/apps/v1/namespaces/default/statefulsets/test-statefulset-nonexistent-ns-update", patch)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusBadRequest), string(body))
		})
	})

	Context("StatefulSet 删除", func() {
		It("删除成功", func() {
			_, err := createStatefulSet("test-statefulset-delete", "default")
			Expect(err).NotTo(HaveOccurred())

			defer func() {
				_, err := makeRequest("DELETE", "/apis/apps/v1/namespaces/default/statefulsets/test-statefulset-delete", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			resp, err := makeRequest("DELETE", "/apis/apps/v1/namespaces/default/statefulsets/test-statefulset-delete", nil)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusOK), string(body))
		})
	})
})

var _ = Describe("DaemonSet 资源接口测试", Label("with-workload-cluster"), func() {
	Context("DaemonSet 创建", func() {
		It("创建成功", func() {
			resp, err := createDaemonSet("test-daemonset-success", "default")
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusCreated), string(body))

			defer func() {
				_, err := makeRequest("DELETE", "/apis/apps/v1/namespaces/default/daemonsets/test-daemonset-success", nil)
				Expect(err).NotTo(HaveOccurred())
			}()
		})

		It("创建失败 - 字段有误", func() {
			daemonSetYAML := map[string]interface{}{
				"apiVersion": "apps/v1",
				"kind":       "DaemonSet",
				"metadata": map[string]interface{}{
					"name":      "test-daemonset-invalid-selector",
					"namespace": "default",
				},
				"spec": map[string]interface{}{
					"selector": map[string]interface{}{
						"matchLabels": map[string]interface{}{
							"app": "test",
						},
					},
					"template": map[string]interface{}{
						"metadata": map[string]interface{}{
							"labels": map[string]interface{}{},
						},
						"spec": map[string]interface{}{
							"containers": []map[string]interface{}{
								{
									"name":  "test-container",
									"image": "nginx:latest",
								},
							},
						},
					},
				},
			}

			resp, err := makeRequest("POST", "/apis/apps/v1/namespaces/default/daemonsets", daemonSetYAML)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusUnprocessableEntity), string(body))
		})

		It("创建失败 - 已存在同名", func() {
			resp1, err := createDaemonSet("test-existing-daemonset", "default")
			Expect(err).NotTo(HaveOccurred())
			defer resp1.Body.Close()
			Expect(resp1.StatusCode).To(Equal(http.StatusCreated))

			defer func() {
				_, err := makeRequest("DELETE", "/apis/apps/v1/namespaces/default/daemonsets/test-existing-daemonset", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			resp2, err := createDaemonSet("test-existing-daemonset", "default")
			Expect(err).NotTo(HaveOccurred())
			defer resp2.Body.Close()
			body, err := io.ReadAll(resp2.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp2.StatusCode).To(Equal(http.StatusConflict), string(body))
		})

		It("创建失败 - Namespace 不存在", func() {
			resp, err := createDaemonSet("test-daemonset-nonexistent-ns", "nonexistent-namespace")
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusNotFound), string(body))
		})
	})

	Context("DaemonSet 查询", func() {
		It("查询成功", func() {
			_, err := createDaemonSet("test-daemonset-query", "default")
			Expect(err).NotTo(HaveOccurred())

			defer func() {
				_, err := makeRequest("DELETE", "/apis/apps/v1/namespaces/default/daemonsets/test-daemonset-query", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			resp, err := makeRequest("GET", "/apis/apps/v1/namespaces/default/daemonsets/test-daemonset-query", nil)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusOK), string(body))

			var daemonSet map[string]interface{}
			err = json.Unmarshal(body, &daemonSet)
			Expect(err).NotTo(HaveOccurred())
			Expect(daemonSet["metadata"].(map[string]interface{})["name"]).To(Equal("test-daemonset-query"))
		})
	})

	Context("DaemonSet 修改", func() {
		It("修改成功", func() {
			_, err := createDaemonSet("test-daemonset-update-success", "default")
			Expect(err).NotTo(HaveOccurred())

			defer func() {
				_, err := makeRequest("DELETE", "/apis/apps/v1/namespaces/default/daemonsets/test-daemonset-update-success", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			patch := createPatch("test-daemonset-update-success", "default")
			resp, err := makeRequest("PATCH", "/apis/apps/v1/namespaces/default/daemonsets/test-daemonset-update-success", patch)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusOK), string(body))
		})

		It("修改失败 - 字段有误", func() {
			_, err := createDaemonSet("test-daemonset-field-error", "default")
			Expect(err).NotTo(HaveOccurred())

			defer func() {
				_, err := makeRequest("DELETE", "/apis/apps/v1/namespaces/default/daemonsets/test-daemonset-field-error", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			patch := createMalformedPatch("test-daemonset-field-error", "default")
			resp, err := makeRequest("PATCH", "/apis/apps/v1/namespaces/default/daemonsets/test-daemonset-field-error", patch)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusUnprocessableEntity), string(body))
		})

		It("修改同一命名空间其他同类资源", func() {
			_, err := createDaemonSet("test-daemonset-other-resource", "default")
			Expect(err).NotTo(HaveOccurred())
			_, err = createDaemonSet("other-daemonset", "default")
			Expect(err).NotTo(HaveOccurred())

			defer func() {
				_, err := makeRequest("DELETE", "/apis/apps/v1/namespaces/default/daemonsets/test-daemonset-other-resource", nil)
				Expect(err).NotTo(HaveOccurred())
				_, err = makeRequest("DELETE", "/apis/apps/v1/namespaces/default/daemonsets/other-daemonset", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			patch := createPatch("other-daemonset", "default")
			resp, err := makeRequest("PATCH", "/apis/apps/v1/namespaces/default/daemonsets/test-daemonset-other-resource", patch)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusBadRequest), string(body))
		})

		It("修改不存在的Namespace", func() {
			_, err := createDaemonSet("test-daemonset-nonexistent-ns", "default")
			Expect(err).NotTo(HaveOccurred())

			defer func() {
				_, err := makeRequest("DELETE", "/apis/apps/v1/namespaces/default/daemonsets/test-daemonset-nonexistent-ns", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			patch := createPatch("test-daemonset-nonexistent-ns", "nonexistent-namespace")
			resp, err := makeRequest("PATCH", "/apis/apps/v1/namespaces/default/daemonsets/test-daemonset-nonexistent-ns", patch)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusBadRequest), string(body))
		})
	})

	Context("DaemonSet 删除", func() {
		It("删除成功", func() {
			_, err := createDaemonSet("test-daemonset-delete", "default")
			Expect(err).NotTo(HaveOccurred())

			resp, err := makeRequest("DELETE", "/apis/apps/v1/namespaces/default/daemonsets/test-daemonset-delete", nil)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusOK), string(body))
		})
	})
})

var _ = Describe("Job 资源接口测试", Label("with-workload-cluster"), func() {
	Context("Job 创建", func() {
		It("创建成功", func() {
			resp, err := createJob("test-job-success", "default")
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusCreated), string(body))

			defer func() {
				_, err := makeRequest("DELETE", "/apis/batch/v1/namespaces/default/jobs/test-job-success", map[string]interface{}{
					"propagationPolicy": "Foreground",
				})
				Expect(err).NotTo(HaveOccurred())
			}()
		})

		It("创建失败 - 字段有误", func() {
			jobYAML := map[string]interface{}{
				"apiVersion": "batch/v1",
				"kind":       "Job",
				"metadata": map[string]interface{}{
					"name":      "test-job-invalid-field",
					"namespace": "default",
				},
				"spec": map[string]interface{}{
					"template": map[string]interface{}{
						"spec": map[string]interface{}{
							"containers": []map[string]interface{}{
								{
									"name":    "test-container",
									"image":   "busybox:latest",
									"command": []string{"echo", "hello"},
								},
							},
							"restartPolicy": "InvalidPolicy",
						},
					},
				},
			}

			resp, err := makeRequest("POST", "/apis/batch/v1/namespaces/default/jobs", jobYAML)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusUnprocessableEntity), string(body))
		})

		It("创建失败 - 已存在同名", func() {
			resp1, err := createJob("test-existing-job", "default")
			Expect(err).NotTo(HaveOccurred())
			defer resp1.Body.Close()
			Expect(resp1.StatusCode).To(Equal(http.StatusCreated))

			defer func() {
				_, err := makeRequest("DELETE", "/apis/batch/v1/namespaces/default/jobs/test-existing-job", map[string]interface{}{
					"propagationPolicy": "Foreground",
				})
				Expect(err).NotTo(HaveOccurred())
			}()

			resp2, err := createJob("test-existing-job", "default")
			Expect(err).NotTo(HaveOccurred())
			defer resp2.Body.Close()
			body, err := io.ReadAll(resp2.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp2.StatusCode).To(Equal(http.StatusConflict), string(body))
		})

		It("创建失败 - Namespace 不存在", func() {
			resp, err := createJob("test-job-nonexistent-ns", "nonexistent-namespace")
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusNotFound), string(body))
		})
	})

	Context("Job 查询", func() {
		It("查询成功", func() {
			_, err := createJob("test-job-query", "default")
			Expect(err).NotTo(HaveOccurred())

			defer func() {
				_, err := makeRequest("DELETE", "/apis/batch/v1/namespaces/default/jobs/test-job-query", map[string]interface{}{
					"propagationPolicy": "Foreground",
				})
				Expect(err).NotTo(HaveOccurred())
			}()

			resp, err := makeRequest("GET", "/apis/batch/v1/namespaces/default/jobs/test-job-query", nil)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusOK), string(body))

			var job map[string]interface{}
			err = json.Unmarshal(body, &job)
			Expect(err).NotTo(HaveOccurred())
			Expect(job["metadata"].(map[string]interface{})["name"]).To(Equal("test-job-query"))
		})
	})

	Context("Job 修改", func() {
		It("修改成功", func() {
			_, err := createJob("test-job-update-success", "default")
			Expect(err).NotTo(HaveOccurred())

			defer func() {
				_, err := makeRequest("DELETE", "/apis/batch/v1/namespaces/default/jobs/test-job-update-success", map[string]interface{}{
					"propagationPolicy": "Foreground",
				})
				Expect(err).NotTo(HaveOccurred())
			}()

			patch := createPatch("test-job-update-success", "default")
			resp, err := makeRequest("PATCH", "/apis/batch/v1/namespaces/default/jobs/test-job-update-success", patch)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusOK), string(body))
		})

		It("修改失败 - 字段有误", func() {
			_, err := createJob("test-job-field-error", "default")
			Expect(err).NotTo(HaveOccurred())

			defer func() {
				_, err := makeRequest("DELETE", "/apis/batch/v1/namespaces/default/jobs/test-job-field-error", map[string]interface{}{
					"propagationPolicy": "Foreground",
				})
				Expect(err).NotTo(HaveOccurred())
			}()

			patch := createMalformedPatch("test-job-field-error", "default")
			resp, err := makeRequest("PATCH", "/apis/batch/v1/namespaces/default/jobs/test-job-field-error", patch)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusUnprocessableEntity), string(body))
		})

		It("修改同一命名空间其他同类资源", func() {
			_, err := createJob("test-job-other-resource", "default")
			Expect(err).NotTo(HaveOccurred())
			_, err = createJob("other-job", "default")
			Expect(err).NotTo(HaveOccurred())

			defer func() {
				_, err := makeRequest("DELETE", "/apis/batch/v1/namespaces/default/jobs/test-job-other-resource", map[string]interface{}{
					"propagationPolicy": "Foreground",
				})
				Expect(err).NotTo(HaveOccurred())
				_, err = makeRequest("DELETE", "/apis/batch/v1/namespaces/default/jobs/other-job", map[string]interface{}{
					"propagationPolicy": "Foreground",
				})
				Expect(err).NotTo(HaveOccurred())
			}()

			patch := createPatch("other-job", "default")
			resp, err := makeRequest("PATCH", "/apis/batch/v1/namespaces/default/jobs/test-job-other-resource", patch)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusBadRequest), string(body))
		})

		It("修改不存在的Namespace", func() {
			_, err := createJob("test-job-nonexistent-ns", "default")
			Expect(err).NotTo(HaveOccurred())

			defer func() {
				_, err := makeRequest("DELETE", "/apis/batch/v1/namespaces/default/jobs/test-job-nonexistent-ns", map[string]interface{}{
					"propagationPolicy": "Foreground",
				})
				Expect(err).NotTo(HaveOccurred())
			}()

			patch := createPatch("test-job-nonexistent-ns", "nonexistent-namespace")
			resp, err := makeRequest("PATCH", "/apis/batch/v1/namespaces/default/jobs/test-job-nonexistent-ns", patch)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusBadRequest), string(body))
		})
	})

	Context("Job 删除", func() {
		It("删除成功", func() {
			_, err := createJob("test-job-delete", "default")
			Expect(err).NotTo(HaveOccurred())

			resp, err := makeRequest("DELETE", "/apis/batch/v1/namespaces/default/jobs/test-job-delete", map[string]interface{}{
				"propagationPolicy": "Foreground",
			})
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusOK), string(body))
		})
	})
})

var _ = Describe("CronJob 资源接口测试", Label("with-workload-cluster"), func() {
	Context("CronJob 创建", func() {
		It("创建成功", func() {
			resp, err := createCronJob("test-cronjob-success", "default")
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusCreated), string(body))

			defer func() {
				_, err := makeRequest("DELETE", "/apis/batch/v1/namespaces/default/cronjobs/test-cronjob-success", nil)
				Expect(err).NotTo(HaveOccurred())
			}()
		})

		It("创建失败 - 字段有误", func() {
			cronJobYAML := map[string]interface{}{
				"apiVersion": "batch/v1",
				"kind":       "CronJob",
				"metadata": map[string]interface{}{
					"name":      "test-cronjob-invalid-field",
					"namespace": "default",
				},
				"spec": map[string]interface{}{
					"jobTemplate": map[string]interface{}{
						"spec": map[string]interface{}{
							"template": map[string]interface{}{
								"spec": map[string]interface{}{
									"containers": []map[string]interface{}{
										{
											"name":    "test-container",
											"image":   "busybox:latest",
											"command": []string{"echo", "hello"},
										},
									},
									"restartPolicy": "InvalidPolicy",
								},
							},
						},
					},
				},
			}

			resp, err := makeRequest("POST", "/apis/batch/v1/namespaces/default/cronjobs", cronJobYAML)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusUnprocessableEntity), string(body))
		})

		It("创建失败 - 已存在同名", func() {
			resp1, err := createCronJob("test-existing-cronjob", "default")
			Expect(err).NotTo(HaveOccurred())
			defer resp1.Body.Close()
			Expect(resp1.StatusCode).To(Equal(http.StatusCreated))

			defer func() {
				_, err := makeRequest("DELETE", "/apis/batch/v1/namespaces/default/cronjobs/test-existing-cronjob", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			resp2, err := createCronJob("test-existing-cronjob", "default")
			Expect(err).NotTo(HaveOccurred())
			defer resp2.Body.Close()
			body, err := io.ReadAll(resp2.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp2.StatusCode).To(Equal(http.StatusConflict), string(body))
		})

		It("创建失败 - Namespace 不存在", func() {
			resp, err := createCronJob("test-cronjob-nonexistent-ns", "nonexistent-namespace")
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			Expect(resp.StatusCode).To(Equal(http.StatusNotFound))
		})
	})

	Context("CronJob 查询", func() {
		It("查询成功", func() {
			_, err := createCronJob("test-cronjob-query", "default")
			Expect(err).NotTo(HaveOccurred())

			defer func() {
				_, err := makeRequest("DELETE", "/apis/batch/v1/namespaces/default/cronjobs/test-cronjob-query", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			resp, err := makeRequest("GET", "/apis/batch/v1/namespaces/default/cronjobs/test-cronjob-query", nil)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusOK), string(body))

			var cronJob map[string]interface{}
			err = json.Unmarshal(body, &cronJob)
			Expect(err).NotTo(HaveOccurred())
			Expect(cronJob["metadata"].(map[string]interface{})["name"]).To(Equal("test-cronjob-query"))
		})
	})

	Context("CronJob 修改", func() {
		It("修改成功", func() {
			_, err := createCronJob("test-cronjob-update-success", "default")
			Expect(err).NotTo(HaveOccurred())

			defer func() {
				_, err := makeRequest("DELETE", "/apis/batch/v1/namespaces/default/cronjobs/test-cronjob-update-success", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			patch := createPatch("test-cronjob-update-success", "default")
			resp, err := makeRequest("PATCH", "/apis/batch/v1/namespaces/default/cronjobs/test-cronjob-update-success", patch)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusOK), string(body))
		})

		It("修改失败 - 字段有误", func() {
			_, err := createCronJob("test-cronjob-field-error", "default")
			Expect(err).NotTo(HaveOccurred())

			defer func() {
				_, err := makeRequest("DELETE", "/apis/batch/v1/namespaces/default/cronjobs/test-cronjob-field-error", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			patch := createMalformedPatch("test-cronjob-field-error", "default")
			resp, err := makeRequest("PATCH", "/apis/batch/v1/namespaces/default/cronjobs/test-cronjob-field-error", patch)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusUnprocessableEntity), string(body))
		})

		It("修改同一命名空间其他同类资源", func() {
			_, err := createCronJob("test-cronjob-other-resource", "default")
			Expect(err).NotTo(HaveOccurred())
			_, err = createCronJob("other-cronjob", "default")
			Expect(err).NotTo(HaveOccurred())

			defer func() {
				_, err := makeRequest("DELETE", "/apis/batch/v1/namespaces/default/cronjobs/test-cronjob-other-resource", map[string]interface{}{
					"propagationPolicy": "Foreground",
				})
				Expect(err).NotTo(HaveOccurred())
				_, err = makeRequest("DELETE", "/apis/batch/v1/namespaces/default/cronjobs/other-cronjob", map[string]interface{}{
					"propagationPolicy": "Foreground",
				})
				Expect(err).NotTo(HaveOccurred())
			}()

			patch := createPatch("other-cronjob", "default")
			resp, err := makeRequest("PATCH", "/apis/batch/v1/namespaces/default/cronjobs/test-cronjob-other-resource", patch)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusBadRequest), string(body))
		})

		It("修改不存在的Namespace", func() {
			_, err := createCronJob("test-cronjob-nonexistent-ns", "default")
			Expect(err).NotTo(HaveOccurred())

			defer func() {
				_, err := makeRequest("DELETE", "/apis/batch/v1/namespaces/default/cronjobs/test-cronjob-nonexistent-ns", map[string]interface{}{
					"propagationPolicy": "Foreground",
				})
				Expect(err).NotTo(HaveOccurred())
			}()

			patch := createPatch("test-cronjob-nonexistent-ns", "nonexistent-namespace")
			resp, err := makeRequest("PATCH", "/apis/batch/v1/namespaces/default/cronjobs/test-cronjob-nonexistent-ns", patch)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusBadRequest), string(body))
		})
	})

	Context("CronJob 删除", func() {
		It("删除成功", func() {
			_, err := createCronJob("test-cronjob-delete", "default")
			Expect(err).NotTo(HaveOccurred())

			resp, err := makeRequest("DELETE", "/apis/batch/v1/namespaces/default/cronjobs/test-cronjob-delete", nil)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusOK), string(body))
		})
	})
})
