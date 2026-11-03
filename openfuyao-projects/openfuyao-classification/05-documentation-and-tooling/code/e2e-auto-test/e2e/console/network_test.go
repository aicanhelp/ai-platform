package console

import (
	"encoding/json"
	"io"
	"net/http"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

var _ = Describe("Service 资源接口测试", Label("with-workload-cluster"), func() {
	Context("Service 创建", func() {
		It("创建成功", func() {
			resp, err := createService("test-service-success", "default")
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusCreated), string(body))

			defer func() {
				_, err := makeRequest("DELETE", "/api/v1/namespaces/default/services/test-service-success", nil)
				Expect(err).NotTo(HaveOccurred())
			}()
		})

		It("创建失败 - 字段有误", func() {
			serviceYAML := map[string]interface{}{
				"apiVersion": "v1",
				"kind":       "Service",
				"metadata": map[string]interface{}{
					"name":      "test-service-invalid-port",
					"namespace": "default",
				},
				"spec": map[string]interface{}{
					"type": "ClusterIP",
					"ports": []map[string]interface{}{
						{
							"port":       -80, // 负数端口
							"targetPort": 8080,
							"protocol":   "TCP",
						},
					},
					"selector": map[string]interface{}{
						"app": "test",
					},
				},
			}

			resp, err := makeRequest("POST", "/api/v1/namespaces/default/services", serviceYAML)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusUnprocessableEntity), string(body))
		})

		It("创建失败 - 已存在同名", func() {
			resp1, err := createService("test-existing-service", "default")
			Expect(err).NotTo(HaveOccurred())
			defer resp1.Body.Close()
			Expect(resp1.StatusCode).To(Equal(http.StatusCreated))

			defer func() {
				_, err := makeRequest("DELETE", "/api/v1/namespaces/default/services/test-existing-service", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			resp2, err := createService("test-existing-service", "default")
			Expect(err).NotTo(HaveOccurred())
			defer resp2.Body.Close()
			body, err := io.ReadAll(resp2.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp2.StatusCode).To(Equal(http.StatusConflict), string(body))
		})

		It("创建失败 - Namespace 不存在", func() {
			resp, err := createService("test-service-nonexistent-ns", "nonexistent-namespace")
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusNotFound), string(body))
		})
	})

	Context("Service 查询", func() {
		It("查询成功", func() {
			_, err := createService("test-service-query", "default")
			Expect(err).NotTo(HaveOccurred())

			defer func() {
				_, err := makeRequest("DELETE", "/api/v1/namespaces/default/services/test-service-query", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			resp, err := makeRequest("GET", "/api/v1/namespaces/default/services/test-service-query", nil)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusOK), string(body))

			var service map[string]interface{}
			err = json.Unmarshal(body, &service)
			Expect(err).NotTo(HaveOccurred())
			Expect(service["metadata"].(map[string]interface{})["name"]).To(Equal("test-service-query"))
		})
	})

	Context("Service 修改", func() {
		It("修改成功", func() {
			_, err := createService("test-service-update-success", "default")
			Expect(err).NotTo(HaveOccurred())

			defer func() {
				_, err := makeRequest("DELETE", "/api/v1/namespaces/default/services/test-service-update-success", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			patch := createPatch("test-service-update-success", "default")
			resp, err := makeRequest("PATCH", "/api/v1/namespaces/default/services/test-service-update-success", patch)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusOK), string(body))
		})

		It("修改失败 - 字段有误", func() {
			_, err := createService("test-service-field-error", "default")
			Expect(err).NotTo(HaveOccurred())

			defer func() {
				_, err := makeRequest("DELETE", "/api/v1/namespaces/default/services/test-service-field-error", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			patch := createMalformedPatch("test-service-field-error", "default")
			resp, err := makeRequest("PATCH", "/api/v1/namespaces/default/services/test-service-field-error", patch)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusUnprocessableEntity), string(body))
		})

		It("修改同一命名空间其他同类资源", func() {
			_, err := createService("test-service-other-resource", "default")
			Expect(err).NotTo(HaveOccurred())
			_, err = createService("other-service", "default")
			Expect(err).NotTo(HaveOccurred())

			defer func() {
				_, err := makeRequest("DELETE", "/api/v1/namespaces/default/services/test-service-other-resource", nil)
				Expect(err).NotTo(HaveOccurred())
				_, err = makeRequest("DELETE", "/api/v1/namespaces/default/services/other-service", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			patch := createPatch("other-service", "default")
			resp, err := makeRequest("PATCH", "/api/v1/namespaces/default/services/test-service-other-resource", patch)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusBadRequest), string(body))
		})

		It("修改不存在的Namespace", func() {
			_, err := createService("test-service-nonexistent-ns", "default")
			Expect(err).NotTo(HaveOccurred())

			defer func() {
				_, err := makeRequest("DELETE", "/api/v1/namespaces/default/services/test-service-nonexistent-ns", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			patch := createPatch("test-service-nonexistent-ns", "nonexistent-namespace")
			resp, err := makeRequest("PATCH", "/api/v1/namespaces/default/services/test-service-nonexistent-ns", patch)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusBadRequest), string(body))
		})
	})

	Context("Service 删除", func() {
		It("删除成功", func() {
			_, err := createService("test-service-delete", "default")
			Expect(err).NotTo(HaveOccurred())

			resp, err := makeRequest("DELETE", "/api/v1/namespaces/default/services/test-service-delete", nil)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusOK), string(body))
		})
	})
})

var _ = Describe("Ingress 资源接口测试", Label("with-workload-cluster"), func() {
	Context("Ingress 创建", func() {
		It("创建成功", func() {
			resp, err := createIngress("test-ingress-success", "default")
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusCreated), string(body))

			defer func() {
				_, err := makeRequest("DELETE", "/apis/networking.k8s.io/v1/namespaces/default/ingresses/test-ingress-success", nil)
				Expect(err).NotTo(HaveOccurred())
			}()
		})

		It("创建失败 - 字段有误", func() {
			ingressYAML := map[string]interface{}{
				"apiVersion": "networking.k8s.io/v1",
				"kind":       "Ingress",
				"metadata": map[string]interface{}{
					"name":      "test-ingress-invalid-host",
					"namespace": "default",
				},
				"spec": map[string]interface{}{
					"rules": []map[string]interface{}{
						{
							"host": "invalid..host.com", // 无效的 host 格式
							"http": map[string]interface{}{
								"paths": []map[string]interface{}{
									{
										"path":     "/",
										"pathType": "Prefix",
										"backend": map[string]interface{}{
											"service": map[string]interface{}{
												"name": "test-service",
												"port": map[string]interface{}{
													"number": 80,
												},
											},
										},
									},
								},
							},
						},
					},
				},
			}

			resp, err := makeRequest("POST", "/apis/networking.k8s.io/v1/namespaces/default/ingresses", ingressYAML)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusUnprocessableEntity), string(body))
		})

		It("创建失败 - 已存在同名", func() {
			resp1, err := createIngress("test-existing-ingress", "default")
			Expect(err).NotTo(HaveOccurred())
			defer resp1.Body.Close()
			Expect(resp1.StatusCode).To(Equal(http.StatusCreated))

			defer func() {
				_, err := makeRequest("DELETE", "/apis/networking.k8s.io/v1/namespaces/default/ingresses/test-existing-ingress", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			resp, err := createIngress("test-existing-ingress", "default")
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusConflict), string(body))
		})

		It("创建失败 - Namespace 不存在", func() {
			resp, err := createIngress("test-ingress-nonexistent-ns", "nonexistent-namespace")
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusNotFound), string(body))
		})
	})

	Context("Ingress 查询", func() {
		It("查询成功", func() {
			_, err := createIngress("test-ingress-query", "default")
			Expect(err).NotTo(HaveOccurred())

			defer func() {
				_, err := makeRequest("DELETE", "/apis/networking.k8s.io/v1/namespaces/default/ingresses/test-ingress-query", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			resp, err := makeRequest("GET", "/apis/networking.k8s.io/v1/namespaces/default/ingresses/test-ingress-query", nil)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusOK), string(body))
		})
	})

	Context("Ingress 修改", func() {
		It("修改成功", func() {
			_, err := createIngress("test-ingress-update-success", "default")
			Expect(err).NotTo(HaveOccurred())

			defer func() {
				_, err := makeRequest("DELETE", "/apis/networking.k8s.io/v1/namespaces/default/ingresses/test-ingress-update-success", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			patch := createPatch("test-ingress-update-success", "default")
			resp, err := makeRequest("PATCH", "/apis/networking.k8s.io/v1/namespaces/default/ingresses/test-ingress-update-success", patch)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusOK), string(body))
		})

		It("修改失败 - 字段有误", func() {
			_, err := createIngress("test-ingress-field-error", "default")
			Expect(err).NotTo(HaveOccurred())

			defer func() {
				_, err := makeRequest("DELETE", "/apis/networking.k8s.io/v1/namespaces/default/ingresses/test-ingress-field-error", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			patch := createMalformedPatch("test-ingress-field-error", "default")
			resp, err := makeRequest("PATCH", "/apis/networking.k8s.io/v1/namespaces/default/ingresses/test-ingress-field-error", patch)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusUnprocessableEntity), string(body))
		})

		It("修改同一命名空间其他同类资源", func() {
			_, err := createIngress("test-ingress-other-resource", "default")
			Expect(err).NotTo(HaveOccurred())
			_, err = createIngress("other-ingress", "default")
			Expect(err).NotTo(HaveOccurred())

			defer func() {
				_, err := makeRequest("DELETE", "/apis/networking.k8s.io/v1/namespaces/default/ingresses/test-ingress-other-resource", nil)
				Expect(err).NotTo(HaveOccurred())
				_, err = makeRequest("DELETE", "/apis/networking.k8s.io/v1/namespaces/default/ingresses/other-ingress", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			patch := createPatch("other-ingress", "default")
			resp, err := makeRequest("PATCH", "/apis/networking.k8s.io/v1/namespaces/default/ingresses/test-ingress-other-resource", patch)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusBadRequest), string(body))
		})

		It("修改不存在的Namespace", func() {
			_, err := createIngress("test-ingress-nonexistent-ns", "default")
			Expect(err).NotTo(HaveOccurred())

			defer func() {
				_, err := makeRequest("DELETE", "/apis/networking.k8s.io/v1/namespaces/default/ingresses/test-ingress-nonexistent-ns", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			patch := createPatch("test-ingress-nonexistent-ns", "nonexistent-namespace")
			resp, err := makeRequest("PATCH", "/apis/networking.k8s.io/v1/namespaces/default/ingresses/test-ingress-nonexistent-ns", patch)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusBadRequest), string(body))
		})
	})

	Context("Ingress 删除", func() {
		It("删除成功", func() {
			_, err := createIngress("test-ingress-delete", "default")
			Expect(err).NotTo(HaveOccurred())

			resp, err := makeRequest("DELETE", "/apis/networking.k8s.io/v1/namespaces/default/ingresses/test-ingress-delete", nil)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusOK), string(body))
		})
	})
})
