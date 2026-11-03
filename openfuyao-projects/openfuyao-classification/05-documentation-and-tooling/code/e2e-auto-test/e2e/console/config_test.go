package console

import (
	"encoding/json"
	"io"
	"net/http"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

var _ = Describe("ConfigMap 资源接口测试", Label("with-workload-cluster"), func() {
	Context("ConfigMap 创建", func() {
		It("创建成功", func() {
			resp, err := createConfigMap("test-configmap-success", "default")
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusCreated), string(body))

			defer func() {
				_, err := makeRequest("DELETE", "/api/v1/namespaces/default/configmaps/test-configmap-success", nil)
				Expect(err).NotTo(HaveOccurred())
			}()
		})

		It("创建失败 - 字段有误", func() {
			configMapYAML := map[string]interface{}{
				"apiVersion": "v1",
				"kind":       "ConfigMap",
				"metadata": map[string]interface{}{
					"name":      "test-configmap-invalid-key",
					"namespace": "default",
				},
				"data": map[string]interface{}{
					"key with spaces": "value", // 非法键名包含空格
				},
			}

			resp, err := makeRequest("POST", "/api/v1/namespaces/default/configmaps", configMapYAML)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusUnprocessableEntity), string(body))
			Expect(err).NotTo(HaveOccurred())
		})

		It("创建失败 - 已存在同名", func() {
			resp1, err := createConfigMap("test-existing-configmap", "default")
			Expect(err).NotTo(HaveOccurred())
			defer resp1.Body.Close()
			Expect(resp1.StatusCode).To(Equal(http.StatusCreated))

			defer func() {
				_, err := makeRequest("DELETE", "/api/v1/namespaces/default/configmaps/test-existing-configmap", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			resp2, err := createConfigMap("test-existing-configmap", "default")
			Expect(err).NotTo(HaveOccurred())
			defer resp2.Body.Close()
			body, err := io.ReadAll(resp2.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp2.StatusCode).To(Equal(http.StatusConflict), string(body))
		})

		It("创建失败 - Namespace 不存在", func() {
			resp, err := createConfigMap("test-configmap-nonexistent-ns", "nonexistent-namespace")
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusNotFound), string(body))
		})
	})

	Context("ConfigMap 查询", func() {
		It("查询成功", func() {
			_, err := createConfigMap("test-configmap-query", "default")
			Expect(err).NotTo(HaveOccurred())

			defer func() {
				_, err := makeRequest("DELETE", "/api/v1/namespaces/default/configmaps/test-configmap-query", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			resp, err := makeRequest("GET", "/api/v1/namespaces/default/configmaps/test-configmap-query", nil)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusOK), string(body))

			var configMap map[string]interface{}
			err = json.Unmarshal(body, &configMap)
			Expect(err).NotTo(HaveOccurred())
			Expect(configMap["metadata"].(map[string]interface{})["name"]).To(Equal("test-configmap-query"))
		})
	})

	Context("ConfigMap 修改", func() {
		It("修改成功", func() {
			_, err := createConfigMap("test-configmap-update-success", "default")
			Expect(err).NotTo(HaveOccurred())

			defer func() {
				_, err := makeRequest("DELETE", "/api/v1/namespaces/default/configmaps/test-configmap-update-success", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			patch := createPatch("test-configmap-update-success", "default")
			resp, err := makeRequest("PATCH", "/api/v1/namespaces/default/configmaps/test-configmap-update-success", patch)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusOK), string(body))
		})

		It("修改失败 - 字段有误", func() {
			_, err := createConfigMap("test-configmap-field-error", "default")
			Expect(err).NotTo(HaveOccurred())

			defer func() {
				_, err := makeRequest("DELETE", "/api/v1/namespaces/default/configmaps/test-configmap-field-error", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			patch := createMalformedPatch("test-configmap-field-error", "default")
			resp, err := makeRequest("PATCH", "/api/v1/namespaces/default/configmaps/test-configmap-field-error", patch)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusUnprocessableEntity), string(body))
		})

		It("修改同一命名空间其他同类资源", func() {
			_, err := createConfigMap("test-configmap-other-resource", "default")
			Expect(err).NotTo(HaveOccurred())
			_, err = createConfigMap("other-configmap", "default")
			Expect(err).NotTo(HaveOccurred())
			defer func() {
				_, err = makeRequest("DELETE", "/api/v1/namespaces/default/configmaps/test-configmap-other-resource", nil)
				Expect(err).NotTo(HaveOccurred())
				_, err := makeRequest("DELETE", "/api/v1/namespaces/default/configmaps/other-configmap", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			patch := createPatch("other-configmap", "default")
			resp, err := makeRequest("PATCH", "/api/v1/namespaces/default/configmaps/test-configmap-other-resource", patch)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusBadRequest), string(body))
		})

		It("修改不存在的Namespace", func() {
			_, err := createConfigMap("test-configmap-nonexistent-ns", "default")
			Expect(err).NotTo(HaveOccurred())

			defer func() {
				_, err := makeRequest("DELETE", "/api/v1/namespaces/default/configmaps/test-configmap-nonexistent-ns", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			patch := createPatch("test-configmap-nonexistent-ns", "nonexistent-namespace")
			resp, err := makeRequest("PATCH", "/api/v1/namespaces/default/configmaps/test-configmap-nonexistent-ns", patch)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusBadRequest), string(body))
		})
	})

	Context("ConfigMap 删除", func() {
		It("删除成功", func() {
			_, err := createConfigMap("test-configmap-delete", "default")
			Expect(err).NotTo(HaveOccurred())

			resp, err := makeRequest("DELETE", "/api/v1/namespaces/default/configmaps/test-configmap-delete", nil)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusOK), string(body))
		})
	})
})

var _ = Describe("Secret 资源接口测试", Label("with-workload-cluster"), func() {
	Context("Secret 创建", func() {
		It("创建成功", func() {
			resp, err := createSecret("test-secret-success", "default")
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusCreated), string(body))

			defer func() {
				_, err := makeRequest("DELETE", "/api/v1/namespaces/default/secrets/test-secret-success", nil)
				Expect(err).NotTo(HaveOccurred())
			}()
		})

		It("创建失败 - 字段有误", func() {
			secretYAML := map[string]interface{}{
				"apiVersion": "v1",
				"kind":       "Secret",
				"metadata": map[string]interface{}{
					"name":      "test-secret-invalid-data",
					"namespace": "default",
				},
				"type": "Opaque",
				"data": map[string]interface{}{
					"key with spaces": "", // 非法键名包含空格
				},
			}

			resp, err := makeRequest("POST", "/api/v1/namespaces/default/secrets", secretYAML)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusUnprocessableEntity), string(body))
		})

		It("创建失败 - 已存在同名", func() {
			resp1, err := createSecret("test-existing-secret", "default")
			Expect(err).NotTo(HaveOccurred())
			defer resp1.Body.Close()
			Expect(resp1.StatusCode).To(Equal(http.StatusCreated))

			defer func() {
				_, err := makeRequest("DELETE", "/api/v1/namespaces/default/secrets/test-existing-secret", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			resp2, err := createSecret("test-existing-secret", "default")
			Expect(err).NotTo(HaveOccurred())
			defer resp2.Body.Close()
			body, err := io.ReadAll(resp2.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp2.StatusCode).To(Equal(http.StatusConflict), string(body))
		})

		It("创建失败 - Namespace 不存在", func() {
			secretYAML := map[string]interface{}{
				"apiVersion": "v1",
				"kind":       "Secret",
				"metadata": map[string]interface{}{
					"name":      "test-secret-nonexistent-ns",
					"namespace": "nonexistent-namespace",
				},
				"type": "Opaque",
				"data": map[string]interface{}{
					"key1": "dmFsdWUx", // base64 encoded "value1"
				},
			}

			resp, err := makeRequest("POST", "/api/v1/namespaces/nonexistent-namespace/secrets", secretYAML)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusNotFound), string(body))
		})
	})

	Context("Secret 查询", func() {
		It("查询成功", func() {
			_, err := createSecret("test-secret-query", "default")
			Expect(err).NotTo(HaveOccurred())

			defer func() {
				_, err := makeRequest("DELETE", "/api/v1/namespaces/default/secrets/test-secret-query", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			resp, err := makeRequest("GET", "/api/v1/namespaces/default/secrets/test-secret-query", nil)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusOK), string(body))
		})
	})

	Context("Secret 修改", func() {
		It("修改成功", func() {
			_, err := createSecret("test-secret-update-success", "default")
			Expect(err).NotTo(HaveOccurred())

			defer func() {
				_, err := makeRequest("DELETE", "/api/v1/namespaces/default/secrets/test-secret-update-success", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			patch := createPatch("test-secret-update-success", "default")
			resp, err := makeRequest("PATCH", "/api/v1/namespaces/default/secrets/test-secret-update-success", patch)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusOK), string(body))
		})

		It("修改失败 - 字段有误", func() {
			_, err := createSecret("test-secret-field-error", "default")
			Expect(err).NotTo(HaveOccurred())

			defer func() {
				_, err := makeRequest("DELETE", "/api/v1/namespaces/default/secrets/test-secret-field-error", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			patch := createMalformedPatch("test-secret-field-error", "default")
			resp, err := makeRequest("PATCH", "/api/v1/namespaces/default/secrets/test-secret-field-error", patch)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusUnprocessableEntity), string(body))
		})

		It("修改同一命名空间其他同类资源", func() {
			_, err := createSecret("test-secret-other-resource", "default")
			Expect(err).NotTo(HaveOccurred())
			_, err = createSecret("other-secret", "default")
			Expect(err).NotTo(HaveOccurred())
			defer func() {
				_, err = makeRequest("DELETE", "/api/v1/namespaces/default/secrets/test-secret-other-resource", nil)
				Expect(err).NotTo(HaveOccurred())
				_, err = makeRequest("DELETE", "/api/v1/namespaces/default/secrets/other-secret", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			patch := createPatch("other-secret", "default")
			resp, err := makeRequest("PATCH", "/api/v1/namespaces/default/secrets/test-secret-other-resource", patch)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusBadRequest), string(body))
		})

		It("修改不存在的Namespace", func() {
			_, err := createSecret("test-secret-field-error", "default")
			Expect(err).NotTo(HaveOccurred())

			defer func() {
				_, err := makeRequest("DELETE", "/api/v1/namespaces/default/secrets/test-secret-field-error", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			patch := createPatch("test-secret-field-error", "nonexistent-namespace")
			resp, err := makeRequest("PATCH", "/api/v1/namespaces/default/secrets/test-secret-field-error", patch)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusBadRequest), string(body))
		})
	})

	Context("Secret 删除", func() {
		It("删除成功", func() {
			_, err := createSecret("test-secret-delete", "default")
			Expect(err).NotTo(HaveOccurred())

			resp, err := makeRequest("DELETE", "/api/v1/namespaces/default/secrets/test-secret-delete", nil)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusOK), string(body))
		})
	})
})
