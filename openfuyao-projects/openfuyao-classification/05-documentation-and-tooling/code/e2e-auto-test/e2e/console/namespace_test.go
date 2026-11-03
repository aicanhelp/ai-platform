package console

import (
	"io"
	"net/http"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

var _ = Describe("Namespace 资源接口测试", Label("with-workload-cluster"), func() {
	Context("Namespace 创建", func() {
		It("创建成功", func() {
			resp, err := createNamespace("test-namespace-success")
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusCreated), string(body))

			defer func() {
				_, err := makeRequest("DELETE", "/api/v1/namespaces/test-namespace-success", nil)
				Expect(err).NotTo(HaveOccurred())
			}()
		})

		It("创建失败 - 字段有误", func() {
			namespaceYAML := map[string]interface{}{
				"apiVersion": "v1",
				"kind":       "Namespace",
				"metadata": map[string]interface{}{
					"name": "TestNamespaceInvalid", // 包含大写字母
				},
			}

			resp, err := makeRequest("POST", "/api/v1/namespaces", namespaceYAML)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusUnprocessableEntity), string(body))
		})

		It("创建失败 - 已存在同名", func() {
			resp1, err := createNamespace("test-existing-namespace")
			Expect(err).NotTo(HaveOccurred())
			defer resp1.Body.Close()

			defer func() {
				_, err := makeRequest("DELETE", "/api/v1/namespaces/test-existing-namespace", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			resp2, err := createNamespace("test-existing-namespace")
			Expect(err).NotTo(HaveOccurred())
			defer resp2.Body.Close()
			Expect(resp2.StatusCode).To(Equal(http.StatusConflict))
		})
	})

	Context("Namespace 查询", func() {
		It("查询成功", func() {
			_, err := createNamespace("test-namespace-query")
			Expect(err).NotTo(HaveOccurred())

			defer func() {
				_, err := makeRequest("DELETE", "/api/v1/namespaces/test-namespace-query", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			resp, err := makeRequest("GET", "/api/v1/namespaces/test-namespace-query", nil)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusOK), string(body))
		})
	})

	Context("Namespace 修改", func() {
		It("修改成功", func() {
			_, err := createNamespace("test-namespace-update-success")
			Expect(err).NotTo(HaveOccurred())

			defer func() {
				_, err := makeRequest("DELETE", "/api/v1/namespaces/test-namespace-update-success", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			patch := createPatch("test-namespace-update-success", "")
			resp, err := makeRequest("PATCH", "/api/v1/namespaces/test-namespace-update-success", patch)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusOK), string(body))
		})

		It("修改失败 - 字段有误", func() {
			_, err := createNamespace("test-namespace-field-error")
			Expect(err).NotTo(HaveOccurred())

			defer func() {
				_, err := makeRequest("DELETE", "/api/v1/namespaces/test-namespace-field-error", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			patch := createMalformedPatch("test-namespace-field-error", "")
			resp, err := makeRequest("PATCH", "/api/v1/namespaces/test-namespace-field-error", patch)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusUnprocessableEntity), string(body))
		})
	})

	Context("Namespace 删除", func() {
		It("删除成功", func() {
			_, err := createNamespace("test-namespace-delete")
			Expect(err).NotTo(HaveOccurred())

			resp, err := makeRequest("DELETE", "/api/v1/namespaces/test-namespace-delete", nil)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusOK), string(body))
		})
	})
})

var _ = Describe("ResourceQuota 资源接口测试", Label("with-workload-cluster"), func() {
	Context("ResourceQuota 创建", func() {
		It("创建成功", func() {
			resp, err := createResourceQuota("test-resourcequota-success", "default")
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusCreated), string(body))

			defer func() {
				_, err := makeRequest("DELETE", "/api/v1/namespaces/default/resourcequotas/test-resourcequota-success", nil)
				Expect(err).NotTo(HaveOccurred())
			}()
		})

		It("创建失败 - 字段有误", func() {
			resourceQuotaYAML := map[string]interface{}{
				"apiVersion": "v1",
				"kind":       "ResourceQuota",
				"metadata": map[string]interface{}{
					"name":      "test-resourcequota-invalid-hard",
					"namespace": "default",
				},
				"spec": map[string]interface{}{
					"hard": map[string]interface{}{
						"requests.cpu": "-100",
					},
				},
			}

			resp, err := makeRequest("POST", "/api/v1/namespaces/default/resourcequotas", resourceQuotaYAML)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusUnprocessableEntity), string(body))
		})

		It("创建失败 - 已存在同名", func() {
			resp1, err := createResourceQuota("test-existing-resourcequota", "default")
			Expect(err).NotTo(HaveOccurred())
			defer resp1.Body.Close()
			Expect(resp1.StatusCode).To(Equal(http.StatusCreated))

			defer func() {
				_, err := makeRequest("DELETE", "/api/v1/namespaces/default/resourcequotas/test-existing-resourcequota", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			resp2, err := createResourceQuota("test-existing-resourcequota", "default")
			Expect(err).NotTo(HaveOccurred())
			defer resp2.Body.Close()
			body, err := io.ReadAll(resp2.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp2.StatusCode).To(Equal(http.StatusConflict), string(body))
		})

		It("创建失败 - Namespace 不存在", func() {
			resp, err := createResourceQuota("test-resourcequota-nonexistent-ns", "nonexistent-namespace")
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusNotFound), string(body))
		})
	})

	Context("ResourceQuota 查询", func() {
		It("查询成功", func() {
			_, err := createResourceQuota("test-resourcequota-query", "default")
			Expect(err).NotTo(HaveOccurred())

			defer func() {
				_, err := makeRequest("DELETE", "/api/v1/namespaces/default/resourcequotas/test-resourcequota-query", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			resp, err := makeRequest("GET", "/api/v1/namespaces/default/resourcequotas/test-resourcequota-query", nil)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusOK), string(body))
		})
	})

	Context("ResourceQuota 修改", func() {
		It("修改成功", func() {
			_, err := createResourceQuota("test-resourcequota-update-success", "default")
			Expect(err).NotTo(HaveOccurred())

			defer func() {
				_, err := makeRequest("DELETE", "/api/v1/namespaces/default/resourcequotas/test-resourcequota-update-success", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			patch := createPatch("test-resourcequota-update-success", "default")
			resp, err := makeRequest("PATCH", "/api/v1/namespaces/default/resourcequotas/test-resourcequota-update-success", patch)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusOK), string(body))
		})

		It("修改失败 - 字段有误", func() {
			_, err := createResourceQuota("test-resourcequota-field-error", "default")
			Expect(err).NotTo(HaveOccurred())

			defer func() {
				_, err := makeRequest("DELETE", "/api/v1/namespaces/default/resourcequotas/test-resourcequota-field-error", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			patch := createMalformedPatch("test-resourcequota-field-error", "default")
			resp, err := makeRequest("PATCH", "/api/v1/namespaces/default/resourcequotas/test-resourcequota-field-error", patch)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusUnprocessableEntity), string(body))
		})

		It("修改同一命名空间其他同类资源", func() {
			_, err := createResourceQuota("test-resourcequota-other-resource", "default")
			Expect(err).NotTo(HaveOccurred())
			_, err = createResourceQuota("other-resourcequota", "default")
			Expect(err).NotTo(HaveOccurred())

			defer func() {
				_, err := makeRequest("DELETE", "/api/v1/namespaces/default/resourcequotas/test-resourcequota-other-resource", nil)
				Expect(err).NotTo(HaveOccurred())
				_, err = makeRequest("DELETE", "/api/v1/namespaces/default/resourcequotas/other-resourcequota", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			patch := createPatch("other-resourcequota", "default")
			resp, err := makeRequest("PATCH", "/api/v1/namespaces/default/resourcequotas/test-resourcequota-other-resource", patch)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusBadRequest), string(body))
		})

		It("修改不存在的Namespace", func() {
			_, err := createResourceQuota("test-resourcequota-nonexistent-ns", "default")
			Expect(err).NotTo(HaveOccurred())

			defer func() {
				_, err := makeRequest("DELETE", "/api/v1/namespaces/default/resourcequotas/test-resourcequota-nonexistent-ns", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			patch := createPatch("test-resourcequota-nonexistent-ns", "nonexistent-namespace")
			resp, err := makeRequest("PATCH", "/api/v1/namespaces/default/resourcequotas/test-resourcequota-nonexistent-ns", patch)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusBadRequest), string(body))
		})
	})

	Context("ResourceQuota 删除", func() {
		It("删除成功", func() {
			_, err := createResourceQuota("test-resourcequota-delete", "default")
			Expect(err).NotTo(HaveOccurred())

			resp, err := makeRequest("DELETE", "/api/v1/namespaces/default/resourcequotas/test-resourcequota-delete", nil)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusOK), string(body))
		})
	})
})

var _ = Describe("LimitRange 资源接口测试", Label("with-workload-cluster"), func() {
	Context("LimitRange 创建", func() {
		It("创建成功", func() {
			resp, err := createLimitRange("test-limitrange-success", "default")
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusCreated), string(body))

			defer func() {
				_, err := makeRequest("DELETE", "/api/v1/namespaces/default/limitranges/test-limitrange-success", nil)
				Expect(err).NotTo(HaveOccurred())
			}()
		})

		It("创建失败 - 字段有误", func() {
			limitRangeYAML := map[string]interface{}{
				"apiVersion": "v1",
				"kind":       "LimitRange",
				"metadata": map[string]interface{}{
					"name":      "test-limitrange-invalid-limits",
					"namespace": "default",
				},
				"spec": map[string]interface{}{
					"limits": []map[string]interface{}{
						{
							"type": "Container",
							"max": map[string]interface{}{
								"cpu": "1",
							},
							"default": map[string]interface{}{
								"cpu": "2", // default.cpu > max.cpu
							},
						},
					},
				},
			}

			resp, err := makeRequest("POST", "/api/v1/namespaces/default/limitranges", limitRangeYAML)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusUnprocessableEntity), string(body))
		})

		It("创建失败 - 已存在同名", func() {
			resp1, err := createLimitRange("test-existing-limitrange", "default")
			Expect(err).NotTo(HaveOccurred())
			defer resp1.Body.Close()
			Expect(resp1.StatusCode).To(Equal(http.StatusCreated))

			defer func() {
				_, err := makeRequest("DELETE", "/api/v1/namespaces/default/limitranges/test-existing-limitrange", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			resp2, err := createLimitRange("test-existing-limitrange", "default")
			Expect(err).NotTo(HaveOccurred())
			defer resp2.Body.Close()
			body, err := io.ReadAll(resp2.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp2.StatusCode).To(Equal(http.StatusConflict), string(body))
		})

		It("创建失败 - Namespace 不存在", func() {
			resp, err := createLimitRange("test-limitrange-nonexistent-ns", "nonexistent-namespace")
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusNotFound), string(body))
		})
	})

	Context("LimitRange 查询", func() {
		It("查询成功", func() {
			_, err := createLimitRange("test-limitrange-query", "default")
			Expect(err).NotTo(HaveOccurred())

			defer func() {
				_, err := makeRequest("DELETE", "/api/v1/namespaces/default/limitranges/test-limitrange-query", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			resp, err := makeRequest("GET", "/api/v1/namespaces/default/limitranges/test-limitrange-query", nil)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusOK), string(body))
		})
	})

	Context("LimitRange 修改", func() {
		It("修改成功", func() {
			_, err := createLimitRange("test-limitrange-update-success", "default")
			Expect(err).NotTo(HaveOccurred())

			defer func() {
				_, err := makeRequest("DELETE", "/api/v1/namespaces/default/limitranges/test-limitrange-update-success", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			patch := createPatch("test-limitrange-update-success", "default")
			resp, err := makeRequest("PATCH", "/api/v1/namespaces/default/limitranges/test-limitrange-update-success", patch)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusOK), string(body))
		})

		It("修改失败 - 字段有误", func() {
			_, err := createLimitRange("test-limitrange-field-error", "default")
			Expect(err).NotTo(HaveOccurred())

			defer func() {
				_, err := makeRequest("DELETE", "/api/v1/namespaces/default/limitranges/test-limitrange-field-error", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			patch := createMalformedPatch("test-limitrange-field-error", "default")
			resp, err := makeRequest("PATCH", "/api/v1/namespaces/default/limitranges/test-limitrange-field-error", patch)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusUnprocessableEntity), string(body))
		})

		It("修改同一命名空间其他同类资源", func() {
			_, err := createLimitRange("test-limitrange-other-resource", "default")
			Expect(err).NotTo(HaveOccurred())
			_, err = createLimitRange("other-limitrange", "default")
			Expect(err).NotTo(HaveOccurred())

			defer func() {
				_, err := makeRequest("DELETE", "/api/v1/namespaces/default/limitranges/other-limitrange", nil)
				Expect(err).NotTo(HaveOccurred())
				_, err = makeRequest("DELETE", "/api/v1/namespaces/default/limitranges/test-limitrange-other-resource", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			patch := createPatch("other-limitrange", "default")
			resp, err := makeRequest("PATCH", "/api/v1/namespaces/default/limitranges/test-limitrange-other-resource", patch)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusBadRequest), string(body))
		})

		It("修改不存在的Namespace", func() {
			_, err := createLimitRange("test-limitrange-nonexistent-ns", "default")
			Expect(err).NotTo(HaveOccurred())

			defer func() {
				_, err := makeRequest("DELETE", "/api/v1/namespaces/default/limitranges/test-limitrange-nonexistent-ns", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			patch := createPatch("test-limitrange-nonexistent-ns", "nonexistent-namespace")
			resp, err := makeRequest("PATCH", "/api/v1/namespaces/default/limitranges/test-limitrange-nonexistent-ns", patch)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusBadRequest), string(body))
		})
	})

	Context("LimitRange 删除", func() {
		It("删除成功", func() {
			_, err := createLimitRange("test-limitrange-delete", "default")
			Expect(err).NotTo(HaveOccurred())

			defer func() {
				_, err := makeRequest("DELETE", "/api/v1/namespaces/default/limitranges/test-limitrange-delete", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			resp, err := makeRequest("DELETE", "/api/v1/namespaces/default/limitranges/test-limitrange-delete", nil)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusOK), string(body))
		})
	})
})
