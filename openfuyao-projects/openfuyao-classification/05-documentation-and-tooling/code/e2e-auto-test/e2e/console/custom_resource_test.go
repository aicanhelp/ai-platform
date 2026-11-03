package console

import (
	"encoding/json"
	"io"
	"net/http"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

var _ = Describe("CustomResourceDefinition 资源接口测试", Label("with-workload-cluster"), func() {
	Context("CustomResourceDefinition 创建", func() {
		It("创建成功", func() {
			resp, err := createCustomResourceDefinition("TestCrdSuccess")
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusCreated), string(body))

			defer func() {
				_, err := makeRequest("DELETE", "/apis/apiextensions.k8s.io/v1/customresourcedefinitions/testcrdsuccesss.example.com", nil)
				Expect(err).NotTo(HaveOccurred())
			}()
		})

		It("创建失败 - 字段有误", func() {
			crdYAML := map[string]interface{}{
				"apiVersion": "apiextensions.k8s.io/v1",
				"kind":       "CustomResourceDefinition",
				"metadata": map[string]interface{}{
					"name": "test-crd-invalids.example.com",
				},
				"spec": map[string]interface{}{
					"group": "example.com",
					"versions": []map[string]interface{}{
						{
							"name":    "v1",
							"served":  true,
							"storage": true,
							"schema": map[string]interface{}{
								"openAPIV3Schema": map[string]interface{}{
									"type": "object",
									"properties": map[string]interface{}{
										"spec": map[string]interface{}{
											"type": "object",
											"properties": map[string]interface{}{
												"replicas": map[string]interface{}{
													"type": "integer",
												},
											},
										},
									},
								},
							},
						},
					},
					"scope": "Namespaced",
					"names": map[string]interface{}{
						"plural":   "testcrdinvalids",
						"singular": "testcrdinvalid",
						"kind":     "TestCRDInvalid", // kind 包含空格
					},
				},
			}

			resp, err := makeRequest("POST", "/apis/apiextensions.k8s.io/v1/customresourcedefinitions", crdYAML)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusUnprocessableEntity), string(body))
		})

		It("创建失败 - 已存在同名", func() {
			resp1, err := createCustomResourceDefinition("TestCrdExisting")
			Expect(err).NotTo(HaveOccurred())
			defer resp1.Body.Close()
			Expect(resp1.StatusCode).To(Equal(http.StatusCreated))

			defer func() {
				_, err := makeRequest("DELETE", "/apis/apiextensions.k8s.io/v1/customresourcedefinitions/testcrdexistings.example.com", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			resp2, err := createCustomResourceDefinition("TestCrdExisting")
			Expect(err).NotTo(HaveOccurred())
			defer resp2.Body.Close()
			body, err := io.ReadAll(resp2.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp2.StatusCode).To(Equal(http.StatusConflict), string(body))
		})
	})

	Context("CustomResourceDefinition 查询", func() {
		It("查询成功", func() {
			_, err := createCustomResourceDefinition("TestCrdQuery")
			Expect(err).NotTo(HaveOccurred())

			defer func() {
				_, err := makeRequest("DELETE", "/apis/apiextensions.k8s.io/v1/customresourcedefinitions/testcrdquerys.example.com", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			resp, err := makeRequest("GET", "/apis/apiextensions.k8s.io/v1/customresourcedefinitions/testcrdquerys.example.com", nil)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusOK), string(body))

			var crd map[string]interface{}
			err = json.Unmarshal(body, &crd)
			Expect(err).NotTo(HaveOccurred())
			Expect(crd["metadata"].(map[string]interface{})["name"]).To(Equal("testcrdquerys.example.com"))
		})
	})

	Context("CustomResourceDefinition 修改", func() {
		It("修改成功", func() {
			_, err := createCustomResourceDefinition("TestCrdUpdateSuccess")
			Expect(err).NotTo(HaveOccurred())

			defer func() {
				_, err := makeRequest("DELETE", "/apis/apiextensions.k8s.io/v1/customresourcedefinitions/testcrdupdatesuccesss.example.com", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			patchCRD := map[string]interface{}{
				"metadata": map[string]interface{}{
					"name": "testcrdupdatesuccesss.example.com",
					"labels": map[string]interface{}{
						"app": "test",
					},
					"annotations": map[string]interface{}{
						"app": "test",
					},
				},
			}
			resp, err := makeRequest("PATCH", "/apis/apiextensions.k8s.io/v1/customresourcedefinitions/testcrdupdatesuccesss.example.com", patchCRD)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusOK), string(body))
		})

		It("修改失败 - 字段有误", func() {
			_, err := createCustomResourceDefinition("TestCrdInvalidUpdate")
			Expect(err).NotTo(HaveOccurred())

			defer func() {
				_, err := makeRequest("DELETE", "/apis/apiextensions.k8s.io/v1/customresourcedefinitions/testcrdinvalidupdates.example.com", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			patch := createMalformedPatch("TestCrdInvalidUpdate", "")
			resp, err := makeRequest("PATCH", "/apis/apiextensions.k8s.io/v1/customresourcedefinitions/testcrdinvalidupdates.example.com", patch)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusUnprocessableEntity), string(body))
		})

		It("修改其他同类资源", func() {
			_, err := createCustomResourceDefinition("TestCrdOtherResource")
			Expect(err).NotTo(HaveOccurred())
			_, err = createCustomResourceDefinition("OtherCrd")
			Expect(err).NotTo(HaveOccurred())

			defer func() {
				_, err := makeRequest("DELETE", "/apis/apiextensions.k8s.io/v1/customresourcedefinitions/testcrdotherresources.example.com", nil)
				Expect(err).NotTo(HaveOccurred())
				_, err = makeRequest("DELETE", "/apis/apiextensions.k8s.io/v1/customresourcedefinitions/othercrds.example.com", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			patch := createPatch("other-cr", "")
			resp, err := makeRequest("PATCH", "/apis/apiextensions.k8s.io/v1/customresourcedefinitions/testcrdotherresources.example.com", patch)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusBadRequest), string(body))
		})
	})

	Context("CustomResourceDefinition 删除", func() {
		It("删除成功", func() {
			_, err := createCustomResourceDefinition("TestCrdDelete")
			Expect(err).NotTo(HaveOccurred())

			defer func() {
				_, err := makeRequest("DELETE", "/apis/apiextensions.k8s.io/v1/customresourcedefinitions/testcrddeletes.example.com", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			resp, err := makeRequest("DELETE", "/apis/apiextensions.k8s.io/v1/customresourcedefinitions/testcrddeletes.example.com", nil)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusOK), string(body))
		})
	})
})

var _ = Describe("CustomResource 资源接口测试", Label("with-workload-cluster"), func() {
	Context("CustomResource 创建", func() {
		It("创建成功", func() {
			_, err := createCustomResourceDefinition("TestCrdSuccess")
			Expect(err).NotTo(HaveOccurred())

			resp, err := createCustomResource("test-cr-success", "default", "TestCrdSuccess")
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusCreated), string(body))

			defer func() {
				_, err := makeRequest("DELETE", "/apis/example.com/v1/namespaces/default/testcrdsuccesss/test-cr-success", nil)
				Expect(err).NotTo(HaveOccurred())
				_, err = makeRequest("DELETE", "/apis/apiextensions.k8s.io/v1/customresourcedefinitions/testcrdsuccesss.example.com", nil)
				Expect(err).NotTo(HaveOccurred())
			}()
		})

		It("创建失败 - 字段有误", func() {
			_, err := createCustomResourceDefinition("TestCrdInvalid")
			Expect(err).NotTo(HaveOccurred())

			defer func() {
				_, err := makeRequest("DELETE", "/apis/apiextensions.k8s.io/v1/customresourcedefinitions/testcrdinvalids.example.com", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			crYAML := map[string]interface{}{
				"apiVersion": "example.com/v1",
				"kind":       "TestCrdInvalid",
				"metadata": map[string]interface{}{
					"name":      "test-cr-invalid",
					"namespace": "default",
				},
				"spec": map[string]interface{}{
					"targets": []map[string]interface{}{
						{
							"name": "test-target",
							"port": -1,
						},
					},
				},
			}

			resp, err := makeRequest("POST", "/apis/example.com/v1/namespaces/default/testcrdinvalids", crYAML)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusUnprocessableEntity), string(body))
		})

		It("创建失败 - 已存在同名", func() {
			_, err := createCustomResourceDefinition("TestCrdExisting")
			Expect(err).NotTo(HaveOccurred())

			resp1, err := createCustomResource("test-cr-existing", "default", "TestCrdExisting")
			Expect(err).NotTo(HaveOccurred())
			defer resp1.Body.Close()
			Expect(resp1.StatusCode).To(Equal(http.StatusCreated))

			defer func() {
				_, err := makeRequest("DELETE", "/apis/example.com/v1/namespaces/default/testcrdexisting/test-cr-existing", nil)
				Expect(err).NotTo(HaveOccurred())
				_, err = makeRequest("DELETE", "/apis/apiextensions.k8s.io/v1/customresourcedefinitions/testcrdexistings.example.com", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			resp2, err := createCustomResource("test-cr-existing", "default", "TestCrdExisting")
			Expect(err).NotTo(HaveOccurred())
			defer resp2.Body.Close()
			body, err := io.ReadAll(resp2.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp2.StatusCode).To(Equal(http.StatusConflict), string(body))
		})

		It("创建失败 - Namespace 不存在", func() {
			_, err := createCustomResourceDefinition("TestCrdNonExistentNamespace")
			Expect(err).NotTo(HaveOccurred())

			defer func() {
				_, err := makeRequest("DELETE", "/apis/apiextensions.k8s.io/v1/customresourcedefinitions/testcrdnonexistentnamespaces.example.com", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			resp, err := createCustomResource("test-cr-nonexistent-namespace", "nonexistent-namespace", "TestCrdNonExistentNamespace")
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusNotFound), string(body))
		})
	})

	Context("CustomResource 查询", func() {
		It("查询成功", func() {
			_, err := createCustomResourceDefinition("TestCrdQuery")
			Expect(err).NotTo(HaveOccurred())

			defer func() {
				_, err := makeRequest("DELETE", "/apis/apiextensions.k8s.io/v1/customresourcedefinitions/testcrdquerys.example.com", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			_, err = createCustomResource("test-cr-query", "default", "TestCrdQuery")
			Expect(err).NotTo(HaveOccurred())

			resp, err := makeRequest("GET", "/apis/example.com/v1/namespaces/default/testcrdquerys/test-cr-query", nil)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusOK), string(body))

			var cr map[string]interface{}
			err = json.Unmarshal(body, &cr)
			Expect(err).NotTo(HaveOccurred())
			Expect(cr["metadata"].(map[string]interface{})["name"]).To(Equal("test-cr-query"))
		})
	})

	Context("CustomResource 修改", func() {
		It("修改成功", func() {
			_, err := createCustomResourceDefinition("TestCrdUpdateSuccess")
			Expect(err).NotTo(HaveOccurred())
			_, err = createCustomResource("test-cr-update-success", "default", "TestCrdUpdateSuccess")
			Expect(err).NotTo(HaveOccurred())

			defer func() {
				_, err := makeRequest("DELETE", "/apis/example.com/v1/namespaces/default/testcrdupdatesuccesss/test-cr-update-success", nil)
				Expect(err).NotTo(HaveOccurred())
				_, err = makeRequest("DELETE", "/apis/apiextensions.k8s.io/v1/customresourcedefinitions/testcrdupdatesuccesss.example.com", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			patch := createPatch("test-cr-update-success", "default")
			resp, err := makeRequest("PATCH", "/apis/example.com/v1/namespaces/default/testcrdupdatesuccesss/test-cr-update-success", patch)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusOK), string(body))
		})

		It("修改失败 - 字段有误", func() {
			_, err := createCustomResourceDefinition("TestCrdInvalidUpdate")
			Expect(err).NotTo(HaveOccurred())
			_, err = createCustomResource("test-cr-invalid-update", "default", "TestCrdInvalidUpdate")
			Expect(err).NotTo(HaveOccurred())

			defer func() {
				_, err := makeRequest("DELETE", "/apis/example.com/v1/namespaces/default/testcrdinvalidupdates/test-cr-invalid-update", nil)
				Expect(err).NotTo(HaveOccurred())
				_, err = makeRequest("DELETE", "/apis/apiextensions.k8s.io/v1/customresourcedefinitions/testcrdinvalidupdates.example.com", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			patch := createMalformedPatch("test-cr-invalid-update", "default")
			resp, err := makeRequest("PATCH", "/apis/example.com/v1/namespaces/default/testcrdinvalidupdates/test-cr-invalid-update", patch)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusUnprocessableEntity), string(body))
		})

		It("修改同一命名空间其他同类资源", func() {
			_, err := createCustomResourceDefinition("TestCrdOtherResource")
			Expect(err).NotTo(HaveOccurred())
			_, err = createCustomResource("test-cr-other-resource", "default", "TestCrdOtherResource")
			Expect(err).NotTo(HaveOccurred())
			_, err = createCustomResource("other-cr", "default", "TestCrdOtherResource")
			Expect(err).NotTo(HaveOccurred())

			defer func() {
				_, err := makeRequest("DELETE", "/apis/example.com/v1/namespaces/default/testcrdotherresources/test-cr-other-resource", nil)
				Expect(err).NotTo(HaveOccurred())
				_, err = makeRequest("DELETE", "/apis/example.com/v1/namespaces/default/testcrdotherresources/other-cr", nil)
				Expect(err).NotTo(HaveOccurred())
				_, err = makeRequest("DELETE", "/apis/apiextensions.k8s.io/v1/customresourcedefinitions/testcrdotherresources.example.com", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			patch := createPatch("other-cr", "default")
			resp, err := makeRequest("PATCH", "/apis/example.com/v1/namespaces/default/testcrdotherresources/test-cr-other-resource", patch)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusBadRequest), string(body))
		})

		It("修改不存在的Namespace", func() {
			_, err := createCustomResourceDefinition("TestCrdNonExistentNamespace")
			Expect(err).NotTo(HaveOccurred())
			_, err = createCustomResource("test-cr-nonexistent-namespace", "default", "TestCrdNonExistentNamespace")
			Expect(err).NotTo(HaveOccurred())

			defer func() {
				_, err := makeRequest("DELETE", "/apis/example.com/v1/namespaces/default/testcrdnonexistentnamespaces/test-cr-nonexistent-namespace", nil)
				Expect(err).NotTo(HaveOccurred())
				_, err = makeRequest("DELETE", "/apis/apiextensions.k8s.io/v1/customresourcedefinitions/testcrdnonexistentnamespaces.example.com", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			patch := createPatch("test-cr-nonexistent-namespace", "nonexistent-namespace")
			resp, err := makeRequest("PATCH", "/apis/example.com/v1/namespaces/default/testcrdnonexistentnamespaces/test-cr-nonexistent-namespace", patch)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusBadRequest), string(body))
		})
	})

	Context("CustomResource 删除", func() {
		It("删除成功", func() {
			_, err := createCustomResourceDefinition("TestCrdDelete")
			Expect(err).NotTo(HaveOccurred())
			_, err = createCustomResource("test-cr-delete", "default", "TestCrdDelete")
			Expect(err).NotTo(HaveOccurred())

			defer func() {
				_, err = makeRequest("DELETE", "/apis/apiextensions.k8s.io/v1/customresourcedefinitions/testcrddeletes.example.com", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			resp, err := makeRequest("DELETE", "/apis/example.com/v1/namespaces/default/testcrddeletes/test-cr-delete", nil)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusOK), string(body))
		})
	})
})
