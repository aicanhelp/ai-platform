package console

import (
	"encoding/json"
	"io"
	"net/http"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

var _ = Describe("Role 资源接口测试", Label("with-workload-cluster"), func() {
	Context("Role 创建", func() {
		It("创建成功", func() {
			resp, err := createRole("test-role-success", "default")
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusCreated), string(body))

			defer func() {
				_, err := makeRequest("DELETE", "/apis/rbac.authorization.k8s.io/v1/namespaces/default/roles/test-role-success", nil)
				Expect(err).NotTo(HaveOccurred())
			}()
		})

		It("创建失败 - 字段有误", func() {
			roleYAML := map[string]interface{}{
				"apiVersion": "rbac.authorization.k8s.io/v1",
				"kind":       "Role",
				"metadata": map[string]interface{}{
					"name":      "test-role-invalid-verbs",
					"namespace": "default",
				},
				"rules": []map[string]interface{}{
					{
						"apiGroups": "",
						"resources": "pods",
						"verbs":     "list",
					},
				},
			}

			resp, err := makeRequest("POST", "/apis/rbac.authorization.k8s.io/v1/namespaces/default/roles", roleYAML)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusBadRequest), string(body))
		})

		It("创建失败 - 已存在同名", func() {
			resp1, err := createRole("test-existing-role", "default")
			Expect(err).NotTo(HaveOccurred())
			defer resp1.Body.Close()
			Expect(resp1.StatusCode).To(Equal(http.StatusCreated))

			defer func() {
				_, err := makeRequest("DELETE", "/apis/rbac.authorization.k8s.io/v1/namespaces/default/roles/test-existing-role", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			resp2, err := createRole("test-existing-role", "default")
			Expect(err).NotTo(HaveOccurred())
			defer resp2.Body.Close()
			body, err := io.ReadAll(resp2.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp2.StatusCode).To(Equal(http.StatusConflict), string(body))
		})

		It("创建失败 - Namespace 不存在", func() {
			resp, err := createRole("test-role-nonexistent-ns", "nonexistent-namespace")
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusNotFound), string(body))
		})
	})

	Context("Role 查询", func() {
		It("查询成功", func() {
			_, err := createRole("test-role-query", "default")
			Expect(err).NotTo(HaveOccurred())

			defer func() {
				_, err := makeRequest("DELETE", "/apis/rbac.authorization.k8s.io/v1/namespaces/default/roles/test-role-query", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			resp, err := makeRequest("GET", "/apis/rbac.authorization.k8s.io/v1/namespaces/default/roles/test-role-query", nil)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusOK), string(body))

			var role map[string]interface{}
			err = json.Unmarshal(body, &role)
			Expect(err).NotTo(HaveOccurred())
			Expect(role["metadata"].(map[string]interface{})["name"]).To(Equal("test-role-query"))
		})
	})

	Context("Role 修改", func() {
		It("修改成功", func() {
			_, err := createRole("test-role-update-success", "default")
			Expect(err).NotTo(HaveOccurred())

			defer func() {
				_, err := makeRequest("DELETE", "/apis/rbac.authorization.k8s.io/v1/namespaces/default/roles/test-role-update-success", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			patch := createPatch("test-role-update-success", "default")
			resp, err := makeRequest("PATCH", "/apis/rbac.authorization.k8s.io/v1/namespaces/default/roles/test-role-update-success", patch)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusOK), string(body))
		})

		It("修改失败 - 字段有误", func() {
			_, err := createRole("test-role-field-error", "default")
			Expect(err).NotTo(HaveOccurred())

			defer func() {
				_, err := makeRequest("DELETE", "/apis/rbac.authorization.k8s.io/v1/namespaces/default/roles/test-role-field-error", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			patch := createMalformedPatch("test-role-field-error", "default")
			resp, err := makeRequest("PATCH", "/apis/rbac.authorization.k8s.io/v1/namespaces/default/roles/test-role-field-error", patch)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusUnprocessableEntity), string(body))
		})

		It("修改同一命名空间其他同类资源", func() {
			_, err := createRole("test-role-other-resource", "default")
			Expect(err).NotTo(HaveOccurred())
			_, err = createRole("other-role", "default")
			Expect(err).NotTo(HaveOccurred())

			defer func() {
				_, err := makeRequest("DELETE", "/apis/rbac.authorization.k8s.io/v1/namespaces/default/roles/test-role-other-resource", nil)
				Expect(err).NotTo(HaveOccurred())
				_, err = makeRequest("DELETE", "/apis/rbac.authorization.k8s.io/v1/namespaces/default/roles/other-role", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			patch := createPatch("other-role", "default")
			resp, err := makeRequest("PATCH", "/apis/rbac.authorization.k8s.io/v1/namespaces/default/roles/test-role-other-resource", patch)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusBadRequest), string(body))
		})

		It("修改不存在的Namespace", func() {
			_, err := createRole("test-role-nonexistent-ns-update", "default")
			Expect(err).NotTo(HaveOccurred())
			defer func() {
				_, err := makeRequest("DELETE", "/apis/rbac.authorization.k8s.io/v1/namespaces/default/roles/test-role-nonexistent-ns-update", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			patch := createPatch("test-role-nonexistent-ns-update", "nonexistent-namespace")
			resp, err := makeRequest("PATCH", "/apis/rbac.authorization.k8s.io/v1/namespaces/default/roles/test-role-nonexistent-ns-update", patch)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusBadRequest), string(body))
		})
	})

	Context("Role 删除", func() {
		It("删除成功", func() {
			_, err := createRole("test-role-delete", "default")
			Expect(err).NotTo(HaveOccurred())

			resp, err := makeRequest("DELETE", "/apis/rbac.authorization.k8s.io/v1/namespaces/default/roles/test-role-delete", nil)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusOK), string(body))
		})
	})
})

var _ = Describe("ClusterRole 资源接口测试", Label("with-workload-cluster"), func() {
	Context("ClusterRole 创建", func() {
		It("创建成功", func() {
			resp, err := createClusterRole("test-clusterrole-success")
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusCreated), string(body))

			defer func() {
				_, err := makeRequest("DELETE", "/apis/rbac.authorization.k8s.io/v1/clusterroles/test-clusterrole-success", nil)
				Expect(err).NotTo(HaveOccurred())
			}()
		})

		It("创建失败 - 字段有误", func() {
			clusterRoleYAML := map[string]interface{}{
				"apiVersion": "rbac.authorization.k8s.io/v1",
				"kind":       "ClusterRole",
				"metadata": map[string]interface{}{
					"name": "test-clusterrole-invalid-verbs",
				},
				"rules": []map[string]interface{}{
					{
						"apiGroups": "",
						"resources": "nodes",
						"verbs":     "invalid-verb",
					},
				},
			}

			resp, err := makeRequest("POST", "/apis/rbac.authorization.k8s.io/v1/clusterroles", clusterRoleYAML)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusBadRequest), string(body))
		})

		It("创建失败 - 已存在同名", func() {
			resp1, err := createClusterRole("test-existing-clusterrole")
			Expect(err).NotTo(HaveOccurred())
			defer resp1.Body.Close()
			Expect(resp1.StatusCode).To(Equal(http.StatusCreated))

			defer func() {
				_, err := makeRequest("DELETE", "/apis/rbac.authorization.k8s.io/v1/clusterroles/test-existing-clusterrole", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			resp2, err := createClusterRole("test-existing-clusterrole")
			Expect(err).NotTo(HaveOccurred())
			defer resp2.Body.Close()
			body, err := io.ReadAll(resp2.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp2.StatusCode).To(Equal(http.StatusConflict), string(body))
		})
	})

	Context("ClusterRole 查询", func() {
		It("查询成功", func() {
			_, err := createClusterRole("test-clusterrole-query")
			Expect(err).NotTo(HaveOccurred())

			defer func() {
				_, err := makeRequest("DELETE", "/apis/rbac.authorization.k8s.io/v1/clusterroles/test-clusterrole-query", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			resp, err := makeRequest("GET", "/apis/rbac.authorization.k8s.io/v1/clusterroles/test-clusterrole-query", nil)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusOK), string(body))

			var clusterRole map[string]interface{}
			err = json.Unmarshal(body, &clusterRole)
			Expect(err).NotTo(HaveOccurred())
			Expect(clusterRole["metadata"].(map[string]interface{})["name"]).To(Equal("test-clusterrole-query"))
		})
	})

	Context("ClusterRole 修改", func() {
		It("修改成功", func() {
			_, err := createClusterRole("test-clusterrole-update-success")
			Expect(err).NotTo(HaveOccurred())

			defer func() {
				_, err := makeRequest("DELETE", "/apis/rbac.authorization.k8s.io/v1/clusterroles/test-clusterrole-update-success", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			patch := createPatch("test-clusterrole-update-success", "")
			resp, err := makeRequest("PATCH", "/apis/rbac.authorization.k8s.io/v1/clusterroles/test-clusterrole-update-success", patch)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusOK), string(body))
		})

		It("修改失败 - 字段有误", func() {
			_, err := createClusterRole("test-clusterrole-field-error")
			Expect(err).NotTo(HaveOccurred())

			defer func() {
				_, err := makeRequest("DELETE", "/apis/rbac.authorization.k8s.io/v1/clusterroles/test-clusterrole-field-error", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			patch := createMalformedPatch("test-clusterrole-field-error", "")
			resp, err := makeRequest("PATCH", "/apis/rbac.authorization.k8s.io/v1/clusterroles/test-clusterrole-field-error", patch)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusUnprocessableEntity), string(body))
		})

		It("修改其他同类资源", func() {
			_, err := createClusterRole("test-clusterrole-other-resource")
			Expect(err).NotTo(HaveOccurred())
			_, err = createClusterRole("other-clusterrole")
			Expect(err).NotTo(HaveOccurred())

			defer func() {
				_, err := makeRequest("DELETE", "/apis/rbac.authorization.k8s.io/v1/clusterroles/test-clusterrole-other-resource", nil)
				Expect(err).NotTo(HaveOccurred())
				_, err = makeRequest("DELETE", "/apis/rbac.authorization.k8s.io/v1/clusterroles/other-clusterrole", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			patch := createPatch("other-clusterrole", "")
			resp, err := makeRequest("PATCH", "/apis/rbac.authorization.k8s.io/v1/clusterroles/test-clusterrole-other-resource", patch)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusBadRequest), string(body))
		})
	})

	Context("ClusterRole 删除", func() {
		It("删除成功", func() {
			_, err := createClusterRole("test-clusterrole-delete")
			Expect(err).NotTo(HaveOccurred())

			resp, err := makeRequest("DELETE", "/apis/rbac.authorization.k8s.io/v1/clusterroles/test-clusterrole-delete", nil)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusOK), string(body))
		})
	})
})

var _ = Describe("RoleBinding 资源接口测试", Label("with-workload-cluster"), func() {
	Context("RoleBinding 创建", func() {
		It("创建成功", func() {
			_, err := createRole("test-role-success", "default")
			Expect(err).NotTo(HaveOccurred())

			resp, err := createRoleBinding("test-rolebinding-success", "test-role-success", "default")
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusCreated), string(body))

			defer func() {
				_, err := makeRequest("DELETE", "/apis/rbac.authorization.k8s.io/v1/namespaces/default/rolebindings/test-rolebinding-success", nil)
				Expect(err).NotTo(HaveOccurred())
				_, err = makeRequest("DELETE", "/apis/rbac.authorization.k8s.io/v1/namespaces/default/roles/test-role-success", nil)
				Expect(err).NotTo(HaveOccurred())
			}()
		})

		It("创建失败 - 字段有误", func() {
			_, err := createRole("test-role-field-error", "default")
			Expect(err).NotTo(HaveOccurred())

			defer func() {
				_, err := makeRequest("DELETE", "/apis/rbac.authorization.k8s.io/v1/namespaces/default/roles/test-role-field-error", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			roleBindingYAML := map[string]interface{}{
				"apiVersion": "rbac.authorization.k8s.io/v1",
				"kind":       "RoleBinding",
				"metadata": map[string]interface{}{
					"name":      "test-rolebinding-field-error",
					"namespace": "default",
				},
				"subjects": []map[string]interface{}{
					{
						"kind":     "User",
						"name":     "test-user",
						"apiGroup": "rbac.authorization.k8s.io",
					},
				},
				"roleRef": map[string]interface{}{
					"kind":     "InvalidRole",
					"name":     "test-role-field-error",
					"apiGroup": "rbac.authorization.k8s.io",
				},
			}

			resp, err := makeRequest("POST", "/apis/rbac.authorization.k8s.io/v1/namespaces/default/rolebindings", roleBindingYAML)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusInternalServerError), string(body)) // InternalServerError due to unsupported role
		})

		It("创建失败 - 已存在同名", func() {
			_, err := createRole("test-role-existing-binding", "default")
			Expect(err).NotTo(HaveOccurred())

			resp, err := createRoleBinding("test-rolebinding-existing-binding", "test-role-existing-binding", "default")
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			Expect(resp.StatusCode).To(Equal(http.StatusCreated))

			defer func() {
				_, err := makeRequest("DELETE", "/apis/rbac.authorization.k8s.io/v1/namespaces/default/rolebindings/test-rolebinding-existing-binding", nil)
				Expect(err).NotTo(HaveOccurred())
				_, err = makeRequest("DELETE", "/apis/rbac.authorization.k8s.io/v1/namespaces/default/roles/test-role-existing-binding", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			resp2, err := createRoleBinding("test-rolebinding-existing-binding", "test-role-existing-binding", "default")
			Expect(err).NotTo(HaveOccurred())
			defer resp2.Body.Close()
			body, err := io.ReadAll(resp2.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp2.StatusCode).To(Equal(http.StatusConflict), string(body))
		})

		It("创建失败 - Namespace 不存在", func() {
			_, err := createRole("test-role-nonexistent-ns", "default")
			Expect(err).NotTo(HaveOccurred())

			defer func() {
				_, err := makeRequest("DELETE", "/apis/rbac.authorization.k8s.io/v1/namespaces/default/roles/test-role-nonexistent-ns", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			resp, err := createRoleBinding("test-rolebinding-nonexistent-ns", "test-role-nonexistent-ns", "nonexistent-namespace")
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusNotFound), string(body))
		})
	})

	Context("RoleBinding 查询", func() {
		It("查询成功", func() {
			_, err := createRole("test-role-query", "default")
			Expect(err).NotTo(HaveOccurred())
			_, err = createRoleBinding("test-rolebinding-query", "test-role-query", "default")
			Expect(err).NotTo(HaveOccurred())

			defer func() {
				_, err := makeRequest("DELETE", "/apis/rbac.authorization.k8s.io/v1/namespaces/default/rolebindings/test-rolebinding-query", nil)
				Expect(err).NotTo(HaveOccurred())
				_, err = makeRequest("DELETE", "/apis/rbac.authorization.k8s.io/v1/namespaces/default/roles/test-role-query", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			resp, err := makeRequest("GET", "/apis/rbac.authorization.k8s.io/v1/namespaces/default/rolebindings/test-rolebinding-query", nil)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusOK), string(body))
		})
	})

	Context("RoleBinding 修改", func() {
		It("修改成功", func() {
			_, err := createRole("test-role-update-success", "default")
			Expect(err).NotTo(HaveOccurred())
			_, err = createRoleBinding("test-rolebinding-update-success", "test-role-update-success", "default")
			Expect(err).NotTo(HaveOccurred())

			defer func() {
				_, err := makeRequest("DELETE", "/apis/rbac.authorization.k8s.io/v1/namespaces/default/roles/test-role-update-success", nil)
				Expect(err).NotTo(HaveOccurred())
				_, err = makeRequest("DELETE", "/apis/rbac.authorization.k8s.io/v1/namespaces/default/rolebindings/test-rolebinding-update-success", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			patch := createPatch("test-rolebinding-update-success", "default")
			resp, err := makeRequest("PATCH", "/apis/rbac.authorization.k8s.io/v1/namespaces/default/rolebindings/test-rolebinding-update-success", patch)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusOK), string(body))
		})

		It("修改失败 - 字段有误", func() {
			_, err := createRole("test-role-field-error", "default")
			Expect(err).NotTo(HaveOccurred())
			_, err = createRoleBinding("test-rolebinding-field-error", "test-role-field-error", "default")
			Expect(err).NotTo(HaveOccurred())

			defer func() {
				_, err := makeRequest("DELETE", "/apis/rbac.authorization.k8s.io/v1/namespaces/default/roles/test-role-field-error", nil)
				Expect(err).NotTo(HaveOccurred())
				_, err = makeRequest("DELETE", "/apis/rbac.authorization.k8s.io/v1/namespaces/default/rolebindings/test-rolebinding-field-error", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			patch := createMalformedPatch("test-rolebinding-field-error", "default")
			resp, err := makeRequest("PATCH", "/apis/rbac.authorization.k8s.io/v1/namespaces/default/rolebindings/test-rolebinding-field-error", patch)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusUnprocessableEntity), string(body))
		})

		It("修改同一命名空间其他同类资源", func() {
			_, err := createRole("test-role-other-resource", "default")
			Expect(err).NotTo(HaveOccurred())
			_, err = createRoleBinding("test-rolebinding-other-resource", "test-role-other-resource", "default")
			Expect(err).NotTo(HaveOccurred())
			_, err = createRoleBinding("other-rolebinding", "test-role-other-resource", "default")
			Expect(err).NotTo(HaveOccurred())

			defer func() {
				_, err := makeRequest("DELETE", "/apis/rbac.authorization.k8s.io/v1/namespaces/default/roles/test-role-other-resource", nil)
				Expect(err).NotTo(HaveOccurred())
				_, err = makeRequest("DELETE", "/apis/rbac.authorization.k8s.io/v1/namespaces/default/rolesbindings/other-role", nil)
				Expect(err).NotTo(HaveOccurred())
				_, err = makeRequest("DELETE", "/apis/rbac.authorization.k8s.io/v1/namespaces/default/rolebindings/test-rolebinding-other-resource", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			patch := createPatch("other-rolebinding", "default")
			resp, err := makeRequest("PATCH", "/apis/rbac.authorization.k8s.io/v1/namespaces/default/rolebindings/test-rolebinding-other-resource", patch)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusBadRequest), string(body))
		})

		It("修改不存在的Namespace", func() {
			_, err := createRole("test-role-nonexistent-ns", "default")
			Expect(err).NotTo(HaveOccurred())
			_, err = createRoleBinding("test-rolebinding-nonexistent-ns", "test-role-nonexistent-ns", "default")
			Expect(err).NotTo(HaveOccurred())

			defer func() {
				_, err := makeRequest("DELETE", "/apis/rbac.authorization.k8s.io/v1/namespaces/default/rolebindings/test-rolebinding-nonexistent-ns", nil)
				Expect(err).NotTo(HaveOccurred())
				_, err = makeRequest("DELETE", "/apis/rbac.authorization.k8s.io/v1/namespaces/default/roles/test-role-nonexistent-ns", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			patch := createPatch("test-rolebinding-nonexistent-ns", "nonexistent-namespace")
			resp, err := makeRequest("PATCH", "/apis/rbac.authorization.k8s.io/v1/namespaces/default/rolebindings/test-rolebinding-nonexistent-ns", patch)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusBadRequest), string(body))
		})
	})

	Context("RoleBinding 删除", func() {
		It("删除成功", func() {
			_, err := createRole("test-role-delete", "default")
			Expect(err).NotTo(HaveOccurred())
			_, err = createRoleBinding("test-rolebinding-delete", "test-role-delete", "default")
			Expect(err).NotTo(HaveOccurred())

			defer func() {
				_, err := makeRequest("DELETE", "/apis/rbac.authorization.k8s.io/v1/namespaces/default/roles/test-role-delete", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			resp, err := makeRequest("DELETE", "/apis/rbac.authorization.k8s.io/v1/namespaces/default/rolebindings/test-rolebinding-delete", nil)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusOK), string(body))
		})
	})
})

var _ = Describe("ClusterRoleBinding 资源接口测试", Label("with-workload-cluster"), func() {
	Context("ClusterRoleBinding 创建", func() {
		It("创建成功", func() {
			_, err := createClusterRole("test-clusterrole-success")
			Expect(err).NotTo(HaveOccurred())

			resp, err := createClusterRoleBinding("test-clusterrolebinding-success", "test-clusterrole-success")
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusCreated), string(body))

			defer func() {
				_, err := makeRequest("DELETE", "/apis/rbac.authorization.k8s.io/v1/clusterrolebindings/test-clusterrolebinding-success", nil)
				Expect(err).NotTo(HaveOccurred())
				_, err = makeRequest("DELETE", "/apis/rbac.authorization.k8s.io/v1/clusterroles/test-clusterrole-success", nil)
				Expect(err).NotTo(HaveOccurred())
			}()
		})

		It("创建失败 - 字段有误", func() {
			_, err := createClusterRole("test-clusterrole-field-error")
			Expect(err).NotTo(HaveOccurred())

			defer func() {
				_, err := makeRequest("DELETE", "/apis/rbac.authorization.k8s.io/v1/clusterroles/test-clusterrole-field-error", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			clusterRoleBindingYAML := map[string]interface{}{
				"apiVersion": "rbac.authorization.k8s.io/v1",
				"kind":       "ClusterRoleBinding",
				"metadata": map[string]interface{}{
					"name": "test-clusterrolebinding-invalid-role",
				},
				"subjects": []map[string]interface{}{
					{
						"kind":     "User",
						"name":     "test-user",
						"apiGroup": "rbac.authorization.k8s.io",
					},
				},
				"roleRef": map[string]interface{}{
					"kind":     "InvalidRole",
					"name":     "test-clusterrole-field-error",
					"apiGroup": "rbac.authorization.k8s.io",
				},
			}

			resp, err := makeRequest("POST", "/apis/rbac.authorization.k8s.io/v1/clusterrolebindings", clusterRoleBindingYAML)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusInternalServerError), string(body)) // InternalServerError due to unsupported role
		})

		It("创建失败 - 已存在同名", func() {
			_, err := createClusterRole("test-clusterrole-existing-binding")
			Expect(err).NotTo(HaveOccurred())

			resp1, err := createClusterRoleBinding("test-clusterrolebinding-existing-binding", "test-clusterrole-existing-binding")
			Expect(err).NotTo(HaveOccurred())
			defer resp1.Body.Close()
			Expect(resp1.StatusCode).To(Equal(http.StatusCreated))

			defer func() {
				_, err := makeRequest("DELETE", "/apis/rbac.authorization.k8s.io/v1/clusterrolebindings/test-clusterrolebinding-existing-binding", nil)
				Expect(err).NotTo(HaveOccurred())
				_, err = makeRequest("DELETE", "/apis/rbac.authorization.k8s.io/v1/clusterroles/test-clusterrole-existing-binding", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			resp2, err := createClusterRoleBinding("test-clusterrolebinding-existing-binding", "test-clusterrole-existing-binding")
			Expect(err).NotTo(HaveOccurred())
			defer resp2.Body.Close()
			body, err := io.ReadAll(resp2.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp2.StatusCode).To(Equal(http.StatusConflict), string(body))
		})
	})

	Context("ClusterRoleBinding 查询", func() {
		It("查询成功", func() {
			_, err := createClusterRole("test-clusterrole-query")
			Expect(err).NotTo(HaveOccurred())
			_, err = createClusterRoleBinding("test-clusterrolebinding-query", "test-clusterrole-query")
			Expect(err).NotTo(HaveOccurred())

			defer func() {
				_, err := makeRequest("DELETE", "/apis/rbac.authorization.k8s.io/v1/clusterrolebindings/test-clusterrolebinding-query", nil)
				Expect(err).NotTo(HaveOccurred())
				_, err = makeRequest("DELETE", "/apis/rbac.authorization.k8s.io/v1/clusterroles/test-clusterrole-query", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			resp, err := makeRequest("GET", "/apis/rbac.authorization.k8s.io/v1/clusterrolebindings/test-clusterrolebinding-query", nil)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusOK), string(body))
		})
	})

	Context("ClusterRoleBinding 修改", func() {
		It("修改成功", func() {
			_, err := createClusterRole("test-clusterrole-update-success")
			Expect(err).NotTo(HaveOccurred())
			_, err = createClusterRoleBinding("test-clusterrolebinding-update-success", "test-clusterrole-update-success")
			Expect(err).NotTo(HaveOccurred())

			defer func() {
				_, err := makeRequest("DELETE", "/apis/rbac.authorization.k8s.io/v1/clusterrolebindings/test-clusterrolebinding-update-success", nil)
				Expect(err).NotTo(HaveOccurred())
				_, err = makeRequest("DELETE", "/apis/rbac.authorization.k8s.io/v1/clusterroles/test-clusterrole-update-success", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			patch := createPatch("test-clusterrolebinding-update-success", "")
			resp, err := makeRequest("PATCH", "/apis/rbac.authorization.k8s.io/v1/clusterrolebindings/test-clusterrolebinding-update-success", patch)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusOK), string(body))
		})

		It("修改失败 - 字段有误", func() {
			_, err := createClusterRole("test-clusterrole-field-error")
			Expect(err).NotTo(HaveOccurred())
			_, err = createClusterRoleBinding("test-clusterrolebinding-field-error", "test-clusterrole-field-error")
			Expect(err).NotTo(HaveOccurred())

			defer func() {
				_, err := makeRequest("DELETE", "/apis/rbac.authorization.k8s.io/v1/clusterrolebindings/test-clusterrolebinding-field-error", nil)
				Expect(err).NotTo(HaveOccurred())
				_, err = makeRequest("DELETE", "/apis/rbac.authorization.k8s.io/v1/clusterroles/test-clusterrole-field-error", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			patch := createMalformedPatch("test-clusterrolebinding-field-error", "")
			resp, err := makeRequest("PATCH", "/apis/rbac.authorization.k8s.io/v1/clusterrolebindings/test-clusterrolebinding-field-error", patch)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusUnprocessableEntity), string(body))
		})

		It("修改其他同类资源", func() {
			_, err := createClusterRole("test-clusterrole-other-resource")
			Expect(err).NotTo(HaveOccurred())
			_, err = createClusterRole("other-clusterrole")
			Expect(err).NotTo(HaveOccurred())

			defer func() {
				_, err := makeRequest("DELETE", "/apis/rbac.authorization.k8s.io/v1/clusterroles/test-clusterrole-other-resource", nil)
				Expect(err).NotTo(HaveOccurred())
				_, err = makeRequest("DELETE", "/apis/rbac.authorization.k8s.io/v1/clusterroles/other-clusterrole", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			patch := createPatch("other-clusterrole", "")
			resp, err := makeRequest("PATCH", "/apis/rbac.authorization.k8s.io/v1/clusterroles/test-clusterrole-other-resource", patch)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusBadRequest), string(body))
		})
	})

	Context("ClusterRoleBinding 删除", func() {
		It("删除成功", func() {
			_, err := createClusterRole("test-clusterrole-delete")
			Expect(err).NotTo(HaveOccurred())
			_, err = createClusterRoleBinding("test-clusterrolebinding-delete", "test-clusterrole-delete")
			Expect(err).NotTo(HaveOccurred())

			defer func() {
				_, err = makeRequest("DELETE", "/apis/rbac.authorization.k8s.io/v1/clusterroles/test-clusterrole-delete", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			resp, err := makeRequest("DELETE", "/apis/rbac.authorization.k8s.io/v1/clusterrolebindings/test-clusterrolebinding-delete", nil)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusOK), string(body))
		})
	})
})

var _ = Describe("ServiceAccount 资源接口测试", Label("with-workload-cluster"), func() {
	Context("ServiceAccount 创建", func() {
		It("创建成功", func() {
			resp, err := createServiceAccount("test-serviceaccount-success", "default")
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusCreated), string(body))

			defer func() {
				_, err := makeRequest("DELETE", "/api/v1/namespaces/default/serviceaccounts/test-serviceaccount-success", nil)
				Expect(err).NotTo(HaveOccurred())
			}()
		})

		It("创建失败 - 字段有误", func() {
			serviceAccountYAML := map[string]interface{}{
				"apiVersion": "v1",
				"kind":       "ServiceAccount",
				"metadata": map[string]interface{}{
					"name":      "test-serviceaccount-invalid-secret",
					"namespace": "default",
					"labels":    "app: test",
				},
			}

			resp, err := makeRequest("POST", "/api/v1/namespaces/default/serviceaccounts", serviceAccountYAML)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusBadRequest), string(body))
		})

		It("创建失败 - 已存在同名", func() {

			resp1, err := createServiceAccount("test-existing-serviceaccount", "default")
			Expect(err).NotTo(HaveOccurred())
			defer resp1.Body.Close()
			Expect(resp1.StatusCode).To(Equal(http.StatusCreated))

			defer func() {
				_, err := makeRequest("DELETE", "/api/v1/namespaces/default/serviceaccounts/test-existing-serviceaccount", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			resp2, err := createServiceAccount("test-existing-serviceaccount", "default")
			Expect(err).NotTo(HaveOccurred())
			defer resp2.Body.Close()
			body, err := io.ReadAll(resp2.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp2.StatusCode).To(Equal(http.StatusConflict), string(body))
		})

		It("创建失败 - Namespace 不存在", func() {
			resp, err := createServiceAccount("test-serviceaccount-nonexistent-ns", "nonexistent-namespace")
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusNotFound), string(body))
		})
	})

	Context("ServiceAccount 查询", func() {
		It("查询成功", func() {
			_, err := createServiceAccount("test-serviceaccount-query", "default")
			Expect(err).NotTo(HaveOccurred())

			defer func() {
				_, err := makeRequest("DELETE", "/api/v1/namespaces/default/serviceaccounts/test-serviceaccount-query", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			resp, err := makeRequest("GET", "/api/v1/namespaces/default/serviceaccounts/test-serviceaccount-query", nil)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusOK), string(body))
		})
	})

	Context("ServiceAccount 修改", func() {
		It("修改成功", func() {
			_, err := createServiceAccount("test-serviceaccount-update-success", "default")
			Expect(err).NotTo(HaveOccurred())

			defer func() {
				_, err := makeRequest("DELETE", "/api/v1/namespaces/default/serviceaccounts/test-serviceaccount-update-success", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			patch := createPatch("test-serviceaccount-update-success", "default")

			resp, err := makeRequest("PATCH", "/api/v1/namespaces/default/serviceaccounts/test-serviceaccount-update-success", patch)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusOK), string(body))
		})

		It("修改失败 - 字段有误", func() {
			_, err := createServiceAccount("test-serviceaccount-field-error", "default")
			Expect(err).NotTo(HaveOccurred())

			defer func() {
				_, err := makeRequest("DELETE", "/api/v1/namespaces/default/serviceaccounts/test-serviceaccount-field-error", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			patch := createMalformedPatch("test-serviceaccount-field-error", "default")
			resp, err := makeRequest("PATCH", "/api/v1/namespaces/default/serviceaccounts/test-serviceaccount-field-error", patch)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusUnprocessableEntity), string(body))
		})

		It("修改同一命名空间其他同类资源", func() {
			_, err := createServiceAccount("test-serviceaccount-other-resource", "default")
			Expect(err).NotTo(HaveOccurred())
			_, err = createServiceAccount("other-serviceaccount", "default")
			Expect(err).NotTo(HaveOccurred())

			defer func() {
				_, err := makeRequest("DELETE", "/api/v1/namespaces/default/serviceaccounts/test-serviceaccount-other-resource", nil)
				Expect(err).NotTo(HaveOccurred())
				_, err = makeRequest("DELETE", "/api/v1/namespaces/default/serviceaccounts/other-serviceaccount", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			patch := createPatch("other-serviceaccount", "default")
			resp, err := makeRequest("PATCH", "/api/v1/namespaces/default/serviceaccounts/test-serviceaccount-other-resource", patch)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusBadRequest), string(body))
		})

		It("修改不存在的Namespace", func() {
			_, err := createServiceAccount("test-serviceaccount-nonexistent-ns-update", "default")
			Expect(err).NotTo(HaveOccurred())

			defer func() {
				_, err := makeRequest("DELETE", "/api/v1/namespaces/default/serviceaccounts/test-serviceaccount-nonexistent-ns-update", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			patch := createPatch("test-serviceaccount-nonexistent-ns-update", "nonexistent-namespace")
			resp, err := makeRequest("PATCH", "/api/v1/namespaces/default/serviceaccounts/test-serviceaccount-nonexistent-ns-update", patch)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusBadRequest), string(body))
		})
	})

	Context("ServiceAccount 删除", func() {
		It("删除成功", func() {
			_, err := createServiceAccount("test-serviceaccount-delete", "default")
			Expect(err).NotTo(HaveOccurred())

			defer func() {
				_, err := makeRequest("DELETE", "/api/v1/namespaces/default/serviceaccounts/test-serviceaccount-delete", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			resp, err := makeRequest("DELETE", "/api/v1/namespaces/default/serviceaccounts/test-serviceaccount-delete", nil)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusOK), string(body))
		})
	})
})
