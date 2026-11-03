package console

import (
	"encoding/json"
	"io"
	"net/http"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

var _ = Describe("PersistentVolume 资源接口测试", Label("with-workload-cluster"), func() {
	Context("PersistentVolume 创建", func() {
		It("创建成功", func() {
			resp, err := createPersistentVolume("test-pv-success")
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusCreated), string(body))

			defer func() {
				_, err := makeRequest("DELETE", "/api/v1/persistentvolumes/test-pv-success", nil)
				Expect(err).NotTo(HaveOccurred())
			}()
		})

		It("创建失败 - 字段有误", func() {
			pvYAML := map[string]interface{}{
				"apiVersion": "v1",
				"kind":       "PersistentVolume",
				"metadata": map[string]interface{}{
					"name": "test-pv-invalid-capacity",
				},
				"spec": map[string]interface{}{
					"capacity": map[string]interface{}{
						"storage": -10,
					},
					"accessModes": []string{
						"ReadWriteOnce",
					},
					"persistentVolumeReclaimPolicy": "Retain",
					"hostPath": map[string]interface{}{
						"path": "/tmp/data",
					},
				},
			}

			resp, err := makeRequest("POST", "/api/v1/persistentvolumes", pvYAML)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusUnprocessableEntity), string(body))
		})

		It("创建失败 - 已存在同名", func() {
			resp1, err := createPersistentVolume("test-existing-pv")
			Expect(err).NotTo(HaveOccurred())
			defer resp1.Body.Close()
			Expect(resp1.StatusCode).To(Equal(http.StatusCreated))

			defer func() {
				_, err := makeRequest("DELETE", "/api/v1/persistentvolumes/test-existing-pv", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			resp2, err := createPersistentVolume("test-existing-pv")
			Expect(err).NotTo(HaveOccurred())
			defer resp2.Body.Close()
			body, err := io.ReadAll(resp2.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp2.StatusCode).To(Equal(http.StatusConflict), string(body))
		})
	})

	Context("PersistentVolume 查询", func() {
		It("查询成功", func() {
			_, err := createPersistentVolume("test-pv-query")
			Expect(err).NotTo(HaveOccurred())

			defer func() {
				_, err := makeRequest("DELETE", "/api/v1/persistentvolumes/test-pv-query", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			resp, err := makeRequest("GET", "/api/v1/persistentvolumes/test-pv-query", nil)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusOK), string(body))

			var pv map[string]interface{}
			err = json.Unmarshal(body, &pv)
			Expect(err).NotTo(HaveOccurred())
			Expect(pv["metadata"].(map[string]interface{})["name"]).To(Equal("test-pv-query"))
		})
	})

	Context("PersistentVolume 修改", func() {
		It("修改成功", func() {
			_, err := createPersistentVolume("test-pv-update-success")
			Expect(err).NotTo(HaveOccurred())

			defer func() {
				_, err := makeRequest("DELETE", "/api/v1/persistentvolumes/test-pv-update-success", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			patch := createPatch("test-pv-update-success", "")

			resp, err := makeRequest("PATCH", "/api/v1/persistentvolumes/test-pv-update-success", patch)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusOK), string(body))
		})

		It("修改失败 - 字段有误", func() {
			_, err := createPersistentVolume("test-pv-field-error")
			Expect(err).NotTo(HaveOccurred())

			defer func() {
				_, err := makeRequest("DELETE", "/api/v1/persistentvolumes/test-pv-field-error", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			patch := createMalformedPatch("test-pv-field-error", "")
			resp, err := makeRequest("PATCH", "/api/v1/persistentvolumes/test-pv-field-error", patch)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusUnprocessableEntity), string(body))
		})

		It("修改其他同类资源", func() {
			_, err := createPersistentVolume("test-pv-other-resource")
			Expect(err).NotTo(HaveOccurred())
			_, err = createPersistentVolume("other-pv")
			Expect(err).NotTo(HaveOccurred())

			defer func() {
				_, err := makeRequest("DELETE", "/api/v1/persistentvolumes/test-pv-other-resource", nil)
				Expect(err).NotTo(HaveOccurred())
				_, err = makeRequest("DELETE", "/api/v1/persistentvolumes/other-pv", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			patch := createPatch("other-pv", "")
			resp, err := makeRequest("PATCH", "/api/v1/persistentvolumes/test-pv-other-resource", patch)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusBadRequest), string(body))
		})
	})

	Context("PersistentVolume 删除", func() {
		It("删除成功", func() {
			_, err := createPersistentVolume("test-pv-delete")
			Expect(err).NotTo(HaveOccurred())

			resp, err := makeRequest("DELETE", "/api/v1/persistentvolumes/test-pv-delete", nil)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusOK), string(body))
		})
	})
})

var _ = Describe("PersistentVolumeClaim 资源接口测试", Label("with-workload-cluster"), func() {
	Context("PersistentVolumeClaim 创建", func() {
		It("创建成功", func() {
			resp, err := createPersistentVolumeClaim("test-pvc-success", "default")
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusCreated), string(body))

			defer func() {
				_, err := makeRequest("DELETE", "/api/v1/namespaces/default/persistentvolumeclaims/test-pvc-success", nil)
				Expect(err).NotTo(HaveOccurred())
			}()
		})

		It("创建失败 - 字段有误", func() {
			pvcYAML := map[string]interface{}{
				"apiVersion": "v1",
				"kind":       "PersistentVolumeClaim",
				"metadata": map[string]interface{}{
					"name":      "test-pvc-invalid-storage",
					"namespace": "default",
				},
				"spec": map[string]interface{}{
					"accessModes": []string{
						"ReadWriteOnce",
					},
					"resources": map[string]interface{}{
						"requests": map[string]interface{}{
							"storage": "invalid-size", // 格式错误
						},
					},
				},
			}

			resp, err := makeRequest("POST", "/api/v1/namespaces/default/persistentvolumeclaims", pvcYAML)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusBadRequest), string(body))
		})

		It("创建失败 - 已存在同名", func() {
			resp1, err := createPersistentVolumeClaim("test-existing-pvc", "default")
			Expect(err).NotTo(HaveOccurred())
			defer resp1.Body.Close()
			Expect(resp1.StatusCode).To(Equal(http.StatusCreated))

			defer func() {
				_, err := makeRequest("DELETE", "/api/v1/namespaces/default/persistentvolumeclaims/test-existing-pvc", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			resp2, err := createPersistentVolumeClaim("test-existing-pvc", "default")
			Expect(err).NotTo(HaveOccurred())
			defer resp2.Body.Close()
			body, err := io.ReadAll(resp2.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp2.StatusCode).To(Equal(http.StatusConflict), string(body))
		})

		It("创建失败 - Namespace 不存在", func() {
			resp, err := createPersistentVolumeClaim("test-pvc-nonexistent-ns", "nonexistent-namespace")
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusNotFound), string(body))
		})
	})

	Context("PersistentVolumeClaim 查询", func() {
		It("查询成功", func() {
			_, err := createPersistentVolumeClaim("test-pvc-query", "default")
			Expect(err).NotTo(HaveOccurred())

			defer func() {
				_, err := makeRequest("DELETE", "/api/v1/namespaces/default/persistentvolumeclaims/test-pvc-query", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			resp, err := makeRequest("GET", "/api/v1/namespaces/default/persistentvolumeclaims/test-pvc-query", nil)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusOK), string(body))

			var pvc map[string]interface{}
			err = json.Unmarshal(body, &pvc)
			Expect(err).NotTo(HaveOccurred())
			Expect(pvc["metadata"].(map[string]interface{})["name"]).To(Equal("test-pvc-query"))
		})
	})

	Context("PersistentVolumeClaim 修改", func() {
		It("修改成功", func() {
			_, err := createPersistentVolumeClaim("test-pvc-update-success", "default")
			Expect(err).NotTo(HaveOccurred())

			defer func() {
				_, err := makeRequest("DELETE", "/api/v1/namespaces/default/persistentvolumeclaims/test-pvc-update-success", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			patch := createPatch("test-pvc-update-success", "default")
			resp, err := makeRequest("PATCH", "/api/v1/namespaces/default/persistentvolumeclaims/test-pvc-update-success", patch)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusOK), string(body))
		})

		It("修改失败 - 字段有误", func() {
			_, err := createPersistentVolumeClaim("test-pvc-field-error", "default")
			Expect(err).NotTo(HaveOccurred())

			defer func() {
				_, err := makeRequest("DELETE", "/api/v1/namespaces/default/persistentvolumeclaims/test-pvc-field-error", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			patch := createMalformedPatch("test-pvc-field-error", "default")
			resp, err := makeRequest("PATCH", "/api/v1/namespaces/default/persistentvolumeclaims/test-pvc-field-error", patch)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusUnprocessableEntity), string(body))
		})

		It("修改其他同类资源", func() {
			_, err := createPersistentVolumeClaim("test-pvc-other-resource", "default")
			Expect(err).NotTo(HaveOccurred())
			_, err = createPersistentVolumeClaim("other-pvc", "default")
			Expect(err).NotTo(HaveOccurred())

			defer func() {
				_, err := makeRequest("DELETE", "/api/v1/namespaces/default/persistentvolumeclaims/test-pvc-other-resource", nil)
				Expect(err).NotTo(HaveOccurred())
				_, err = makeRequest("DELETE", "/api/v1/namespaces/default/persistentvolumeclaims/other-pvc", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			patch := createPatch("other-pvc", "default")
			resp, err := makeRequest("PATCH", "/api/v1/namespaces/default/persistentvolumeclaims/test-pvc-other-resource", patch)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusBadRequest), string(body))
		})

		It("修改不存在的Namespace", func() {
			_, err := createPersistentVolumeClaim("test-pvc-nonexistent-ns-update", "default")
			Expect(err).NotTo(HaveOccurred())

			defer func() {
				_, err := makeRequest("DELETE", "/api/v1/namespaces/default/persistentvolumeclaims/test-pvc-nonexistent-ns-update", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			patch := createPatch("test-pvc-nonexistent-ns-update", "nonexistent-namespace")
			resp, err := makeRequest("PATCH", "/api/v1/namespaces/default/persistentvolumeclaims/test-pvc-nonexistent-ns-update", patch)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusBadRequest), string(body))
		})
	})

	Context("PersistentVolumeClaim 删除", func() {
		It("删除成功", func() {
			_, err := createPersistentVolumeClaim("test-pvc-delete", "default")
			Expect(err).NotTo(HaveOccurred())

			resp, err := makeRequest("DELETE", "/api/v1/namespaces/default/persistentvolumeclaims/test-pvc-delete", nil)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusOK), string(body))
		})
	})
})

var _ = Describe("StorageClass 资源接口测试", Label("with-workload-cluster"), func() {
	Context("StorageClass 创建", func() {
		It("创建成功", func() {
			resp, err := createStorageClass("test-storageclass-success")
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusCreated), string(body))

			defer func() {
				_, err := makeRequest("DELETE", "/apis/storage.k8s.io/v1/storageclasses/test-storageclass-success", nil)
				Expect(err).NotTo(HaveOccurred())
			}()
		})

		It("创建失败 - 字段有误", func() {
			scYAML := map[string]interface{}{
				"apiVersion": "storage.k8s.io/v1",
				"kind":       "StorageClass",
				"metadata": map[string]interface{}{
					"name": "test-storageclass-empty-provisioner",
				},
				"provisioner": "", // 空的 provisioner
			}

			resp, err := makeRequest("POST", "/apis/storage.k8s.io/v1/storageclasses", scYAML)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusUnprocessableEntity), string(body))
		})

		It("创建失败 - 已存在同名", func() {
			resp1, err := createStorageClass("test-existing-storageclass")
			Expect(err).NotTo(HaveOccurred())
			defer resp1.Body.Close()
			Expect(resp1.StatusCode).To(Equal(http.StatusCreated))

			defer func() {
				_, err := makeRequest("DELETE", "/apis/storage.k8s.io/v1/storageclasses/test-existing-storageclass", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			resp2, err := createStorageClass("test-existing-storageclass")
			Expect(err).NotTo(HaveOccurred())
			defer resp2.Body.Close()
			body, err := io.ReadAll(resp2.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp2.StatusCode).To(Equal(http.StatusConflict), string(body))
		})
	})

	Context("StorageClass 查询", func() {
		It("查询成功", func() {
			_, err := createStorageClass("test-storageclass-query")
			Expect(err).NotTo(HaveOccurred())

			defer func() {
				_, err := makeRequest("DELETE", "/apis/storage.k8s.io/v1/storageclasses/test-storageclass-query", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			resp, err := makeRequest("GET", "/apis/storage.k8s.io/v1/storageclasses/test-storageclass-query", nil)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusOK), string(body))

			var sc map[string]interface{}
			err = json.Unmarshal(body, &sc)
			Expect(err).NotTo(HaveOccurred())
			Expect(sc["metadata"].(map[string]interface{})["name"]).To(Equal("test-storageclass-query"))
		})
	})

	Context("StorageClass 修改", func() {
		It("修改成功", func() {
			_, err := createStorageClass("test-storageclass-update-success")
			Expect(err).NotTo(HaveOccurred())

			defer func() {
				_, err := makeRequest("DELETE", "/apis/storage.k8s.io/v1/storageclasses/test-storageclass-update-success", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			patch := createPatch("test-storageclass-update-success", "")
			resp, err := makeRequest("PATCH", "/apis/storage.k8s.io/v1/storageclasses/test-storageclass-update-success", patch)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusOK), string(body))
		})

		It("修改失败 - 字段有误", func() {
			_, err := createStorageClass("test-storageclass-field-error")
			Expect(err).NotTo(HaveOccurred())

			defer func() {
				_, err := makeRequest("DELETE", "/apis/storage.k8s.io/v1/storageclasses/test-storageclass-field-error", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			patch := createMalformedPatch("test-storageclass-field-error", "")
			resp, err := makeRequest("PATCH", "/apis/storage.k8s.io/v1/storageclasses/test-storageclass-field-error", patch)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusUnprocessableEntity), string(body))
		})

		It("修改其他同类资源", func() {
			_, err := createStorageClass("test-storageclass-other-resource")
			Expect(err).NotTo(HaveOccurred())
			_, err = createStorageClass("other-storageclass")
			Expect(err).NotTo(HaveOccurred())

			defer func() {
				_, err := makeRequest("DELETE", "/apis/storage.k8s.io/v1/storageclasses/test-storageclass-other-resource", nil)
				Expect(err).NotTo(HaveOccurred())
				_, err = makeRequest("DELETE", "/apis/storage.k8s.io/v1/storageclasses/other-storageclass", nil)
				Expect(err).NotTo(HaveOccurred())
			}()

			patch := createPatch("other-storageclass", "")
			resp, err := makeRequest("PATCH", "/apis/storage.k8s.io/v1/storageclasses/test-storageclass-other-resource", patch)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusBadRequest), string(body))
		})
	})

	Context("StorageClass 删除", func() {
		It("删除成功", func() {
			_, err := createStorageClass("test-storageclass-delete")
			Expect(err).NotTo(HaveOccurred())

			resp, err := makeRequest("DELETE", "/apis/storage.k8s.io/v1/storageclasses/test-storageclass-delete", nil)
			Expect(err).NotTo(HaveOccurred())
			defer resp.Body.Close()
			body, err := io.ReadAll(resp.Body)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusOK), string(body))
		})
	})
})
