package local

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"

	"plugin-management-service/pkg/constant"
	"plugin-management-service/pkg/utils/httputil"
)

var _ = Describe("Plugin Management Functional Test", func() {
	var (
		pluginName = "test-plugin-functional"
		pluginGVR  = schema.GroupVersionResource{
			Group:    "console.openfuyao.com",
			Version:  "v1beta1",
			Resource: "consoleplugins",
		}
	)

	createPluginResource := func(name string, enabled bool) {
		plugin := &unstructured.Unstructured{
			Object: map[string]interface{}{
				"apiVersion": "console.openfuyao.com/v1beta1",
				"kind":       "ConsolePlugin",
				"metadata": map[string]interface{}{
					"name": name,
					"annotations": map[string]interface{}{
						"meta.helm.sh/release-name": "test-release-123",
					},
				},
				"spec": map[string]interface{}{
					"pluginName":  name,
					"displayName": "测试扩展组件",
					"entrypoint":  "/test-entry",
					"enabled":     enabled,
					"backend": map[string]interface{}{
						"type": "Service",
						"service": map[string]interface{}{
							"name":      "test-svc",
							"namespace": "default",
							"port":      int64(80),
						},
					},
				},
			},
		}
		_, err := client.Resource(pluginGVR).Create(context.Background(), plugin, metav1.CreateOptions{})
		Expect(err).NotTo(HaveOccurred())
	}

	deletePluginResource := func(name string) {
		_ = client.Resource(pluginGVR).Delete(context.Background(), name, metav1.DeleteOptions{})
	}

	BeforeEach(func() {
		deletePluginResource(pluginName)
	})

	Context("【应用运维-扩展详情-001】详情基础信息展示", func() {
		It("当插件存在时，GET 详情接口应返回正确的元数据和状态", func() {
			createPluginResource(pluginName, true)

			url := fmt.Sprintf("%s/rest/plugin-management/v1beta1/consoleplugins/%s", baseURL, pluginName)
			resp, err := http.Get(url)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(int(http.StatusOK)))

			var result httputil.ResponseJson
			err = json.NewDecoder(resp.Body).Decode(&result)
			Expect(err).NotTo(HaveOccurred())
			Expect(result.Code).To(Equal(int32(constant.Success)))

			data := result.Data.(map[string]interface{})
			Expect(data["pluginName"]).To(Equal(pluginName))
			Expect(data["displayName"]).To(Equal("测试扩展组件"))
			Expect(data["enabled"]).To(BeTrue())
			Expect(data["release"]).To(Equal("test-release-123"))
		})
	})

	Context("【应用运维-扩展详情-002/024】扩展界面启停与编辑生效", func() {
		It("发送 POST 请求修改 enabled 状态，应能持久化到 K8s 并同步通过 GET 接口查看到", func() {
			// 1. 预置启用状态
			createPluginResource(pluginName, true)

			// 2. 发起停用请求 (模拟点击开关或编辑保存)
			url := fmt.Sprintf("%s/rest/plugin-management/v1beta1/consoleplugins/%s/enabled", baseURL, pluginName)
			body, _ := json.Marshal(map[string]interface{}{
				"pluginName": pluginName,
				"enabled":    false,
			})
			
			req, _ := http.NewRequest(http.MethodPost, url, bytes.NewBuffer(body))
			req.Header.Set("Content-Type", "application/json")
			
			resp, err := http.DefaultClient.Do(req)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.StatusCode).To(Equal(http.StatusOK))

			// 3. 校验 K8s 资源层面的持久化结果 (Mock 外部依赖的真实反馈)
			updated, err := client.Resource(pluginGVR).Get(context.Background(), pluginName, metav1.GetOptions{})
			Expect(err).NotTo(HaveOccurred())
			enabled, found, _ := unstructured.NestedBool(updated.Object, "spec", "enabled")
			Expect(found).To(BeTrue())
			Expect(enabled).To(BeFalse())

			// 4. 校验业务 GET 接口返回的一致性
			respGet, _ := http.Get(url) // 访问 /enabled 接口
			var result httputil.ResponseJson
			json.NewDecoder(respGet.Body).Decode(&result)
			Expect(result.Data.(map[string]interface{})["enabled"]).To(BeFalse())
		})
	})
})
