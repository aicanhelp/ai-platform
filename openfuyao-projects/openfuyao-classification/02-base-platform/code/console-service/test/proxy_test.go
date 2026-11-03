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

package test

import (
	"context"
	"io"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"

	"console-service/pkg/auth"
	"console-service/pkg/constant"
	"console-service/pkg/plugin"
)

var _ = Describe("验证检查", func() {

	It("未登录", func() {
		req, err := http.NewRequest("GET", serverAddr, nil)
		Expect(err).NotTo(HaveOccurred())

		resp, err := testHttpClient.Do(req)
		Expect(err).NotTo(HaveOccurred())
		defer resp.Body.Close()

		Expect(resp.StatusCode).To(Equal(http.StatusSeeOther))
		location := resp.Header.Get("Location")
		u, err := url.Parse(location)
		Expect(err).NotTo(HaveOccurred())
		Expect(u.Path).To(Equal("/rest/auth/login"))
	})

	It("无效的sessionID", func() {
		// 创建临时的过期 session
		err := auth.StoreSession(k8sClient, expiredSession, false)
		Expect(err).NotTo(HaveOccurred())

		req, err := http.NewRequest("GET", serverAddr, nil)
		Expect(err).NotTo(HaveOccurred())
		req.AddCookie(&http.Cookie{Name: "sessionID", Value: "non-existing-sessionID"})

		resp, err := testHttpClient.Do(req)
		Expect(err).NotTo(HaveOccurred())
		defer resp.Body.Close()

		Expect(resp.StatusCode).To(Equal(http.StatusSeeOther))
		location := resp.Header.Get("Location")
		u, err := url.Parse(location)
		Expect(err).NotTo(HaveOccurred())
		Expect(u.Path).To(Equal("/rest/auth/login"))
	})

	It("有效的sessionID cookie", func() {
		// 创建可刷新的 session
		err := auth.StoreSession(k8sClient, validSession, false)
		Expect(err).NotTo(HaveOccurred())

		req, err := http.NewRequest("GET", serverAddr, nil)
		Expect(err).NotTo(HaveOccurred())
		req.AddCookie(&http.Cookie{Name: "sessionID", Value: validSession.GetSessionID()})

		resp, err := testHttpClient.Do(req)
		Expect(err).NotTo(HaveOccurred())
		defer resp.Body.Close()

		Expect(resp.StatusCode).To(Equal(http.StatusOK))
		body, err := io.ReadAll(resp.Body)
		Expect(err).NotTo(HaveOccurred())
		Expect(string(body)).To(ContainSubstring("console-website"))
	})

	It("有效的sessionID cookie, token 需刷新", func() {
		// 创建可刷新的 session
		err := auth.StoreSession(k8sClient, refreshableSession, false)
		Expect(err).NotTo(HaveOccurred())

		req, err := http.NewRequest("GET", serverAddr, nil)
		Expect(err).NotTo(HaveOccurred())
		req.AddCookie(&http.Cookie{Name: "sessionID", Value: refreshableSession.GetSessionID()})

		resp, err := testHttpClient.Do(req)
		Expect(err).NotTo(HaveOccurred())
		defer resp.Body.Close()

		Expect(resp.StatusCode).To(Equal(http.StatusOK))
		body, err := io.ReadAll(resp.Body)
		Expect(err).NotTo(HaveOccurred())
		Expect(string(body)).To(ContainSubstring("console-website"))
	})
})

var _ = Describe("接口转发", func() {

	It("kube-apiserver", func() {
		req, err := http.NewRequest("GET", serverAddr+"/clusters/default/api/kubernetes/api/v1/namespaces", nil)
		Expect(err).NotTo(HaveOccurred())
		req.AddCookie(&http.Cookie{Name: "sessionID", Value: validSession.GetSessionID()})

		resp, err := testHttpClient.Do(req)
		Expect(err).NotTo(HaveOccurred())
		defer resp.Body.Close()

		Expect(resp.StatusCode).To(Equal(http.StatusUnauthorized)) // 没有真实 bearer token, 预期 401
		body, err := io.ReadAll(resp.Body)
		Expect(err).NotTo(HaveOccurred())
		Expect(string(body)).To(ContainSubstring("Unauthorized"))
	})

	It("kube-apiserver多集群", func() {
		// 创建 mock karmada 资源，模拟多集群环境
		_, err := k8sClient.CoreV1().Namespaces().Create(context.Background(), &corev1.Namespace{
			ObjectMeta: metav1.ObjectMeta{
				Name: constant.KaramdaNamespace,
			},
		}, metav1.CreateOptions{})
		Expect(err).NotTo(HaveOccurred())
		_, err = k8sClient.CoreV1().Services(constant.KaramdaNamespace).Create(context.Background(), &corev1.Service{
			ObjectMeta: metav1.ObjectMeta{
				Name:      constant.KaramdaAPIServer,
				Namespace: constant.KaramdaNamespace,
			},
			Spec: corev1.ServiceSpec{
				Ports: []corev1.ServicePort{
					{
						Name:     "https",
						Port:     443,
						Protocol: corev1.ProtocolTCP,
					},
				},
			},
		}, metav1.CreateOptions{})
		Expect(err).NotTo(HaveOccurred())
		defer func() {
			err := k8sClient.CoreV1().Services(constant.KaramdaNamespace).Delete(context.Background(), constant.KaramdaAPIServer, metav1.DeleteOptions{})
			Expect(err).NotTo(HaveOccurred())
			err = k8sClient.CoreV1().Namespaces().Delete(context.Background(), constant.KaramdaNamespace, metav1.DeleteOptions{})
			Expect(err).NotTo(HaveOccurred())
		}()

		req, err := http.NewRequest("GET", serverAddr+"/clusters/default/api/kubernetes/api/v1/namespaces", nil)
		Expect(err).NotTo(HaveOccurred())
		req.AddCookie(&http.Cookie{Name: "sessionID", Value: validSession.GetSessionID()})

		resp, err := testHttpClient.Do(req)
		Expect(err).NotTo(HaveOccurred())
		defer resp.Body.Close()
		Expect(resp.StatusCode).To(Equal(http.StatusInternalServerError))
	})

	It("告警", func() {
		req, err := http.NewRequest("GET", serverAddr+"/rest/alert/v1/test", nil)
		Expect(err).NotTo(HaveOccurred())
		req.AddCookie(&http.Cookie{Name: "sessionID", Value: validSession.GetSessionID()})

		resp, err := testHttpClient.Do(req)
		Expect(err).NotTo(HaveOccurred())
		defer resp.Body.Close()

		Expect(resp.StatusCode).To(Equal(http.StatusOK))
		body, err := io.ReadAll(resp.Body)
		Expect(err).NotTo(HaveOccurred())
		Expect(string(body)).To(Equal("alert"))
	})

	It("监控", func() {
		req, err := http.NewRequest("GET", serverAddr+"/rest/monitoring/test", nil)
		Expect(err).NotTo(HaveOccurred())
		req.AddCookie(&http.Cookie{Name: "sessionID", Value: validSession.GetSessionID()})

		resp, err := testHttpClient.Do(req)
		Expect(err).NotTo(HaveOccurred())
		defer resp.Body.Close()

		Expect(resp.StatusCode).To(Equal(http.StatusOK))
		body, err := io.ReadAll(resp.Body)
		Expect(err).NotTo(HaveOccurred())
		Expect(string(body)).To(Equal("monitoring"))
	})

	It("webterminal", func() {
		req, err := http.NewRequest("GET", serverAddr+"/rest/webterminal/test", nil)
		Expect(err).NotTo(HaveOccurred())
		req.AddCookie(&http.Cookie{Name: "sessionID", Value: validSession.GetSessionID()})

		resp, err := testHttpClient.Do(req)
		Expect(err).NotTo(HaveOccurred())
		defer resp.Body.Close()

		Expect(resp.StatusCode).To(Equal(http.StatusOK))
		body, err := io.ReadAll(resp.Body)
		Expect(err).NotTo(HaveOccurred())
		Expect(string(body)).To(Equal("webterminal"))
	})

	It("扩展管理", func() {
		req, err := http.NewRequest("GET", serverAddr+"/rest/plugin-management/test", nil)
		Expect(err).NotTo(HaveOccurred())
		req.AddCookie(&http.Cookie{Name: "sessionID", Value: validSession.GetSessionID()})

		resp, err := testHttpClient.Do(req)
		Expect(err).NotTo(HaveOccurred())
		defer resp.Body.Close()

		Expect(resp.StatusCode).To(Equal(http.StatusOK))
		body, err := io.ReadAll(resp.Body)
		Expect(err).NotTo(HaveOccurred())
		Expect(string(body)).To(Equal("plugin-management"))
	})

	It("应用管理", func() {
		req, err := http.NewRequest("GET", serverAddr+"/rest/application-management/test", nil)
		Expect(err).NotTo(HaveOccurred())
		req.AddCookie(&http.Cookie{Name: "sessionID", Value: validSession.GetSessionID()})

		resp, err := testHttpClient.Do(req)
		Expect(err).NotTo(HaveOccurred())
		defer resp.Body.Close()

		Expect(resp.StatusCode).To(Equal(http.StatusOK))
		body, err := io.ReadAll(resp.Body)
		Expect(err).NotTo(HaveOccurred())
		Expect(string(body)).To(Equal("application-management"))
	})

	It("应用市场", func() {
		req, err := http.NewRequest("GET", serverAddr+"/rest/marketplace/test", nil)
		Expect(err).NotTo(HaveOccurred())
		req.AddCookie(&http.Cookie{Name: "sessionID", Value: validSession.GetSessionID()})

		resp, err := testHttpClient.Do(req)
		Expect(err).NotTo(HaveOccurred())
		defer resp.Body.Close()

		Expect(resp.StatusCode).To(Equal(http.StatusOK))
		body, err := io.ReadAll(resp.Body)
		Expect(err).NotTo(HaveOccurred())
		Expect(string(body)).To(Equal("marketplace"))
	})

	It("用户管理", func() {
		req, err := http.NewRequest("GET", serverAddr+"/rest/user/test", nil)
		Expect(err).NotTo(HaveOccurred())
		req.AddCookie(&http.Cookie{Name: "sessionID", Value: validSession.GetSessionID()})

		resp, err := testHttpClient.Do(req)
		Expect(err).NotTo(HaveOccurred())
		defer resp.Body.Close()

		Expect(resp.StatusCode).To(Equal(http.StatusOK))
		body, err := io.ReadAll(resp.Body)
		Expect(err).NotTo(HaveOccurred())
		Expect(string(body)).To(Equal("user-management"))
	})

	It("oauth（修改密码）", func() {
		// 修改密码会自动登出，需要创建临时的有效 session
		err := auth.StoreSession(k8sClient, &auth.SessionStore{
			"AccessToken":   []byte("xxxxxxxx"),
			"RefreshToken":  []byte("xxxxxxxx"),
			"SessionID":     []byte("xxxxxxxx"),
			"AccessExpiry":  []byte(strconv.FormatInt(time.Now().Add(2*time.Hour).Unix(), 10)),
			"RefreshExpiry": []byte(strconv.FormatInt(time.Now().Add(2*time.Hour).Unix(), 10)),
		}, false)
		Expect(err).NotTo(HaveOccurred())

		requestBody := `{"oldPassword": "oldPassword", "newPassword": "newPassword"}`
		req, err := http.NewRequest("POST", serverAddr+"/password", strings.NewReader(requestBody))
		Expect(err).NotTo(HaveOccurred())
		req.AddCookie(&http.Cookie{Name: "sessionID", Value: "xxxxxxxx"})

		resp, err := testHttpClient.Do(req)
		Expect(err).NotTo(HaveOccurred())
		defer resp.Body.Close()

		Expect(resp.StatusCode).To(Equal(http.StatusOK))
		body, err := io.ReadAll(resp.Body)
		Expect(err).NotTo(HaveOccurred())
		Expect(string(body)).To(Equal("password"))

		// 对应的 session 应被删除
		_, err = auth.GetSession(k8sClient, "xxxxxxxx")
		Expect(err).To(HaveOccurred())
	})

	It("扩展组件请求", func() {
		// 创建 ConsolePlugin CR
		testConsolePlugin := &plugin.ConsolePlugin{
			TypeMeta: metav1.TypeMeta{
				APIVersion: "console.openfuyao.com/v1beta1",
				Kind:       "ConsolePlugin",
			},
			ObjectMeta: metav1.ObjectMeta{
				Name: "test-plugin",
			},
			Spec: plugin.ConsolePluginSpec{
				PluginName:  "test-plugin",
				DisplayName: "Test Plugin",
				Entrypoint:  "/",
				Backend: &plugin.ConsolePluginBackend{
					Type: plugin.ServiceBackendType,
					Service: &plugin.ConsolePluginService{
						Name:      "test-plugin",
						Namespace: "test-namespace",
						BasePath:  "/test-base-path",
						Port:      8080,
					},
				},
				Enabled: true,
			},
		}

		// 将 ConsolePlugin 转换为 unstructured 对象
		pluginUnstructured, err := runtime.DefaultUnstructuredConverter.ToUnstructured(testConsolePlugin)
		Expect(err).NotTo(HaveOccurred())

		unstructuredObj := &unstructured.Unstructured{Object: pluginUnstructured}

		// 定义 GVR
		gvr := schema.GroupVersionResource{
			Group:    "console.openfuyao.com",
			Version:  "v1beta1",
			Resource: "consoleplugins",
		}

		// 创建资源
		_, err = dynamicClient.Resource(gvr).Create(context.Background(), unstructuredObj, metav1.CreateOptions{})
		Expect(err).NotTo(HaveOccurred())

		req, err := http.NewRequest("GET", serverAddr+"/rest/test-plugin/test", nil)
		Expect(err).NotTo(HaveOccurred())
		req.AddCookie(&http.Cookie{Name: "sessionID", Value: validSession.GetSessionID()})

		resp, err := testHttpClient.Do(req)
		Expect(err).NotTo(HaveOccurred())
		defer resp.Body.Close()

		Expect(resp.StatusCode).To(Equal(http.StatusNotFound)) // 暂时无法 mock 扩展组件后端，预期 404
	})

	It("前端静态资源", func() {
		req, err := http.NewRequest("GET", serverAddr+"/test", nil)
		Expect(err).NotTo(HaveOccurred())
		req.AddCookie(&http.Cookie{Name: "sessionID", Value: validSession.GetSessionID()})

		resp, err := testHttpClient.Do(req)
		Expect(err).NotTo(HaveOccurred())
		defer resp.Body.Close()

		Expect(resp.StatusCode).To(Equal(http.StatusOK))
		body, err := io.ReadAll(resp.Body)
		Expect(err).NotTo(HaveOccurred())
		Expect(string(body)).To(Equal("console-website"))
	})
})
