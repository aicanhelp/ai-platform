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
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"strconv"
	"testing"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	corev1 "k8s.io/api/core/v1"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	crdclientset "k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"sigs.k8s.io/controller-runtime/pkg/envtest"
	sigsyaml "sigs.k8s.io/yaml"

	"console-service/cmd/config"
	"console-service/pkg/auth"
	"console-service/pkg/client/k8s"
	"console-service/pkg/server"
	"console-service/pkg/server/runtime"
	"console-service/pkg/utils/iputil"
)

const (
	serverScheme = "http"
	serverPort   = 19020
)

// serverHost prefers the IPv6 loopback to exercise IPv6 paths and falls back
// to 127.0.0.1 when the host has no IPv6 support.
var serverHost = func() string {
	if l, err := net.Listen("tcp", net.JoinHostPort("::1", "0")); err == nil {
		_ = l.Close()
		return "::1"
	}
	return "127.0.0.1"
}()

var (
	ctx            context.Context
	serverAddr     string
	testHttpClient *http.Client
	testEnv        *envtest.Environment
	k8sCfg         *rest.Config
	k8sClient      *kubernetes.Clientset
	dynamicClient  dynamic.Interface
	stopFunc       func()
)

var (
	oauthServer                 *httptest.Server
	consoleWebsiteServer        *httptest.Server
	alertServer                 *httptest.Server
	monitoringServer            *httptest.Server
	webterminalServer           *httptest.Server
	applicationManagementServer *httptest.Server
	marketplaceServer           *httptest.Server
	pluginManagementServer      *httptest.Server
	userManagementServer        *httptest.Server
)

var validSession = &auth.SessionStore{
	"AccessToken":   []byte("12345678"),
	"RefreshToken":  []byte("12345678"),
	"SessionID":     []byte("12345678"),
	"AccessExpiry":  []byte(strconv.FormatInt(time.Now().Add(5*time.Minute).Unix(), 10)),
	"RefreshExpiry": []byte(strconv.FormatInt(time.Now().Add(2*time.Hour).Unix(), 10)),
}

var refreshableSession = &auth.SessionStore{
	"AccessToken":   []byte("87654321"),
	"RefreshToken":  []byte("87654321"),
	"SessionID":     []byte("87654321"),
	"AccessExpiry":  []byte(strconv.FormatInt(time.Now().Add(-5*time.Hour).Unix(), 10)),
	"RefreshExpiry": []byte(strconv.FormatInt(time.Now().Add(2*time.Hour).Unix(), 10)),
}

var expiredSession = &auth.SessionStore{
	"AccessToken":   []byte("00000000"),
	"RefreshToken":  []byte("00000000"),
	"SessionID":     []byte("00000000"),
	"AccessExpiry":  []byte(strconv.FormatInt(time.Unix(0, 0).Unix(), 10)),
	"RefreshExpiry": []byte(strconv.FormatInt(time.Unix(0, 0).Unix(), 10)),
}

var _ = BeforeSuite(func() {
	ctx = context.Background()
	serverAddr = iputil.FormatURL(serverScheme, serverHost, serverPort)

	By("启动 mock 后端服务器")
	consoleWebsiteServer = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("console-website"))
	}))
	alertServer = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("alert"))
	}))
	monitoringServer = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("monitoring"))
	}))
	webterminalServer = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("webterminal"))
	}))
	applicationManagementServer = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("application-management"))
	}))
	marketplaceServer = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("marketplace"))
	}))
	pluginManagementServer = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("plugin-management"))
	}))
	userManagementServer = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("user-management"))
	}))

	By("启动 mock OAuth 服务器")
	oauthServer = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/oauth2/oauth/token" && r.Method == "POST" {
			// 检查是否是有效的授权码
			if err := r.ParseForm(); err != nil {
				http.Error(w, "Bad Request", http.StatusBadRequest)
				return
			}

			code := r.FormValue("code")
			if code == "invalidcode" {
				// 无效验证码，返回错误
				http.Error(w, "invalid_grant", http.StatusBadRequest)
				return
			}

			// 有效的验证码，返回token
			w.Header().Set("Content-Type", "application/json")
			response := `{
				"access_token": "fake-access-token",
				"token_type": "Bearer",
				"expires_in": 3600,
				"refresh_token": "fake-refresh-token",
				"refresh_token_expires_in": 7200
			}`
			w.Write([]byte(response))
			return
		} else if r.URL.Path == "/oauth2/auth/password/fuyaoPasswordProvider" && r.Method == "POST" {
			w.WriteHeader(http.StatusOK)
			w.Write([]byte("password"))
			return
		} else {
			http.NotFound(w, r)
		}
	}))

	By("创建 envtest 环境")
	testEnv = &envtest.Environment{
		CRDInstallOptions: envtest.CRDInstallOptions{
			CleanUpAfterUse: true,
		},
	}

	var err error
	k8sCfg, err = testEnv.Start()
	Expect(err).NotTo(HaveOccurred())
	Expect(k8sCfg).NotTo(BeNil())

	k8sClient, err = kubernetes.NewForConfig(k8sCfg)
	Expect(err).NotTo(HaveOccurred())
	Expect(k8sClient).NotTo(BeNil())

	dynamicClient, err = dynamic.NewForConfig(k8sCfg)
	Expect(err).NotTo(HaveOccurred())
	Expect(dynamicClient).NotTo(BeNil())

	By("注册 ConsolePlugin CRD")
	crdPath := "../charts/console-service/crds/consoleplugins.crd.yaml"
	crdBytes, err := os.ReadFile(crdPath)
	Expect(err).NotTo(HaveOccurred())
	var crd apiextensionsv1.CustomResourceDefinition
	Expect(sigsyaml.Unmarshal(crdBytes, &crd)).To(Succeed())
	apiExtClient, err := crdclientset.NewForConfig(k8sCfg)
	Expect(err).NotTo(HaveOccurred())
	_, err = apiExtClient.ApiextensionsV1().CustomResourceDefinitions().Create(ctx, &crd, metav1.CreateOptions{})
	Expect(err).NotTo(HaveOccurred())

	By("创建 openfuyao-system 命名空间")
	_, err = k8sClient.CoreV1().Namespaces().Create(ctx, &corev1.Namespace{
		ObjectMeta: metav1.ObjectMeta{
			Name: "openfuyao-system",
		},
	}, metav1.CreateOptions{})
	Expect(err).NotTo(HaveOccurred())

	By("创建 configmap")
	_, err = k8sClient.CoreV1().ConfigMaps("openfuyao-system").Create(ctx, &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "console-service-config",
			Namespace: "openfuyao-system",
		},
		Data: map[string]string{
			"oauth-server-host":           oauthServer.URL,
			"console-service-host":        "http://console-service:9020",
			"console-website-host":        consoleWebsiteServer.URL,
			"alert-host":                  alertServer.URL,
			"monitoring-host":             monitoringServer.URL,
			"webterminal-host":            webterminalServer.URL,
			"application-management-host": applicationManagementServer.URL,
			"marketplace-host":            marketplaceServer.URL,
			"plugin-management-host":      pluginManagementServer.URL,
			"user-management-host":        userManagementServer.URL,
			"insecure-skip-verify":        "true",
			"server-name":                 "",
		},
	}, metav1.CreateOptions{})
	Expect(err).NotTo(HaveOccurred())

	By("创建 symmetric key")
	_, err = k8sClient.CoreV1().Secrets("openfuyao-system").Create(ctx, &corev1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "console-service-token-key",
			Namespace: "openfuyao-system",
		},
		Data: map[string][]byte{
			"console-service-symmetric-key": []byte("xxxxxxxxxxxxxxxx"),
		},
	}, metav1.CreateOptions{})
	Expect(err).NotTo(HaveOccurred())

	By("创建 session-secret 命名空间")
	_, err = k8sClient.CoreV1().Namespaces().Create(ctx, &corev1.Namespace{
		ObjectMeta: metav1.ObjectMeta{
			Name: "session-secret",
		},
	}, metav1.CreateOptions{})
	Expect(err).NotTo(HaveOccurred())
	err = auth.StoreSession(k8sClient, validSession, false)
	Expect(err).NotTo(HaveOccurred())

	By("设置 OAuth 环境变量")
	Expect(os.Setenv("OAUTH_CLIENT_ID", "test-client-id")).To(Succeed())
	Expect(os.Setenv("OAUTH_CLIENT_SECRET", "test-client-secret")).To(Succeed())

	By("启动 console-service 主进程(由测试注入 envtest 配置和测试端口)")
	// 不调用 config.NewRunConfig()：其内部 NewKubernetesCfg→GetKubeConfig 在缺少
	// in-cluster/kubeconfig 时会 Fatalf。DT 用 envtest 的 rest.Config 直接组装 RunConfig。
	importedConfig := newDTRunConfig(k8sCfg, serverHost, serverPort)

	serverCtx, cancel := context.WithCancel(context.Background())

	consoleServer, err := server.NewServer(importedConfig, serverCtx)
	Expect(err).NotTo(HaveOccurred())
	go func() {
		_ = consoleServer.Run(serverCtx)
	}()

	testHttpClient = &http.Client{
		CheckRedirect: func(req *http.Request, via []*http.Request) error {
			return http.ErrUseLastResponse
		},
	}

	ready := false
	for i := 0; i < 30; i++ {
		if testHttpClient != nil {
			resp, err := testHttpClient.Get(serverAddr + "/rest/auth/login")
			if err == nil {
				resp.Body.Close()
				ready = true
				break
			}
			time.Sleep(1000 * time.Millisecond)
		}
	}
	Expect(ready).To(BeTrue(), "console-service 服务未成功拉起")

	stopFunc = func() {
		cancel()
		if testEnv != nil {
			testEnv.Stop()
		}
	}

})

var _ = AfterSuite(func() {
	By("停止 mock 服务器")
	for _, server := range []*httptest.Server{
		oauthServer,
		consoleWebsiteServer,
		alertServer,
		monitoringServer,
		webterminalServer,
		applicationManagementServer,
		marketplaceServer,
		pluginManagementServer,
		userManagementServer,
	} {
		if server != nil {
			server.Close()
		}
	}

	By("清除 OAuth 环境变量")
	Expect(os.Unsetenv("OAUTH_CLIENT_ID")).To(Succeed())
	Expect(os.Unsetenv("OAUTH_CLIENT_SECRET")).To(Succeed())

	if stopFunc != nil {
		stopFunc()
	}
})

// newDTRunConfig builds RunConfig for DT without calling config.NewRunConfig(),
// which would Fatalf when neither in-cluster config nor ~/.kube/config exists.
func newDTRunConfig(envtestCfg *rest.Config, bindAddress string, insecurePort int) *config.RunConfig {
	serverCfg := runtime.NewServerConfig()
	serverCfg.BindAddress = bindAddress
	serverCfg.InsecurePort = insecurePort
	serverCfg.SecurePort = 0
	return &config.RunConfig{
		Server: serverCfg,
		KubernetesCfg: &k8s.KubernetesCfg{
			KubeConfig: envtestCfg,
			QPS:        1e6,
			Burst:      1e6,
		},
	}
}

func TestTest(t *testing.T) {
	RegisterFailHandler(Fail)
	RunSpecs(t, "Test Suite")
}
