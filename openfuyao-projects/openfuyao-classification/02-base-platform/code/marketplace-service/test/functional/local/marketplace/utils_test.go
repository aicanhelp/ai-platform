/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */
package marketplace

import (
	"archive/tar"
	"bytes"
	"compress/gzip"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"time"
	"unicode"

	"github.com/onsi/gomega"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/kubernetes"
	clientSetFake "k8s.io/client-go/kubernetes/fake"
	"k8s.io/client-go/rest"

	helmv1 "marketplace-service/pkg/api/marketplace/v1beta1"
	"marketplace-service/pkg/constant"
	"marketplace-service/pkg/helm"
	helmModel "marketplace-service/pkg/models/helm"
	param "marketplace-service/pkg/server/param"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

const (
	apiBasePath      = "/rest/marketplace/v1beta1"
	mockHelmRepoPort = 9038
)

type SecretRef struct {
	Name string `json:"name"`
}

// 创建仓库的请求结构
type RepoRequest struct {
	Name                  string `json:"name"`                     // 仓库显示名称
	URL                   string `json:"url"`                      // 仓库 URL
	Username              []byte `json:"username,omitempty"`       // 用户名（用于需要认证的仓库）
	Password              []byte `json:"password,omitempty"`       // 密码（用于需要认证的仓库）
	CertFile              string `json:"certFile,omitempty"`       // TLS 证书文件内容
	KeyFile               string `json:"keyFile,omitempty"`        // TLS 私钥文件内容
	CAFile                string `json:"caFile,omitempty"`         // CA 证书文件内容
	InsecureSkipTLSVerify bool   `json:"insecure_skip_tls_verify"` // 是否跳过 TLS 验证
	PassCredentialsAll    bool   `json:"pass_credentials_all"`     // 是否将凭据传递给所有域
}

// 创建仓库的响应结构
type RepoResponse struct {
	Code int    `json:"code"`
	Msg  string `json:"msg"`
	Data struct {
		APIVersion string `json:"apiVersion"`
		Kind       string `json:"kind"`
		Metadata   struct {
			Name              string            `json:"name"`
			CreationTimestamp time.Time         `json:"creationTimestamp"`
			Annotations       map[string]string `json:"annotations,omitempty"`
		} `json:"metadata"`
		Spec struct {
			DisplayName           string    `json:"displayName"`
			URL                   string    `json:"url"`
			Description           string    `json:"description,omitempty"`
			BasicAuth             SecretRef `json:"basicAuth"`
			CA                    SecretRef `json:"ca"`
			InsecureSkipTLSVerify bool      `json:"insecureSkipTLSVerify"`
			PassCredentialsAll    bool      `json:"passCredentialsAll"`
			TLS                   SecretRef `json:"tls"`
		} `json:"spec"`
	} `json:"data"`
}

// cleanupRepos 清理测试过程中创建的 repo CR。
// 这是通过调用测试用的 HTTP server（fake mux）走 DeleteRepo 接口来清理。
func cleanupRepos(ctx context.Context, apiURL string, httpClient *http.Client, createdRepos []string) {
	if len(createdRepos) == 0 {
		return
	}
	for _, name := range createdRepos {
		req, err := http.NewRequestWithContext(ctx, http.MethodDelete,
			fmt.Sprintf("%s%s/helm-repos/%s", apiURL, apiBasePath, name), nil)
		if err != nil {
			continue
		}
		resp, err := httpClient.Do(req)
		if err != nil {
			continue
		}
		if err := resp.Body.Close(); err != nil {
		}
	}
}

// 测试上下文，包含所有测试需要的共享资源
type testContext struct {
	ctx        context.Context
	apiURL     string
	httpClient *http.Client

	// handler 仅用于复用部分字段/断言（真实业务逻辑在 helm.Operation 中）
	handler interface{} // *v1beta1.Handler

	// httptest server：承载 fake route（net/http mux），用于让测试用例用 HTTP 方式访问真实业务逻辑
	server *httptest.Server

	// 外部依赖 mock：模拟远端 Helm Repo（index.yaml）和本地 Harbor（charts API + index.yaml）
	mockHelmRepo *MockHelmRepositoryServer
	mockHarbor   *MockHarborServer

	repoURL      string
	createdRepos []string

	// 外部依赖 mock：K8s 资源
	dynamicClient interface{} // dynamic.Interface，实际为 *MockDynamicClient
	k8sClient     interface{} // kubernetes.Interface
}

// setupTestContext 搭建一套“轻量功能测试环境”：
// - mock 外部 Helm Repo（index.yaml）
// - mock 外部 Harbor（用于本地仓库上传/删除 + index.yaml）
// - fake k8s clientset/dynamic client（提供 ConfigMap/Secret/CRD）
// - 用 net/http mux 手写路由，将真实 helm.Operation 暴露成 HTTP API（绕开 go-restful 初始化副作用）
func setupTestContext(helmRepoConfig func(*MockHelmRepositoryServer), k8sOptFns ...mockK8sOption) *testContext {
	// 重置 helm 包内部的全局同步状态，避免用例间 repoAsyncTaskMap 污染导致 SyncRepo 返回 400(in_progress)
	helm.ResetSyncStatusForTest()

	ctx := context.Background()

	// 1) 启动 Mock Helm Repository（外部依赖）
	mockHelmRepo := NewMockHelmRepositoryServer(mockHelmRepoPort)
	if helmRepoConfig != nil {
		helmRepoConfig(mockHelmRepo)
	}
	if err := mockHelmRepo.Start(); err != nil {
		_ = mockHelmRepo.Stop()
		time.Sleep(300 * time.Millisecond)
		gomega.Expect(mockHelmRepo.Start()).To(gomega.Succeed(), "Failed to start MockHelmRepo after retry, port %d may be in use", mockHelmRepoPort)
	}
	repoURL := mockHelmRepo.GetURL()

	// 2) 创建 fake dynamic client（用于 CRD）
	scheme := runtime.NewScheme()
	if err := helmModel.AddHelmChartRepositoryToScheme(scheme); err != nil {
		gomega.Expect(err).NotTo(gomega.HaveOccurred())
	}

	gvr := schema.GroupVersionResource{
		Group:    constant.CRDRepoGroup,
		Version:  constant.CRDRepoVersion,
		Resource: constant.CRDRepoResource,
	}
	// 这里只需要 list kind 字符串；Group/Version 在 NewMockDynamicClient 的 map value 中不会被读取
	dynamicClient := NewMockDynamicClient(scheme, map[schema.GroupVersionResource]string{
		gvr: constant.CRDRepoListKind,
	})

	// 3) 启动 Mock Harbor（外部依赖：本地 Harbor API）
	mockHarbor := NewMockHarborServer(0, "library")
	gomega.Expect(mockHarbor.Start()).To(gomega.Succeed())

	// 4) 创建 fake clientset：提供 ConfigMap/Secret（业务逻辑读取这些配置）
	k8sClient := createMockK8sClientSet(mockHarbor.GetURL(), k8sOptFns...)

	// 5) 创建真实 helm operation，并注入 fake k8s client
	mockK8sConfig := &rest.Config{Host: "https://mock-kube"}
	op, err := helm.NewHelmOperation(mockK8sConfig)
	gomega.Expect(err).NotTo(gomega.HaveOccurred())
	if injector, ok := op.(interface {
		SetDynamicClient(client dynamic.Interface)
		SetK8sClient(client kubernetes.Interface)
	}); ok {
		injector.SetDynamicClient(dynamicClient)
		injector.SetK8sClient(k8sClient)
	} else {
		panic(fmt.Sprintf("helm operation (%T) does not implement SetDynamicClient/SetK8sClient methods", op))
	}

	// 6) 用 net/http mux 注册 fake routes，将真实业务逻辑暴露成 HTTP
	mux := http.NewServeMux()
	registerFakeMarketplaceRoutes(mux, op)
	server := httptest.NewServer(mux)

	return &testContext{
		ctx:           ctx,
		apiURL:        server.URL,
		httpClient:    &http.Client{Timeout: 30 * time.Second},
		handler:       &helmv1.Handler{HelmHandler: op},
		server:        server,
		mockHelmRepo:  mockHelmRepo,
		mockHarbor:    mockHarbor,
		repoURL:       repoURL,
		createdRepos:  make([]string, 0),
		dynamicClient: dynamicClient,
		k8sClient:     k8sClient,
	}
}

// createMockK8sClientSet 通过 client-go fake clientset 构造“最小可用”的 K8s 环境：
// - marketplace-service-config ConfigMap（包含 chart-limit、local-harbor-host 等）
// - local-harbor-core Secret（包含 Harbor 管理员密码，Upload/Delete chart 需要）
type mockK8sOptions struct {
	skipConfigMap bool
	skipSecret    bool
	chartLimit    string
	// 允许覆盖 marketplace-scenes（不传则使用默认）
	marketplaceScenes string
}

type mockK8sOption func(*mockK8sOptions)

func withSkipConfigMap() mockK8sOption {
	return func(o *mockK8sOptions) { o.skipConfigMap = true }
}

func withSkipSecret() mockK8sOption {
	return func(o *mockK8sOptions) { o.skipSecret = true }
}

func withChartLimit(v string) mockK8sOption {
	return func(o *mockK8sOptions) { o.chartLimit = v }
}

// createMockK8sClientSet 通过 client-go fake clientset 构造“最小可用”的 K8s 环境：
// - marketplace-service-config ConfigMap（包含 chart-limit、local-harbor-host 等）
// - local-harbor-core Secret（包含 Harbor 管理员密码，Upload/Delete chart 需要）
func createMockK8sClientSet(mockHarborURL string, optFns ...mockK8sOption) *clientSetFake.Clientset {
	clientset := clientSetFake.NewSimpleClientset()

	opts := &mockK8sOptions{}
	for _, fn := range optFns {
		if fn != nil {
			fn(opts)
		}
	}

	defaultScenes := `
            - fuyao-select
            - compute-power-engine-plugin
            - artificial-intelligence
            - computing
            - database
            - developer-tool
            - CI/CD
            - monitor
            - log
            - network
            - observability
            - security
            - storage
        `

	scenesYAML := defaultScenes
	if strings.TrimSpace(opts.marketplaceScenes) != "" {
		scenesYAML = opts.marketplaceScenes
	}

	chartLimit := "100"
	if strings.TrimSpace(opts.chartLimit) != "" {
		chartLimit = opts.chartLimit
	}

	if !opts.skipConfigMap {
		configmap := &v1.ConfigMap{
			ObjectMeta: metav1.ObjectMeta{
				Name:      constant.MarketplaceServiceConfigmap,
				Namespace: constant.MarketplaceServiceDefaultNamespace,
			},
			Data: map[string]string{
				"chart-limit":                  chartLimit,
				"official-harbor-tags-URL":     "/v2/openfuyao-catalog-tags",
				"local-harbor-display-name":    "local",
				"local-harbor-host":            mockHarborURL,
				"local-harbor-project":         "/chartrepo/library",
				"official-harbor-display-name": "openFuyao",
				"official-harbor-host":         "https://cr.openfuyao.cn",
				"official-harbor-project":      "/chartrepo/openfuyao-catalog",
				"marketplace-scenes":           scenesYAML,
				"oauth-server-host":            "https://oauth-server.openfuyao-system.svc.cluster.local",
				"marketplace-service-host":     "http://marketplace-service.openfuyao-system.svc.cluster.local",
				"console-website-host":         "http://console-website.openfuyao-system.svc.cluster.local",
				"alert-host":                   "http://alertmanager-main.monitoring.svc.cluster.local:9093",
				"monitoring-host":              "http://monitoring-service.openfuyao-system.svc.cluster.local:80",
			},
		}
		_, _ = clientset.CoreV1().ConfigMaps(constant.MarketplaceServiceDefaultNamespace).Create(context.Background(), configmap, metav1.CreateOptions{})
	}

	if !opts.skipSecret {
		secret := &v1.Secret{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "local-harbor-core",
				Namespace: constant.MarketplaceServiceDefaultNamespace,
			},
			Data: map[string][]byte{
				"HARBOR_ADMIN_PASSWORD": []byte("harbor-password"),
			},
		}
		_, _ = clientset.CoreV1().Secrets(constant.MarketplaceServiceDefaultNamespace).Create(context.Background(), secret, metav1.CreateOptions{})
	}

	return clientset
}

// teardown 释放 tc 在单个用例中创建的资源。
// 注意：UploadChart/DeleteChart 会异步触发 repo sync，因此关闭 mock server 的时机要谨慎。
func (tc *testContext) teardown() {
	defer func() {
		if tc.mockHelmRepo != nil {
			_ = tc.mockHelmRepo.Stop()
		}
		if tc.mockHarbor != nil {
			_ = tc.mockHarbor.Stop()
		}
	}()

	if dc, ok := tc.dynamicClient.(dynamic.Interface); ok {
		for _, name := range tc.createdRepos {
			_ = dc.Resource(schema.GroupVersionResource{
				Group:    constant.CRDRepoGroup,
				Version:  constant.CRDRepoVersion,
				Resource: constant.CRDRepoResource,
			}).Delete(context.Background(), strings.ToLower(name), metav1.DeleteOptions{})
		}
	} else {
		cleanupRepos(tc.ctx, tc.apiURL, tc.httpClient, tc.createdRepos)
	}

	if tc.server != nil {
		tc.server.Close()
	}
}

func (tc *testContext) createRepoRequest(reqBody RepoRequest) (*http.Request, error) {
	return tc.createRepoRequestWithAuth(reqBody, true)
}

func (tc *testContext) createRepoRequestWithAuth(reqBody RepoRequest, withAuth bool) (*http.Request, error) {
	reqBodyJSON, err := json.Marshal(reqBody)
	if err != nil {
		return nil, err
	}

	req, err := http.NewRequestWithContext(tc.ctx, http.MethodPost,
		fmt.Sprintf("%s%s/helm-repos", tc.apiURL, apiBasePath),
		bytes.NewBuffer(reqBodyJSON))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	if withAuth {
		req.Header.Set("Authorization", "Bearer valid-token-12345")
	}
	return req, nil
}

// 发送HTTP请求并返回响应
func (tc *testContext) sendRequest(req *http.Request) (*http.Response, []byte, error) {
	resp, err := tc.httpClient.Do(req)
	if err != nil {
		return nil, nil, err
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	return resp, respBody, err
}

// 创建仓库并返回响应
func (tc *testContext) createRepo(reqBody RepoRequest) (*RepoResponse, *http.Response, error) {
	req, err := tc.createRepoRequest(reqBody)
	gomega.Expect(err).NotTo(gomega.HaveOccurred())

	resp, respBody, err := tc.sendRequest(req)
	if err != nil {
		return nil, resp, err
	}

	var repoResp RepoResponse
	err = json.Unmarshal(respBody, &repoResp)
	return &repoResp, resp, err
}

// 删除仓库并返回响应
func (tc *testContext) deleteRepo(repoName string) (*http.Response, []byte, error) {
	return tc.deleteRepoWithAuth(repoName, true)
}

func (tc *testContext) deleteRepoWithAuth(repoName string, withAuth bool) (*http.Response, []byte, error) {
	req, err := http.NewRequestWithContext(tc.ctx, http.MethodDelete,
		fmt.Sprintf("%s%s/helm-repos/%s", tc.apiURL, apiBasePath, repoName), nil)
	if err != nil {
		return nil, nil, err
	}
	if withAuth {
		req.Header.Set("Authorization", "Bearer valid-token-12345")
	}
	return tc.sendRequest(req)
}

// 验证成功响应
func verifySuccessResponse(resp *http.Response, repoResp *RepoResponse, expectedDisplayName, expectedURL string) {
	gomega.Expect(resp.StatusCode).To(gomega.Equal(http.StatusCreated))
	gomega.Expect(repoResp.Code).To(gomega.Equal(200))
	gomega.Expect(repoResp.Msg).To(gomega.ContainSubstring("created"))
	gomega.Expect(repoResp.Data.Kind).To(gomega.Equal("HelmChartRepository"))
	gomega.Expect(repoResp.Data.APIVersion).To(gomega.Equal("console.openfuyao.com/v1beta1"))
	if expectedDisplayName != "" {
		gomega.Expect(repoResp.Data.Spec.DisplayName).To(gomega.Equal(expectedDisplayName))
	}
	if expectedURL != "" {
		gomega.Expect(repoResp.Data.Spec.URL).To(gomega.Equal(expectedURL))
	}
	gomega.Expect(repoResp.Data.Metadata.Name).NotTo(gomega.BeEmpty())
}

// 验证错误响应
func verifyErrorResponse(resp *http.Response, respBody []byte, expectedCode int, expectedMsgSubstring string) {
	gomega.Expect(resp.StatusCode).To(gomega.Equal(expectedCode))

	var errorResp struct {
		Code int    `json:"code"`
		Msg  string `json:"msg"`
	}
	err := json.Unmarshal(respBody, &errorResp)
	gomega.Expect(err).NotTo(gomega.HaveOccurred())
	gomega.Expect(errorResp.Code).To(gomega.Equal(expectedCode))
	if expectedMsgSubstring != "" {
		gomega.Expect(errorResp.Msg).To(gomega.ContainSubstring(expectedMsgSubstring))
	}
}

// 验证删除成功响应
func verifyDeleteSuccessResponse(resp *http.Response, respBody []byte, expectedRepoName string) {
	gomega.Expect(resp.StatusCode).To(gomega.Equal(http.StatusOK))

	var deleteResp struct {
		Code int         `json:"code"`
		Msg  string      `json:"msg"`
		Data interface{} `json:"data"`
	}
	err := json.Unmarshal(respBody, &deleteResp)
	gomega.Expect(err).NotTo(gomega.HaveOccurred())
	gomega.Expect(deleteResp.Code).To(gomega.Equal(200))
	gomega.Expect(deleteResp.Msg).To(gomega.ContainSubstring("deleted"))
	if expectedRepoName != "" {
		gomega.Expect(deleteResp.Msg).To(gomega.ContainSubstring(expectedRepoName))
	}
	gomega.Expect(deleteResp.Data).To(gomega.BeNil())
}

// 同步响应结构
type SyncResponse struct {
	Code int         `json:"code"`
	Msg  string      `json:"msg"`
	Data interface{} `json:"data"`
}

// 同步状态响应结构
type SyncStatusResponse struct {
	Code int    `json:"code"`
	Msg  string `json:"msg"`
}

// 同步单个仓库并返回响应
func (tc *testContext) syncRepo(repoName string) (*http.Response, []byte, error) {
	return tc.syncRepoWithAuth(repoName, true)
}

func (tc *testContext) syncRepoWithAuth(repoName string, withAuth bool) (*http.Response, []byte, error) {
	req, err := http.NewRequestWithContext(tc.ctx, http.MethodPost,
		fmt.Sprintf("%s%s/helm-repos/%s/sync", tc.apiURL, apiBasePath, repoName), nil)
	if err != nil {
		return nil, nil, err
	}
	if withAuth {
		req.Header.Set("Authorization", "Bearer valid-token-12345")
	}
	return tc.sendRequest(req)
}

// 获取仓库同步状态
func (tc *testContext) getRepoSyncStatus(repoName string) (*http.Response, []byte, error) {
	req, err := http.NewRequestWithContext(tc.ctx, http.MethodGet,
		fmt.Sprintf("%s%s/helm-repos/%s/sync", tc.apiURL, apiBasePath, repoName), nil)
	if err != nil {
		return nil, nil, err
	}
	req.Header.Set("Authorization", "Bearer valid-token-12345")
	return tc.sendRequest(req)
}

// 获取最新 Chart 列表（GET /helm-charts）
func (tc *testContext) getLatestCharts(rawQuery string) (*http.Response, []byte, error) {
	url := fmt.Sprintf("%s%s/helm-charts", tc.apiURL, apiBasePath)
	if strings.TrimSpace(rawQuery) != "" {
		url = url + "?" + strings.TrimPrefix(rawQuery, "?")
	}
	req, err := http.NewRequestWithContext(tc.ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, nil, err
	}
	req.Header.Set("Authorization", "Bearer valid-token-12345")
	return tc.sendRequest(req)
}

// 获取 Chart 指定版本信息
func (tc *testContext) getChartVersion(repoName, chartName, version string) (*http.Response, []byte, error) {
	req, err := http.NewRequestWithContext(tc.ctx, http.MethodGet,
		fmt.Sprintf("%s%s/helm-repos/%s/charts/%s/versions/%s", tc.apiURL, apiBasePath, repoName, chartName, version), nil)
	if err != nil {
		return nil, nil, err
	}
	req.Header.Set("Authorization", "Bearer valid-token-12345")
	return tc.sendRequest(req)
}

// 获取仓库列表（GET /helm-repos）
func (tc *testContext) listRepos(rawQuery string) (*http.Response, []byte, error) {
	url := fmt.Sprintf("%s%s/helm-repos", tc.apiURL, apiBasePath)
	if rawQuery != "" {
		url = url + "?" + rawQuery
	}
	req, err := http.NewRequestWithContext(tc.ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, nil, err
	}
	req.Header.Set("Authorization", "Bearer valid-token-12345")
	return tc.sendRequest(req)
}

// 获取仓库详情（GET /helm-repos/{repo}）
func (tc *testContext) getRepo(repoName string) (*http.Response, []byte, error) {
	req, err := http.NewRequestWithContext(tc.ctx, http.MethodGet,
		fmt.Sprintf("%s%s/helm-repos/%s", tc.apiURL, apiBasePath, repoName), nil)
	if err != nil {
		return nil, nil, err
	}
	req.Header.Set("Authorization", "Bearer valid-token-12345")
	return tc.sendRequest(req)
}

// 上传 chart
func (tc *testContext) uploadChart(chartContent []byte, chartFilename string, withAuth bool) (*http.Response, []byte, error) {
	body := &bytes.Buffer{}
	writer := multipart.NewWriter(body)
	part, err := writer.CreateFormFile("chart", chartFilename)
	if err != nil {
		return nil, nil, err
	}
	_, err = io.Copy(part, bytes.NewReader(chartContent))
	if err != nil {
		return nil, nil, err
	}
	if err := writer.Close(); err != nil {
		return nil, nil, err
	}

	req, err := http.NewRequestWithContext(tc.ctx, http.MethodPost, fmt.Sprintf("%s%s/helm-charts", tc.apiURL, apiBasePath), body)
	if err != nil {
		return nil, nil, err
	}
	req.Header.Set("Content-Type", writer.FormDataContentType())
	if withAuth {
		req.Header.Set("Authorization", "Bearer valid-token-12345")
	}

	return tc.sendRequest(req)
}

func (tc *testContext) deleteChart(chartName string, version string, withAuth bool) (*http.Response, []byte, error) {
	url := fmt.Sprintf("%s%s/helm-charts/%s", tc.apiURL, apiBasePath, chartName)
	if version != "" {
		url = fmt.Sprintf("%s/versions/%s", url, version)
	}
	req, err := http.NewRequestWithContext(tc.ctx, http.MethodDelete, url, nil)
	if err != nil {
		return nil, nil, err
	}
	if withAuth {
		req.Header.Set("Authorization", "Bearer valid-token-12345")
	}
	return tc.sendRequest(req)
}

func (tc *testContext) waitForSyncComplete(repoName string, timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		resp, respBody, err := tc.getRepoSyncStatus(repoName)
		if err != nil {
			return err
		}
		if resp.StatusCode == http.StatusBadRequest {
			time.Sleep(500 * time.Millisecond)
			continue
		}
		var statusResp SyncStatusResponse
		if err := json.Unmarshal(respBody, &statusResp); err != nil {
			return err
		}
		if statusResp.Msg == "complete" {
			return nil
		}
		if statusResp.Msg == "failed" {
			return fmt.Errorf("sync failed: %s", statusResp.Msg)
		}
		// 如果状态是 in_progress，继续等待
		time.Sleep(500 * time.Millisecond)
	}
	return fmt.Errorf("sync timeout after %v", timeout)
}

func verifySyncSuccessResponse(resp *http.Response, respBody []byte, expectedRepoName string) {
	gomega.Expect(resp.StatusCode).To(gomega.Equal(http.StatusAccepted))
	var syncResp SyncResponse
	err := json.Unmarshal(respBody, &syncResp)
	gomega.Expect(err).NotTo(gomega.HaveOccurred())
	gomega.Expect(syncResp.Code).To(gomega.Equal(200))
	gomega.Expect(syncResp.Msg).To(gomega.ContainSubstring("syncronizing"))
	if expectedRepoName != "" {
		gomega.Expect(syncResp.Msg).To(gomega.ContainSubstring(expectedRepoName))
	}
}

// 通过 net/http ServeMux 暴露测试用 API。
func registerFakeMarketplaceRoutes(mux *http.ServeMux, op helm.Operation) {
	writeJSON := func(w http.ResponseWriter, status int, v interface{}) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		_ = json.NewEncoder(w).Encode(v)
	}

	parseBool := func(v string) bool {
		vv := strings.ToLower(strings.TrimSpace(v))
		return vv == "true" || vv == "1" || vv == "yes"
	}

	// list & create
	mux.HandleFunc(apiBasePath+"/helm-repos", func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodGet:
			q := &param.Query{Pagination: param.GetNoPagination()}
			limit, err := strconv.Atoi(r.URL.Query().Get(param.Limit))
			if err != nil {
				limit = -1
			}
			page, err := strconv.Atoi(r.URL.Query().Get(param.Page))
			if err != nil {
				page = 1
			}
			q.Pagination = &param.Pagination{Limit: limit, Offset: (page - 1) * limit, CurrentPage: page}
			q.SortBy = r.URL.Query().Get(param.SortType)
			order := r.URL.Query().Get(param.Order)
			q.Ascending = strings.ToLower(order) != "desc"
			if sortBy := r.URL.Query().Get("sortBy"); strings.TrimSpace(sortBy) != "" {
				q.SortBy = sortBy
			}
			if ascStr := r.URL.Query().Get("ascending"); strings.TrimSpace(ascStr) != "" {
				q.Ascending = parseBool(ascStr)
			}

			repo := r.URL.Query().Get(param.Repository)
			result, status := op.ListRepo(q, repo)
			writeJSON(w, status, result)
		case http.MethodPost:
			var reqBody RepoRequest
			if err := json.NewDecoder(r.Body).Decode(&reqBody); err != nil {
				writeJSON(w, http.StatusBadRequest, map[string]interface{}{"code": http.StatusBadRequest, "msg": "invalid input"})
				return
			}
			repoEntry := &helmModel.SafeRepoEntry{
				Name:                  reqBody.Name,
				URL:                   reqBody.URL,
				Username:              reqBody.Username,
				Password:              reqBody.Password,
				CertFile:              reqBody.CertFile,
				KeyFile:               reqBody.KeyFile,
				CAFile:                reqBody.CAFile,
				InsecureSkipTLSVerify: reqBody.InsecureSkipTLSVerify,
				PassCredentialsAll:    reqBody.PassCredentialsAll,
			}
			result, status := op.CreateRepo(repoEntry)
			writeJSON(w, status, result)
		default:
			w.WriteHeader(http.StatusMethodNotAllowed)
		}
	})

	// /helm-repos/... 子路由（包含 /{repo}、/{repo}/sync 以及 /{repo}/charts/...）
	mux.HandleFunc(apiBasePath+"/helm-repos/", func(w http.ResponseWriter, r *http.Request) {
		path := strings.TrimPrefix(r.URL.Path, apiBasePath+"/helm-repos/")
		path = strings.Trim(path, "/")
		if path == "" {
			if r.Method == http.MethodGet {
				q := &param.Query{}
				limit, err := strconv.Atoi(r.URL.Query().Get(param.Limit))
				if err != nil {
					limit = -1
				}
				page, err := strconv.Atoi(r.URL.Query().Get(param.Page))
				if err != nil {
					page = 1
				}
				q.Pagination = &param.Pagination{Limit: limit, Offset: (page - 1) * limit, CurrentPage: page}
				q.SortBy = r.URL.Query().Get(param.SortType)
				order := r.URL.Query().Get(param.Order)
				q.Ascending = strings.ToLower(order) != "desc"
				if sortBy := r.URL.Query().Get("sortBy"); strings.TrimSpace(sortBy) != "" {
					q.SortBy = sortBy
				}
				if ascStr := r.URL.Query().Get("ascending"); strings.TrimSpace(ascStr) != "" {
					q.Ascending = parseBool(ascStr)
				}
				repo := r.URL.Query().Get(param.Repository)
				result, status := op.ListRepo(q, repo)
				writeJSON(w, status, result)
			} else {
				w.WriteHeader(http.StatusMethodNotAllowed)
			}
			return
		}
		parts := strings.Split(path, "/")
		repoName := parts[0]

		// /{repo}/sync
		if len(parts) == 2 && parts[1] == "sync" {
			switch r.Method {
			case http.MethodGet:
				result, status := op.GetRepoSyncStatus(repoName)
				writeJSON(w, status, result)
			case http.MethodPost:
				result, status := op.SyncRepo(repoName)
				writeJSON(w, status, result)
			default:
				w.WriteHeader(http.StatusMethodNotAllowed)
			}
			return
		}

		// /{repo}/charts/{chart}/versions/{version}
		if len(parts) == 5 && parts[1] == "charts" && parts[3] == "versions" {
			chartName := parts[2]
			version := parts[4]
			if r.Method == http.MethodGet {
				result, status := op.GetChartVersion(repoName, chartName, version)
				writeJSON(w, status, result)
				return
			}
			w.WriteHeader(http.StatusMethodNotAllowed)
			return
		}

		// /{repo}
		switch r.Method {
		case http.MethodGet:
			result, status := op.GetRepo(repoName)
			writeJSON(w, status, result)
		case http.MethodPut:
			var reqBody RepoRequest
			if err := json.NewDecoder(r.Body).Decode(&reqBody); err != nil {
				writeJSON(w, http.StatusBadRequest, map[string]interface{}{"code": http.StatusBadRequest, "msg": "invalid input"})
				return
			}
			repoEntry := &helmModel.SafeRepoEntry{
				Name:                  repoName,
				URL:                   reqBody.URL,
				Username:              reqBody.Username,
				Password:              reqBody.Password,
				CertFile:              reqBody.CertFile,
				KeyFile:               reqBody.KeyFile,
				CAFile:                reqBody.CAFile,
				InsecureSkipTLSVerify: reqBody.InsecureSkipTLSVerify,
				PassCredentialsAll:    reqBody.PassCredentialsAll,
			}
			result, status := op.UpdateRepo(repoEntry)
			writeJSON(w, status, result)
		case http.MethodDelete:
			result, status := op.DeleteRepo(repoName)
			writeJSON(w, status, result)
		default:
			w.WriteHeader(http.StatusMethodNotAllowed)
		}
	})

	// /helm-charts（GET 获取最新 Chart 列表，POST 上传 chart）
	mux.HandleFunc(apiBasePath+"/helm-charts", func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodGet:
			q := &param.Query{}
			limit, err := strconv.Atoi(r.URL.Query().Get(param.Limit))
			if err != nil {
				limit = -1
			}
			page, err := strconv.Atoi(r.URL.Query().Get(param.Page))
			if err != nil {
				page = 1
			}
			q.Pagination = &param.Pagination{Limit: limit, Offset: (page - 1) * limit, CurrentPage: page}
			q.SortBy = r.URL.Query().Get(param.SortType)
			order := r.URL.Query().Get(param.Order)
			q.Ascending = strings.ToLower(order) != "desc"

			searchParam := &helmModel.ChartSearchParam{Query: q}
			if repos := r.URL.Query()[param.Repository]; len(repos) > 0 {
				searchParam.Repositories = make([]string, 0, len(repos))
				for _, repo := range repos {
					if repo != "" {
						searchParam.Repositories = append(searchParam.Repositories, repo)
					}
				}
			}
			if chart := r.URL.Query().Get(param.Chart); chart != "" {
				searchParam.Chart = chart
			}
			if types := r.URL.Query()[param.AppType]; len(types) > 0 {
				searchParam.Types = make([]string, 0, len(types))
				for _, t := range types {
					if t != "" {
						searchParam.Types = append(searchParam.Types, t)
					}
				}
			}
			if scenes := r.URL.Query()[param.Scene]; len(scenes) > 0 {
				searchParam.Scene = make([]string, 0, len(scenes))
				for _, scene := range scenes {
					if scene != "" {
						searchParam.Scene = append(searchParam.Scene, scene)
					}
				}
			}

			result, status := op.GetLatestCharts(searchParam)
			writeJSON(w, status, result)
		case http.MethodPost:
			formFile, fileHeader, err := r.FormFile("chart")
			if err != nil {
				writeJSON(w, http.StatusBadRequest, map[string]interface{}{
					"code": constant.ClientError,
					"msg":  "Failed to get chart",
				})
				return
			}
			defer formFile.Close()
			result, status := op.UploadChart(formFile, fileHeader)
			writeJSON(w, status, result)
		default:
			w.WriteHeader(http.StatusMethodNotAllowed)
		}
	})

	// /helm-charts/... 子路由：DELETE /helm-charts/{chartName} 以及 DELETE /helm-charts/{chartName}/versions/{version}
	mux.HandleFunc(apiBasePath+"/helm-charts/", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodDelete {
			w.WriteHeader(http.StatusMethodNotAllowed)
			return
		}

		path := strings.TrimPrefix(r.URL.Path, apiBasePath+"/helm-charts/")
		path = strings.Trim(path, "/")
		parts := strings.Split(path, "/")
		if len(parts) == 0 || parts[0] == "" {
			writeJSON(w, http.StatusBadRequest, map[string]interface{}{
				"code": constant.ClientError,
				"msg":  "please provide chart name for delete",
			})
			return
		}

		chartName := parts[0]
		if len(parts) >= 3 && parts[1] == "versions" {
			version := parts[2]
			result, status := op.DeleteChartVersion(chartName, version)
			writeJSON(w, status, result)
			return
		}

		result, status := op.DeleteChart(chartName)
		writeJSON(w, status, result)
	})
}

// 直接往 fake dynamic client 写入 HelmChartRepository CR（用于 ListRepo/排序/分页等用例的前置数据准备）
func (tc *testContext) seedRepoCR(displayName, repoURL string) error {
	dc, ok := tc.dynamicClient.(dynamic.Interface)
	if !ok {
		return fmt.Errorf("dynamicClient is not dynamic.Interface")
	}

	repoCR := &helmModel.HelmChartRepository{
		TypeMeta: metav1.TypeMeta{
			Kind:       constant.CRDRepoKind,
			APIVersion: fmt.Sprintf("%s/%s", constant.CRDRepoGroup, constant.CRDRepoVersion),
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: strings.ToLower(displayName),
		},
		Spec: helmModel.HelmChartRepositorySpec{
			DisplayName: displayName,
			URL:         repoURL,
		},
	}

	unstructuredObj, err := runtime.DefaultUnstructuredConverter.ToUnstructured(repoCR)
	if err != nil {
		return fmt.Errorf("failed to convert typed repo to unstructured: %w", err)
	}
	u := &unstructured.Unstructured{Object: unstructuredObj}

	_, err = dc.Resource(schema.GroupVersionResource{Group: constant.CRDRepoGroup, Version: constant.CRDRepoVersion, Resource: constant.CRDRepoResource}).
		Create(context.Background(), u, metav1.CreateOptions{})

	if err == nil {
		tc.createdRepos = append(tc.createdRepos, displayName)
	}
	return err
}

// 排序断言辅助函数
func firstRuneRank(s string) int {
	if s == "" {
		return 4
	}
	r := []rune(s)[0]
	if unicode.IsDigit(r) {
		return 2
	}
	if unicode.IsLetter(r) {
		return 1
	}
	if unicode.IsPunct(r) || unicode.IsSymbol(r) {
		return 0
	}
	return 3
}

func compareChartNameMock(a, b string) int {
	getPriority := func(r rune) int {
		switch {
		case unicode.IsSymbol(r) || unicode.IsPunct(r):
			return 0
		case unicode.IsLetter(r):
			return 1
		case unicode.IsDigit(r):
			return 2
		default:
			return 3
		}
	}

	for i := 0; i < len(a) && i < len(b); i++ {
		ra, rb := rune(a[i]), rune(b[i])
		pa, pb := getPriority(ra), getPriority(rb)

		if pa != pb {
			if pa < pb {
				return -1
			}
			return 1
		}

		if ra != rb {
			if unicode.ToLower(ra) == unicode.ToLower(rb) {
				if ra < rb {
					return -1
				}
				return 1
			}
			if unicode.ToLower(ra) < unicode.ToLower(rb) {
				return -1
			}
			return 1
		}
	}

	if len(a) < len(b) {
		return -1
	}
	if len(a) > len(b) {
		return 1
	}
	return 0
}

// ==================== Chart 测试数据构造辅助函数====================

// 创建一个简单的测试 chart tar.gz 文件
func createTestChartTgz(chartName, version string, size int) ([]byte, error) {
	if size == 0 {
		size = 1024 
	}

	var buf bytes.Buffer
	gw := gzip.NewWriter(&buf)
	tw := tar.NewWriter(gw)

	// Chart.yaml
	chartYAML := `apiVersion: v2
name: ` + chartName + `
version: ` + version + `
description: A test chart
type: application
`
	if err := writeTarEntry(tw, chartName+"/Chart.yaml", []byte(chartYAML)); err != nil {
		return nil, err
	}

	// values.yaml
	valuesYAML := `# Default values
replicaCount: 1
`
	if err := writeTarEntry(tw, chartName+"/values.yaml", []byte(valuesYAML)); err != nil {
		return nil, err
	}

	// templates/deployment.yaml
	deploymentYAML := `apiVersion: apps/v1
kind: Deployment
metadata:
  name: ` + chartName + `
spec:
  replicas: 1
`
	if err := writeTarEntry(tw, chartName+"/templates/deployment.yaml", []byte(deploymentYAML)); err != nil {
		return nil, err
	}

	if size > 1024 {
		padding := make([]byte, size-1024)
		if err := writeTarEntry(tw, chartName+"/padding.txt", padding); err != nil {
			return nil, err
		}
	}

	if err := tw.Close(); err != nil {
		return nil, err
	}
	if err := gw.Close(); err != nil {
		return nil, err
	}

	return buf.Bytes(), nil
}

// writeTarEntry 写入一个 tar 条目
func writeTarEntry(tw *tar.Writer, name string, data []byte) error {
	hdr := &tar.Header{
		Name: name,
		Size: int64(len(data)),
		Mode: 0644,
	}
	if err := tw.WriteHeader(hdr); err != nil {
		return err
	}
	_, err := tw.Write(data)
	return err
}

// 创建一个无效的 chart 文件
func createInvalidChartTgz() []byte {
	return []byte("this is not a valid tar.gz file")
}

// 建一个超过 2MB 的 chart 文件
func createLargeChartTgz(chartName, version string) ([]byte, error) {
	return createTestChartTgz(chartName, version, 2*1024*1024+512*1024)
}
