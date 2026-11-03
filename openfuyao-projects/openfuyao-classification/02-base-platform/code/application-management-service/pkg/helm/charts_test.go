/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

// Package helm
package helm

import (
	"context"
	"crypto/tls"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"helm.sh/helm/v3/pkg/action"
	"github.com/spf13/viper"
	"github.com/stretchr/testify/assert"
	"helm.sh/helm/v3/pkg/chart"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/client-go/rest"

	"application-management-service/pkg/constant"
	"application-management-service/pkg/models/helm"
)

// HTTP状态码常量
const (
	statusNotFound = 404
)

// 测试数据常量
const (
	testServerHost   = "https://test-marketplace.example.com"
	testChartVersion = "1.0.0"
	testRelease      = "test-release"
	minTLSVersion    = tls.VersionTLS12
)

// 测试响应常量
const (
	testResponseCode = 200
	testErrorCode    = 500
	testSuccessMsg   = "success"
)

// 配置常量
const (
	marketplaceKey   = "marketplace-service-host"
	chartDescription = "Test Chart Description"
	templateName     = "templates/deployment.yaml"
	templateContent  = "apiVersion: apps/v1\nkind: Deployment"
	valuesContent    = `{"replicas": 3}`
)

// ResponseStruct 定义API响应结构
type ResponseStruct struct {
	Code int    `json:"code"`
	Msg  string `json:"msg"`
	Data struct {
		Chart *chart.Chart `json:"chart"`
	} `json:"data"`
}

// TestCase 定义通用测试用例结构
type TestCase struct {
	name           string
	host           string
	repoName       string
	chartName      string
	chartVersion   string
	expectedSubstr string
}

// HTTPTestCase 定义HTTP测试用例结构
type HTTPTestCase struct {
	name         string
	statusCode   int
	responseBody []byte
	expectError  bool
	skipVerify   bool
}

// createTestChart 创建测试用的Chart
func createTestChart() *chart.Chart {
	return &chart.Chart{
		Metadata: &chart.Metadata{
			Name:        testChartName,
			Version:     testChartVersion,
			Description: chartDescription,
		},
		Templates: []*chart.File{
			{
				Name: templateName,
				Data: []byte(templateContent),
			},
		},
	}
}

// createTestKubeConfig 创建测试用的Kubernetes配置
func createTestKubeConfig() *rest.Config {
	return &rest.Config{
		Host: "https://test-cluster:6443",
		TLSClientConfig: rest.TLSClientConfig{
			Insecure: true,
		},
	}
}

// createFakeKubeClient 创建测试用的Kubernetes客户端
func createFakeKubeClient() kubernetes.Interface {
	return fake.NewSimpleClientset()
}

// createReleaseHelmClient 创建测试用的helm客户端
func createChartTestHelmClient() (*helmClient, error) {
	config := createTestKubeConfig()
	if config == nil {
		return nil, assert.AnError
	}

	client := &helmClient{}

	return client, nil
}

// setupTestEnvironment 设置测试环境
func setupTestEnvironment(t *testing.T) {
	// 设置测试环境变量
	t.Setenv("KUBECONFIG", "")

	// 清理viper配置
	viper.Reset()
}

// createReleaseTestDeployParam 创建测试用的部署参数
func createChartTestDeployParam() *helm.DeployParam {
	return &helm.DeployParam{
		Namespace: testNamespace,
		Release:   testRelease,
		ChartName: testChartName,
		Values:    valuesContent,
	}
}

// createResponseBody 创建响应体
func createResponseBody(code int, message string, chart *chart.Chart) ([]byte, error) {
	response := ResponseStruct{
		Code: code,
		Msg:  message,
	}
	if chart != nil {
		response.Data.Chart = chart
	}

	return json.Marshal(response)
}

// createTestServer 创建测试HTTP服务器
func createTestServer(statusCode int, responseBody []byte) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(statusCode)
		if len(responseBody) != 0 {
			w.Write(responseBody)
		}
	}))
}

// setupViperConfig 设置测试用的viper配置
func setupViperConfig(serverURL string) {
	viper.Set(marketplaceKey, serverURL)
}

func TestEnvironmentSetup(t *testing.T) {
	setupTestEnvironment(t)

	// 验证环境设置
	kubeConfig := createTestKubeConfig()
	assert.NotNil(t, kubeConfig)
	assert.NotEmpty(t, kubeConfig.Host)

	kubeClient := createFakeKubeClient()
	assert.NotNil(t, kubeClient)

	// 验证viper配置被重置
	viper.Set("test-key", "test-value")
	assert.Equal(t, "test-value", viper.GetString("test-key"))

	setupTestEnvironment(t)
	assert.Empty(t, viper.GetString("test-key"))
}

func TestKubernetesClientCreation(t *testing.T) {
	// 测试防止空指针引用的辅助函数
	config := createTestKubeConfig()
	assert.NotNil(t, config)
	assert.Equal(t, "https://test-cluster:6443", config.Host)

	client := createFakeKubeClient()
	assert.NotNil(t, client)

	// 验证客户端可以安全使用
	ctx := context.Background()
	namespaces, err := client.CoreV1().Namespaces().List(ctx, metav1.ListOptions{})
	assert.NoError(t, err)
	assert.NotNil(t, namespaces)
}

func TestHelmClientCreation(t *testing.T) {
	helmClient, err := createChartTestHelmClient()
	assert.NoError(t, err)
	assert.NotNil(t, helmClient)
}

// TestGetRollBack_WaitConfigApplied covers getRollBack: wait/timeout/maxHistory branches.
func TestGetRollBack_WaitConfigApplied(t *testing.T) {
	deployParam := createReleaseTestDeployParam()
	_ = deployParam // keep helper usage consistent with other helm tests

	clientset := fake.NewSimpleClientset(
		&corev1.ConfigMap{
			ObjectMeta: metav1.ObjectMeta{
				Name:      constant.ApplicationManagementServiceConfigmap,
				Namespace: constant.ApplicationManagementServiceDefaultNamespace,
			},
			Data: map[string]string{
				"marketplace-service-host": "http://marketplace-service",
				"helm-timeout":             "5s",
				"helm-wait-charts":         "- test-chart\n",
			},
		},
	)
	client := createTestHelmClientWithFake(clientset)

	rollback := client.getRollBack(&action.Configuration{}, 1, 10, testChartName)
	assert.NotNil(t, rollback)
	assert.True(t, rollback.Wait)
	assert.True(t, rollback.WaitForJobs)
	assert.Equal(t, time.Second*5, rollback.Timeout)
	assert.Equal(t, 10, rollback.MaxHistory)
}

// TestGetRollBack_TimeoutParseError_DoesNotEnableWait covers timeout parse failure branch.
func TestGetRollBack_TimeoutParseError_DoesNotEnableWait(t *testing.T) {
	clientset := fake.NewSimpleClientset(
		&corev1.ConfigMap{
			ObjectMeta: metav1.ObjectMeta{
				Name:      constant.ApplicationManagementServiceConfigmap,
				Namespace: constant.ApplicationManagementServiceDefaultNamespace,
			},
			Data: map[string]string{
				"marketplace-service-host": "http://marketplace-service",
				"helm-timeout":             "bad-duration",
				"helm-wait-charts":         "- test-chart\n",
			},
		},
	)
	client := createTestHelmClientWithFake(clientset)

	// maxHistory <= 0 should fall back to defaultRollbackHisNums.
	rollback := client.getRollBack(&action.Configuration{}, 1, 0, testChartName)
	assert.NotNil(t, rollback)
	assert.False(t, rollback.Wait)
	assert.False(t, rollback.WaitForJobs)
	assert.Equal(t, time.Duration(0), rollback.Timeout)
	assert.Equal(t, defaultRollbackHisNums, rollback.MaxHistory)
}

func TestGetHarborHttpClient(t *testing.T) {
	client := getHarborHttpClient()

	assert.NotNil(t, client)
	assert.Equal(t, time.Second*constant.DefaultHttpRequestSeconds, client.Timeout)

	transport, ok := client.Transport.(*http.Transport)
	assert.True(t, ok)
	assert.NotNil(t, transport)

	tlsConfig := transport.TLSClientConfig
	assert.NotNil(t, tlsConfig)
	assert.Equal(t, uint16(minTLSVersion), tlsConfig.MinVersion)
	assert.True(t, tlsConfig.InsecureSkipVerify)
}

func TestGetHttpRequestResponse(t *testing.T) {
	testChart := createTestChart()
	successBody, err := createResponseBody(testResponseCode, testSuccessMsg, testChart)
	assert.NoError(t, err)

	testCases := []HTTPTestCase{
		{
			name:         "success response",
			statusCode:   statusOK,
			responseBody: successBody,
			expectError:  false,
			skipVerify:   true,
		},
		{
			name:         "success without skip verify",
			statusCode:   statusOK,
			responseBody: successBody,
			expectError:  false,
			skipVerify:   false,
		},
		{
			name:         "server error",
			statusCode:   statusInternalServerError,
			responseBody: []byte("Internal Server Error"),
			expectError:  false,
			skipVerify:   true,
		},
		{
			name:         "not found",
			statusCode:   statusNotFound,
			responseBody: []byte("Not Found"),
			expectError:  false,
			skipVerify:   true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			server := createTestServer(tc.statusCode, tc.responseBody)
			defer server.Close()

			req, err := http.NewRequest(http.MethodGet, server.URL, nil)
			assert.NoError(t, err)

			resp, body, err := getHttpRequestResponse(req, tc.skipVerify)

			if tc.expectError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.NotNil(t, resp)
				assert.NotNil(t, body)
				assert.Equal(t, tc.statusCode, resp.StatusCode)
			}
		})
	}
}

type integrationTestCase struct {
	name         string
	statusCode   int
	responseCode int
	responseMsg  string
	expectError  bool
}

func TestHelmClientIntegration(t *testing.T) {
	testChart := createTestChart()

	integrationTestCases := []integrationTestCase{
		{
			name:         "success",
			statusCode:   statusOK,
			responseCode: testResponseCode,
			responseMsg:  testSuccessMsg,
			expectError:  false,
		},
		{
			name:         "http error",
			statusCode:   statusNotFound,
			responseCode: testResponseCode,
			responseMsg:  testSuccessMsg,
			expectError:  true,
		},
	}

	for _, tc := range integrationTestCases {
		t.Run(tc.name, func(t *testing.T) {
			runSingleIntegrationTest(t, tc, testChart)
		})
	}
}

func runSingleIntegrationTest(t *testing.T, tc integrationTestCase, testChart interface{}) {
	var responseBody []byte
	var err error

	if tc.statusCode == statusOK {
		chart, ok := testChart.(*chart.Chart)
		if !ok {
			t.Fatal("testChart type assertion failed: expected *chart.Chart")
		}
		responseBody, err = createResponseBody(tc.responseCode, tc.responseMsg, chart)
	} else {
		responseBody = []byte(tc.responseMsg)
	}
	assert.NoError(t, err)

	server := createTestServer(tc.statusCode, responseBody)
	defer server.Close()

	setupViperConfig(server.URL)

	marketplaceURL := getMarketplaceURL(server.URL, testRepoName, testChartName, testChartVersion)
	assert.NotEmpty(t, marketplaceURL)

	req, err := http.NewRequest(http.MethodGet, marketplaceURL, nil)
	assert.NoError(t, err)

	resp, body, err := getHttpRequestResponse(req, true)
	assert.NoError(t, err)

	validateIntegrationResponse(t, tc, resp, body)
}

func validateIntegrationResponse(t *testing.T, tc integrationTestCase, resp *http.Response, body []byte) {
	if tc.expectError && tc.statusCode != statusOK {
		assert.NotEqual(t, statusOK, resp.StatusCode)
		return
	}

	assert.Equal(t, statusOK, resp.StatusCode)

	if tc.statusCode == statusOK {
		var responseStruct ResponseStruct
		err := json.Unmarshal(body, &responseStruct)

		if tc.responseCode == testResponseCode {
			assert.NoError(t, err)
			assert.Equal(t, tc.responseCode, responseStruct.Code)
		} else {
			assert.NoError(t, err)
			assert.Equal(t, testErrorCode, responseStruct.Code)
		}
	}
}

func TestInvalidJSONResponse(t *testing.T) {
	invalidJSON := []byte("invalid json")
	server := createTestServer(statusOK, invalidJSON)
	defer server.Close()

	req, err := http.NewRequest(http.MethodGet, server.URL, nil)
	assert.NoError(t, err)

	resp, body, err := getHttpRequestResponse(req, true)
	assert.NoError(t, err)
	assert.Equal(t, statusOK, resp.StatusCode)

	var responseStruct ResponseStruct
	err = json.Unmarshal(body, &responseStruct)
	assert.Error(t, err)
}

func TestDeployRequestStructure(t *testing.T) {
	chart := createTestChart()
	deployParam := createChartTestDeployParam()

	deployRequest := &DeployRequest{
		Chart:       chart,
		DeployParam: deployParam,
	}

	assert.NotNil(t, deployRequest)
	assert.NotNil(t, deployRequest.Chart)
	assert.NotNil(t, deployRequest.DeployParam)
	assert.Equal(t, testChartName, deployRequest.Chart.Metadata.Name)
	assert.Equal(t, testChartVersion, deployRequest.Chart.Metadata.Version)
	assert.Equal(t, testNamespace, deployRequest.DeployParam.Namespace)
	assert.Equal(t, testRelease, deployRequest.DeployParam.Release)
}

func TestConstants(t *testing.T) {
	constantTestCases := []struct {
		actual   string
		expected string
	}{
		{token, "token"},
		{authorization, "Authorization"},
		{wwwAuthenticate, "Www-Authenticate"},
		{realmParam, "realm"},
		{serviceParam, "service"},
		{scopeParam, "scope"},
	}

	for _, tc := range constantTestCases {
		assert.Equal(t, tc.expected, tc.actual)
	}
}

func TestGetMarketplaceURL(t *testing.T) {
	url := getMarketplaceURL("http://marketplace", "repoA", "chartA", "1.2.3")
	assert.Equal(t, "http://marketplace/helm-repos/repoA/charts/chartA/versions/1.2.3/files?fileType=raw", url)
}

func TestGetHttpRequestResponse_InvalidRequestURL(t *testing.T) {
	req, err := http.NewRequest(http.MethodGet, "http://example.com", nil)
	assert.NoError(t, err)
	req.URL.Host = "%"

	resp, body, err := getHttpRequestResponse(req, true)
	assert.Error(t, err)
	assert.Nil(t, resp)
	assert.Nil(t, body)
}

func TestGetHttpRequestResponse_ConnectionRefused(t *testing.T) {
	req, err := http.NewRequest(http.MethodGet, "http://127.0.0.1:1", nil)
	assert.NoError(t, err)

	resp, body, err := getHttpRequestResponse(req, true)
	assert.Error(t, err)
	assert.Nil(t, resp)
	assert.Nil(t, body)
}

func TestCreateResponseBodyWithNilChart(t *testing.T) {
	body, err := createResponseBody(testResponseCode, testSuccessMsg, nil)
	assert.NoError(t, err)
	assert.NotNil(t, body)

	var response ResponseStruct
	err = json.Unmarshal(body, &response)
	assert.NoError(t, err)
	assert.Equal(t, testResponseCode, response.Code)
	assert.Nil(t, response.Data.Chart)
}

func TestChartStructure(t *testing.T) {
	chart := createTestChart()

	assert.NotNil(t, chart)
	assert.NotNil(t, chart.Metadata)
	assert.Equal(t, testChartName, chart.Metadata.Name)
	assert.Equal(t, testChartVersion, chart.Metadata.Version)
	assert.Equal(t, chartDescription, chart.Metadata.Description)

	assert.NotNil(t, chart.Templates)
	assert.Len(t, chart.Templates, 1)
	assert.Equal(t, templateName, chart.Templates[0].Name)
	assert.Equal(t, []byte(templateContent), chart.Templates[0].Data)
}

// BenchmarkGetMarketplaceURL 基准测试URL构造
func BenchmarkGetMarketplaceURL(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		url := getMarketplaceURL(testServerHost, testRepoName, testChartName, testChartVersion)
		if url == "" {
			b.Fatal("URL should not be empty")
		}
	}
}

// BenchmarkGetHarborHttpClient 基准测试HTTP客户端创建
func BenchmarkGetHarborHttpClient(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		client := getHarborHttpClient()
		if client == nil {
			b.Fatal("Failed to create HTTP client")
		}
	}
}
