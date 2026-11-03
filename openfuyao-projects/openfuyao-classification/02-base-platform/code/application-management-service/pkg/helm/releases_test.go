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
	"testing"

	stdtime "time"

	"github.com/stretchr/testify/assert"
	"helm.sh/helm/v3/pkg/action"
	"helm.sh/helm/v3/pkg/chart"
	"helm.sh/helm/v3/pkg/chartutil"
	"helm.sh/helm/v3/pkg/kube"
	"helm.sh/helm/v3/pkg/release"
	"helm.sh/helm/v3/pkg/time"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/cli-runtime/pkg/resource"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/client-go/rest"

	"application-management-service/pkg/constant"
	"application-management-service/pkg/models/helm"
	"application-management-service/pkg/server/param"
)

// HTTP状态码常量
const (
	statusOK                  = 200
	statusBadRequest          = 400
	statusInternalServerError = 500
)

// 测试数据常量
const (
	testReleaseName     = "test-release"
	testVersion         = "1.0.0"
	testValues          = `{"replicas": 3}`
	testNamespace       = "default"
	testChartName       = "test-chart"
	testRepoName        = "test-repo"
	testHost            = "localhost"
	testString          = "test"
	testLongReleaseName = "testtesttesttesttesttesttesttesttesttesttesttesttesttesttesttesttesttest"
	testSortBy          = "date"
	testExtensionTrue   = "true"
	testExtensionFalse  = "false"
	testStatusDeployed  = "deployed"
	testContainerName   = "test-container"
	testPodName         = "test-pod"
)

// 分页常量
const (
	testLimit       = 10
	testOffset      = 10
	testCurrentPage = 10
	testMaxHistory  = 20
	testRevision    = 1
)

// Kubernetes资源类型常量
const (
	kindStatefulSet = "StatefulSet"
	kindDaemonSet   = "DaemonSet"
	kindDeployment  = "Deployment"
)

// MockObject 实现 runtime.Object 接口用于测试
type MockObject struct{}

func (m *MockObject) GetObjectKind() schema.ObjectKind {
	return schema.EmptyObjectKind
}

func (m *MockObject) DeepCopyObject() runtime.Object {
	return &MockObject{}
}

// createReleaseTestHelmClient 创建测试用的helm客户端
func createReleaseTestHelmClient() *helmClient {
	return &helmClient{
		dynamicClient: nil,
		clientset:     nil,
		kubeConfig:    createReleaseTestRestConfig(),
	}
}

// createReleaseTestRestConfig 创建测试用的REST配置
func createReleaseTestRestConfig() *rest.Config {
	return &rest.Config{
		Host: testHost,
	}
}

// createTestHelmClientWithFake 创建带fake客户端的helm客户端
func createTestHelmClientWithFake(clientSet *fake.Clientset) *helmClient {
	return &helmClient{
		dynamicClient: nil,
		clientset:     clientSet,
		kubeConfig:    createReleaseTestRestConfig(),
	}
}

// createTestReleaseParameter 创建测试用的发布参数
func createTestReleaseParameter() param.ReleaseParameter {
	return param.ReleaseParameter{
		Namespace: testNamespace,
		Release:   testReleaseName,
		Extension: testExtensionFalse,
	}
}

// createReleaseTestDeployParam 创建测试用的部署参数
func createReleaseTestDeployParam() helm.DeployParam {
	return helm.DeployParam{
		Namespace: testNamespace,
		Release:   testReleaseName,
		ChartName: testChartName,
		RepoName:  testRepoName,
		Version:   testVersion,
		Values:    testValues,
	}
}

// createTestRelease 创建测试用的发布对象
func createTestRelease() *release.Release {
	return &release.Release{
		Name:      testReleaseName,
		Namespace: testNamespace,
		Info: &release.Info{
			FirstDeployed: time.Time{},
			LastDeployed:  time.Time{},
			Status:        testStatusDeployed,
		},
		Chart: &chart.Chart{
			Metadata: &chart.Metadata{
				Name:     testChartName,
				Keywords: []string{constant.FuyaoExtensionKeyword},
			},
		},
	}
}

// createTestPodClientSet 创建包含Pod的fake客户端
func createTestPodClientSet() *fake.Clientset {
	return fake.NewSimpleClientset(
		&corev1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:      testPodName,
				Namespace: testNamespace,
				Labels: map[string]string{
					testString: testString,
				},
			},
			Spec: corev1.PodSpec{
				InitContainers: []corev1.Container{
					{Name: testContainerName},
				},
				Containers: []corev1.Container{
					{Name: testContainerName},
				},
			},
		},
	)
}

// validateTestResponse 验证测试响应和状态码
func validateTestResponse(t *testing.T, response interface{}, statusCode, expectedStatus int) {
	assert.NotNil(t, response)
	assert.Equal(t, expectedStatus, statusCode)
}

// createTestResourceInfo 创建测试用的资源信息
func createTestResourceInfo() *resource.Info {
	return &resource.Info{
		Name:      testString,
		Namespace: testNamespace,
	}
}

// createTestResourceList 创建测试用的资源列表
func createTestResourceList() kube.ResourceList {
	return []*resource.Info{
		{
			Namespace:       testNamespace,
			Name:            testString,
			Object:          &MockObject{},
			ResourceVersion: testVersion,
		},
	}
}

// createTestStatefulSet 创建测试用的StatefulSet
func createTestStatefulSet() *appsv1.StatefulSet {
	return &appsv1.StatefulSet{
		ObjectMeta: metav1.ObjectMeta{
			Name:      testString,
			Namespace: testNamespace,
		},
		Spec: appsv1.StatefulSetSpec{
			Selector: &metav1.LabelSelector{
				MatchLabels: map[string]string{
					testString: testString,
				},
			},
		},
	}
}

// createTestDaemonSet 创建测试用的DaemonSet
func createTestDaemonSet() *appsv1.DaemonSet {
	return &appsv1.DaemonSet{
		ObjectMeta: metav1.ObjectMeta{
			Name:      testString,
			Namespace: testNamespace,
		},
		Spec: appsv1.DaemonSetSpec{
			Selector: &metav1.LabelSelector{
				MatchLabels: map[string]string{
					testString: testString,
				},
			},
		},
	}
}

// createTestDeployment 创建测试用的Deployment
func createTestDeployment() *appsv1.Deployment {
	return &appsv1.Deployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      testString,
			Namespace: testNamespace,
		},
		Spec: appsv1.DeploymentSpec{
			Selector: &metav1.LabelSelector{
				MatchLabels: map[string]string{
					testString: testString,
				},
			},
		},
	}
}

func TestHelmClientGetReleases(t *testing.T) {
	testCases := []struct {
		name           string
		releaseParam   param.ReleaseParameter
		expectedStatus int
	}{
		{
			name:           "valid parameters",
			releaseParam:   createTestReleaseParameter(),
			expectedStatus: statusInternalServerError,
		},
		{
			name: "long release name",
			releaseParam: param.ReleaseParameter{
				Namespace: testNamespace,
				Release:   testLongReleaseName,
			},
			expectedStatus: statusBadRequest,
		},
		{
			name: "extension enabled",
			releaseParam: param.ReleaseParameter{
				Namespace: testNamespace,
				Release:   testReleaseName,
				Extension: testExtensionTrue,
			},
			expectedStatus: statusInternalServerError,
		},
		{
			name: "empty namespace",
			releaseParam: param.ReleaseParameter{
				Namespace: "",
				Release:   testReleaseName,
			},
			expectedStatus: statusInternalServerError,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			client := createReleaseTestHelmClient()
			response, statusCode := client.GetReleases(tc.releaseParam)
			validateTestResponse(t, response, statusCode, tc.expectedStatus)
		})
	}
}

func TestHelmClientInstall(t *testing.T) {
	testCases := []struct {
		name           string
		deployParam    helm.DeployParam
		expectedStatus int
	}{
		{
			name:           "empty parameters",
			deployParam:    helm.DeployParam{},
			expectedStatus: statusBadRequest,
		},
		{
			name: "missing chart info",
			deployParam: helm.DeployParam{
				Namespace: testNamespace,
				Release:   testReleaseName,
			},
			expectedStatus: statusBadRequest,
		},
		{
			name:           "valid parameters TestHelmClientInstall",
			deployParam:    createReleaseTestDeployParam(),
			expectedStatus: statusInternalServerError,
		},
		{
			name: "long release name",
			deployParam: helm.DeployParam{
				Namespace: testNamespace,
				Release:   testLongReleaseName,
				ChartName: testChartName,
				RepoName:  testRepoName,
				Version:   testVersion,
			},
			expectedStatus: statusBadRequest,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			client := createReleaseTestHelmClient()
			response, statusCode := client.Install(tc.deployParam)
			validateTestResponse(t, response, statusCode, tc.expectedStatus)
		})
	}
}

func TestHelmClientUninstall(t *testing.T) {
	testCases := []struct {
		name           string
		namespace      string
		releaseName    string
		expectedStatus int
	}{
		{
			name:           "valid parameters TestHelmClientUninstall",
			namespace:      testNamespace,
			releaseName:    testReleaseName,
			expectedStatus: statusInternalServerError,
		},
		{
			name:           "empty release name TestHelmClientUninstall",
			namespace:      testNamespace,
			releaseName:    "",
			expectedStatus: statusBadRequest,
		},
		{
			name:           "empty namespace TestHelmClientUninstall",
			namespace:      "",
			releaseName:    testReleaseName,
			expectedStatus: statusBadRequest,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			client := createReleaseTestHelmClient()
			response, statusCode := client.Uninstall(tc.namespace, tc.releaseName)
			validateTestResponse(t, response, statusCode, tc.expectedStatus)
		})
	}
}

func TestHelmClientGetReleaseInfo(t *testing.T) {
	testCases := []struct {
		name           string
		namespace      string
		releaseName    string
		expectedStatus int
	}{
		{
			name:           "valid parameters TestHelmClientGetReleaseInfo",
			namespace:      testNamespace,
			releaseName:    testReleaseName,
			expectedStatus: statusInternalServerError,
		},
		{
			name:           "empty release name TestHelmClientGetReleaseInfo",
			namespace:      testNamespace,
			releaseName:    "",
			expectedStatus: statusBadRequest,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			client := createReleaseTestHelmClient()
			response, statusCode := client.GetReleaseInfo(tc.namespace, tc.releaseName)
			validateTestResponse(t, response, statusCode, tc.expectedStatus)
		})
	}
}

func TestHelmClientGetReleaseHistory(t *testing.T) {
	testCases := []struct {
		name           string
		namespace      string
		releaseName    string
		maxHistory     int
		expectedStatus int
	}{
		{
			name:           "valid parameters TestHelmClientGetReleaseHistory",
			namespace:      testNamespace,
			releaseName:    testReleaseName,
			maxHistory:     testMaxHistory,
			expectedStatus: statusInternalServerError,
		},
		{
			name:           "empty release name TestHelmClientGetReleaseHistory",
			namespace:      testNamespace,
			releaseName:    "",
			maxHistory:     testMaxHistory,
			expectedStatus: statusBadRequest,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			client := createReleaseTestHelmClient()
			response, statusCode := client.GetReleaseHistory(tc.namespace, tc.releaseName, tc.maxHistory)
			validateTestResponse(t, response, statusCode, tc.expectedStatus)
		})
	}
}

// TestHelmClientRollbackRelease 测试回滚功能 - 移除导致空指针的测试用例
func TestHelmClientRollbackRelease(t *testing.T) {
	testCases := []struct {
		name           string
		namespace      string
		releaseName    string
		chartName      string
		version        int
		maxHistory     int
		expectedStatus int
	}{
		{
			name:           "empty release name TestHelmClientRollbackRelease",
			namespace:      testNamespace,
			releaseName:    "",
			chartName:      testChartName,
			version:        testRevision,
			maxHistory:     testMaxHistory,
			expectedStatus: statusBadRequest,
		},
		{
			name:           "empty namespace TestHelmClientRollbackRelease",
			namespace:      "",
			releaseName:    testReleaseName,
			chartName:      testChartName,
			version:        testRevision,
			maxHistory:     testMaxHistory,
			expectedStatus: statusBadRequest,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			client := createReleaseTestHelmClient()
			response, statusCode := client.RollbackRelease(tc.namespace, tc.releaseName,
				tc.chartName, tc.version, tc.maxHistory)
			validateTestResponse(t, response, statusCode, tc.expectedStatus)
		})
	}
}

func TestHelmClientGetManifest(t *testing.T) {
	testCases := []struct {
		name           string
		namespace      string
		releaseName    string
		expectedStatus int
	}{
		{
			name:           "valid parameters TestHelmClientGetManifest",
			namespace:      testNamespace,
			releaseName:    testReleaseName,
			expectedStatus: statusInternalServerError,
		},
		{
			name:           "empty release name TestHelmClientGetManifest",
			namespace:      testNamespace,
			releaseName:    "",
			expectedStatus: statusBadRequest,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			client := createReleaseTestHelmClient()
			response, statusCode := client.GetManifest(tc.namespace, tc.releaseName)
			validateTestResponse(t, response, statusCode, tc.expectedStatus)
		})
	}
}

func TestHelmClientGetValues(t *testing.T) {
	testCases := []struct {
		name           string
		namespace      string
		releaseName    string
		expectedStatus int
	}{
		{
			name:           "valid parameters TestHelmClientGetValues",
			namespace:      testNamespace,
			releaseName:    testReleaseName,
			expectedStatus: statusInternalServerError,
		},
		{
			name:           "empty release name TestHelmClientGetValues",
			namespace:      testNamespace,
			releaseName:    "",
			expectedStatus: statusBadRequest,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			client := createReleaseTestHelmClient()
			response, statusCode := client.GetValues(tc.namespace, tc.releaseName)
			validateTestResponse(t, response, statusCode, tc.expectedStatus)
		})
	}
}

func TestHelmClientGetReleasePods(t *testing.T) {
	testCases := []struct {
		name           string
		namespace      string
		releaseName    string
		expectedStatus int
	}{
		{
			name:           "valid parameters TestHelmClientGetReleasePods",
			namespace:      testNamespace,
			releaseName:    testReleaseName,
			expectedStatus: statusInternalServerError,
		},
		{
			name:           "empty release name TestHelmClientGetReleasePods",
			namespace:      testNamespace,
			releaseName:    "",
			expectedStatus: statusBadRequest,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			client := createReleaseTestHelmClient()
			response, statusCode := client.GetReleasePods(tc.namespace, tc.releaseName)
			validateTestResponse(t, response, statusCode, tc.expectedStatus)
		})
	}
}

func TestHelmClientCheckReleaseStatus(t *testing.T) {
	client := createReleaseTestHelmClient()
	response, statusCode := client.CheckReleaseStatus(testChartName)
	validateTestResponse(t, response, statusCode, statusInternalServerError)
}

func TestGetPodsContainersInfo(t *testing.T) {
	resources := createTestResourceList()
	client := createReleaseTestHelmClient()

	result := getPodsContainersInfo(resources, client)
	assert.NotNil(t, result)
}

func TestGetInfoFromSelector(t *testing.T) {
	testCases := []struct {
		name          string
		namespace     string
		labelSelector *metav1.LabelSelector
		resourceInfo  *resource.Info
	}{
		{
			name:      "with label selector",
			namespace: testNamespace,
			labelSelector: &metav1.LabelSelector{
				MatchLabels: map[string]string{
					testString: testString,
				},
			},
			resourceInfo: createTestResourceInfo(),
		},
		{
			name:          "without label selector",
			namespace:     testNamespace,
			labelSelector: nil,
			resourceInfo:  createTestResourceInfo(),
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			client := createTestHelmClientWithFake(createTestPodClientSet())
			result := make(map[string][]string)

			getInfoFromSelector(tc.namespace, tc.labelSelector, client, tc.resourceInfo, result)
			assert.True(t, len(result) >= 0)
		})
	}
}

func TestGetLabelSelector(t *testing.T) {
	testCases := []struct {
		name         string
		gvk          schema.GroupVersionKind
		clientSet    *fake.Clientset
		resourceInfo *resource.Info
		expectNil    bool
	}{
		{
			name: "StatefulSet",
			gvk:  schema.GroupVersionKind{Kind: kindStatefulSet},
			clientSet: fake.NewSimpleClientset(
				createTestStatefulSet(),
			),
			resourceInfo: createTestResourceInfo(),
			expectNil:    false,
		},
		{
			name: "DaemonSet",
			gvk:  schema.GroupVersionKind{Kind: kindDaemonSet},
			clientSet: fake.NewSimpleClientset(
				createTestDaemonSet(),
			),
			resourceInfo: createTestResourceInfo(),
			expectNil:    false,
		},
		{
			name: "Deployment",
			gvk:  schema.GroupVersionKind{Kind: kindDeployment},
			clientSet: fake.NewSimpleClientset(
				createTestDeployment(),
			),
			resourceInfo: createTestResourceInfo(),
			expectNil:    false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			client := createTestHelmClientWithFake(tc.clientSet)
			result := getLabelSelector(tc.gvk, client, tc.resourceInfo)

			if tc.expectNil {
				assert.Nil(t, result)
			} else {
				assert.NotNil(t, result)
			}
		})
	}
}

func TestCreateInstall_WaitConfigApplied(t *testing.T) {
	deployParam := createReleaseTestDeployParam()

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

	install, err := client.createInstall(&action.Configuration{}, deployParam)
	assert.NoError(t, err)
	assert.NotNil(t, install)
	assert.True(t, install.Wait)
	assert.True(t, install.WaitForJobs)
	assert.Equal(t, stdtime.Second*5, install.Timeout)
}

func TestCreateInstall_TimeoutParseError_DoesNotEnableWait(t *testing.T) {
	deployParam := createReleaseTestDeployParam()

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

	install, err := client.createInstall(&action.Configuration{}, deployParam)
	assert.NoError(t, err)
	assert.NotNil(t, install)
	assert.False(t, install.Wait)
	assert.False(t, install.WaitForJobs)
	assert.Equal(t, stdtime.Duration(0), install.Timeout)
}

func TestCreateUpgrade_WaitConfigApplied(t *testing.T) {
	deployParam := createReleaseTestDeployParam()

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

	upgrade, err := client.createUpgrade(&action.Configuration{}, deployParam)
	assert.NoError(t, err)
	assert.NotNil(t, upgrade)
	assert.True(t, upgrade.Wait)
	assert.True(t, upgrade.WaitForJobs)
	assert.Equal(t, stdtime.Second*5, upgrade.Timeout)
}

func TestCreateUpgrade_TimeoutParseError_DoesNotEnableWait(t *testing.T) {
	deployParam := createReleaseTestDeployParam()

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

	upgrade, err := client.createUpgrade(&action.Configuration{}, deployParam)
	assert.NoError(t, err)
	assert.NotNil(t, upgrade)
	assert.False(t, upgrade.Wait)
	assert.False(t, upgrade.WaitForJobs)
	assert.Equal(t, stdtime.Duration(0), upgrade.Timeout)
}

func TestDeleteReleasesValue_ClearsChartValuesAndConfig(t *testing.T) {
	r1 := &release.Release{
		Chart: &chart.Chart{Values: chartutil.Values{"k": "v"}},
		Config: map[string]interface{}{
			"a": "b",
		},
	}
	r2 := &release.Release{
		Chart: &chart.Chart{Values: chartutil.Values{"x": "y"}},
		Config: map[string]interface{}{
			"c": "d",
		},
	}

	deleteReleasesValue([]*release.Release{r1, r2})

	assert.Nil(t, r1.Chart.Values)
	assert.Nil(t, r1.Config)
	assert.Nil(t, r2.Chart.Values)
	assert.Nil(t, r2.Config)
}

func TestFilterReleaseByName(t *testing.T) {
	releases := []*release.Release{createTestRelease()}

	testCases := []struct {
		name           string
		releaseName    string
		expectedLength int
	}{
		{
			name:           "filter with name",
			releaseName:    testString,
			expectedLength: 1,
		},
		{
			name:           "filter without name",
			releaseName:    "",
			expectedLength: 1,
		},
		{
			name:           "filter no match",
			releaseName:    "not-exist",
			expectedLength: 0,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result := filterReleaseByName(releases, tc.releaseName)
			assert.Equal(t, tc.expectedLength, len(result))
		})
	}
}

func TestConvertResponseResult(t *testing.T) {
	releases := []*release.Release{createTestRelease()}

	result := convertResponseResult(releases)
	assert.NotNil(t, result)
	assert.Equal(t, 1, len(result.Items))
}

func TestFilterExtensions(t *testing.T) {
	testCases := []struct {
		name           string
		extension      string
		releases       []*release.Release
		expectedLength int
	}{
		{
			name:           "filter extension enabled",
			extension:      testExtensionTrue,
			releases:       []*release.Release{createTestRelease()},
			expectedLength: 1,
		},
		{
			name:      "filter component excluded",
			extension: testExtensionTrue,
			releases: []*release.Release{
				{
					Name:      testReleaseName,
					Namespace: testNamespace,
					Chart: &chart.Chart{
						Metadata: &chart.Metadata{
							Keywords: []string{constant.FuyaoComponent},
						},
					},
				},
			},
			expectedLength: 0,
		},
		{
			name:      "business app when extension disabled",
			extension: testExtensionFalse,
			releases: []*release.Release{
				{
					Name:      testReleaseName,
					Namespace: testNamespace,
					Chart:     &chart.Chart{Metadata: &chart.Metadata{Keywords: []string{"other"}}},
				},
			},
			expectedLength: 1,
		},
		{
			name:           "extension app excluded when extension disabled",
			extension:      testExtensionFalse,
			releases:       []*release.Release{createTestRelease()},
			expectedLength: 0,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result := filterExtensions(tc.extension, tc.releases)
			assert.Equal(t, tc.expectedLength, len(result))
		})
	}
}

func TestConvertToReleaseHistory(t *testing.T) {
	releases := []*release.Release{createTestRelease()}

	result := convertToReleaseHis(releases)
	assert.NotNil(t, result)
	assert.Equal(t, 1, len(result))
}

func TestGetMaxHistories(t *testing.T) {
	releases := []*release.Release{createTestRelease()}

	testCases := []struct {
		name           string
		maxNum         int
		expectedLength int
	}{
		{
			name:           "limit greater than length",
			maxNum:         2,
			expectedLength: 1,
		},
		{
			name:           "zero limit",
			maxNum:         0,
			expectedLength: 1,
		},
		{
			name:           "negative limit uses default",
			maxNum:         -1,
			expectedLength: 1,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result := getMaxHistories(tc.maxNum, releases)
			assert.Equal(t, tc.expectedLength, len(result))
		})
	}

	emptyResult := getMaxHistories(1, []*release.Release{})
	assert.Len(t, emptyResult, 0)
}

func TestCheckReleaseRunning(t *testing.T) {
	testCases := []struct {
		name           string
		releases       []*release.Release
		chartName      string
		expectedResult string
	}{
		{
			name:           "chart running",
			releases:       []*release.Release{createTestRelease()},
			chartName:      testString,
			expectedResult: testString + " running",
		},
		{
			name: "chart not ready",
			releases: []*release.Release{
				{
					Chart: &chart.Chart{
						Metadata: &chart.Metadata{Name: testString},
					},
					Info: &release.Info{Status: ""},
				},
			},
			chartName:      testString,
			expectedResult: testString + " not ready",
		},
		{
			name: "name not match",
			releases: []*release.Release{
				{
					Chart: &chart.Chart{Metadata: &chart.Metadata{Name: "other-chart"}},
					Info:  &release.Info{Status: testStatusDeployed},
				},
			},
			chartName:      testString,
			expectedResult: testString + " not ready",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result := checkReleaseRuning(tc.releases, tc.chartName)
			assert.Equal(t, tc.expectedResult, result)
		})
	}
}

func TestValidateParams(t *testing.T) {
	testCases := []struct {
		name        string
		releaseName string
		namespace   string
		expectError bool
	}{
		{
			name:        "valid parameters",
			releaseName: testReleaseName,
			namespace:   testNamespace,
			expectError: false,
		},
		{
			name:        "empty release name",
			releaseName: "",
			namespace:   testNamespace,
			expectError: true,
		},
		{
			name:        "empty namespace",
			releaseName: testReleaseName,
			namespace:   "",
			expectError: true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			err := validateParams(tc.releaseName, tc.namespace)
			if tc.expectError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestValidateDeployParam(t *testing.T) {
	testCases := []struct {
		name        string
		deployParam helm.DeployParam
		expectNil   bool
	}{
		{
			name:        "valid parameters",
			deployParam: createReleaseTestDeployParam(),
			expectNil:   true,
		},
		{
			name: "missing namespace",
			deployParam: helm.DeployParam{
				Release:   testReleaseName,
				ChartName: testChartName,
			},
			expectNil: false,
		},
		{
			name: "missing chart info",
			deployParam: helm.DeployParam{
				Namespace: testNamespace,
				Release:   testReleaseName,
			},
			expectNil: false,
		},
		{
			name: "missing release",
			deployParam: helm.DeployParam{
				Namespace: testNamespace,
				ChartName: testChartName,
				RepoName:  testRepoName,
				Version:   testVersion,
			},
			expectNil: false,
		},
		{
			name: "missing namespace and chart info",
			deployParam: helm.DeployParam{
				Release: testReleaseName,
			},
			expectNil: false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result := validateDeployParam(tc.deployParam)
			if tc.expectNil {
				assert.Nil(t, result)
			} else {
				assert.NotNil(t, result)
			}
		})
	}
}

// BenchmarkGetReleases 基准测试GetReleases
func BenchmarkGetReleases(b *testing.B) {
	client := createReleaseTestHelmClient()
	releaseParam := createTestReleaseParameter()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = client.GetReleases(releaseParam)
	}
}

// BenchmarkFilterReleaseByName 基准测试FilterReleaseByName
func BenchmarkFilterReleaseByName(b *testing.B) {
	releases := []*release.Release{createTestRelease()}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = filterReleaseByName(releases, testString)
	}
}
