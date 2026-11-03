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

/*
Package helm include helm related struct
*/
package helm

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"

	"application-management-service/pkg/constant"
)

const (
	testDisplayName           = "test-repo"
	testURL                   = "https://test.example.com"
	testSecretName            = "test-secret"
	testConfigMapName         = "test-configmap"
	testUsername              = "testuser"
	testPassword              = "testpass"
	testCertFile              = "/path/to/cert"
	testKeyFile               = "/path/to/key"
	testCAFile                = "/path/to/ca"
	testLocalHarborHost       = "harbor.local.com"
	testOfficialHarborHost    = "harbor.official.com"
	testProjectName           = "test-project"
	testOAuthHost             = "oauth.example.com"
	testConsoleHost           = "console.example.com"
	testAlertHost             = "alert.example.com"
	testMonitoringHost        = "monitoring.example.com"
	testTagsURL               = "https://harbor.official.com/api/v2.0/projects/test/repositories/charts/tags"
	testServerName            = "test-server"
	testChartLimit            = "100"
	testMarketPlaceHost       = "marketplace.example.com"
	testHelmTimeout           = "300s"
	testChartName             = "nginx"
	maxLineLength             = 120
	maxFunctionLines          = 50
	maxDuplicateLines         = 10
	expectedMarketplaceScenes = 2
	expectedRepositoryCount   = 2
	expectedTaggedChartCount  = 3
)

func createTestChartRepository() *ChartRepository {
	return &ChartRepository{
		TypeMeta: metav1.TypeMeta{
			Kind:       "ChartRepository",
			APIVersion: "v1",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      testDisplayName,
			Namespace: "default",
		},
		Spec: ChartRepositorySpec{
			DisplayName: testDisplayName,
			URL:         testURL,
			BasicAuth: SecretReference{
				Name: testSecretName,
			},
			TLS: SecretReference{
				Name: testSecretName,
			},
			CA: ConfigMapReference{
				Name: testConfigMapName,
			},
			InsecureSkipTLSVerify: true,
			PassCredentialsAll:    false,
		},
	}
}

func createTestChartRepositorySpec() ChartRepositorySpec {
	return ChartRepositorySpec{
		DisplayName: testDisplayName,
		URL:         testURL,
		CA: ConfigMapReference{
			Name: testConfigMapName,
		},
		BasicAuth: SecretReference{
			Name: testSecretName,
		},
		TLS: SecretReference{
			Name: testSecretName,
		},
		InsecureSkipTLSVerify: false,
		PassCredentialsAll:    true,
	}
}

func createTestSecretReference() SecretReference {
	return SecretReference{
		Name: testSecretName,
	}
}

func createTestConfigMapReference() ConfigMapReference {
	return ConfigMapReference{
		Name: testConfigMapName,
	}
}

func TestChartRepositoryDeepCopy(t *testing.T) {
	original := createTestChartRepository()
	if original == nil {
		t.Fatal("Failed to create test ChartRepository")
	}

	copied := original.DeepCopy()
	if copied == nil {
		t.Fatal("DeepCopy returned nil")
	}

	if copied.Spec.DisplayName != original.Spec.DisplayName {
		t.Errorf("Expected DisplayName %s, got %s",
			original.Spec.DisplayName, copied.Spec.DisplayName)
	}

	if copied.Spec.URL != original.Spec.URL {
		t.Errorf("Expected URL %s, got %s", original.Spec.URL, copied.Spec.URL)
	}
}

func TestChartRepositoryDeepCopyInto(t *testing.T) {
	original := createTestChartRepository()
	if original == nil {
		t.Fatal("Failed to create test ChartRepository")
	}

	var target ChartRepository
	original.DeepCopyInto(&target)

	if target.Spec.DisplayName != original.Spec.DisplayName {
		t.Errorf("Expected DisplayName %s, got %s",
			original.Spec.DisplayName, target.Spec.DisplayName)
	}

	if target.Spec.BasicAuth.Name != original.Spec.BasicAuth.Name {
		t.Errorf("Expected BasicAuth.Name %s, got %s",
			original.Spec.BasicAuth.Name, target.Spec.BasicAuth.Name)
	}
}

func TestChartRepositoryDeepCopyObject(t *testing.T) {
	original := createTestChartRepository()
	if original == nil {
		t.Fatal("Failed to create test ChartRepository")
	}

	obj := original.DeepCopyObject()
	if obj == nil {
		t.Fatal("DeepCopyObject returned nil")
	}

	copied, ok := obj.(*ChartRepository)
	if !ok {
		t.Fatal("DeepCopyObject returned wrong type")
	}

	if copied.Spec.URL != original.Spec.URL {
		t.Errorf("Expected URL %s, got %s", original.Spec.URL, copied.Spec.URL)
	}
}

func TestChartRepositorySpecDeepCopy(t *testing.T) {
	original := createTestChartRepositorySpec()
	copied := original.DeepCopy()

	if copied == nil {
		t.Fatal("DeepCopy returned nil")
	}

	if copied.DisplayName != original.DisplayName {
		t.Errorf("Expected DisplayName %s, got %s",
			original.DisplayName, copied.DisplayName)
	}

	if copied.InsecureSkipTLSVerify != original.InsecureSkipTLSVerify {
		t.Errorf("Expected InsecureSkipTLSVerify %v, got %v",
			original.InsecureSkipTLSVerify, copied.InsecureSkipTLSVerify)
	}
}

func TestChartRepositorySpecDeepCopyInto(t *testing.T) {
	original := createTestChartRepositorySpec()
	var target ChartRepositorySpec

	original.DeepCopyInto(&target)

	if target.URL != original.URL {
		t.Errorf("Expected URL %s, got %s", original.URL, target.URL)
	}

	if target.PassCredentialsAll != original.PassCredentialsAll {
		t.Errorf("Expected PassCredentialsAll %v, got %v",
			original.PassCredentialsAll, target.PassCredentialsAll)
	}
}

func TestSecretReferenceDeepCopy(t *testing.T) {
	original := createTestSecretReference()
	copied := original.DeepCopy()

	if copied == nil {
		t.Fatal("DeepCopy returned nil")
	}

	if copied.Name != original.Name {
		t.Errorf("Expected Name %s, got %s", original.Name, copied.Name)
	}
}

func TestSecretReferenceDeepCopyInto(t *testing.T) {
	original := createTestSecretReference()
	var target SecretReference

	original.DeepCopyInto(&target)

	if target.Name != original.Name {
		t.Errorf("Expected Name %s, got %s", original.Name, target.Name)
	}
}

func TestConfigMapReferenceDeepCopy(t *testing.T) {
	original := createTestConfigMapReference()
	copied := original.DeepCopy()

	if copied == nil {
		t.Fatal("DeepCopy returned nil")
	}

	if copied.Name != original.Name {
		t.Errorf("Expected Name %s, got %s", original.Name, copied.Name)
	}
}

func TestConfigMapReferenceDeepCopyInto(t *testing.T) {
	original := createTestConfigMapReference()
	var target ConfigMapReference

	original.DeepCopyInto(&target)

	if target.Name != original.Name {
		t.Errorf("Expected Name %s, got %s", original.Name, target.Name)
	}
}

func TestAddHelmChartRepositoryToScheme(t *testing.T) {
	scheme := runtime.NewScheme()
	if scheme == nil {
		t.Fatal("Failed to create runtime scheme")
	}

	err := AddHelmChartRepositoryToScheme(scheme)
	if err != nil {
		t.Errorf("AddHelmChartRepositoryToScheme failed: %v", err)
	}

	gvk := schema.GroupVersionKind{
		Group:   constant.CRDRepoGroup,
		Version: constant.CRDRepoVersion,
		Kind:    "ChartRepository",
	}

	if !scheme.Recognizes(gvk) {
		t.Error("Scheme does not recognize ChartRepository type")
	}
}

func TestSafeRepoEntryCreation(t *testing.T) {
	entry := SafeRepoEntry{
		Name:                  testDisplayName,
		URL:                   testURL,
		Username:              []byte(testUsername),
		Password:              []byte(testPassword),
		CertFile:              testCertFile,
		KeyFile:               testKeyFile,
		CAFile:                testCAFile,
		InsecureSkipTLSVerify: true,
		PassCredentialsAll:    false,
	}

	if entry.Name != testDisplayName {
		t.Errorf("Expected Name %s, got %s", testDisplayName, entry.Name)
	}

	if entry.URL != testURL {
		t.Errorf("Expected URL %s, got %s", testURL, entry.URL)
	}

	if string(entry.Username) != testUsername {
		t.Errorf("Expected Username %s, got %s", testUsername, string(entry.Username))
	}

	if !entry.InsecureSkipTLSVerify {
		t.Error("Expected InsecureSkipTLSVerify to be true")
	}
}

func TestApplicationManagementServiceConfigCreation(t *testing.T) {
	config := ApplicationManagementServiceConfig{
		LocalHarborDisplayName:    testDisplayName,
		LocalHarborHost:           testLocalHarborHost,
		LocalHarborProject:        testProjectName,
		OfficialHarborDisplayName: testDisplayName,
		OfficialHarborHost:        testOfficialHarborHost,
		OfficialHarborProject:     testProjectName,
		MarketplaceScenes:         []string{"scene1", "scene2"},
		OAuthServerHost:           testOAuthHost,
		ConsoleServerHost:         testConsoleHost,
		ConsoleWebsiteHost:        testConsoleHost,
		AlertHost:                 testAlertHost,
		MonitoringHost:            testMonitoringHost,
		OfficialHarborTagsURL:     testTagsURL,
		InsecureSkipVerify:        "true",
		ServerName:                testServerName,
		ChartLimit:                testChartLimit,
		MarketPlaceServiceHost:    testMarketPlaceHost,
		HelmTimeout:               testHelmTimeout,
		HelmWaitCharts:            []string{testChartName},
	}

	if config.LocalHarborHost != testLocalHarborHost {
		t.Errorf("Expected LocalHarborHost %s, got %s",
			testLocalHarborHost, config.LocalHarborHost)
	}

	if len(config.MarketplaceScenes) != expectedMarketplaceScenes {
		t.Errorf("Expected %d marketplace scenes, got %d",
			expectedMarketplaceScenes, len(config.MarketplaceScenes))
	}

	if config.HelmTimeout != testHelmTimeout {
		t.Errorf("Expected HelmTimeout %s, got %s", testHelmTimeout, config.HelmTimeout)
	}
}

func TestOfficialTagsResponseCreation(t *testing.T) {
	response := OfficialTagsResponse{
		Name:         testChartName,
		TaggedCharts: []string{"v1.0.0", "v1.1.0", "latest"},
	}

	if response.Name != testChartName {
		t.Errorf("Expected Name %s, got %s", testChartName, response.Name)
	}

	if len(response.TaggedCharts) != expectedTaggedChartCount {
		t.Errorf("Expected %d tagged charts, got %d",
			expectedTaggedChartCount, len(response.TaggedCharts))
	}

	expectedFirstTag := "v1.0.0"
	if response.TaggedCharts[0] != expectedFirstTag {
		t.Errorf("Expected first tag %s, got %s",
			expectedFirstTag, response.TaggedCharts[0])
	}
}

func TestChartRepositoryListCreation(t *testing.T) {
	repo1 := createTestChartRepository()
	repo2 := createTestChartRepository()
	if repo1 == nil || repo2 == nil {
		t.Fatal("Failed to create test repositories")
	}

	repo2.ObjectMeta.Name = "second-repo"

	list := ChartRepositoryList{
		TypeMeta: metav1.TypeMeta{
			Kind:       "ChartRepositoryList",
			APIVersion: "v1",
		},
		ListMeta: metav1.ListMeta{},
		Items:    []ChartRepository{*repo1, *repo2},
	}

	if len(list.Items) != expectedRepositoryCount {
		t.Errorf("Expected %d items, got %d", expectedRepositoryCount, len(list.Items))
	}

	if list.Items[1].ObjectMeta.Name != "second-repo" {
		t.Errorf("Expected second repo name 'second-repo', got %s",
			list.Items[1].ObjectMeta.Name)
	}
}
