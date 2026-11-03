/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *           http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package kube

import (
	"context"
	"errors"
	"os"
	"testing"
	"time"

	"github.com/agiledragon/gomonkey/v2"
	"helm.sh/helm/v3/pkg/chart"
	corev1 "k8s.io/api/core/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"

	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkeaddon "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/addon"
	bkeinit "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/initialize"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/constant"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/log"
)

const (
	testChartTimeout = 10
)

func TestChartConstants(t *testing.T) {
	if releaseNotFound != "release: not found" {
		t.Errorf("Expected releaseNotFound to be 'release: not found', got %s", releaseNotFound)
	}
}

func TestClientGetChartTimeout(t *testing.T) {
	c := &Client{}

	tests := []struct {
		name  string
		addon *confv1beta1.Product
		want  time.Duration
	}{
		{
			name:  "addon with timeout",
			addon: &confv1beta1.Product{Timeout: testChartTimeout},
			want:  testChartTimeout * time.Minute,
		},
		{
			name:  "addon without timeout",
			addon: &confv1beta1.Product{Timeout: 0},
			want:  bkeinit.DefaultAddonTimeout,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := c.getChartTimeout(tt.addon)
			if got != tt.want {
				t.Errorf("getChartTimeout() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestIsOCIReference(t *testing.T) {
	tests := []struct {
		name string
		ref  string
		want bool
	}{
		{
			name: "oci reference",
			ref:  "oci://registry.example.com/charts/mychart",
			want: true,
		},
		{
			name: "http reference",
			ref:  "http://example.com/chart.tgz",
			want: false,
		},
		{
			name: "https reference",
			ref:  "https://example.com/chart.tgz",
			want: false,
		},
		{
			name: "empty reference",
			ref:  "",
			want: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := isOCIReference(tt.ref)
			if got != tt.want {
				t.Errorf("isOCIReference() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestConvertToOCIURL(t *testing.T) {
	tests := []struct {
		name    string
		repoURL string
		want    string
	}{
		{
			name:    "https with chartrepo",
			repoURL: "https://192.168.100.202:30043/chartrepo/library",
			want:    "oci://192.168.100.202:30043/library",
		},
		{
			name:    "http with chartrepo",
			repoURL: "http://example.com/chartrepo/myrepo",
			want:    "oci://example.com/myrepo",
		},
		{
			name:    "no protocol",
			repoURL: "registry.example.com/chartrepo/charts",
			want:    "oci://registry.example.com/charts",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := convertToOCIURL(tt.repoURL)
			if got != tt.want {
				t.Errorf("convertToOCIURL() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestExtractRegistryHost(t *testing.T) {
	tests := []struct {
		name   string
		ociRef string
		want   string
	}{
		{
			name:   "oci with path",
			ociRef: "oci://registry.example.com/charts/mychart",
			want:   "registry.example.com",
		},
		{
			name:   "oci with port",
			ociRef: "oci://192.168.1.1:5000/library/nginx",
			want:   "192.168.1.1:5000",
		},
		{
			name:   "no oci prefix",
			ociRef: "registry.example.com/charts",
			want:   "registry.example.com",
		},
		{
			name:   "no path",
			ociRef: "oci://registry.example.com",
			want:   "registry.example.com",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := extractRegistryHost(tt.ociRef)
			if got != tt.want {
				t.Errorf("extractRegistryHost() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestAuthConfigSetters(t *testing.T) {
	auth := NewAuthConfig()

	testUsername := []byte("testuser")
	testPassword := []byte("testpass")
	testCaFile := []byte("ca-content")
	testCertFile := []byte("cert-content")
	testKeyFile := []byte("key-content")
	testCaPath := "/path/to/ca"
	testCertPath := "/path/to/cert"
	testKeyPath := "/path/to/key"

	auth.SetUsername(testUsername).
		SetPassword(testPassword).
		SetCaFile(testCaFile).
		SetCertFile(testCertFile).
		SetKeyFile(testKeyFile).
		SetCaFilePath(testCaPath).
		SetCertFilePath(testCertPath).
		SetKeyFilePath(testKeyPath).
		SetInsecureSkipTLSVerify(true)

	if string(auth.Username) != string(testUsername) {
		t.Errorf("Username = %v, want %v", string(auth.Username), string(testUsername))
	}
	if string(auth.Password) != string(testPassword) {
		t.Errorf("Password = %v, want %v", string(auth.Password), string(testPassword))
	}
	if string(auth.CaFile) != string(testCaFile) {
		t.Errorf("CaFile = %v, want %v", string(auth.CaFile), string(testCaFile))
	}
	if string(auth.CertFile) != string(testCertFile) {
		t.Errorf("CertFile = %v, want %v", string(auth.CertFile), string(testCertFile))
	}
	if string(auth.KeyFile) != string(testKeyFile) {
		t.Errorf("KeyFile = %v, want %v", string(auth.KeyFile), string(testKeyFile))
	}
	if auth.CaFilePath != testCaPath {
		t.Errorf("CaFilePath = %v, want %v", auth.CaFilePath, testCaPath)
	}
	if auth.CertFilePath != testCertPath {
		t.Errorf("CertFilePath = %v, want %v", auth.CertFilePath, testCertPath)
	}
	if auth.KeyFilePath != testKeyPath {
		t.Errorf("KeyFilePath = %v, want %v", auth.KeyFilePath, testKeyPath)
	}
	if !auth.InsecureSkipTLSVerify {
		t.Errorf("InsecureSkipTLSVerify = %v, want true", auth.InsecureSkipTLSVerify)
	}
}

func TestAuthConfigCleanup(t *testing.T) {
	auth := NewAuthConfig()
	auth.Username = []byte("user")
	auth.Password = []byte("pass")
	auth.CaFile = []byte("ca")
	auth.CertFile = []byte("cert")
	auth.KeyFile = []byte("key")

	auth.Cleanup()

	if len(auth.Username) != 0 {
		t.Errorf("Username not cleaned, got %v", auth.Username)
	}
	if len(auth.Password) != 0 {
		t.Errorf("Password not cleaned, got %v", auth.Password)
	}
	if len(auth.CaFile) != 0 {
		t.Errorf("CaFile not cleaned, got %v", auth.CaFile)
	}
	if len(auth.CertFile) != 0 {
		t.Errorf("CertFile not cleaned, got %v", auth.CertFile)
	}
	if len(auth.KeyFile) != 0 {
		t.Errorf("KeyFile not cleaned, got %v", auth.KeyFile)
	}
}

type mockClient struct {
	client.Client
	getFunc func(ctx context.Context, key client.ObjectKey, obj client.Object, opts ...client.GetOption) error
}

func (m *mockClient) Get(ctx context.Context, key client.ObjectKey, obj client.Object, opts ...client.GetOption) error {
	if m.getFunc != nil {
		return m.getFunc(ctx, key, obj, opts...)
	}
	return nil
}

func TestClientGetDataFromCMByKey(t *testing.T) {
	c := &Client{Log: nil}

	const (
		testNamespace = "test-ns"
		testName      = "test-cm"
		testKey       = "test-key"
		testValue     = "test-value"
	)

	tests := []struct {
		name      string
		mockFunc  func(ctx context.Context, key client.ObjectKey, obj client.Object, opts ...client.GetOption) error
		wantValue string
		wantErr   bool
	}{
		{
			name: "success",
			mockFunc: func(ctx context.Context, key client.ObjectKey, obj client.Object, opts ...client.GetOption) error {
				cm := obj.(*corev1.ConfigMap)
				cm.Data = map[string]string{testKey: testValue}
				return nil
			},
			wantValue: testValue,
			wantErr:   false,
		},
		{
			name: "configmap not found",
			mockFunc: func(ctx context.Context, key client.ObjectKey, obj client.Object, opts ...client.GetOption) error {
				return errors.New("not found")
			},
			wantValue: "",
			wantErr:   true,
		},
		{
			name: "key not found",
			mockFunc: func(ctx context.Context, key client.ObjectKey, obj client.Object, opts ...client.GetOption) error {
				cm := obj.(*corev1.ConfigMap)
				cm.Data = map[string]string{"other-key": "other-value"}
				return nil
			},
			wantValue: "",
			wantErr:   false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mockCli := &mockClient{getFunc: tt.mockFunc}
			got, err := c.getDataFromCMByKey(testName, testNamespace, testKey, mockCli)
			if (err != nil) != tt.wantErr {
				t.Errorf("getDataFromCMByKey() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if got != tt.wantValue {
				t.Errorf("getDataFromCMByKey() = %v, want %v", got, tt.wantValue)
			}
		})
	}
}

func TestClientGetDataFromSecretByKeys(t *testing.T) {
	c := &Client{Log: nil}

	const (
		testNamespace = "test-ns"
		testName      = "test-secret"
		testKey1      = "key1"
		testKey2      = "key2"
	)

	testValue1 := []byte("value1")
	testValue2 := []byte("value2")

	tests := []struct {
		name     string
		keys     []string
		mockFunc func(ctx context.Context, key client.ObjectKey, obj client.Object, opts ...client.GetOption) error
		wantData map[string][]byte
	}{
		{
			name: "all keys found",
			keys: []string{testKey1, testKey2},
			mockFunc: func(ctx context.Context, key client.ObjectKey, obj client.Object, opts ...client.GetOption) error {
				secret := obj.(*corev1.Secret)
				secret.Data = map[string][]byte{
					testKey1: testValue1,
					testKey2: testValue2,
				}
				return nil
			},
			wantData: map[string][]byte{
				testKey1: testValue1,
				testKey2: testValue2,
			},
		},
		{
			name: "key not found",
			keys: []string{testKey1, "missing-key"},
			mockFunc: func(ctx context.Context, key client.ObjectKey, obj client.Object, opts ...client.GetOption) error {
				secret := obj.(*corev1.Secret)
				secret.Data = map[string][]byte{testKey1: testValue1}
				return nil
			},
			wantData: map[string][]byte{
				testKey1:      testValue1,
				"missing-key": []byte(""),
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mockCli := &mockClient{getFunc: tt.mockFunc}
			got, _ := c.getDataFromSecretByKeys(testName, testNamespace, tt.keys, mockCli)
			for k, v := range tt.wantData {
				if string(got[k]) != string(v) {
					t.Errorf("getDataFromSecretByKeys() key %s = %v, want %v", k, got[k], v)
				}
			}
		})
	}
}

func TestClientGetChartValues(t *testing.T) {
	c := &Client{Log: nil}

	const (
		testNamespace = "test-ns"
		testName      = "test-cm"
		testYaml      = "key: value\nfoo: bar"
	)

	tests := []struct {
		name     string
		addon    *confv1beta1.Product
		mockFunc func(ctx context.Context, key client.ObjectKey, obj client.Object, opts ...client.GetOption) error
		wantErr  bool
	}{
		{
			name: "no values config",
			addon: &confv1beta1.Product{
				Name:               "test-addon",
				ValuesConfigMapRef: nil,
			},
			wantErr: false,
		},
		{
			name: "with values config",
			addon: &confv1beta1.Product{
				Name: "test-addon",
				ValuesConfigMapRef: &confv1beta1.ValuesConfigMapRef{
					Name:      testName,
					Namespace: testNamespace,
					ValuesKey: constant.ValuesYamlKey,
				},
			},
			mockFunc: func(ctx context.Context, key client.ObjectKey, obj client.Object, opts ...client.GetOption) error {
				cm := obj.(*corev1.ConfigMap)
				cm.Data = map[string]string{constant.ValuesYamlKey: testYaml}
				return nil
			},
			wantErr: false,
		},
		{
			name: "configmap not found",
			addon: &confv1beta1.Product{
				Name: "test-addon",
				ValuesConfigMapRef: &confv1beta1.ValuesConfigMapRef{
					Name:      testName,
					Namespace: testNamespace,
				},
			},
			mockFunc: func(ctx context.Context, key client.ObjectKey, obj client.Object, opts ...client.GetOption) error {
				return errors.New("not found")
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mockCli := &mockClient{getFunc: tt.mockFunc}
			_, err := c.getChartValues(tt.addon, testNamespace, mockCli)
			if (err != nil) != tt.wantErr {
				t.Errorf("getChartValues() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestClientGetChartRepoTLSCerts(t *testing.T) {
	c := &Client{Log: nil}

	const (
		testNamespace = "test-ns"
		testName      = "test-secret"
	)

	testCaData := []byte("ca-data")
	testCertData := []byte("cert-data")
	testKeyData := []byte("key-data")

	tests := []struct {
		name     string
		cfg      bkeinit.BkeConfig
		mockFunc func(ctx context.Context, key client.ObjectKey, obj client.Object, opts ...client.GetOption) error
		wantErr  bool
	}{
		{
			name: "no tls secret",
			cfg: bkeinit.BkeConfig{
				Cluster: confv1beta1.Cluster{
					ChartRepo: confv1beta1.Repo{
						TlsSecretRef: nil,
					},
				},
			},
			wantErr: false,
		},
		{
			name: "with tls secret",
			cfg: bkeinit.BkeConfig{
				Cluster: confv1beta1.Cluster{
					ChartRepo: confv1beta1.Repo{
						TlsSecretRef: &confv1beta1.TlsSecretRef{
							Name:      testName,
							Namespace: testNamespace,
						},
					},
				},
			},
			mockFunc: func(ctx context.Context, key client.ObjectKey, obj client.Object, opts ...client.GetOption) error {
				secret := obj.(*corev1.Secret)
				secret.Data = map[string][]byte{
					constant.CaKey:   testCaData,
					constant.CertKey: testCertData,
					constant.KeyKey:  testKeyData,
				}
				return nil
			},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mockCli := &mockClient{getFunc: tt.mockFunc}
			_, err := c.getChartRepoTLSCerts(tt.cfg, testNamespace, mockCli)
			if (err != nil) != tt.wantErr {
				t.Errorf("getChartRepoTLSCerts() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestClientGetChartRepoLoginInfo(t *testing.T) {
	c := &Client{Log: nil}

	const (
		testNamespace = "test-ns"
		testName      = "test-secret"
	)

	testUsername := []byte("testuser")
	testPassword := []byte("testpass")

	tests := []struct {
		name     string
		cfg      bkeinit.BkeConfig
		mockFunc func(ctx context.Context, key client.ObjectKey, obj client.Object, opts ...client.GetOption) error
		wantErr  bool
	}{
		{
			name: "no auth secret",
			cfg: bkeinit.BkeConfig{
				Cluster: confv1beta1.Cluster{
					ChartRepo: confv1beta1.Repo{
						AuthSecretRef: nil,
					},
				},
			},
			wantErr: false,
		},
		{
			name: "with auth secret",
			cfg: bkeinit.BkeConfig{
				Cluster: confv1beta1.Cluster{
					ChartRepo: confv1beta1.Repo{
						AuthSecretRef: &confv1beta1.AuthSecretRef{
							Name:      testName,
							Namespace: testNamespace,
						},
					},
				},
			},
			mockFunc: func(ctx context.Context, key client.ObjectKey, obj client.Object, opts ...client.GetOption) error {
				secret := obj.(*corev1.Secret)
				secret.Data = map[string][]byte{
					constant.UsernameKey: testUsername,
					constant.PasswordKey: testPassword,
				}
				return nil
			},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mockCli := &mockClient{getFunc: tt.mockFunc}
			username, password, err := c.getChartRepoLoginInfo(tt.cfg, testNamespace, mockCli)
			if (err != nil) != tt.wantErr {
				t.Errorf("getChartRepoLoginInfo() error = %v, wantErr %v", err, tt.wantErr)
			}
			if tt.cfg.Cluster.ChartRepo.AuthSecretRef != nil {
				if string(username) != string(testUsername) {
					t.Errorf("username = %v, want %v", string(username), string(testUsername))
				}
				if string(password) != string(testPassword) {
					t.Errorf("password = %v, want %v", string(password), string(testPassword))
				}
			}
		})
	}
}

func TestClientInstallChartAddon(t *testing.T) {
	c := &Client{Log: nil}

	const (
		testNamespace = "test-ns"
		testAddonName = "test-addon"
	)

	addon := &confv1beta1.Product{
		Name:      testAddonName,
		Namespace: testNamespace,
	}

	tests := []struct {
		name         string
		addonOperate bkeaddon.AddonOperate
		expectErr    bool
	}{
		{
			name:         "unknown operation",
			addonOperate: bkeaddon.AddonOperate("unknown"),
			expectErr:    true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := c.installChartAddon(addon, tt.addonOperate, testNamespace, bkeinit.BkeConfig{}, nil)
			if (err != nil) != tt.expectErr {
				t.Errorf("installChartAddon() error = %v, expectErr %v", err, tt.expectErr)
			}
		})
	}
}

func TestGetCertTmpPath(t *testing.T) {
	const (
		numZero  = 0
		numThree = 3
	)

	tests := []struct {
		name    string
		auth    *AuthConfig
		wantErr bool
	}{
		{
			name: "empty certs",
			auth: &AuthConfig{
				CaFile:   []byte(""),
				CertFile: []byte(""),
				KeyFile:  []byte(""),
			},
			wantErr: false,
		},
		{
			name: "with all certs",
			auth: &AuthConfig{
				CaFile:   []byte("ca-content"),
				CertFile: []byte("cert-content"),
				KeyFile:  []byte("key-content"),
			},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			auth, tmpDir, err := getCertTmpPath(tt.auth)
			if (err != nil) != tt.wantErr {
				t.Errorf("getCertTmpPath() error = %v, wantErr %v", err, tt.wantErr)
			}
			if tmpDir != "" {
				defer func() {
					_ = os.RemoveAll(tmpDir)
				}()
			}
			if err == nil {
				hasCA := len(tt.auth.CaFile) > numZero
				hasCert := len(tt.auth.CertFile) > numZero
				hasKey := len(tt.auth.KeyFile) > numZero

				if hasCA || hasCert || hasKey {
					if auth.CaFilePath == "" && auth.CertFilePath == "" && auth.KeyFilePath == "" {
						t.Errorf("At least one file path should be set")
					}
				}
			}
		})
	}
}

func TestFetchChartUniversal(t *testing.T) {
	auth := NewAuthConfig()

	tests := []struct {
		name      string
		chartRepo string
		chartName string
		version   string
		wantOCI   bool
	}{
		{
			name:      "oci reference",
			chartRepo: "oci://registry.example.com",
			chartName: "mychart",
			version:   "1.0.0",
			wantOCI:   true,
		},
		{
			name:      "http reference",
			chartRepo: "http://example.com/charts",
			chartName: "mychart",
			version:   "1.0.0",
			wantOCI:   false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patches := gomonkey.ApplyFunc(FetchChartOCI,
				func(repoURL, chartName, version string, auth *AuthConfig, logger *log.Logger) (*chart.Chart, error) {
					return nil, errors.New("mock oci error")
				})
			defer patches.Reset()

			patches.ApplyFunc(FetchChartTraditional,
				func(repoURL, chartName, version string, auth *AuthConfig, logger *log.Logger) (*chart.Chart, error) {
					return nil, errors.New("mock traditional error")
				})

			_, err := FetchChartUniversal(tt.chartRepo, tt.chartName, tt.version, auth, nil)
			if err == nil {
				t.Errorf("FetchChartUniversal() expected error, got nil")
			}
		})
	}
}

func TestNewAuthConfig(t *testing.T) {
	auth := NewAuthConfig()
	if auth == nil {
		t.Error("NewAuthConfig() returned nil")
	}
	if len(auth.Username) != 0 || len(auth.Password) != 0 {
		t.Error("NewAuthConfig() should initialize with empty credentials")
	}
	if auth.InsecureSkipTLSVerify {
		t.Error("NewAuthConfig() should initialize with InsecureSkipTLSVerify=false")
	}
}

func TestFetchChartPackageError(t *testing.T) {
	c := &Client{Log: nil}

	addon := &confv1beta1.Product{
		Name:    "test-chart",
		Version: "1.0.0",
	}

	cfg := bkeinit.BkeConfig{
		Cluster: confv1beta1.Cluster{
			ChartRepo: confv1beta1.Repo{
				Domain: "example.com",
			},
		},
	}

	mockCli := &mockClient{
		getFunc: func(ctx context.Context, key client.ObjectKey, obj client.Object, opts ...client.GetOption) error {
			return errors.New("mock error")
		},
	}

	patches := gomonkey.ApplyFunc(FetchChartUniversal,
		func(chartRepo, chartName, version string, auth *AuthConfig, logger *log.Logger) (*chart.Chart, error) {
			return nil, errors.New("mock fetch error")
		})
	defer patches.Reset()

	_, err := c.fetchChartPackage(addon, cfg, "test-ns", mockCli)
	if err == nil {
		t.Error("fetchChartPackage() expected error, got nil")
	}
}

func TestGetChartValuesWithInvalidYaml(t *testing.T) {
	c := &Client{Log: nil}

	const invalidYaml = "invalid: yaml: content: ["

	addon := &confv1beta1.Product{
		Name: "test-addon",
		ValuesConfigMapRef: &confv1beta1.ValuesConfigMapRef{
			Name:      "test-cm",
			Namespace: "test-ns",
		},
	}

	mockCli := &mockClient{
		getFunc: func(ctx context.Context, key client.ObjectKey, obj client.Object, opts ...client.GetOption) error {
			cm := obj.(*corev1.ConfigMap)
			cm.Data = map[string]string{constant.ValuesYamlKey: invalidYaml}
			return nil
		},
	}

	_, err := c.getChartValues(addon, "test-ns", mockCli)
	if err == nil {
		t.Error("getChartValues() with invalid yaml expected error, got nil")
	}
}

func TestGetChartValuesWithEmptyNamespace(t *testing.T) {
	c := &Client{Log: nil}

	addon := &confv1beta1.Product{
		Name: "test-addon",
		ValuesConfigMapRef: &confv1beta1.ValuesConfigMapRef{
			Name:      "test-cm",
			Namespace: "",
		},
	}

	mockCli := &mockClient{
		getFunc: func(ctx context.Context, key client.ObjectKey, obj client.Object, opts ...client.GetOption) error {
			cm := obj.(*corev1.ConfigMap)
			cm.Data = map[string]string{constant.ValuesYamlKey: "key: value"}
			return nil
		},
	}

	_, err := c.getChartValues(addon, "test-ns", mockCli)
	if err != nil {
		t.Errorf("getChartValues() unexpected error: %v", err)
	}
}

func TestGetChartRepoTLSCertsWithCustomKeys(t *testing.T) {
	c := &Client{Log: nil}

	const (
		customCaKey   = "custom-ca"
		customCertKey = "custom-cert"
		customKeyKey  = "custom-key"
	)

	testCaData := []byte("ca-data")
	testCertData := []byte("cert-data")
	testKeyData := []byte("key-data")

	cfg := bkeinit.BkeConfig{
		Cluster: confv1beta1.Cluster{
			ChartRepo: confv1beta1.Repo{
				TlsSecretRef: &confv1beta1.TlsSecretRef{
					Name:      "test-secret",
					Namespace: "test-ns",
					CaKey:     customCaKey,
					CertKey:   customCertKey,
					KeyKey:    customKeyKey,
				},
			},
		},
	}

	mockCli := &mockClient{
		getFunc: func(ctx context.Context, key client.ObjectKey, obj client.Object, opts ...client.GetOption) error {
			secret := obj.(*corev1.Secret)
			secret.Data = map[string][]byte{
				customCaKey:   testCaData,
				customCertKey: testCertData,
				customKeyKey:  testKeyData,
			}
			return nil
		},
	}

	auth, err := c.getChartRepoTLSCerts(cfg, "test-ns", mockCli)
	if err != nil {
		t.Errorf("getChartRepoTLSCerts() unexpected error: %v", err)
	}
	if string(auth.CaFile) != string(testCaData) {
		t.Errorf("CaFile = %v, want %v", string(auth.CaFile), string(testCaData))
	}
}

func TestGetChartRepoLoginInfoWithCustomKeys(t *testing.T) {
	c := &Client{Log: nil}

	const (
		customUsernameKey = "custom-user"
		customPasswordKey = "custom-pass"
	)

	testUsername := []byte("testuser")
	testPassword := []byte("testpass")

	cfg := bkeinit.BkeConfig{
		Cluster: confv1beta1.Cluster{
			ChartRepo: confv1beta1.Repo{
				AuthSecretRef: &confv1beta1.AuthSecretRef{
					Name:        "test-secret",
					Namespace:   "test-ns",
					UsernameKey: customUsernameKey,
					PasswordKey: customPasswordKey,
				},
			},
		},
	}

	mockCli := &mockClient{
		getFunc: func(ctx context.Context, key client.ObjectKey, obj client.Object, opts ...client.GetOption) error {
			secret := obj.(*corev1.Secret)
			secret.Data = map[string][]byte{
				customUsernameKey: testUsername,
				customPasswordKey: testPassword,
			}
			return nil
		},
	}

	username, password, err := c.getChartRepoLoginInfo(cfg, "test-ns", mockCli)
	if err != nil {
		t.Errorf("getChartRepoLoginInfo() unexpected error: %v", err)
	}
	if string(username) != string(testUsername) {
		t.Errorf("username = %v, want %v", string(username), string(testUsername))
	}
	if string(password) != string(testPassword) {
		t.Errorf("password = %v, want %v", string(password), string(testPassword))
	}
}
