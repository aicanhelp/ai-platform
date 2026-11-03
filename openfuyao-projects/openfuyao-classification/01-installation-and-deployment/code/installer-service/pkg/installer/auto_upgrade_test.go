/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 * http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package installer

import (
	"archive/tar"
	"compress/gzip"
	"os"
	"path/filepath"
	"strings"
	"testing"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	dynamicfake "k8s.io/client-go/dynamic/fake"
	fake "k8s.io/client-go/kubernetes/fake"
)

// Test constants
const (
	testValidDir             = "testdata"
	testValidFileName        = "testfile"
	testNonExistentPath      = "/non/existent/path"
	testEmptyString          = ""
	testContent              = "test content"
	testFileName             = "test.txt"
	testFileMode             = 0644
	testPatchName            = "test"
	testScriptPath           = "/test/path"
	testScriptName           = "testPatch"
	testInvalidGzipFileName  = "invalid.gz"
	testInvalidTarGzFileName = "invalid.tar.gz"
	testTempFilePrefix       = "testfile"
	testTarGzFileName        = "test.tar.gz"
)

// Test data setup
func setupTestData(t *testing.T) (string, string) {
	testDir := t.TempDir()
	tarGzPath := filepath.Join(testDir, testTarGzFileName)

	if err := createValidTarGzFile(tarGzPath); err != nil {
		t.Fatalf("Failed to create test tar.gz file: %v", err)
	}

	return testDir, tarGzPath
}

// createValidTarGzFile creates a valid tar.gz file for testing
func createValidTarGzFile(filePath string) error {
	file, err := os.OpenFile(filePath, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, testFileMode)
	if err != nil {
		return err
	}
	defer file.Close()

	gzipWriter := gzip.NewWriter(file)
	defer gzipWriter.Close()

	tarWriter := tar.NewWriter(gzipWriter)
	defer tarWriter.Close()

	header := &tar.Header{
		Name: testFileName,
		Mode: testFileMode,
		Size: int64(len(testContent)),
	}

	if err := tarWriter.WriteHeader(header); err != nil {
		return err
	}

	_, err = tarWriter.Write([]byte(testContent))
	return err
}

// TestValidateInputParameters tests the validateInputParameters function
func TestValidateInputParameters(t *testing.T) {
	testCases := []struct {
		name       string
		patchPath  string
		patchName  string
		expectErr  bool
		errMessage string
	}{
		{
			name:       "valid parameters",
			patchPath:  "/valid/path",
			patchName:  "validName",
			expectErr:  false,
			errMessage: "",
		},
		{
			name:       "empty patch path",
			patchPath:  testEmptyString,
			patchName:  "validName",
			expectErr:  true,
			errMessage: "patch path cannot be empty",
		},
		{
			name:       "empty patch name",
			patchPath:  "/valid/path",
			patchName:  testEmptyString,
			expectErr:  true,
			errMessage: "patch name cannot be empty",
		},
		{
			name:       "both parameters empty",
			patchPath:  testEmptyString,
			patchName:  testEmptyString,
			expectErr:  true,
			errMessage: "patch path cannot be empty",
		},
	}

	runParameterTestCases(t, testCases)
}

// runParameterTestCases executes test cases for parameter validation
func runParameterTestCases(t *testing.T, testCases []struct {
	name       string
	patchPath  string
	patchName  string
	expectErr  bool
	errMessage string
}) {
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			err := validateInputParameters(tc.patchPath, tc.patchName)
			assertErrorCondition(t, err, tc.expectErr, tc.errMessage)
		})
	}
}

// TestNormalizePath tests the normalizePath function
func TestNormalizePath(t *testing.T) {
	testCases := []struct {
		name        string
		inputPath   string
		expectErr   bool
		expectClean bool
	}{
		{
			name:        "valid absolute path",
			inputPath:   "/tmp/test",
			expectErr:   false,
			expectClean: true,
		},
		{
			name:        "empty path",
			inputPath:   testEmptyString,
			expectErr:   true,
			expectClean: false,
		},
		{
			name:        "relative path",
			inputPath:   "relative/path",
			expectErr:   false,
			expectClean: true,
		},
	}

	runPathTestCases(t, testCases)
}

// runPathTestCases executes test cases for path normalization
func runPathTestCases(t *testing.T, testCases []struct {
	name        string
	inputPath   string
	expectErr   bool
	expectClean bool
}) {
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result, err := normalizePath(tc.inputPath)

			if tc.expectErr {
				if err == nil {
					t.Errorf("Expected error but got none")
				}
			} else {
				if err != nil {
					t.Errorf("Unexpected error: %v", err)
				}
				if tc.expectClean && strings.Contains(result, "..") {
					t.Errorf("Path contains suspicious pattern: %s", result)
				}
				if result == "" {
					t.Errorf("Expected non-empty result")
				}
			}
		})
	}
}

// TestNormalizeFileName tests the normalizeFileName function
func TestNormalizeFileName(t *testing.T) {
	testCases := []struct {
		name        string
		inputName   string
		expectErr   bool
		errContains string
	}{
		{
			name:        "valid file name",
			inputName:   "valid-file-name.tar.gz",
			expectErr:   false,
			errContains: "",
		},
		{
			name:        "empty file name",
			inputName:   testEmptyString,
			expectErr:   true,
			errContains: "file name cannot be empty",
		},
		{
			name:        "file name with command injection attempt",
			inputName:   "file; rm -rf /",
			expectErr:   true,
			errContains: "suspicious character",
		},
	}

	runFileNameTestCases(t, testCases)
}

// runFileNameTestCases executes test cases for file name normalization
func runFileNameTestCases(t *testing.T, testCases []struct {
	name        string
	inputName   string
	expectErr   bool
	errContains string
}) {
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result, err := normalizeFileName(tc.inputName)

			if tc.expectErr {
				if err == nil {
					t.Errorf("Expected error but got none")
				} else if !strings.Contains(err.Error(), tc.errContains) {
					t.Errorf("Expected error containing '%s', got '%v'", tc.errContains, err)
				}
			} else {
				if err != nil {
					t.Errorf("Unexpected error: %v", err)
				}
				if result == "" {
					t.Errorf("Expected non-empty result")
				}
				if result != filepath.Base(tc.inputName) {
					t.Errorf("Expected normalized name '%s', got '%s'",
						filepath.Base(tc.inputName), result)
				}
			}
		})
	}
}

// TestValidateDirectoryExists tests the validateDirectoryExists function
func TestValidateDirectoryExists(t *testing.T) {
	tempDir := t.TempDir()

	testCases := []struct {
		name        string
		path        string
		expectErr   bool
		errContains string
	}{
		{
			name:        "existing directory",
			path:        tempDir,
			expectErr:   false,
			errContains: "",
		},
		{
			name:        "non-existent directory",
			path:        testNonExistentPath,
			expectErr:   true,
			errContains: "path does not exist",
		},
		{
			name:        "path is a file not directory",
			path:        createTempFile(t),
			expectErr:   true,
			errContains: "path is not a directory",
		},
	}

	runDirectoryTestCases(t, testCases)
}

// runDirectoryTestCases executes test cases for directory validation
func runDirectoryTestCases(t *testing.T, testCases []struct {
	name        string
	path        string
	expectErr   bool
	errContains string
}) {
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			err := validateDirectoryExists(tc.path)
			assertErrorCondition(t, err, tc.expectErr, tc.errContains)
		})
	}
}

// TestValidateTarGzFormat tests the validateTarGzFormat function
func TestValidateTarGzFormat(t *testing.T) {
	testDir, validTarGzPath := setupTestData(t)

	testCases := []struct {
		name        string
		setupFile   func(t *testing.T) string
		expectErr   bool
		errContains string
	}{
		{
			name: "invalid gzip format",
			setupFile: func(t *testing.T) string {
				return createInvalidGzipFile(t, testDir)
			},
			expectErr:   true,
			errContains: "invalid gzip format",
		},
		{
			name: "valid tar.gz file",
			setupFile: func(t *testing.T) string {
				return validTarGzPath
			},
			expectErr:   false,
			errContains: "",
		},
		{
			name: "invalid tar format",
			setupFile: func(t *testing.T) string {
				return createInvalidTarFile(t, testDir)
			},
			expectErr:   true,
			errContains: "invalid tar format",
		},
	}

	runTarGzFormatTestCases(t, testCases)
}

// runTarGzFormatTestCases executes test cases for tar.gz format validation
func runTarGzFormatTestCases(t *testing.T, testCases []struct {
	name        string
	setupFile   func(t *testing.T) string
	expectErr   bool
	errContains string
}) {
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			filePath := tc.setupFile(t)
			err := validateTarGzFormat(filePath)
			assertErrorCondition(t, err, tc.expectErr, tc.errContains)
		})
	}
}

// TestValidateTarGzFile tests the validateTarGzFile function
func TestValidateTarGzFile(t *testing.T) {
	testDir, validTarGzPath := setupTestData(t)

	testCases := []struct {
		name        string
		setupFile   func(t *testing.T) string
		expectErr   bool
		errContains string
	}{
		{
			name: "valid tar.gz file",
			setupFile: func(t *testing.T) string {
				return validTarGzPath
			},
			expectErr:   false,
			errContains: "",
		},
		{
			name: "non-existent file",
			setupFile: func(t *testing.T) string {
				return filepath.Join(testDir, "nonexistent.tar.gz")
			},
			expectErr:   true,
			errContains: "file does not exist",
		},
		{
			name: "directory instead of file",
			setupFile: func(t *testing.T) string {
				return testDir
			},
			expectErr:   true,
			errContains: "file is not a regular file",
		},
	}

	runTarGzFileTestCases(t, testCases)
}

// runTarGzFileTestCases executes test cases for tar.gz file validation
func runTarGzFileTestCases(t *testing.T, testCases []struct {
	name        string
	setupFile   func(t *testing.T) string
	expectErr   bool
	errContains string
}) {
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			filePath := tc.setupFile(t)
			err := validateTarGzFile(filePath)
			assertErrorCondition(t, err, tc.expectErr, tc.errContains)
		})
	}
}

// TestValidatePatchFiles tests the validatePatchFiles function
func TestValidatePatchFiles(t *testing.T) {
	testDir, _ := setupTestData(t)

	testCases := []struct {
		name        string
		patchPath   string
		patchName   string
		expectErr   bool
		errContains string
	}{
		{
			name:        "valid patch files",
			patchPath:   testDir,
			patchName:   testPatchName,
			expectErr:   false,
			errContains: "",
		},
		{
			name:        "empty patch path",
			patchPath:   testEmptyString,
			patchName:   testPatchName,
			expectErr:   true,
			errContains: "patch path cannot be empty",
		},
		{
			name:        "non-existent directory",
			patchPath:   testNonExistentPath,
			patchName:   testPatchName,
			expectErr:   true,
			errContains: "path does not exist",
		},
		{
			name:        "non-existent tar.gz file",
			patchPath:   testDir,
			patchName:   "nonexistent",
			expectErr:   true,
			errContains: "file does not exist",
		},
	}

	runPatchFilesTestCases(t, testCases)
}

// runPatchFilesTestCases executes test cases for patch files validation
func runPatchFilesTestCases(t *testing.T, testCases []struct {
	name        string
	patchPath   string
	patchName   string
	expectErr   bool
	errContains string
}) {
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			err := validatePatchFiles(tc.patchPath, tc.patchName)
			assertErrorCondition(t, err, tc.expectErr, tc.errContains)
		})
	}
}

// TestBuildUpgradeScript tests the buildUpgradeScript function
func TestBuildUpgradeScript(t *testing.T) {
	script := buildUpgradeScript(testScriptPath, testScriptName, "127.0.0.1", false)

	if script == "" {
		t.Error("Expected non-empty script")
	}

	if !strings.Contains(script, testScriptPath) {
		t.Error("Script should contain the test path")
	}

	if !strings.Contains(script, testScriptName) {
		t.Error("Script should contain the test name")
	}

	assertScriptContainsRequiredParts(t, script)
}

// assertScriptContainsRequiredParts verifies script contains all required parts
func assertScriptContainsRequiredParts(t *testing.T, script string) {
	expectedParts := []string{
		"PATCH_PATH",
		"PATCH_NAME",
		"Check offline package file",
		"Extract offline package",
		"Sync images",
		"Copy specified binary files",
	}

	for _, part := range expectedParts {
		if !strings.Contains(script, part) {
			t.Errorf("Script should contain '%s'", part)
		}
	}
}

// Helper functions

// assertErrorCondition asserts error conditions in tests
func assertErrorCondition(t *testing.T, err error, expectErr bool, errContains string) {
	if expectErr {
		if err == nil {
			t.Errorf("Expected error but got none")
		} else if errContains != "" && !strings.Contains(err.Error(), errContains) {
			t.Errorf("Expected error containing '%s', got '%v'", errContains, err)
		}
	} else {
		if err != nil {
			t.Errorf("Expected no error, got %v", err)
		}
	}
}

// createTempFile creates a temporary file and returns its path
func createTempFile(t *testing.T) string {
	tempFile, err := os.CreateTemp(t.TempDir(), testTempFilePrefix)
	if err != nil {
		t.Fatalf("Failed to create temp file: %v", err)
	}
	defer tempFile.Close()

	return tempFile.Name()
}

// createInvalidGzipFile creates an invalid gzip file for testing
func createInvalidGzipFile(t *testing.T, dir string) string {
	filePath := filepath.Join(dir, testInvalidGzipFileName)
	err := os.WriteFile(filePath, []byte("not a gzip file"), testFileMode)
	if err != nil {
		t.Fatalf("Failed to create invalid gzip file: %v", err)
	}
	return filePath
}

// createInvalidTarFile creates an invalid tar file for testing
func createInvalidTarFile(t *testing.T, dir string) string {
	filePath := filepath.Join(dir, testInvalidTarGzFileName)
	file, err := os.OpenFile(filePath, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, testFileMode)
	if err != nil {
		t.Fatalf("Failed to create file: %v", err)
	}
	defer file.Close()

	gzipWriter := gzip.NewWriter(file)
	defer gzipWriter.Close()

	// Write invalid tar data
	_, err = gzipWriter.Write([]byte("invalid tar data"))
	if err != nil {
		t.Fatalf("Failed to write invalid tar data: %v", err)
	}

	return filePath
}

// TestGetBootstrapIpFromUnstructured tests extraction of bootstrap IP from JSON/YAML
func TestGetBootstrapIpFromUnstructured(t *testing.T) {
	cases := []struct {
		name    string
		input   string
		wantIP  string
		wantErr bool
	}{
		{name: "valid", input: `{"apiVersion":"v1","kind":"BKECluster","spec":{"clusterConfig":{"cluster":{"imageRepo":{"ip":"192.168.0.10"}}}}}`, wantIP: "192.168.0.10", wantErr: false},
		{name: "invalid-ip", input: `{"apiVersion":"v1","kind":"BKECluster","spec":{"clusterConfig":{"cluster":{"imageRepo":{"ip":"not-an-ip"}}}}}`, wantErr: true},
		{name: "missing-field", input: `{"apiVersion":"v1","kind":"BKECluster","spec":{}}`, wantErr: true},
		{name: "empty-ip", input: `{"apiVersion":"v1","kind":"BKECluster","spec":{"clusterConfig":{"cluster":{"imageRepo":{"ip":""}}}}}`, wantErr: true},
		{name: "raw-not-object", input: "just a plain scalar string", wantErr: true},
		{name: "decode-fail", input: ": invalid yaml: :::", wantErr: true},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			c := &installerClient{}
			ip, err := c.getBootstrapIpFromUnstructured(tc.input)
			if tc.wantErr {
				if err == nil {
					t.Fatalf("expected error, got none")
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if ip != tc.wantIP {
				t.Fatalf("expected ip %s, got %q", tc.wantIP, ip)
			}
		})
	}
}

// TestGetBootstrapIpFromCluster uses a fake dynamic client to ensure getBootstrapIpFromCluster
// returns the expected IP when the BKECluster CR exists in the cluster namespace.
func TestGetBootstrapIpFromCluster(t *testing.T) {
	cases := []struct {
		name      string
		setupObjs []*unstructured.Unstructured
		cluster   string
		wantIP    string
		wantErr   bool
		errSubstr string
	}{
		{
			name: "success",
			setupObjs: []*unstructured.Unstructured{
				{Object: map[string]interface{}{"apiVersion": "bke.bocloud.com/v1beta1", "kind": "BKECluster", "metadata": map[string]interface{}{"name": "ok", "namespace": "ok"}, "spec": map[string]interface{}{"clusterConfig": map[string]interface{}{"cluster": map[string]interface{}{"imageRepo": map[string]interface{}{"ip": "10.0.0.5"}}}}}},
			},
			cluster: "ok",
			wantIP:  "10.0.0.5",
		},
		{name: "dynamic-nil", setupObjs: nil, cluster: "x", wantErr: true, errSubstr: "dynamic client is nil"},
		{name: "missing-field", setupObjs: []*unstructured.Unstructured{{Object: map[string]interface{}{"apiVersion": "bke.bocloud.com/v1beta1", "kind": "BKECluster", "metadata": map[string]interface{}{"name": "c", "namespace": "c"}, "spec": map[string]interface{}{}}}}, cluster: "c", wantErr: true, errSubstr: "missing field"},
		{name: "empty-ip", setupObjs: []*unstructured.Unstructured{{Object: map[string]interface{}{"apiVersion": "bke.bocloud.com/v1beta1", "kind": "BKECluster", "metadata": map[string]interface{}{"name": "c2", "namespace": "c2"}, "spec": map[string]interface{}{"clusterConfig": map[string]interface{}{"cluster": map[string]interface{}{"imageRepo": map[string]interface{}{"ip": ""}}}}}}}, cluster: "c2", wantErr: true, errSubstr: "is empty"},
		{name: "invalid-ip", setupObjs: []*unstructured.Unstructured{{Object: map[string]interface{}{"apiVersion": "bke.bocloud.com/v1beta1", "kind": "BKECluster", "metadata": map[string]interface{}{"name": "c3", "namespace": "c3"}, "spec": map[string]interface{}{"clusterConfig": map[string]interface{}{"cluster": map[string]interface{}{"imageRepo": map[string]interface{}{"ip": "not-ip"}}}}}}}, cluster: "c3", wantErr: true, errSubstr: "invalid IP address format"},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			var c *installerClient
			if tc.setupObjs == nil {
				c = &installerClient{}
			} else {
				scheme := runtime.NewScheme()
				_ = corev1.AddToScheme(scheme)
				bkeGVR := schema.GroupVersionResource{Group: "bke.bocloud.com", Version: "v1beta1", Resource: "bkeclusters"}
				listKinds := map[schema.GroupVersionResource]string{bkeGVR: "BKEClusterList"}
				// convert []*unstructured.Unstructured to []runtime.Object accepted by fake
				objs := make([]runtime.Object, 0, len(tc.setupObjs))
				for _, u := range tc.setupObjs {
					objs = append(objs, u)
				}
				dyn := dynamicfake.NewSimpleDynamicClientWithCustomListKinds(scheme, listKinds, objs...)
				c = &installerClient{dynamicClient: dyn}
			}

			ip, err := c.getBootstrapIpFromCluster(tc.cluster)
			if tc.wantErr {
				if err == nil {
					t.Fatalf("expected error, got none")
				}
				if tc.errSubstr != "" && !strings.Contains(err.Error(), tc.errSubstr) {
					t.Fatalf("expected error containing %q, got %v", tc.errSubstr, err)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if ip != tc.wantIP {
				t.Fatalf("expected ip %s, got %q", tc.wantIP, ip)
			}
		})
	}
}

func TestAutoUpgradePatchPrepare(t *testing.T) {
	cases := []struct {
		name    string
		setup   func(t *testing.T) (*installerClient, string, AutoUpgradeRequest)
		wantErr bool
		substr  string
	}{
		{
			name: "empty-request",
			setup: func(t *testing.T) (*installerClient, string, AutoUpgradeRequest) {
				return &installerClient{}, "any", AutoUpgradeRequest{PatchDir: "", PatchName: ""}
			},
			wantErr: true,
			substr:  "patchDir and patchName cannot be empty",
		},
		{
			name: "dynamic-nil",
			setup: func(t *testing.T) (*installerClient, string, AutoUpgradeRequest) {
				testDir, _ := setupTestData(t)
				cs := fake.NewSimpleClientset(&corev1.ConfigMap{ObjectMeta: metav1.ObjectMeta{Name: BKEConfigCmKey().Name, Namespace: BKEConfigCmKey().Namespace}, Data: map[string]string{"otherRepo": "repo"}})
				return &installerClient{clientset: cs}, "any", AutoUpgradeRequest{PatchDir: testDir, PatchName: testPatchName}
			},
			wantErr: true,
			substr:  "get bootstrap ip failed",
		},
		{
			name: "get-configmap-fail",
			setup: func(t *testing.T) (*installerClient, string, AutoUpgradeRequest) {
				testDir, _ := setupTestData(t)
				clusterName := "c"
				bc := &unstructured.Unstructured{Object: map[string]interface{}{"apiVersion": "bke.bocloud.com/v1beta1", "kind": "BKECluster", "metadata": map[string]interface{}{"name": clusterName, "namespace": clusterName}, "spec": map[string]interface{}{"clusterConfig": map[string]interface{}{"cluster": map[string]interface{}{"imageRepo": map[string]interface{}{"ip": "1.2.3.5"}}}}}}
				scheme := runtime.NewScheme()
				_ = corev1.AddToScheme(scheme)
				bkeGVR := schema.GroupVersionResource{Group: "bke.bocloud.com", Version: "v1beta1", Resource: "bkeclusters"}
				listKinds := map[schema.GroupVersionResource]string{bkeGVR: "BKEClusterList"}
				dyn := dynamicfake.NewSimpleDynamicClientWithCustomListKinds(scheme, listKinds, bc)
				cs := fake.NewSimpleClientset()
				return &installerClient{dynamicClient: dyn, clientset: cs}, clusterName, AutoUpgradeRequest{PatchDir: testDir, PatchName: testPatchName}
			},
			wantErr: true,
			substr:  "get configmap",
		},
		{
			name: "is-online-fail",
			setup: func(t *testing.T) (*installerClient, string, AutoUpgradeRequest) {
				testDir, _ := setupTestData(t)
				clusterName := "c2"
				bc := &unstructured.Unstructured{Object: map[string]interface{}{"apiVersion": "bke.bocloud.com/v1beta1", "kind": "BKECluster", "metadata": map[string]interface{}{"name": clusterName, "namespace": clusterName}, "spec": map[string]interface{}{"clusterConfig": map[string]interface{}{"cluster": map[string]interface{}{"imageRepo": map[string]interface{}{"ip": "1.2.3.6"}}}}}}
				scheme := runtime.NewScheme()
				_ = corev1.AddToScheme(scheme)
				bkeGVR := schema.GroupVersionResource{Group: "bke.bocloud.com", Version: "v1beta1", Resource: "bkeclusters"}
				listKinds := map[schema.GroupVersionResource]string{bkeGVR: "BKEClusterList"}
				dyn := dynamicfake.NewSimpleDynamicClientWithCustomListKinds(scheme, listKinds, bc)
				cm := &corev1.ConfigMap{ObjectMeta: metav1.ObjectMeta{Name: BKEConfigCmKey().Name, Namespace: BKEConfigCmKey().Namespace}, Data: map[string]string{}}
				cs := fake.NewSimpleClientset(cm)
				return &installerClient{dynamicClient: dyn, clientset: cs}, clusterName, AutoUpgradeRequest{PatchDir: testDir, PatchName: testPatchName}
			},
			wantErr: true,
			substr:  "judge is online failed",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			c, cluster, req := tc.setup(t)
			err := c.AutoUpgradePatchPrepare(cluster, req)
			if tc.wantErr {
				if err == nil || (tc.substr != "" && !strings.Contains(err.Error(), tc.substr)) {
					t.Fatalf("expected error containing %q, got: %v", tc.substr, err)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
		})
	}
}
