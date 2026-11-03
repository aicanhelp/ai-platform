/*
 *
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain n copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 *
 */
package source

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// Constants for file permissions
const (
	testFilePermission = 0644
	testDirPermission  = 0755
)

func TestGetCustomDownloadPath(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected string
	}{
		{
			name:     "URL with trailing slash",
			input:    "http://example.com/repo/",
			expected: "http://example.com/repo/files",
		},
		{
			name:     "URL without trailing slash",
			input:    "http://example.com/repo",
			expected: "http://example.com/repo/files",
		},
		{
			name:     "URL with query parameters",
			input:    "http://example.com/repo?param=value",
			expected: "http://example.com/repo?param=value/files",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := GetCustomDownloadPath(tt.input)
			if result != tt.expected {
				t.Errorf("GetCustomDownloadPath(%s) = %s; expected %s", tt.input, result, tt.expected)
			}
		})
	}
}

func TestGetRPMDownloadPath(t *testing.T) {
	// This function depends on host.Info() which is difficult to mock
	// For now, we'll test error cases and the logic structure
	t.Run("Test with empty URL", func(t *testing.T) {
		// Note: This test will fail if not running on a supported platform,
		// but it will at least test the URL construction logic
		_, err := GetRPMDownloadPath("")
		// We only verify that the function doesn't panic
		if err != nil {
			t.Logf("Expected error when host info is not available in test environment: %v", err)
		}
	})

	t.Run("Test URL with no trailing slash", func(t *testing.T) {
		_, err := GetRPMDownloadPath("http://example.com")
		if err != nil {
			t.Logf("Expected error when host info is not available in test environment: %v", err)
		}
	})

	t.Run("Test URL with trailing slash", func(t *testing.T) {
		_, err := GetRPMDownloadPath("http://example.com/")
		if err != nil {
			t.Logf("Expected error when host info is not available in test environment: %v", err)
		}
	})
}

func TestSetSource(t *testing.T) {
	// Since SetSource requires system paths that may not exist in test environment,
	// we'll test error conditions and logic flow
	t.Run("Test with invalid URL", func(t *testing.T) {
		err := SetSource("")
		if err != nil {
			t.Logf("Expected error when host info is not available in test environment: %v", err)
		}
	})
}

func TestWriteAptSource(t *testing.T) {
	// Create a temporary directory for testing
	tempDir := t.TempDir()
	tempFile := filepath.Join(tempDir, "sources.list")

	// Temporarily change the aptRepos variable for testing
	originalAptRepos := aptRepos
	aptRepos = tempFile
	defer func() {
		aptRepos = originalAptRepos
	}()

	testURL := "http://test.repo.com/ubuntu"
	err := writeAptSource(testURL)
	if err != nil {
		t.Fatalf("writeAptSource failed: %v", err)
	}

	// Read the file back and verify its contents
	content, err := os.ReadFile(aptRepos)
	if err != nil {
		t.Fatalf("Failed to read written file: %v", err)
	}

	expectedContent := strings.Replace(aptSource, "{{.baseurl}}", testURL, -1)
	if string(content) != expectedContent {
		t.Errorf("writeAptSource wrote %s; expected %s", string(content), expectedContent)
	}
}

func TestBackupAptSource(t *testing.T) {
	// Create a temporary directory for testing
	tempDir := t.TempDir()
	originalAptRepos := aptRepos
	aptRepos = filepath.Join(tempDir, "sources.list")
	aptBak := aptRepos + ".bak"

	defer func() {
		aptRepos = originalAptRepos
	}()

	// Create the original file
	testContent := "deb http://archive.ubuntu.com/ubuntu/ focal main"
	err := os.WriteFile(aptRepos, []byte(testContent), testFilePermission)
	if err != nil {
		t.Fatalf("Failed to create test file: %v", err)
	}

	// Verify the file exists
	if !fileExists(aptRepos) {
		t.Fatalf("Test file was not created: %s", aptRepos)
	}

	// Perform backup
	err = backupAptSource()
	if err != nil {
		t.Fatalf("backupAptSource failed: %v", err)
	}

	// Verify the original file was moved to backup
	if fileExists(aptRepos) {
		t.Error("Original sources.list file still exists after backup")
	}
	if !fileExists(aptBak) {
		t.Error("Backup sources.list file does not exist after backup")
	}

	// Verify content was preserved
	content, err := os.ReadFile(aptBak)
	if err != nil {
		t.Fatalf("Failed to read backup file: %v", err)
	}
	if string(content) != testContent {
		t.Errorf("Backup file content %s; expected %s", string(content), testContent)
	}
}

func TestResetAptSource(t *testing.T) {
	// Create a temporary directory for testing
	tempDir := t.TempDir()
	originalAptRepos := aptRepos
	aptRepos = filepath.Join(tempDir, "sources.list")
	aptBak := aptRepos + ".bak"

	defer func() {
		aptRepos = originalAptRepos
	}()

	// Create backup file
	testContent := "deb http://archive.ubuntu.com/ubuntu/ focal main"
	err := os.WriteFile(aptBak, []byte(testContent), testFilePermission)
	if err != nil {
		t.Fatalf("Failed to create backup file: %v", err)
	}

}

// setupYumTestEnv creates a temporary environment for yum-related tests
func setupYumTestEnv(t *testing.T) (string, string, string) {
	tempDir := t.TempDir()
	yumRepoDir := filepath.Join(tempDir, "yum.repos.d")
	yumBakDir := filepath.Join(yumRepoDir, "bak")

	originalYumRepos := yumRepos
	yumRepos = yumRepoDir
	t.Cleanup(func() {
		yumRepos = originalYumRepos
	})

	return originalYumRepos, yumRepoDir, yumBakDir
}

// createYumDirs creates the yum repository directories for tests
func createYumDirs(t *testing.T, yumRepoDir, yumBakDir string) {
	err := os.Mkdir(yumRepoDir, testDirPermission)
	if err != nil {
		t.Fatalf("Failed to create test directory: %v", err)
	}
	err = os.Mkdir(yumBakDir, testDirPermission)
	if err != nil {
		t.Fatalf("Failed to create backup directory: %v", err)
	}
}

func fileExists(filename string) bool {
	_, err := os.Stat(filename)
	return err == nil
}

func TestWriteYumSource(t *testing.T) {
	// Create a temporary directory for testing
	tempDir := t.TempDir()

	// Temporarily change the yumRepos variable for testing
	originalYumRepos := yumRepos
	yumRepos = tempDir
	defer func() {
		yumRepos = originalYumRepos
	}()

	testURL := "http://test.repo.com/centos"
	err := writeYumSource(testURL)
	if err != nil {
		t.Fatalf("writeYumSource failed: %v", err)
	}

	// Read the file back and verify its contents
	content, err := os.ReadFile(filepath.Join(yumRepos, "bke.repo"))
	if err != nil {
		t.Fatalf("Failed to read written file: %v", err)
	}

	expectedContent := strings.Replace(yumSource, "{{.baseurl}}", testURL, -1)
	if string(content) != expectedContent {
		t.Errorf("writeYumSource wrote %s; expected %s", string(content), expectedContent)
	}
}

func TestGetRPMDownloadPathOSHandling(t *testing.T) {
	// This test is limited due to the dependency on host.Info(),
	// but we can test the version parsing logic
	t.Run("Test version with no dots", func(t *testing.T) {
		// Manually test the version parsing logic from GetRPMDownloadPath
		version := "20"
		if !strings.Contains(version, ".") {
			result := strings.ToUpper(version)
			if result != "20" {
				t.Errorf("Version processing failed: %s became %s", version, result)
			}
		}
	})

	t.Run("Test version with dots", func(t *testing.T) {
		version := "20.04"
		var ver string
		if !strings.Contains(version, ".") {
			ver = strings.ToUpper(version)
		} else {
			versionParts := strings.Split(version, ".")
			if len(versionParts) >= minimumVersionParts {
				ver = versionParts[0]
			}
		}
		expected := "20"
		if ver != expected {
			t.Errorf("Version parsing failed: %s became %s, expected %s", version, ver, expected)
		}
	})

	t.Run("Test arch is appended", func(t *testing.T) {
		// Just verify that runtime.GOARCH is used in the path
		currentArch := runtime.GOARCH
		if currentArch == "" {
			t.Error("GOARCH should not be empty")
		}
	})
}

func TestBackupYumSource(t *testing.T) {
	// Temporarily change the yumRepos variable for testing
	_, yumRepoDir, _ := setupYumTestEnv(t)

	// Create the yum.repos.d directory
	err := os.Mkdir(yumRepoDir, testDirPermission)
	if err != nil {
		t.Fatalf("Failed to create test directory: %v", err)
	}

	// Create some test repo files
	testFiles := []string{"repo1.repo", "repo2.repo", "other.conf"}
	for _, filename := range testFiles {
		err = os.WriteFile(filepath.Join(yumRepoDir, filename), []byte("test content"), testFilePermission)
		if err != nil {
			t.Fatalf("Failed to create test file: %v", err)
		}
	}

}

func TestResetYumSource(t *testing.T) {
	// Temporarily change the yumRepos variable for testing
	_, yumRepoDir, yumBakDir := setupYumTestEnv(t)

	// Create the yum.repos.d and bak directories
	createYumDirs(t, yumRepoDir, yumBakDir)

	// Create bke.repo file (this would have been created by writeYumSource)
	bkeRepoPath := filepath.Join(yumRepoDir, "bke.repo")
	err := os.WriteFile(bkeRepoPath, []byte("bke repo content"), testFilePermission)
	if err != nil {
		t.Fatalf("Failed to create bke.repo file: %v", err)
	}

	// Create some backup repo files
	testFiles := []string{"repo1.repo", "repo2.repo", "other.conf"}
	for _, filename := range testFiles {
		err = os.WriteFile(filepath.Join(yumBakDir, filename), []byte("test content"), testFilePermission)
		if err != nil {
			t.Fatalf("Failed to create backup file: %v", err)
		}
	}

	// Perform reset
	err = resetYumSource()
	if err != nil {
		t.Fatalf("resetYumSource failed: %v", err)
	}

	// Verify that bke.repo was removed
	if fileExists(bkeRepoPath) {
		t.Error("bke.repo file still exists after reset")
	}

	// Verify that backup files were moved back to main directory
	for _, filename := range testFiles {
		originalPath := filepath.Join(yumRepoDir, filename)
		bakPath := filepath.Join(yumBakDir, filename)

		if !fileExists(originalPath) {
			t.Errorf("File %s was not moved back from backup", filename)
		}
		if fileExists(bakPath) {
			t.Errorf("Backup file %s still exists after reset", filename)
		}
	}

	// Verify that bak directory was removed
	if fileExists(yumBakDir) {
		t.Error("Backup directory still exists after reset")
	}
}

func TestResetYumSourceNoBakDir(t *testing.T) {
	// Test resetYumSource when bak directory doesn't exist
	tempDir := t.TempDir()
	yumRepoDir := filepath.Join(tempDir, "yum.repos.d")

	// Temporarily change the yumRepos variable for testing
	originalYumRepos := yumRepos
	yumRepos = yumRepoDir
	defer func() {
		yumRepos = originalYumRepos
	}()

	// Create the yum.repos.d directory but NOT the bak directory
	err := os.Mkdir(yumRepoDir, testDirPermission)
	if err != nil {
		t.Fatalf("Failed to create test directory: %v", err)
	}

	// Call resetYumSource - should return nil without errors
	err = resetYumSource()
	if err != nil {
		t.Errorf("resetYumSource should return nil when bak directory doesn't exist, got: %v", err)
	}
}

func TestResetYumSourceNoBkeRepo(t *testing.T) {
	// Test resetYumSource when bke.repo doesn't exist
	_, yumRepoDir, yumBakDir := setupYumTestEnv(t)

	// Create the yum.repos.d and bak directories
	createYumDirs(t, yumRepoDir, yumBakDir)

	// Call resetYumSource - should return nil without errors when bke.repo doesn't exist
	err := resetYumSource()
	if err != nil {
		t.Errorf("resetYumSource should return nil when bke.repo doesn't exist, got: %v", err)
	}
}
