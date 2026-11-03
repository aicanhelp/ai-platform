/******************************************************************
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * cluster-api-provider-bke is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package scriptshelper

import (
	"os"
	"path/filepath"
	"testing"
)

func TestIsScriptFile(t *testing.T) {
	tests := []struct {
		name     string
		filename string
		expected bool
	}{
		{"shell script", "test.sh", true},
		{"python script", "test.py", true},
		{"text file", "test.txt", false},
		{"no extension", "test", false},
		{"empty", "", false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := isScriptFile(tt.filename)
			if result != tt.expected {
				t.Errorf("expected %v, got %v", tt.expected, result)
			}
		})
	}
}

func TestCollectScriptFiles(t *testing.T) {
	tmpDir := t.TempDir()

	os.WriteFile(filepath.Join(tmpDir, "test.sh"), []byte("#!/bin/bash\necho test"), 0644)
	os.WriteFile(filepath.Join(tmpDir, "test.py"), []byte("print('test')"), 0644)
	os.WriteFile(filepath.Join(tmpDir, "test.txt"), []byte("text"), 0644)

	scripts, err := CollectScriptFiles(tmpDir)
	if err != nil {
		t.Fatalf("CollectScriptFiles failed: %v", err)
	}

	if len(scripts) != 2 {
		t.Errorf("expected 2 scripts, got %d", len(scripts))
	}

	if _, ok := scripts["test.sh"]; !ok {
		t.Error("test.sh not found")
	}
	if _, ok := scripts["test.py"]; !ok {
		t.Error("test.py not found")
	}
}

func TestCollectScriptFiles_NonExistentDir(t *testing.T) {
	_, err := CollectScriptFiles("/nonexistent/path/that/does/not/exist")
	if err == nil {
		t.Error("expected error for non-existent directory")
	}
}

func TestReadAndNormalizeScript(t *testing.T) {
	tmpDir := t.TempDir()
	testFile := filepath.Join(tmpDir, "test.sh")

	content := "#!/bin/bash\r\necho test\r\n"
	os.WriteFile(testFile, []byte(content), 0644)

	result, err := readAndNormalizeScript(testFile)
	if err != nil {
		t.Fatalf("readAndNormalizeScript failed: %v", err)
	}

	expected := "#!/bin/bash\necho test\n"
	if result != expected {
		t.Errorf("expected %q, got %q", expected, result)
	}
}

func TestReadAndNormalizeScript_NonExistent(t *testing.T) {
	_, err := readAndNormalizeScript("/nonexistent/file.sh")
	if err == nil {
		t.Error("expected error for non-existent file")
	}
}

func TestBuildScriptConfigMap(t *testing.T) {
	name := "test.sh"
	content := "#!/bin/bash\necho test"

	cm := buildScriptConfigMap(name, content)

	if cm.Name != name {
		t.Errorf("expected name %s, got %s", name, cm.Name)
	}
	if cm.Namespace != "cluster-system" {
		t.Errorf("expected namespace cluster-system, got %s", cm.Namespace)
	}
	if cm.Data[name] != content {
		t.Error("content mismatch")
	}
}

func TestCollectScriptFiles_WithSubdirectories(t *testing.T) {
	tmpDir := t.TempDir()
	subDir := filepath.Join(tmpDir, "subdir")
	os.Mkdir(subDir, 0755)

	os.WriteFile(filepath.Join(tmpDir, "root.sh"), []byte("#!/bin/bash"), 0644)
	os.WriteFile(filepath.Join(subDir, "sub.sh"), []byte("#!/bin/bash"), 0644)
	os.WriteFile(filepath.Join(subDir, "sub.py"), []byte("print('test')"), 0644)

	scripts, err := CollectScriptFiles(tmpDir)
	if err != nil {
		t.Fatalf("CollectScriptFiles failed: %v", err)
	}

	if len(scripts) != 3 {
		t.Errorf("expected 3 scripts, got %d", len(scripts))
	}
}

func TestReadAndNormalizeScript_UnixLineEndings(t *testing.T) {
	tmpDir := t.TempDir()
	testFile := filepath.Join(tmpDir, "test.sh")

	content := "#!/bin/bash\necho test\n"
	os.WriteFile(testFile, []byte(content), 0644)

	result, err := readAndNormalizeScript(testFile)
	if err != nil {
		t.Fatalf("readAndNormalizeScript failed: %v", err)
	}

	if result != content {
		t.Errorf("content should remain unchanged for Unix line endings")
	}
}

func TestIsScriptFile_EdgeCases(t *testing.T) {
	tests := []struct {
		name     string
		filename string
		expected bool
	}{
		{"uppercase .SH", "test.SH", false},
		{"uppercase .PY", "test.PY", false},
		{"multiple dots", "test.backup.sh", true},
		{"sh in middle", "test.sh.bak", false},
		{"py in middle", "test.py.bak", false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := isScriptFile(tt.filename)
			if result != tt.expected {
				t.Errorf("expected %v, got %v", tt.expected, result)
			}
		})
	}
}

func TestCollectScriptFiles_EmptyDirectory(t *testing.T) {
	tmpDir := t.TempDir()

	scripts, err := CollectScriptFiles(tmpDir)
	if err != nil {
		t.Fatalf("CollectScriptFiles failed: %v", err)
	}

	if len(scripts) != 0 {
		t.Errorf("expected 0 scripts in empty directory, got %d", len(scripts))
	}
}
