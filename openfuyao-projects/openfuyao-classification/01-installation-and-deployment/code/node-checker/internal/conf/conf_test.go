/*
Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
*/

// conf package provides configuration loading functions for oschecktool
package conf

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"openfuyao.com/oscheck/internal/utils"
)

// 日志初始化和清理
func InitLogForTest(t *testing.T) (func(), string) {
	rootdir := t.TempDir()
	err := os.MkdirAll(rootdir, utils.DirModeOwnerReadWriteExec)
	if err != nil {
		fmt.Printf("clean: failed to create log dir: %v, use current dir", err)
		rootdir = "."
	}
	logPath := filepath.Join(rootdir, "test.log")
	release := utils.InitLogFile(logPath)
	return release, logPath
}

// 写入yaml测试文件
func writeYamlFile(dir, filename, content string) (string, error) {
	filePath := filepath.Join(dir, filename)
	if err := os.WriteFile(filePath, []byte(content), utils.FileModeOnlyOwnerReadWrite); err != nil {
		return "", err
	}
	return filePath, nil
}

func TestLoadCheckItemFromFile(t *testing.T) {
	release, logPath := InitLogForTest(t)
	defer func() {
		release()
		err := os.Remove(logPath)
		if err != nil {
			fmt.Printf("clean: failed to remove log file: %v", err)
		}
	}()
	testDir := t.TempDir()

	t.Run("合法yaml", func(t *testing.T) {
		content := `
name: testitem
desc: testdesc
kind: testkind
doc: testdoc
spec:
  key: value
`
		filePath, err := writeYamlFile(testDir, "valid_item.yaml", content)
		if err != nil {
			t.Fatalf("failed to write yaml: %v", err)
		}
		item, err := loadCheckItemFromFile(filePath)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if item.Name != "testitem" {
			t.Errorf("unexpected item name: %v", item.Name)
		}
	})

	t.Run("yaml格式错误", func(t *testing.T) {
		content := `
name: testitem
desc: testdesc
kind: testkind
doc: testdoc
spec
  key: value
`
		filePath, err := writeYamlFile(testDir, "invalid_item.yaml", content)
		if err != nil {
			t.Fatalf("failed to write yaml: %v", err)
		}
		_, err = loadCheckItemFromFile(filePath)
		if err == nil {
			t.Errorf("expected error for invalid yaml, got nil")
		}
	})

	t.Run("空文件", func(t *testing.T) {
		filePath, err := writeYamlFile(testDir, "empty_item.yaml", "")
		if err != nil {
			t.Fatalf("failed to write yaml: %v", err)
		}
		checkItem, err := loadCheckItemFromFile(filePath)
		if err == nil {
			t.Errorf("expected error for empty file, got %v, error: %v", checkItem.Name, err)
		}
	})
}

func TestLoadCheckItems(t *testing.T) {
	release, logPath := InitLogForTest(t)
	defer func() {
		release()
		err := os.Remove(logPath)
		if err != nil {
			fmt.Printf("clean: failed to remove log file: %v", err)
		}
	}()
	testDir := t.TempDir()

	t.Run("包含合法和非法item", func(t *testing.T) {
		validContent := `
name: testitem
desc: testdesc
kind: testkind
doc: testdoc
spec:
  key: value
`
		invalidContent := `
name: testitem
desc: testdesc
kind: testkind
doc: testdoc
spec
  key: value
`
		_, err := writeYamlFile(testDir, "valid_item.yaml", validContent)
		if err != nil {
			t.Fatalf("failed to write yaml: %v", err)
		}
		_, err = writeYamlFile(testDir, "invalid_item.yaml", invalidContent)
		if err != nil {
			t.Fatalf("failed to write yaml: %v", err)
		}
		_, err = loadCheckItems(testDir)
		if err == nil {
			t.Errorf("expected error for bad item, got nil")
		}
	})
}

func TestLoadCheckSetFromFile(t *testing.T) {
	release, logPath := InitLogForTest(t)
	defer func() {
		release()
		err := os.Remove(logPath)
		if err != nil {
			fmt.Printf("clean: failed to remove log file: %v", err)
		}
	}()
	testDir := t.TempDir()

	t.Run("合法set", func(t *testing.T) {
		itemContent := `
name: testitem
desc: testdesc
kind: testkind
doc: testdoc
spec:
  key: value
`
		setContent := `
name: testset
desc: testdesc
items: [testitem]
include: []
`
		itemPath, err := writeYamlFile(testDir, "valid_item.yaml", itemContent)
		if err != nil {
			t.Fatalf("failed to write yaml: %v", err)
		}
		item, err := loadCheckItemFromFile(itemPath)
		if err != nil {
			t.Fatalf("failed to load valid item: %v", err)
		}
		checkItems := make(map[string]CheckItem)
		checkItems[item.Name] = item

		setPath, err := writeYamlFile(testDir, "valid_set.yaml", setContent)
		if err != nil {
			t.Fatalf("failed to write yaml: %v", err)
		}
		_, err = loadCheckSetFromFile(setPath, checkItems)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
	})

	t.Run("set引用不存在item", func(t *testing.T) {
		setContent := `
name: testset
desc: testdesc
items: [notexistitem]
include: []
`
		setPath, err := writeYamlFile(testDir, "missing_item_set.yaml", setContent)
		if err != nil {
			t.Fatalf("failed to write yaml: %v", err)
		}
		checkItems := make(map[string]CheckItem)
		_, err = loadCheckSetFromFile(setPath, checkItems)
		if err == nil {
			t.Errorf("expected error for missing item, got nil")
		}
	})

	t.Run("set格式错误", func(t *testing.T) {
		setContent := `
name: testset
desc: testdesc
items
  - testitem
include: []
`
		setPath, err := writeYamlFile(testDir, "invalid_set.yaml", setContent)
		if err != nil {
			t.Fatalf("failed to write yaml: %v", err)
		}
		checkItems := make(map[string]CheckItem)
		checkItems["testitem"] = CheckItem{Name: "testitem"}
		_, err = loadCheckSetFromFile(setPath, checkItems)
		if err == nil {
			t.Errorf("expected error for invalid set yaml, got nil")
		}
	})

	t.Run("空set文件", func(t *testing.T) {
		setPath, err := writeYamlFile(testDir, "empty_set.yaml", "")
		if err != nil {
			t.Fatalf("failed to write yaml: %v", err)
		}
		checkItems := make(map[string]CheckItem)
		_, err = loadCheckSetFromFile(setPath, checkItems)
		if err == nil {
			t.Errorf("expected error for empty set file, got nil")
		}
	})
}

func TestLoadCheckSets(t *testing.T) {
	release, logPath := InitLogForTest(t)
	defer func() {
		release()
		err := os.Remove(logPath)
		if err != nil {
			fmt.Printf("clean: failed to remove log file: %v", err)
		}
	}()
	testDir := t.TempDir()

	t.Run("包含合法和非法set", func(t *testing.T) {
		item := CheckItem{
			Name: "testitem",
		}
		checkItems := make(map[string]CheckItem)
		checkItems["testitem"] = item

		validSetContent := `
name: testset
desc: testdesc
items: [testitem]
include: []
`
		invalidSetContent := `
name: testset
desc: testdesc
items
  - testitem
include: []
`
		validSetPath, err := writeYamlFile(testDir, "valid_set.yaml", validSetContent)
		if err != nil {
			t.Fatalf("failed to write yaml: %v", err)
		}
		_, err = writeYamlFile(testDir, "invalid_set.yaml", invalidSetContent)
		if err != nil {
			t.Fatalf("failed to write yaml: %v", err)
		}
		_, err = loadCheckSets(testDir, checkItems)
		if err == nil {
			t.Errorf("expected error for bad set, got nil")
		}
		err = os.Remove(validSetPath)
		if err != nil {
			fmt.Printf("clean: failed to remove valid set file: %v", err)
		}
	})
}
