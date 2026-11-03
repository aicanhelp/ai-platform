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

// check  package provides check functions for oschecktool
package checker

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"openfuyao.com/oscheck/internal/conf"
	"openfuyao.com/oscheck/internal/utils"
)

// helper: 写入模拟的 kv 文件
func writeKVFile(dir, filename string, lines []string) (string, error) {
	filePath := filepath.Join(dir, filename)
	f, err := utils.CreateFile(filePath, utils.FileModeOnlyOwnerReadWrite, true)
	if err != nil {
		return "", err
	}
	defer utils.CloseAll(nil, f)
	f.Chmod(utils.FileModeOnlyOwnerReadWrite)
	for _, line := range lines {
		_, err := f.WriteString(line + "\n")
		if err != nil {
			return "", err
		}
	}
	return filePath, nil
}

func TestKVCheckerInit(t *testing.T) {
	release, logPath := InitLogForTest(t)
	defer func() {
		release()
		err := os.Remove(logPath)
		if err != nil {
			t.Errorf("Remove log file failed: %v", err)
		}
	}()

	ctx := Context{}
	t.Run("合法场景", func(t *testing.T) {
		spec := map[string]interface{}{
			"source": map[string]interface{}{
				"file": []string{"testkv.txt"},
			},
			"kvParser": map[string]interface{}{
				"reSplitter": "\\=",
			},
			"subItems": []map[string]interface{}{
				{
					"name":   "key1",
					"type":   "int",
					"expect": "123",
					"doc":    "doc1",
				},
			},
		}
		item := conf.CheckItem{
			Name: "kvtest",
			Spec: spec,
			Doc:  "doc",
		}
		checker := &kvChecker{}
		err := checker.Init(ctx, item)
		if err != nil {
			t.Errorf("Init failed: %v", err)
		}
		if len(checker.subCheckers) != 1 {
			t.Errorf("expected 1 subChecker, got %d", len(checker.subCheckers))
		}
	})

	t.Run("subItems缺少name", func(t *testing.T) {
		badSpec := map[string]interface{}{
			"source": map[string]interface{}{
				"file": []string{"testkv.txt"},
			},
			"subItems": []map[string]interface{}{
				{
					"type":   "int",
					"expect": "123",
				},
			},
		}
		badItem := conf.CheckItem{
			Name: "kvtest",
			Spec: badSpec,
		}
		checker := &kvChecker{}
		err := checker.Init(ctx, badItem)
		if err == nil {
			t.Errorf("expected error for missing name, got nil")
		}
	})

	t.Run("kvParser正则错误", func(t *testing.T) {
		badParserSpec := map[string]interface{}{
			"source": map[string]interface{}{
				"file": []string{"testkv.txt"},
			},
			"kvParser": map[string]interface{}{
				"reSplitter": "[",
			},
			"subItems": []map[string]interface{}{
				{
					"name":   "key1",
					"type":   "regex",
					"expect": "123(",
				},
			},
		}
		badParserItem := conf.CheckItem{
			Name: "kvtest",
			Spec: badParserSpec,
		}
		checker := &kvChecker{}
		err := checker.Init(ctx, badParserItem)
		if err == nil {
			t.Errorf("expected error for bad kvParser, got nil")
		}
	})
}

func TestKVCheckerCheckFileSource(t *testing.T) {
	release, logPath := InitLogForTest(t)
	defer func() {
		release()
		err := os.Remove(logPath)
		if err != nil {
			fmt.Printf("Remove log file failed: %v", err)
		}
	}()
	tmpDir := t.TempDir()
	ctx := Context{}

	t.Run("正常校验", func(t *testing.T) {
		kvLines := []string{
			"key1=123",
			"key2=456",
		}
		kvFile, err := writeKVFile(tmpDir, "testkv.txt", kvLines)
		if err != nil {
			t.Fatalf("failed to write kv file: %v", err)
		}
		spec := map[string]interface{}{
			"source": map[string]interface{}{
				"file": []string{kvFile},
			},
			"kvParser": map[string]interface{}{
				"reSplitter": "\\=",
			},
			"subItems": []map[string]interface{}{
				{
					"name":   "key1",
					"type":   "int",
					"expect": "123",
					"doc":    "doc1",
				},
				{
					"name":   "key2",
					"type":   "int",
					"expect": "999",
					"doc":    "doc2",
				},
			},
		}
		item := conf.CheckItem{
			Name: "kvtest",
			Spec: spec,
			Doc:  "doc",
		}
		checker := &kvChecker{}
		if err := checker.Init(ctx, item); err != nil {
			t.Fatalf("Init failed: %v", err)
		}
		result := checker.Check()
		if result.Result != ResultInvalid {
			t.Errorf("expected CheckInvalid, got %v", result.Result)
		}
		if len(result.SubItems) != len(spec["subItems"].([]map[string]interface{})) {
			t.Errorf("expected 2 subItems, got %d", len(result.SubItems))
		}
		if result.SubItems[0].Result != ResultValid {
			t.Errorf("expected key1 valid, got %v", result.SubItems[0].Result)
		}
		if result.SubItems[1].Result != ResultInvalid {
			t.Errorf("expected key2 invalid, got %v", result.SubItems[1].Result)
		}
	})

	t.Run("文件不存在", func(t *testing.T) {
		badSpec := map[string]interface{}{
			"source": map[string]interface{}{
				"file": []string{filepath.Join(tmpDir, "notexist.txt")},
			},
			"kvParser": map[string]interface{}{
				"reSplitter": "\\=",
			},
			"subItems": []map[string]interface{}{
				{
					"name":   "key1",
					"type":   "int",
					"expect": "123",
				},
			},
		}
		badItem := conf.CheckItem{
			Name: "kvtest",
			Spec: badSpec,
		}
		checker := &kvChecker{}
		if err := checker.Init(ctx, badItem); err != nil {
			t.Fatalf("Init failed: %v", err)
		}
		result := checker.Check()
		if result.Result != ResultError {
			t.Errorf("expected CheckError for missing file, got %v", result.Result)
		}
	})
}
