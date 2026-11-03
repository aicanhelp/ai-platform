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

// report package provides report functions for oschecktool
package report

import (
	"os"
	"testing"
	"time"

	"openfuyao.com/oscheck/internal/checker"
)

func makeCheckResult(items []checker.ItemRslt, result checker.ResultType) checker.Result {
	return checker.Result{
		CheckSet:  "testset",
		Result:    result,
		StartTime: time.Now(),
		EndTime:   time.Now(),
		Items:     items,
	}
}

func makeSubItem(key, expect, real, doc string, result checker.ResultType) checker.SubItemRslt {
	return checker.SubItemRslt{
		Key:    key,
		Expect: expect,
		Real:   real,
		Result: result,
		Doc:    doc,
	}
}

func makeItem(key, doc string, result checker.ResultType, subItems ...checker.SubItemRslt) checker.ItemRslt {
	return checker.ItemRslt{
		Key:      key,
		Result:   result,
		Doc:      doc,
		SubItems: subItems,
	}
}

func TestGenerateReport(t *testing.T) {
	t.Run("Yaml report with sub items", func(t *testing.T) {
		tmpDir := t.TempDir()
		rslt := makeCheckResult(
			[]checker.ItemRslt{
				makeItem("item1", "desc1", checker.ResultValid,
					makeSubItem("sub1", "exp1", "real1", "subdesc1", checker.ResultValid),
				),
				makeItem("item2", "desc2", checker.ResultInvalid,
					makeSubItem("sub2", "exp2", "real2", "subdesc2", checker.ResultInvalid),
				),
			},
			checker.ResultValid,
		)
		filePath, err := GenerateReport(rslt, tmpDir, yamlType)
		if err != nil {
			t.Fatalf("Yaml report failed: %v", err)
		}
		if _, err := os.Stat(filePath); err != nil {
			t.Errorf("Yaml report file not found: %v", err)
		}
	})

	t.Run("Yaml report without sub items", func(t *testing.T) {
		tmpDir := t.TempDir()
		rslt := makeCheckResult(
			[]checker.ItemRslt{
				makeItem("item1", "desc1", checker.ResultValid),
				makeItem("item2", "desc2", checker.ResultInvalid),
			},
			checker.ResultInvalid,
		)
		filePath, err := GenerateReport(rslt, tmpDir, yamlType)
		if err != nil {
			t.Fatalf("Yaml report failed: %v", err)
		}
		if _, err := os.Stat(filePath); err != nil {
			t.Errorf("Yaml report file not found: %v", err)
		}
	})

	t.Run("Yaml report with empty result", func(t *testing.T) {
		tmpDir := t.TempDir()
		rslt := makeCheckResult([]checker.ItemRslt{}, checker.ResultValid)
		filePath, err := GenerateReport(rslt, tmpDir, yamlType)
		if err != nil {
			t.Fatalf("Yaml report failed: %v", err)
		}
		if _, err := os.Stat(filePath); err != nil {
			t.Errorf("Yaml report file not found: %v", err)
		}
	})

	t.Run("CSV report with sub items", func(t *testing.T) {
		tmpDir := t.TempDir()
		rslt := makeCheckResult(
			[]checker.ItemRslt{
				makeItem("item1", "desc1", checker.ResultValid,
					makeSubItem("sub1", "exp1", "real1", "subdesc1", checker.ResultValid),
					makeSubItem("sub2", "exp2", "real2", "subdesc2", checker.ResultInvalid),
				),
				makeItem("item2", "desc2", checker.ResultError),
			},
			checker.ResultInvalid,
		)
		filePath, err := GenerateReport(rslt, tmpDir, csvType)
		if err != nil {
			t.Fatalf("CSV report failed: %v", err)
		}
		if _, err := os.Stat(filePath); err != nil {
			t.Errorf("CSV report file not found: %v", err)
		}
	})

	t.Run("CSV report without sub items", func(t *testing.T) {
		tmpDir := t.TempDir()
		rslt := makeCheckResult(
			[]checker.ItemRslt{
				makeItem("item1", "desc1", checker.ResultValid),
				makeItem("item2", "desc2", checker.ResultInvalid),
			},
			checker.ResultInvalid,
		)
		filePath, err := GenerateReport(rslt, tmpDir, csvType)
		if err != nil {
			t.Fatalf("CSV report failed: %v", err)
		}
		if _, err := os.Stat(filePath); err != nil {
			t.Errorf("CSV report file not found: %v", err)
		}
	})

	t.Run("CSV report with empty result", func(t *testing.T) {
		tmpDir := t.TempDir()
		rslt := makeCheckResult([]checker.ItemRslt{}, checker.ResultError)
		filePath, err := GenerateReport(rslt, tmpDir, csvType)
		if err != nil {
			t.Fatalf("CSV report failed: %v", err)
		}
		if _, err := os.Stat(filePath); err != nil {
			t.Errorf("CSV report file not found: %v", err)
		}
	})

	t.Run("Plain report with sub items", func(t *testing.T) {
		tmpDir := t.TempDir()
		rslt := makeCheckResult(
			[]checker.ItemRslt{
				makeItem("item1", "desc1", checker.ResultValid,
					makeSubItem("sub1", "exp1", "real1", "subdesc1", checker.ResultValid),
					makeSubItem("sub2", "exp2", "real2", "subdesc2", checker.ResultInvalid),
				),
				makeItem("item2", "desc2", checker.ResultError),
			},
			checker.ResultInvalid,
		)
		filePath, err := GenerateReport(rslt, tmpDir, plainType)
		if err != nil {
			t.Fatalf("Plain report failed: %v", err)
		}
		// filePath 可能为空（输出到stdout），但如果有文件名则检查文件存在
		if filePath != "" {
			if _, err := os.Stat(filePath); err != nil {
				t.Errorf("Plain report file not found: %v", err)
			}
		}
	})
	t.Run("Plain report with sub items to stdout", func(t *testing.T) {

		rslt := makeCheckResult(
			[]checker.ItemRslt{
				makeItem("item1", "desc1", checker.ResultValid,
					makeSubItem("sub1", "exp1", "real1", "subdesc1 that content is very lengh and have \n in the content"+
						"and file the content with useless string hhhhhhhhhhhhhhhhhhhhhhhhhhhhh", checker.ResultValid),
					makeSubItem("sub2", "exp2", "real2", "subdesc2", checker.ResultInvalid),
				),
				makeItem("item2", "desc2", checker.ResultError),
			},
			checker.ResultInvalid,
		)
		filePath, err := GenerateReport(rslt, "", plainType)
		if err != nil {
			t.Fatalf("Plain report failed: %v", err)
		}
		// filePath 可能为空（输出到stdout），但如果有文件名则检查文件存在
		if filePath != "" {
			if _, err := os.Stat(filePath); err != nil {
				t.Errorf("Plain report file not found: %v", err)
			}
		}
	})

	t.Run("Plain report without sub items", func(t *testing.T) {
		tmpDir := t.TempDir()
		rslt := makeCheckResult(
			[]checker.ItemRslt{
				makeItem("item1", "desc1", checker.ResultValid),
				makeItem("item2", "desc2", checker.ResultInvalid),
			},
			checker.ResultInvalid,
		)
		filePath, err := GenerateReport(rslt, tmpDir, plainType)
		if err != nil {
			t.Fatalf("Plain report failed: %v", err)
		}
		if filePath != "" {
			if _, err := os.Stat(filePath); err != nil {
				t.Errorf("Plain report file not found: %v", err)
			}
		}
	})

	t.Run("Plain report with empty result", func(t *testing.T) {
		tmpDir := t.TempDir()
		rslt := makeCheckResult([]checker.ItemRslt{}, checker.ResultError)
		filePath, err := GenerateReport(rslt, tmpDir, plainType)
		if err != nil {
			t.Fatalf("Plain report failed: %v", err)
		}
		if filePath != "" {
			if _, err := os.Stat(filePath); err != nil {
				t.Errorf("Plain report file not found: %v", err)
			}
		}
	})

	t.Run("Unsupported report type", func(t *testing.T) {
		tmpDir := t.TempDir()
		rslt := makeCheckResult([]checker.ItemRslt{}, checker.ResultValid)
		_, err := GenerateReport(rslt, tmpDir, "txt")
		if err == nil {
			t.Errorf("Expected error for unsupported report type")
		}
	})
}
