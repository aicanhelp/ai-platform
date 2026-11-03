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

// helper: 创建模拟的 /proc/modules 文件
func writeProcModules(dir string, content string) error {
	procModuleFile := filepath.Join(dir, procModulesPath)
	baseDir := filepath.Dir(procModuleFile)
	if err := os.MkdirAll(baseDir, utils.DirModeOwnerReadWriteExec); err != nil {
		return err
	}
	return os.WriteFile(procModuleFile, []byte(content), utils.FileModeOwnerReadWriteExec)
}

// helper: 创建模拟的 /sys/module 目录和模块
func writeSysModule(dir string, modules []string) error {
	sysPath := filepath.Join(dir, sysModulePath)
	for _, m := range modules {
		if err := os.MkdirAll(filepath.Join(sysPath, m), utils.DirModeOwnerReadWriteExec); err != nil {
			return err
		}
	}
	return nil
}

// helper: 创建模拟的 /sys/module/<module>/<detail> 文件
func writeModuleDetail(dir, module, detail, value string) error {
	detailDir := filepath.Join(dir, sysModulePath, module)
	if err := os.MkdirAll(detailDir, utils.DirModeOwnerReadWriteExec); err != nil {
		return err
	}
	detailPath := filepath.Join(detailDir, detail)
	return os.WriteFile(detailPath, []byte(value), utils.FileModeOwnerReadWriteExec)
}

func InitLogForTest(t *testing.T) (func(), string) {
	rootdir := t.TempDir()
	err := os.MkdirAll(rootdir, utils.DirModeOwnerReadWriteExec)
	if err != nil {
		fmt.Printf("Create log dir failed: %v", err)
	}
	logPath := filepath.Join(rootdir, "test.log")
	release := utils.InitLogFile(logPath)
	return release, logPath
}

func TestKernelModuleCheckerInit(t *testing.T) {
	release, logPath := InitLogForTest(t)
	defer func() {
		release()
		err := os.Remove(logPath)
		if err != nil {
			fmt.Printf("Remove log file failed: %v", err)
		}
	}()

	ctx := Context{}

	t.Run("合法场景", func(t *testing.T) {
		spec := []map[string]interface{}{
			{
				"name": "mod1",
				"doc":  "test doc",
				"detail": []map[string]interface{}{
					{
						"name":   "param1",
						"type":   "int",
						"expect": 123,
						"doc":    "detail doc",
					},
				},
			},
		}
		item := conf.CheckItem{
			Name: "kernel",
			Spec: spec,
			Doc:  "kernel doc",
		}
		checker := &kernelModuleChecker{}
		err := checker.Init(ctx, item)
		if err != nil {
			t.Errorf("Init failed: %v", err)
		}
		if len(checker.subCheckers) != 1 {
			t.Errorf("expected 1 subChecker, got %d", len(checker.subCheckers))
		}
	})

	t.Run("缺少name字段", func(t *testing.T) {
		badSpec := []map[string]interface{}{
			{
				"doc": "no name",
			},
		}
		badItem := conf.CheckItem{
			Name: "kernel",
			Spec: badSpec,
		}
		checker := &kernelModuleChecker{}
		err := checker.Init(ctx, badItem)
		if err == nil {
			t.Errorf("expected error for missing name, got nil")
		}
	})

	t.Run("detail格式错误", func(t *testing.T) {
		badDetailSpec := []map[string]interface{}{
			{
				"name": "mod1",
				"detail": []map[string]interface{}{
					{
						"name": "",
						"type": "",
					},
				},
			},
		}
		badDetailItem := conf.CheckItem{
			Name: "kernel",
			Spec: badDetailSpec,
		}
		checker := &kernelModuleChecker{}
		err := checker.Init(ctx, badDetailItem)
		if err == nil {
			t.Errorf("expected error for bad detail, got nil")
		}
	})
}

func TestKernelModuleCheckerCheck(t *testing.T) {
	release, logPath := InitLogForTest(t)
	defer func() {
		release()
		err := os.Remove(logPath)
		if err != nil {
			fmt.Printf("Remove log file failed: %v", err)
		}
	}()

	t.Run("正常场景", func(t *testing.T) {
		RootPath := t.TempDir()
		ctx := Context{
			RootPath: RootPath,
		}
		procContent := "mod1 0 0 0 0 Live 0\n"
		if err := writeProcModules(RootPath, procContent); err != nil {
			t.Fatalf("failed to write proc_modules: %v", err)
		}
		if err := writeSysModule(RootPath, []string{"mod1"}); err != nil {
			t.Fatalf("failed to write sys_module: %v", err)
		}
		if err := writeModuleDetail(RootPath, "mod1", "param1", "123"); err != nil {
			t.Fatalf("failed to write module detail: %v", err)
		}

		spec := []map[string]interface{}{
			{
				"name": "mod1",
				"doc":  "test doc",
				"detail": []map[string]interface{}{
					{
						"name":   "param1",
						"type":   "int",
						"expect": "123",
						"doc":    "detail doc",
					},
				},
			},
		}
		item := conf.CheckItem{
			Name: "kernel",
			Spec: spec,
			Doc:  "kernel doc",
		}
		checker := &kernelModuleChecker{}
		checker.Init(ctx, item)
		result := checker.Check()
		if result.Result != ResultValid {
			t.Errorf("expected CheckValid, got %v", result.Result)
		}
		if len(result.SubItems) != 1 {
			t.Errorf("expected 1 subItem, got %d", len(result.SubItems))
		}
		if len(result.SubItems) > 0 && result.SubItems[0].Result != ResultValid {
			t.Errorf("expected subItem CheckValid, got %v", result.SubItems[0].Result)
		}
	})

	t.Run("模块不存在", func(t *testing.T) {
		RootPath := t.TempDir()
		ctx := Context{
			RootPath: RootPath,
		}
		if err := writeProcModules(RootPath, ""); err != nil {
			t.Fatalf("failed to write proc_modules: %v", err)
		}
		if err := writeModuleDetail(RootPath, "mod2", "param1", "123"); err != nil {
			t.Fatalf("failed to write module detail: %v", err)
		}
		spec := []map[string]interface{}{
			{
				"name": "mod1",
				"doc":  "test doc",
				"detail": []map[string]interface{}{
					{
						"name":   "param1",
						"type":   "int",
						"expect": "123",
						"doc":    "detail doc",
					},
				},
			},
		}
		item := conf.CheckItem{
			Name: "kernel",
			Spec: spec,
			Doc:  "kernel doc",
		}
		checker := &kernelModuleChecker{}
		checker.Init(ctx, item)
		result := checker.Check()
		if result.Result != ResultInvalid {
			t.Errorf("expected CheckInvalid for missing module, got %v", result.Result)
		}
	})

	t.Run("参数读取失败", func(t *testing.T) {
		RootPath := t.TempDir()
		ctx := Context{
			RootPath: RootPath,
		}

		procContent := "mod1 0 0 0 0 Live 0\n"
		if err := writeProcModules(RootPath, procContent); err != nil {
			t.Fatalf("failed to write proc_modules: %v", err)
		}
		if err := writeSysModule(RootPath, []string{"mod1"}); err != nil {
			t.Fatalf("failed to write sys_module: %v", err)
		}
		// 不写param1文件
		spec := []map[string]interface{}{
			{
				"name": "mod1",
				"doc":  "test doc",
				"detail": []map[string]interface{}{
					{
						"name":   "param1",
						"type":   "int",
						"expect": "123",
						"doc":    "detail doc",
					},
				},
			},
		}
		item := conf.CheckItem{
			Name: "kernel",
			Spec: spec,
			Doc:  "kernel doc",
		}
		checker := &kernelModuleChecker{}
		checker.Init(ctx, item)
		result := checker.Check()
		if result.Result != ResultInvalid {
			t.Errorf("expected CheckInvalid for param read fail, got %v", result.Result)
		}
	})
}
