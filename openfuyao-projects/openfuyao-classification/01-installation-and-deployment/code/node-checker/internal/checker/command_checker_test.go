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
	"reflect"
	"testing"

	"github.com/agiledragon/gomonkey/v2"

	"openfuyao.com/oscheck/internal/conf"
	"openfuyao.com/oscheck/internal/utils"
)

func TestCommandCheckerInit(t *testing.T) {
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
		spec := []map[string]interface{}{
			{
				"name":    "testcmd",
				"command": "echo 123",
				"type":    "int",
				"expect":  "123",
				"doc":     "doc1",
			},
		}
		item := conf.CheckItem{
			Name: "cmdtest",
			Spec: spec,
			Doc:  "doc",
		}
		checker := &commandChecker{}
		err := checker.Init(ctx, item)
		if err != nil {
			t.Errorf("Init failed: %v", err)
		}
		if len(checker.subCheckers) != 1 {
			t.Errorf("expected 1 subChecker, got %d", len(checker.subCheckers))
		}
	})

	t.Run("缺少command字段", func(t *testing.T) {
		spec := []map[string]interface{}{
			{
				"name":   "testcmd",
				"type":   "int",
				"expect": "123",
			},
		}
		item := conf.CheckItem{
			Name: "cmdtest",
			Spec: spec,
		}
		checker := &commandChecker{}
		err := checker.Init(ctx, item)
		if err == nil {
			t.Errorf("expected error for missing command, got nil")
		}
	})

	t.Run("valueChecker类型错误", func(t *testing.T) {
		spec := []map[string]interface{}{
			{
				"name":    "testcmd",
				"command": "echo 123",
				"type":    "notype",
				"expect":  "123",
			},
		}
		item := conf.CheckItem{
			Name: "cmdtest",
			Spec: spec,
		}
		checker := &commandChecker{}
		err := checker.Init(ctx, item)
		if err == nil {
			t.Errorf("expected error for invalid valueChecker type, got nil")
		}
	})
}

func TestCommandCheckerCheck(t *testing.T) {
	release, logPath := InitLogForTest(t)
	defer func() {
		release()
		err := os.Remove(logPath)
		if err != nil {
			fmt.Printf("Remove log file failed: %v", err)
		}
	}()

	ctx := Context{}

	t.Run("命令执行成功且匹配", func(t *testing.T) {

		patches := gomonkey.ApplyMethod(reflect.TypeOf(&utils.Command{}), "Exec",
			func(_ *utils.Command) (*utils.CommandExecResult, error) {
				return &utils.CommandExecResult{
					Stdout: "123",
				}, nil
			})
		defer patches.Reset()

		spec := []map[string]interface{}{
			{
				"name":    "testcmd",
				"command": "echo 123",
				"type":    "int",
				"expect":  "123",
			},
		}
		item := conf.CheckItem{
			Name: "cmdtest",
			Spec: spec,
		}
		checker := &commandChecker{}
		_ = checker.Init(ctx, item)
		result := checker.Check()
		if result.Result != ResultValid {
			t.Errorf("expected CheckValid, got %v", result.Result)
		}
	})

	t.Run("命令执行成功但不匹配", func(t *testing.T) {
		patches := gomonkey.ApplyMethod(reflect.TypeOf(&utils.Command{}), "Exec",
			func(_ *utils.Command) (*utils.CommandExecResult, error) {
				return &utils.CommandExecResult{
					Stdout: "456",
				}, nil
			})
		defer patches.Reset()

		spec := []map[string]interface{}{
			{
				"name":    "testcmd",
				"command": "echo 456",
				"type":    "int",
				"expect":  "123",
			},
		}
		item := conf.CheckItem{
			Name: "cmdtest",
			Spec: spec,
		}
		checker := &commandChecker{}
		_ = checker.Init(ctx, item)
		result := checker.Check()
		if result.Result != ResultInvalid {
			t.Errorf("expected CheckInvalid, got %v", result.Result)
		}
	})

	t.Run("命令执行结果不符合预期", func(t *testing.T) {
		patches := gomonkey.ApplyMethod(reflect.TypeOf(&utils.Command{}), "Exec",
			func(_ *utils.Command) (*utils.CommandExecResult, error) {
				return &utils.CommandExecResult{
					Stdout:   "123",
					ExitCode: 1,
				}, nil
			})
		defer patches.Reset()

		spec := []map[string]interface{}{
			{
				"name":    "testcmd",
				"command": "badcmd",
				"type":    "int",
				"expect":  "333",
			},
		}
		item := conf.CheckItem{
			Name: "cmdtest",
			Spec: spec,
		}
		checker := &commandChecker{}
		_ = checker.Init(ctx, item)
		result := checker.Check()
		if result.Result != ResultInvalid {
			t.Errorf("expected CheckInvalid for command error, got %v", result.Result)
		}
	})

	t.Run("多个子项混合结果", func(t *testing.T) {
		patches := gomonkey.ApplyMethod(reflect.TypeOf(&utils.Command{}), "Exec",
			func(_ *utils.Command) (*utils.CommandExecResult, error) {
				return &utils.CommandExecResult{
					Stdout: "123",
				}, nil
			})
		defer patches.Reset()

		spec := []map[string]interface{}{
			{
				"name":    "okcmd",
				"command": "echo 123",
				"type":    "int",
				"expect":  "123",
			},
			{
				"name":    "failcmd",
				"command": "badcmd",
				"type":    "int",
				"expect":  "333",
			},
		}
		item := conf.CheckItem{
			Name: "cmdtest",
			Spec: spec,
		}
		checker := &commandChecker{}
		_ = checker.Init(ctx, item)
		result := checker.Check()
		if result.Result != ResultInvalid {
			t.Errorf("expected CheckInvalid for mixed result, got %v", result.Result)
		}
		if len(result.SubItems) != len(spec) {
			t.Errorf("expected %d subItems, got %d", len(spec), len(result.SubItems))
		}
	})
}
