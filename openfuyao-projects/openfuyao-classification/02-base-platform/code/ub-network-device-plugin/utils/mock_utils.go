/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package utils

import (
	"context"
	"encoding/json"
	"fmt"
	"io/fs"
	"os"
	"os/exec"
	"time"

	"github.com/agiledragon/gomonkey/v2"
	"go.opentelemetry.io/otel/trace"
	cri "k8s.io/cri-api/pkg/apis"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	critesting "k8s.io/cri-api/pkg/apis/testing"
	remote "k8s.io/cri-client/pkg"
	"k8s.io/klog/v2"
)

type FakeFileInfo struct {
	name     string
	contents string
	mode     fs.FileMode
	offset   int
}

func (f *FakeFileInfo) Name() string {
	return f.name
}

func (f *FakeFileInfo) Size() int64 {
	return int64(len(f.contents))
}

func (f *FakeFileInfo) Mode() fs.FileMode {
	return f.mode
}

func (f *FakeFileInfo) ModTime() time.Time {
	return time.Time{}
}

func (f *FakeFileInfo) IsDir() bool {
	return false
}

func (f *FakeFileInfo) Sys() any {
	return nil
}

type MockUtils struct {
	parseJsonStrPatches            *gomonkey.Patches
	StatPatches                    *gomonkey.Patches
	ChmodPatches                   *gomonkey.Patches
	GetPodUIDByDeviceIDPatches     *gomonkey.Patches
	NewRemoteRuntimeServicePatches *gomonkey.Patches
	GetSandboxesPatches            *gomonkey.Patches
	GetSandboxPidPatches           *gomonkey.Patches
	CombinedOutputPatches          *gomonkey.Patches

	ResetFunc []func()
}

func (m *MockUtils) Reset() {
	for _, fn := range m.ResetFunc {
		fn()
	}
}

func (m *MockUtils) mockParseJsonStr(jsonStr string) {
	m.parseJsonStrPatches = gomonkey.ApplyFunc(parseJsonStr, func(string) (map[string]interface{}, error) {
		data := make(map[string]interface{})
		_ = json.Unmarshal([]byte(jsonStr), &data)
		return data, nil
	})
	m.ResetFunc = append(m.ResetFunc, m.parseJsonStrPatches.Reset)
}

func (m *MockUtils) mockStat(name string) {
	switch name {
	case "file not exist":
		m.StatPatches = gomonkey.ApplyFunc(os.Stat, func(string) (os.FileInfo, error) {
			return nil, os.ErrNotExist
		})
	case "file already has permission":
		m.StatPatches = gomonkey.ApplyFunc(os.Stat, func(string) (os.FileInfo, error) {
			return &FakeFileInfo{
				mode: Permission,
			}, nil
		})
	case "file's permission can be set":
		m.StatPatches = gomonkey.ApplyFunc(os.Stat, func(string) (os.FileInfo, error) {
			return &FakeFileInfo{
				mode: 0555,
			}, nil
		})
	default:
	}

	m.ResetFunc = append(m.ResetFunc, m.StatPatches.Reset)
}

func (m *MockUtils) mockChmod() {
	m.ChmodPatches = gomonkey.ApplyFunc(os.Chmod, func(string, os.FileMode) error {
		return nil
	})

	m.ResetFunc = append(m.ResetFunc, m.ChmodPatches.Reset)
}

func (m *MockUtils) MockGetPodUIDByDeviceID() {
	m.GetPodUIDByDeviceIDPatches = gomonkey.ApplyFunc(GetPodUIDByDeviceID, func(string, string) (string, error) {
		return "", nil
	})

	m.ResetFunc = append(m.ResetFunc, m.GetPodUIDByDeviceIDPatches.Reset)
}

func (m *MockUtils) MockNewRemoteRuntimeService() {
	m.NewRemoteRuntimeServicePatches = gomonkey.ApplyFunc(remote.NewRemoteRuntimeService, func(string, time.Duration, trace.TracerProvider, *klog.Logger) (cri.RuntimeService, error) {
		return critesting.NewFakeRuntimeService(), nil
	})

	m.ResetFunc = append(m.ResetFunc, m.NewRemoteRuntimeServicePatches.Reset)
}

func (m *MockUtils) MockGetSandboxes() {
	m.GetSandboxesPatches = gomonkey.ApplyFunc(GetSandboxes, func(context.Context, cri.RuntimeService, string) ([]*runtimeapi.PodSandbox, error) {
		return []*runtimeapi.PodSandbox{
			{Id: "test1"},
		}, nil
	})

	m.ResetFunc = append(m.ResetFunc, m.GetSandboxesPatches.Reset)
}

func (m *MockUtils) MockGetSandboxPid() {
	m.GetSandboxPidPatches = gomonkey.ApplyFunc(GetSandboxPid, func(context.Context, cri.RuntimeService, string) (string, error) {
		return "pid", nil
	})

	m.ResetFunc = append(m.ResetFunc, m.GetSandboxPidPatches.Reset)
}

func (m *MockUtils) MockCombinedOutput(name string) {
	switch name {
	case "setUrmaDevNetworkNS err":
		m.CombinedOutputPatches = gomonkey.ApplyFunc((*exec.Cmd).CombinedOutput, func(c *exec.Cmd) ([]byte, error) {
			return []byte{}, fmt.Errorf("exec failed: %w", os.ErrPermission)
		})
	default:
		m.CombinedOutputPatches = gomonkey.ApplyFunc((*exec.Cmd).CombinedOutput, func(c *exec.Cmd) ([]byte, error) {
			return []byte{}, nil
		})
	}

	m.ResetFunc = append(m.ResetFunc, m.CombinedOutputPatches.Reset)
}
