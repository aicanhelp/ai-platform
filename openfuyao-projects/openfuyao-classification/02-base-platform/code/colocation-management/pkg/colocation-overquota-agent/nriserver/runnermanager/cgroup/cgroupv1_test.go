/*
* Copyright (c) 2024 China Unicom Digital Technology Co., Ltd.
* openFuyao is licensed under Mulan PSL v2.
* You can use this software according to the terms and conditions of the Mulan PSL v2.
* You may obtain a copy of Mulan PSL v2 at:
*          http://license.coscl.org.cn/MulanPSL2
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
* EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
* MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
* See the Mulan PSL v2 for more details.
* Author: YuXiang Guo
* Date: 2025-09-08
 */

// Package cgroup
package cgroup

import (
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"

	"openfuyao.com/colocation-management/pkg/common"
	"openfuyao.com/colocation-management/pkg/utils"
)

// CgroupHelper 定义helper接口
type CgroupHelper interface {
	CgroupFileRead(path string) ([]byte, error)
	CgroupFIleWrite(path string, data string) error
	IsCpuSetEqual(current, new string) bool
}

// RealCgroupHelper 模拟实际的helper实现
type RealCgroupHelper struct{}

func (r *RealCgroupHelper) CgroupFileRead(path string) ([]byte, error) {
	return nil, nil
}

func (r *RealCgroupHelper) CgroupFIleWrite(path string, data string) error {
	return nil
}

func (r *RealCgroupHelper) IsCpuSetEqual(current, new string) bool {
	return current == new
}

// MockCgroupHelper 用于模拟 CgroupHelper的功能
type MockCgroupHelper struct {
	mock.Mock
}

func (m *MockCgroupHelper) CgroupFileRead(path string) ([]byte, error) {
	args := m.Called(path)
	return args.Get(0).([]byte), args.Error(1)
}

func (m *MockCgroupHelper) CgroupFIleWrite(path string, data string) error {
	args := m.Called(path, data)
	return args.Error(0)
}

func (m *MockCgroupHelper) IsCpuSetEqual(current, new string) bool {
	args := m.Called(current, new)
	return args.Bool(0)
}

// 修改fakev1CgroupManager
type fakev1CgroupManager struct {
	Config Config
	helper CgroupHelper
}

// NewFakeV1CgroupManagerWithHelper 创建带有自定义helper的v1cgroup manager
func NewFakeV1CgroupManagerWithHelper(helper CgroupHelper) *fakev1CgroupManager {
	return &fakev1CgroupManager{
		helper: helper,
	}
}

// SetCpuSet 修改原始的SetCpuSet方法以使用注入的helper
func (cm *fakev1CgroupManager) SetCpuSet(cgroupPath string, resource utils.LinuxContainerResources) error {
	if resource.CPUSet == nil {
		return nil
	}
	if *resource.CPUSet == "" {
		return nil
	}

	read, err := cm.helper.CgroupFileRead(CgroupAbsPath(cgroupPath, "", common.CPUSetPath))
	if err != nil {
		return err
	}
	if !cm.helper.IsCpuSetEqual(string(read), *resource.CPUSet) {
		err = cm.helper.CgroupFIleWrite(CgroupAbsPath(cgroupPath, "",
			common.CPUSetPath), *resource.CPUSet)
		if err != nil {
			return err
		}
	}
	return nil
}

// SetCpuSetMem 修改原始的SetCpuSetMem方法以使用注入的helper
func (cm *fakev1CgroupManager) SetCpuSetMem(cgroupPath string, resource utils.LinuxContainerResources) error {
	if resource.CPUSetMem == nil {
		return nil
	}
	if *resource.CPUSetMem == "" {
		return nil
	}

	read, err := cm.helper.CgroupFileRead(CgroupAbsPath(cgroupPath, "", common.CPUSetMemPath))
	if err != nil {
		return err
	}
	if strings.TrimSpace(string(read)) != strings.TrimSpace(*resource.CPUSetMem) {
		err = cm.helper.CgroupFIleWrite(CgroupAbsPath(cgroupPath, "",
			common.CPUSetMemPath), *resource.CPUSetMem)
		if err != nil {
			return err
		}
	}
	return nil
}

// SetCpuShares 修改原始的SetCpuShares方法以使用注入的helper
func (cm *fakev1CgroupManager) SetCpuShares(cgroupPath string, resource utils.LinuxContainerResources) error {
	if resource.CPUShares == nil {
		return nil
	}

	read, err := cm.helper.CgroupFileRead(CgroupAbsPath(cgroupPath, "", common.CPUSharesPath))
	if err != nil {
		return err
	}
	if string(read) == strconv.FormatInt(*resource.CPUShares, DecimalBase) {
		return nil
	}
	err = cm.helper.CgroupFIleWrite(CgroupAbsPath(cgroupPath, "",
		common.CPUSharesPath), strconv.FormatInt(*resource.CPUShares, DecimalBase))
	if err != nil {
		return err
	}
	return nil
}

// SetCpuQuota 修改原始的SetCpuQuota方法以使用注入的helper
func (cm *fakev1CgroupManager) SetCpuQuota(cgroupPath string, resource utils.LinuxContainerResources) error {
	if resource.CFSQuota == nil {
		return nil
	}

	read, err := cm.helper.CgroupFileRead(CgroupAbsPath(cgroupPath, "", common.CPUQuotaPath))
	if err != nil {
		return err
	}

	if string(read) == strconv.FormatInt(*resource.CFSQuota, DecimalBase) {
		return nil
	}
	err = cm.helper.CgroupFIleWrite(CgroupAbsPath(cgroupPath, "",
		common.CPUQuotaPath), strconv.FormatInt(*resource.CFSQuota, DecimalBase))
	if err != nil {
		return err
	}
	return nil
}

// SetMemoryLimit 修改原始的SetMemoryLimit方法以使用注入的helper
func (cm *fakev1CgroupManager) SetMemoryLimit(cgroupPath string, resource utils.LinuxContainerResources) error {
	if resource.MemoryLimit == nil {
		return nil
	}

	read, err := cm.helper.CgroupFileRead(CgroupAbsPath(cgroupPath, "", common.MemoryLimitPath))
	if err != nil {
		return err
	}

	if string(read) == strconv.FormatInt(*resource.MemoryLimit, DecimalBase) {
		return nil
	}
	err = cm.helper.CgroupFIleWrite(CgroupAbsPath(cgroupPath, "",
		common.MemoryLimitPath), strconv.FormatInt(*resource.MemoryLimit, DecimalBase))
	if err != nil {
		return err
	}

	return nil
}

// 测试
func TestSetCpuSet(t *testing.T) {
	tests := []struct {
		name        string
		cgroupPath  string
		resource    utils.LinuxContainerResources
		setupMock   func(mockHelper *MockCgroupHelper)
		expectError bool
	}{
		{
			name:       "nil CPUSet",
			cgroupPath: "test/path",
			resource:   utils.LinuxContainerResources{CPUSet: nil},
			setupMock:  func(mockHelper *MockCgroupHelper) {},
		},
		{
			name:       "empty CPUSet",
			cgroupPath: "test/path",
			resource:   utils.LinuxContainerResources{CPUSet: func() *string { s := ""; return &s }()},
			setupMock:  func(mockHelper *MockCgroupHelper) {},
		},
		{
			name:       "CPUSet already set correctly",
			cgroupPath: "test/path",
			resource:   utils.LinuxContainerResources{CPUSet: func() *string { s := "0-3"; return &s }()},
			setupMock: func(mockHelper *MockCgroupHelper) {
				mockHelper.On("CgroupFileRead", mock.Anything).Return([]byte("0-3"), nil)
				mockHelper.On("IsCpuSetEqual", "0-3", "0-3").Return(true)
			},
		},
		{
			name:       "CPUSet needs update",
			cgroupPath: "test/path",
			resource:   utils.LinuxContainerResources{CPUSet: func() *string { s := "0-3"; return &s }()},
			setupMock: func(mockHelper *MockCgroupHelper) {
				mockHelper.On("CgroupFileRead", mock.Anything).Return([]byte("0-2"), nil)
				mockHelper.On("IsCpuSetEqual", "0-2", "0-3").Return(false)
				mockHelper.On("CgroupFIleWrite", mock.Anything, "0-3").Return(nil)
			},
		},
		{
			name:        "read error",
			cgroupPath:  "test/path",
			resource:    utils.LinuxContainerResources{CPUSet: func() *string { s := "0-3"; return &s }()},
			expectError: true,
			setupMock: func(mockHelper *MockCgroupHelper) {
				mockHelper.On("CgroupFileRead", mock.Anything).Return([]byte(""), assert.AnError)
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mockHelper := &MockCgroupHelper{}
			tt.setupMock(mockHelper)

			manager := NewFakeV1CgroupManagerWithHelper(mockHelper)
			err := manager.SetCpuSet(tt.cgroupPath, tt.resource)

			if tt.expectError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
			mockHelper.AssertExpectations(t)
		})
	}
}

func TestSetCpuShares(t *testing.T) {
	shares := int64(1024)
	sharesStr := strconv.FormatInt(shares, DecimalBase)

	tests := []struct {
		name        string
		cgroupPath  string
		resource    utils.LinuxContainerResources
		setupMock   func(mockHelper *MockCgroupHelper)
		expectError bool
	}{
		{
			name:       "nil CPUShares",
			cgroupPath: "test/path",
			resource:   utils.LinuxContainerResources{CPUShares: nil},
			setupMock:  func(mockHelper *MockCgroupHelper) {},
		},
		{
			name:       "CPUShares already set correctly",
			cgroupPath: "test/path",
			resource:   utils.LinuxContainerResources{CPUShares: &shares},
			setupMock: func(mockHelper *MockCgroupHelper) {
				mockHelper.On("CgroupFileRead", mock.Anything).Return([]byte(sharesStr), nil)
			},
		},
		{
			name:       "CPUShares needs update",
			cgroupPath: "test/path",
			resource:   utils.LinuxContainerResources{CPUShares: &shares},
			setupMock: func(mockHelper *MockCgroupHelper) {
				mockHelper.On("CgroupFileRead", mock.Anything).Return([]byte("512"), nil)
				mockHelper.On("CgroupFIleWrite", mock.Anything, sharesStr).Return(nil)
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mockHelper := &MockCgroupHelper{}
			tt.setupMock(mockHelper)

			manager := NewFakeV1CgroupManagerWithHelper(mockHelper)
			err := manager.SetCpuShares(tt.cgroupPath, tt.resource)

			if tt.expectError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
			mockHelper.AssertExpectations(t)
		})
	}
}

// TestCgroupAbsPath 测试CgroupAbsPath函数
func TestCgroupAbsPath(t *testing.T) {
	tests := []struct {
		name         string
		rootPath     string
		relativePath string
		cgroupPath   string
		expected     string
	}{
		{
			name:         "basic path",
			rootPath:     "kubepods.slice",
			relativePath: "kubepods-podxxxxxx.slice",
			cgroupPath:   common.CPUSharesPath,
			expected: filepath.Join(common.DefaultCgroupRoot, common.CgroupCPUDir,
				"kubepods.slice", "kubepods-podxxxxxx.slice", common.CPUSharesPath),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := CgroupAbsPath(tt.rootPath, tt.relativePath, tt.cgroupPath)
			assert.Equal(t, tt.expected, result)
		})
	}
}
