/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package protector

import (
	"errors"
	"testing"
	"time"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"
	v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/dynamic"

	appconfig "openfuyao/oauth-server/cmd/oauth-server/app/config"
	"openfuyao/oauth-server/pkg/fuyaouser"
	"openfuyao/oauth-server/pkg/i18n"
)

const (
	failTimes   = 3
	lockMinutes = 5
)

// 测试 CheckLocked 用户不存在
func TestCheckLockedUserNotFound(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	// 模拟 GetUserInfo 返回错误
	patches.ApplyFunc(
		fuyaouser.GetUserInfo,
		func(_ dynamic.Interface, username string) (*fuyaouser.User, error) {
			assert.Equal(t, "testuser", username)
			return nil, errors.New("user not found")
		},
	)

	// 创建保护器
	protector := NewLoginUserProtector(
		nil, // 不需要实际客户端
		&appconfig.IPProtectorConfig{
			LockDuration: lockMinutes * time.Minute,
			FailTimes:    failTimes,
		},
	)

	// 调用函数
	locked, msg := protector.CheckLocked("testuser", "zh-CN")

	// 验证结果
	assert.True(t, locked)
	assert.Contains(t, msg, i18n.T("zh-CN", "error.username_or_password"))
}

// 测试 CheckLocked 用户被锁定
func TestCheckLockedUserLocked(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	// 创建锁定用户
	lockedUser := &fuyaouser.User{
		Status: fuyaouser.UserStatus{
			LockStatus:      "Locked",
			LockedTimestamp: &v1.Time{Time: time.Now().Add(-1 * time.Minute)},
		},
	}

	// 模拟 GetUserInfo 返回锁定用户
	patches.ApplyFunc(
		fuyaouser.GetUserInfo,
		func(_ dynamic.Interface, username string) (*fuyaouser.User, error) {
			return lockedUser, nil
		},
	)

	// 创建保护器
	protector := NewLoginUserProtector(
		nil,
		&appconfig.IPProtectorConfig{
			FailTimes:    failTimes,
			LockDuration: lockMinutes * time.Minute,
		},
	)

	// 调用函数
	locked, msg := protector.CheckLocked("testuser", "zh-CN")

	// 验证结果
	assert.True(t, locked)
	assert.Contains(t, msg, "4") // 剩余分钟数 (5-1=4)
}

// 测试 CheckLocked 用户未锁定
func TestCheckLockedUserNotLocked(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	// 创建未锁定用户
	unlockedUser := &fuyaouser.User{
		Status: fuyaouser.UserStatus{
			LockStatus: "Unlocked",
		},
	}

	// 模拟 GetUserInfo 返回未锁定用户
	patches.ApplyFunc(
		fuyaouser.GetUserInfo,
		func(_ dynamic.Interface, username string) (*fuyaouser.User, error) {
			return unlockedUser, nil
		},
	)

	// 创建保护器
	protector := NewLoginUserProtector(nil, &appconfig.IPProtectorConfig{})

	// 调用函数
	locked, msg := protector.CheckLocked("testuser", "zh-CN")

	// 验证结果
	assert.False(t, locked)
	assert.Empty(t, msg)
}

// 测试 AddFailedLogin 获取用户失败
func TestAddFailedLoginGetUserFailure(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	// 模拟 GetUserInfo 返回错误
	patches.ApplyFunc(
		fuyaouser.GetUserInfo,
		func(_ dynamic.Interface, username string) (*fuyaouser.User, error) {
			return nil, errors.New("get user failed")
		},
	)

	// 创建保护器
	protector := NewLoginUserProtector(nil, &appconfig.IPProtectorConfig{})

	// 调用函数
	attempts := protector.AddFailedLogin("testuser")

	// 验证结果
	assert.Equal(t, 0, attempts)
}

// 测试 AddFailedLogin 更新用户失败
func TestAddFailedLoginUpdateFailure(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	// 模拟 UpdateUserInfo 返回错误
	patches.ApplyFunc(
		fuyaouser.UpdateUserInfo,
		func(_ dynamic.Interface, _ *fuyaouser.User) error {
			return errors.New("update failed")
		},
	)

	// 模拟 GetUserInfo 返回用户
	patches.ApplyFunc(
		fuyaouser.GetUserInfo,
		func(_ dynamic.Interface, username string) (*fuyaouser.User, error) {
			return &fuyaouser.User{}, nil
		},
	)

	// 创建保护器
	protector := NewLoginUserProtector(nil, &appconfig.IPProtectorConfig{})

	// 调用函数
	attempts := protector.AddFailedLogin("testuser")

	// 验证结果
	assert.Equal(t, 0, attempts)
}

// 测试 AddFailedLogin 剩余尝试次数为负
func TestAddFailedLoginNegativeAttempts(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	// 模拟 GetUserInfo 返回用户
	patches.ApplyFunc(
		fuyaouser.GetUserInfo,
		func(_ dynamic.Interface, username string) (*fuyaouser.User, error) {
			user := &fuyaouser.User{}
			user.Status.RemainAttempts = -1 // 负值
			return user, nil
		},
	)

	// 模拟更新用户成功
	patches.ApplyFunc(
		fuyaouser.UpdateUserInfo,
		func(_ dynamic.Interface, _ *fuyaouser.User) error {
			return nil
		},
	)

	// 创建保护器
	protector := NewLoginUserProtector(nil, &appconfig.IPProtectorConfig{})

	// 调用函数
	attempts := protector.AddFailedLogin("testuser")

	// 验证结果
	assert.Equal(t, 0, attempts)
}

// 测试 Unlock 获取用户失败
func TestUnlockGetUserFailure(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	// 模拟 GetUserInfo 返回错误
	patches.ApplyFunc(
		fuyaouser.GetUserInfo,
		func(_ dynamic.Interface, username string) (*fuyaouser.User, error) {
			return nil, errors.New("get user failed")
		},
	)

	// 创建保护器
	protector := NewLoginUserProtector(nil, &appconfig.IPProtectorConfig{})

	// 调用函数
	protector.Unlock("testuser")
}

// 测试 Unlock 更新用户失败
func TestUnlockUpdateFailure(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	// 模拟 GetUserInfo 返回用户
	patches.ApplyFunc(
		fuyaouser.GetUserInfo,
		func(_ dynamic.Interface, username string) (*fuyaouser.User, error) {
			return &fuyaouser.User{}, nil
		},
	)

	// 模拟 UpdateUserInfo 返回错误
	patches.ApplyFunc(
		fuyaouser.UpdateUserInfo,
		func(_ dynamic.Interface, _ *fuyaouser.User) error {
			return errors.New("update failed")
		},
	)

	// 创建保护器
	protector := NewLoginUserProtector(nil, &appconfig.IPProtectorConfig{})

	// 调用函数
	protector.Unlock("testuser")
}

// 测试 Unlock 成功解锁
func TestUnlockSuccess(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	// 创建锁定用户
	lockedUser := &fuyaouser.User{
		Spec: fuyaouser.UserSpec{
			FailedLoginRecords: []v1.Time{
				{Time: time.Now()},
				{Time: time.Now()},
			},
		},
		Status: fuyaouser.UserStatus{
			LockStatus: "Locked",
		},
	}

	// 模拟 GetUserInfo 返回用户
	patches.ApplyFunc(
		fuyaouser.GetUserInfo,
		func(_ dynamic.Interface, username string) (*fuyaouser.User, error) {
			return lockedUser, nil
		},
	)

	// 捕获更新后的用户
	var updatedUser *fuyaouser.User
	patches.ApplyFunc(
		fuyaouser.UpdateUserInfo,
		func(_ dynamic.Interface, user *fuyaouser.User) error {
			updatedUser = user
			return nil
		},
	)

	// 创建保护器
	protector := NewLoginUserProtector(nil, &appconfig.IPProtectorConfig{})

	// 调用函数
	protector.Unlock("testuser")

	// 验证结果
	if updatedUser != nil && len(updatedUser.Spec.FailedLoginRecords) != 0 {
		t.Errorf("expected user.Spec.FailedLoginRecords to be nil")
	}
}
