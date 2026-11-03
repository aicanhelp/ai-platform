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
	"math"
	"strconv"
	"time"

	v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/dynamic"

	"openfuyao/oauth-server/cmd/oauth-server/app/config"
	"openfuyao/oauth-server/pkg/constants"
	"openfuyao/oauth-server/pkg/fuyaouser"
	"openfuyao/oauth-server/pkg/i18n"
	"openfuyao/oauth-server/pkg/zlog"
)

// LoginUserProtector is the structure for IPProtector
type LoginUserProtector struct {
	dynamicClient dynamic.Interface
	failTimes     int
	LockDuration  time.Duration
}

// NewLoginUserProtector inits LoginIPProtector
func NewLoginUserProtector(client dynamic.Interface, config *config.IPProtectorConfig) *LoginUserProtector {
	return &LoginUserProtector{
		dynamicClient: client,
		failTimes:     config.FailTimes,
		LockDuration:  config.LockDuration,
	}
}

// CheckLocked checks whether username is locked and return the remaining locked time if it's locked
func (p *LoginUserProtector) CheckLocked(username, locale string) (bool, string) {
	// fetch user cr
	userCR, err := fuyaouser.GetUserInfo(p.dynamicClient, username)
	if err != nil {
		zlog.LogErrorf("user %s does not exist, err: %v", username, err)
		return true, i18n.T(locale, "error.username_or_password")
	}

	if userCR.Status.LockStatus == "Locked" {
		remainingTime := math.Ceil(userCR.Status.LockedTimestamp.Add(p.LockDuration).Sub(time.Now()).Minutes())
		errString := i18n.TFormattable(locale, "error.login_blocked",
			strconv.FormatInt(int64(remainingTime), constants.Decimal))
		return true, errString
	}

	return false, ""
}

// AddFailedLogin adds a new failed sample to the user and return attempts left to try logging in
func (p *LoginUserProtector) AddFailedLogin(username string) int {
	// fetch user cr
	userCR, err := fuyaouser.GetUserInfo(p.dynamicClient, username)
	if err != nil {
		zlog.LogErrorf("cannot fetch user %s from userCRs, err: %v", username, err)
		return 0
	}

	failedLoginRecords := append(userCR.Spec.FailedLoginRecords, v1.Time{Time: time.Now()})
	userCR.Spec.FailedLoginRecords = failedLoginRecords
	if err = fuyaouser.UpdateUserInfo(p.dynamicClient, userCR); err != nil {
		zlog.LogErrorf("cannot update user cr, err: %v", err)
		return 0
	}

	// refetch user the get the latest remaining attempts
	// sleep 0.05s to get the newest userCR
	const sleepMilliSeconds = 50
	time.Sleep(sleepMilliSeconds * time.Millisecond)
	userCR, err = fuyaouser.GetUserInfo(p.dynamicClient, username)
	if err != nil {
		zlog.LogErrorf("cannot fetch user %s from userCRs, err: %v", username, err)
		return 0
	}

	remainAttempts := userCR.Status.RemainAttempts
	if remainAttempts < 0 {
		remainAttempts = 0
	}

	return remainAttempts
}

// Unlock releases the locked user
func (p *LoginUserProtector) Unlock(username string) {
	// fetch user cr
	userCR, err := fuyaouser.GetUserInfo(p.dynamicClient, username)
	if err != nil {
		zlog.LogErrorf("cannot fetch user %s from userCRs, err: %v", username, err)
		return
	}

	// reset user cr
	userCR.Spec.FailedLoginRecords = []v1.Time{}

	// update user
	if err = fuyaouser.UpdateUserInfo(p.dynamicClient, userCR); err != nil {
		zlog.LogErrorf("cannot update user cr, err: %v", err)
		return
	}

	return
}
