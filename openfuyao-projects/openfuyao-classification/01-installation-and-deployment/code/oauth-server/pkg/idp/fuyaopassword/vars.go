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

package fuyaopassword

import (
	"time"

	"k8s.io/apimachinery/pkg/runtime"
	dfake "k8s.io/client-go/dynamic/fake"

	"openfuyao/oauth-server/cmd/oauth-server/app/config"
	"openfuyao/oauth-server/pkg/fuyaouser"
	"openfuyao/oauth-server/pkg/protector"
)

const (
	failTimes        = 5
	failMinute       = 5
	lockMinute       = 20
	loginStoreMaxAge = 300

	testValidToken = "valid-access-token-for-testing"
)

var (
	fakeDynamicClient  = dfake.NewSimpleDynamicClient(runtime.NewScheme())
	fakeLoginProtector = protector.NewLoginUserProtector(fakeDynamicClient, &config.IPProtectorConfig{
		FailTimes:    failTimes,
		FailDuration: time.Minute * failMinute,
		LockDuration: time.Minute * lockMinute,
	})
	mockUserInfo *fuyaouser.User
)
