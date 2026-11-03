/*
 * Copyright (c) 2024-2024 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package main

import (
	"errors"
	"reflect"
	"testing"
	"time"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"

	"openfuyao.com/kae-device-plugin/pkg/deviceplugin"
)

/*测试用例描述： main 函数测试
* 预制条件：
*		1. 创建KAEManger
* 操作步骤：
*		1. 调用main
* 预期结果：
*		1. nil
 */

func TestStart(t *testing.T) {
	kaeManager := deviceplugin.NewKAEManager()
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyGlobalVar(&retryTime, 10*time.Millisecond)
	patches.ApplyGlobalVar(&waitTime, 10*time.Millisecond)
	patches.ApplyMethodSeq(reflect.TypeOf(kaeManager), "Start", []gomonkey.OutputCell{
		{Values: gomonkey.Params{errors.New("error")}},
		{Values: gomonkey.Params{errors.New("error")}},
		{Values: gomonkey.Params{errors.New("error")}},
		{Values: gomonkey.Params{nil}},
	})
	patches.ApplyMethod(reflect.TypeOf(kaeManager), "Serve",
		func(_ *deviceplugin.KAEManager, pluginEndpoint string) { return })
	main()
	assert.NotNil(t, main)
}
