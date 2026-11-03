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

package signal

import (
	"context"
)

var globalContext context.Context

func FySetGlobalContext(stopCh <-chan struct{}) {
	if globalContext != nil {
		return
	}
	globalContext = FyConvertStopChanToContext(stopCh)
}

func FySetGlobalContextForce(stopCh <-chan struct{}) {
	globalContext = FyConvertStopChanToContext(stopCh)
}

func FyGlobalContext() context.Context {
	if globalContext != nil {
		return globalContext
	}
	// if global context does not exist, create one and return it
	ctx, _ := context.WithCancel(context.Background())
	return ctx
}

func FyConvertStopChanToContext(stopCh <-chan struct{}) context.Context {
	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		<-stopCh
		cancel()
	}()
	return ctx
}
