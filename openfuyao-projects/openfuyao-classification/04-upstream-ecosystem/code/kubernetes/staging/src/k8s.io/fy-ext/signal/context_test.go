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
	"testing"
	"time"
)

func TestFyConvertStopChanToContext(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	stopCh := ctx.Done()

	ctxNew := FyConvertStopChanToContext(stopCh)

	count := 0
	go func() {
		<-ctxNew.Done()
		count++
	}()

	time.Sleep(100 * time.Millisecond)
	if count != 0 {
		t.Errorf("expect count is 0")
	}

	cancel()
	time.Sleep(100 * time.Millisecond)
	if count != 1 {
		t.Errorf("expect count is 1")
	}
}
