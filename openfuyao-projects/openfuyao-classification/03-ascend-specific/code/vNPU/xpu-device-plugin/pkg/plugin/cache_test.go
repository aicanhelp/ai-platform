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

package plugin

import (
	"testing"

	"huawei.com/vxpu-device-plugin/pkg/plugin/xpu"
)

func TestNewDeviceCache(t *testing.T) {
	dc := NewDeviceCache()
	if dc == nil {
		t.Errorf("new device cache is nil")
	}
}

func TestAddNotifyChannel(t *testing.T) {
	dc := NewDeviceCache()
	testChan := make(chan *xpu.Device)
	dc.AddNotifyChannel("test", testChan)
	if dc.notifyCh["test"] != testChan {
		t.Errorf("error adding notify channel")
	}
}

func TestRemoveNotifyChannel(t *testing.T) {
	dc := NewDeviceCache()
	testChan := make(chan *xpu.Device)
	dc.AddNotifyChannel("test", testChan)
	if dc.notifyCh["test"] != testChan {
		t.Errorf("error adding notify channel")
	}
	dc.RemoveNotifyChannel("test")
	if dc.notifyCh["test"] == testChan {
		t.Errorf("error deleting notify channel")
	}
}

func TestStop(t *testing.T) {
	dc := NewDeviceCache()
	dc.Stop()
	select {
	case <-dc.stopCh:
	default:
		t.Error("Channel is not closed")
	}
}

func TestNotifyLoopStop(t *testing.T) {
	dc := NewDeviceCache()
	testCh := make(chan int)
	go func() {
		dc.notifyLoop()
		testCh <- 1
	}()
	dc.stopCh <- 1
	select {
	case <-testCh:
	default:
		t.Error("No notification in stop channel")
	}
}

func TestNotifyLoopUnhealthy(t *testing.T) {
	dc := NewDeviceCache()
	testChan := make(chan *xpu.Device)
	dc.AddNotifyChannel("test", testChan)
	go dc.notifyLoop()
	dc.unhealthy <- &xpu.Device{}
	select {
	case <-dc.notifyCh["test"]:
	default:
		t.Error("No notification in devices channel")
	}
	dc.stopCh <- 1
}
