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
* Date: 2025-09-15
 */

package utils

import (
	"sync"
	"testing"
	"time"
)

func TestNotifySyncDone(t *testing.T) {
	cacheSyncChan = make(chan struct{}, 3)

	var wg sync.WaitGroup
	wg.Add(1)

	go func() {
		defer wg.Done()
		WaitCacheSync()
	}()

	NotifySyncDone()

	wgDone := make(chan struct{})
	go func() {
		wg.Wait()
		close(wgDone)
	}()

	timer := time.NewTimer(100 * time.Millisecond)
	defer timer.Stop()

	select {
	case <-wgDone:
	case <-timer.C:
		t.Error("WaitCacheSync did not return after NotifySyncDone")
	}
}

func TestWaitCacheSyncMultipleTimes(t *testing.T) {
	cacheSyncChan = make(chan struct{}, 3)

	NotifySyncDone()

	timer := time.NewTimer(100 * time.Millisecond)
	defer timer.Stop()

	for i := 0; i < 3; i++ {
		done := make(chan struct{})
		go func() {
			WaitCacheSync()
			close(done)
		}()

		if !timer.Stop() {
			select {
			case <-timer.C:
			default:
			}
		}
		timer.Reset(100 * time.Millisecond)

		select {
		case <-done:
		case <-timer.C:
			t.Errorf("WaitCacheSync blocked on iteration %d", i)
		}
	}

	done := make(chan struct{})
	go func() {
		WaitCacheSync()
		close(done)
	}()

	if !timer.Stop() {
		select {
		case <-timer.C:
		default:
		}
	}
	timer.Reset(100 * time.Millisecond)

	select {
	case <-done:
	case <-timer.C:
		t.Error("WaitCacheSync blocked after channel is closed and empty")
	}
}

func TestNotifySyncDoneSendsThreeItems(t *testing.T) {
	cacheSyncChan = make(chan struct{}, 3)

	NotifySyncDone()

	for i := 0; i < 3; i++ {
		select {
		case _, ok := <-cacheSyncChan:
			if !ok {
				t.Errorf("Expected to read item %d but channel was closed", i)
			}
		default:
			t.Errorf("Expected 3 items in channel but only got %d", i)
		}
	}

	_, ok := <-cacheSyncChan
	if ok {
		t.Error("Expected channel to be closed after reading all items")
	}
}

func TestWaitCacheSyncBeforeNotify(t *testing.T) {
	cacheSyncChan = make(chan struct{}, 3)

	done := make(chan struct{})
	go func() {
		WaitCacheSync()
		close(done)
	}()

	time.Sleep(10 * time.Millisecond)

	NotifySyncDone()

	timer := time.NewTimer(100 * time.Millisecond)
	defer timer.Stop()

	select {
	case <-done:
	case <-timer.C:
		t.Error("WaitCacheSync did not return after NotifySyncDone")
	}
}

func TestConcurrentAccess(t *testing.T) {
	cacheSyncChan = make(chan struct{}, 3)

	var wg sync.WaitGroup
	const numGoroutines = 10

	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			WaitCacheSync()
		}()
	}

	NotifySyncDone()

	done := make(chan struct{})
	go func() {
		wg.Wait()
		close(done)
	}()

	timer := time.NewTimer(100 * time.Millisecond)
	defer timer.Stop()

	select {
	case <-done:
	case <-timer.C:
		t.Error("Not all WaitCacheSync calls returned in time")
	}
}
