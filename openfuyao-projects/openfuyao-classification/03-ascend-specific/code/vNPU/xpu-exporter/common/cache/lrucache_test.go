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

package cache

import (
	"fmt"
	"math"
	"testing"
	"time"

	"huawei.com/xpu-exporter/common"
)

const (
	cacheTime = 100
)

func TestSetAndGet(t *testing.T) {
	cache := New(1)
	err := cache.Set("testkey1", "1", cacheTime*time.Millisecond)
	common.AssertEquals(nil, err, t)

	v, _ := cache.Get("testkey1")
	common.AssertEquals("1", v, t)

	<-time.After(time.Millisecond * cacheTime)
	v, _ = cache.Get("testkey1")
	common.AssertEquals(nil, v, t)
}

func TestDelete(t *testing.T) {
	cache := New(1)
	err := cache.Set("testkey1", "1", time.Duration(neverExpire))
	common.AssertEquals(nil, err, t)
	v, _ := cache.Get("testkey1")
	common.AssertEquals("1", v, t)

	cache.Delete("testkey1")
	v, _ = cache.Get("testkey1")
	common.AssertEquals(nil, v, t)
}

func TestIncreaseAndDecrease(t *testing.T) {
	c := &lruCache{}
	_, err := c.increment("test", time.Minute)
	common.AssertEquals(notInitErr, err, t)
	_, err = c.decrement("test", time.Minute)
	common.AssertEquals(notInitErr, err, t)

	cache := New(1)
	v, err := cache.IncreaseOne("testkey", time.Minute)
	common.AssertEquals(countInit, v, t)
	common.AssertEquals(nil, err, t)
	v, err = cache.IncreaseOne("testkey", time.Minute)
	common.AssertEquals(countInit+1, v, t)
	common.AssertEquals(nil, err, t)

	v, err = cache.DecreaseOne("testkey", time.Minute)
	common.AssertEquals(countInit, v, t)
	common.AssertEquals(nil, err, t)

	v, err = cache.DecreaseOne("testkey", time.Minute)
	common.AssertEquals(int64(0), v, t)
	common.AssertEquals(nil, err, t)

	v, err = cache.DecreaseOne("testkey", time.Microsecond)
	common.AssertEquals(int64(-1), v, t)
	common.AssertEquals(nil, err, t)
	<-time.After(time.Millisecond * 1)
	v, err = cache.DecreaseOne("testkey", time.Microsecond)
	common.AssertEquals(int64(0), v, t)
	common.AssertEquals(nil, err, t)

	v, err = cache.DecreaseOne("testkey1", time.Duration(neverExpire))
	common.AssertEquals(int64(0), v, t)
	common.AssertEquals(nil, err, t)

	value, _ := cache.Get("testkey1")
	common.AssertEquals(int64(0), value.(int64), t)
}

func TestSetIfNotExist(t *testing.T) {
	cache := New(1)
	ret := cache.SetIfNotExist("testkey", "test", cacheTime*time.Millisecond)
	common.AssertEquals(true, ret, t)

	ret = cache.SetIfNotExist("testkey", "test", cacheTime*time.Millisecond)
	common.AssertEquals(false, ret, t)

	<-time.After(time.Millisecond * cacheTime)
	ret = cache.SetIfNotExist("testkey", "test", cacheTime*time.Millisecond)
	common.AssertEquals(true, ret, t)
}

func TestNewWithInvalidMaxEntry(t *testing.T) {
	if New(0) != nil {
		t.Errorf("expected nil for maxEntry=0")
	}
	if New(-1) != nil {
		t.Errorf("expected nil for maxEntry=-1")
	}
}

func TestSetOnNilCache(t *testing.T) {
	var cl *ConcurrencyLRUCache
	common.AssertEquals(notInitErr, cl.Set("k", "v", time.Minute), t)
	cl = &ConcurrencyLRUCache{}
	common.AssertEquals(notInitErr, cl.Set("k", "v", time.Minute), t)
}

func TestSetWithInvalidExpireTime(t *testing.T) {
	cache := New(segmentCount)
	common.AssertEquals(paraErr, cache.Set("k", "v", time.Duration(neverExpire)-1), t)
	common.AssertEquals(paraErr, cache.Set("k", "v", tenYears+1), t)
}

func TestGetOnNilCache(t *testing.T) {
	var cl *ConcurrencyLRUCache
	_, err := cl.Get("k")
	common.AssertEquals(notInitErr, err, t)
	cl = &ConcurrencyLRUCache{}
	_, err = cl.Get("k")
	common.AssertEquals(notInitErr, err, t)
}

func TestGetKeyNotFound(t *testing.T) {
	cache := New(segmentCount)
	_, err := cache.Get("nonexistent")
	if err == nil {
		t.Errorf("expected error for nonexistent key")
	}
}

func TestDeleteOnNilCache(t *testing.T) {
	var cl *ConcurrencyLRUCache
	cl.Delete("k")
	cl = &ConcurrencyLRUCache{}
	cl.Delete("k")
}

func TestSetIfNotExistOnNilCache(t *testing.T) {
	var cl *ConcurrencyLRUCache
	common.AssertEquals(false, cl.SetIfNotExist("k", "v", time.Minute), t)
	cl = &ConcurrencyLRUCache{}
	common.AssertEquals(false, cl.SetIfNotExist("k", "v", time.Minute), t)
}

func TestSetIfNotExistWithInvalidExpireTime(t *testing.T) {
	cache := New(segmentCount)
	common.AssertEquals(false, cache.SetIfNotExist("k", "v", time.Duration(neverExpire)-1), t)
	common.AssertEquals(false, cache.SetIfNotExist("k", "v", tenYears+1), t)
}

func TestIncreaseOneNilAndInvalid(t *testing.T) {
	var cl *ConcurrencyLRUCache
	_, err := cl.IncreaseOne("k", time.Minute)
	common.AssertEquals(paraErr, err, t)
	cl = &ConcurrencyLRUCache{}
	_, err = cl.IncreaseOne("k", time.Minute)
	common.AssertEquals(paraErr, err, t)
	cache := New(segmentCount)
	_, err = cache.IncreaseOne("k", 0)
	common.AssertEquals(paraErr, err, t)
}

func TestDecreaseOneNilAndInvalid(t *testing.T) {
	var cl *ConcurrencyLRUCache
	_, err := cl.DecreaseOne("k", time.Minute)
	common.AssertEquals(paraErr, err, t)
	cl = &ConcurrencyLRUCache{}
	_, err = cl.DecreaseOne("k", time.Minute)
	common.AssertEquals(paraErr, err, t)
	cache := New(segmentCount)
	_, err = cache.DecreaseOne("k", 0)
	common.AssertEquals(paraErr, err, t)
}

func TestValidateAllBranches(t *testing.T) {
	var cl *ConcurrencyLRUCache
	common.AssertEquals(paraErr, validate(cl, time.Minute), t)
	cl2 := &ConcurrencyLRUCache{}
	common.AssertEquals(paraErr, validate(cl2, time.Minute), t)
	cache := New(segmentCount)
	common.AssertEquals(paraErr, validate(cache, 0), t)
	common.AssertEquals(paraErr, validate(cache, time.Duration(neverExpire)-time.Nanosecond), t)
	common.AssertEquals(nil, validate(cache, time.Duration(neverExpire)), t)
	common.AssertEquals(nil, validate(cache, time.Minute), t)
}

func TestSetValueOnNilLRU(t *testing.T) {
	c := &lruCache{}
	if c.setValue("k", "v", time.Minute) == nil {
		t.Errorf("expected error")
	}
}

func TestGetValueOnNilLRU(t *testing.T) {
	c := &lruCache{}
	_, err := c.getValue("k")
	if err == nil {
		t.Errorf("expected error")
	}
}

func TestGetValueKeyNotFound(t *testing.T) {
	c := New(segmentCount).cacheBucket[0]
	_, err := c.getValue("nonexistent")
	if err == nil {
		t.Errorf("expected error")
	}
}

func TestGetValueExpired(t *testing.T) {
	c := New(segmentCount).cacheBucket[0]
	err := c.setValue("k", "v", time.Microsecond)
	common.AssertEquals(nil, err, t)
	time.Sleep(time.Millisecond)
	_, err = c.getValue("k")
	if err == nil {
		t.Errorf("expected error for expired key")
	}
}

func TestSetValueConvertFailed(t *testing.T) {
	c := New(segmentCount).cacheBucket[0]
	c.mu.Lock()
	e := c.PushFront("bad")
	c.elemIndex["bk"] = e
	c.mu.Unlock()
	err := c.setValue("bk", "v", time.Minute)
	if err == nil {
		t.Errorf("expected error")
	}
}

func TestGetValueConvertFailed(t *testing.T) {
	c := New(segmentCount).cacheBucket[0]
	c.mu.Lock()
	e := c.PushFront("bad")
	c.elemIndex["bk"] = e
	c.mu.Unlock()
	_, err := c.getValue("bk")
	if err == nil {
		t.Errorf("expected error")
	}
}

func TestDelValueOnNilLRU(t *testing.T) {
	c := &lruCache{}
	c.delValue("k")
}

func TestDelValueKeyNotFound(t *testing.T) {
	c := New(segmentCount).cacheBucket[0]
	c.delValue("nonexistent")
}

func TestIncrementOnNilLRU(t *testing.T) {
	c := &lruCache{}
	_, err := c.increment("k", time.Minute)
	common.AssertEquals(notInitErr, err, t)
}

func TestIncrementExpiredValue(t *testing.T) {
	c := New(segmentCount).cacheBucket[0]
	err := c.setValue("k", int64(5), time.Microsecond)
	common.AssertEquals(nil, err, t)
	time.Sleep(time.Millisecond)
	v, err := c.increment("k", time.Minute)
	common.AssertEquals(countInit, v, t)
	common.AssertEquals(nil, err, t)
}

func TestIncrementConvertFailed(t *testing.T) {
	c := New(segmentCount).cacheBucket[0]
	c.mu.Lock()
	e := c.PushFront("bad")
	c.elemIndex["bk"] = e
	c.mu.Unlock()
	v, err := c.increment("bk", time.Minute)
	common.AssertEquals(countInit, v, t)
	common.AssertEquals(nil, err, t)
}

func TestIncrementMaxInt64(t *testing.T) {
	cache := New(segmentCount)
	err := cache.Set("maxkey", int64(math.MaxInt64), time.Duration(neverExpire))
	common.AssertEquals(nil, err, t)
	v, err := cache.IncreaseOne("maxkey", time.Minute)
	common.AssertEquals(int64(0), v, t)
	if err == nil {
		t.Errorf("expected error for MaxInt64")
	}
}

func TestIncrementNonInt64Value(t *testing.T) {
	cache := New(segmentCount)
	err := cache.Set("strkey", "strval", time.Duration(neverExpire))
	common.AssertEquals(nil, err, t)
	v, err := cache.IncreaseOne("strkey", time.Minute)
	common.AssertEquals(int64(0), v, t)
	if err == nil {
		t.Errorf("expected error")
	}
}

func TestDecrementOnNilLRU(t *testing.T) {
	c := &lruCache{}
	_, err := c.decrement("k", time.Minute)
	common.AssertEquals(notInitErr, err, t)
}

func TestDecrementExpiredValue(t *testing.T) {
	c := New(segmentCount).cacheBucket[0]
	err := c.setValue("k", int64(5), time.Microsecond)
	common.AssertEquals(nil, err, t)
	time.Sleep(time.Millisecond)
	v, err := c.decrement("k", time.Minute)
	common.AssertEquals(int64(0), v, t)
	common.AssertEquals(nil, err, t)
}

func TestDecrementConvertFailed(t *testing.T) {
	c := New(segmentCount).cacheBucket[0]
	c.mu.Lock()
	e := c.PushFront("bad")
	c.elemIndex["bk"] = e
	c.mu.Unlock()
	v, err := c.decrement("bk", time.Minute)
	common.AssertEquals(int64(0), v, t)
	common.AssertEquals(nil, err, t)
}

func TestDecrementMinInt64(t *testing.T) {
	cache := New(segmentCount)
	err := cache.Set("minkey", int64(math.MinInt64), time.Duration(neverExpire))
	common.AssertEquals(nil, err, t)
	v, err := cache.DecreaseOne("minkey", time.Minute)
	common.AssertEquals(int64(0), v, t)
	if err == nil {
		t.Errorf("expected error")
	}
}

func TestDecrementNonInt64Value(t *testing.T) {
	cache := New(segmentCount)
	err := cache.Set("strkey", "strval", time.Duration(neverExpire))
	common.AssertEquals(nil, err, t)
	v, err := cache.DecreaseOne("strkey", time.Minute)
	common.AssertEquals(int64(0), v, t)
	if err == nil {
		t.Errorf("expected error")
	}
}

func TestSetIfNotExistOnNilLRU(t *testing.T) {
	c := &lruCache{}
	common.AssertEquals(false, c.setIfNotExist("k", "v", time.Minute), t)
}

func TestSetIfNotExistConvertFailed(t *testing.T) {
	c := New(segmentCount).cacheBucket[0]
	c.mu.Lock()
	e := c.PushFront("bad")
	c.elemIndex["bk"] = e
	c.mu.Unlock()
	common.AssertEquals(false, c.setIfNotExist("bk", "v", time.Minute), t)
}

func TestSetIfNotExistExpiredKey(t *testing.T) {
	c := New(segmentCount).cacheBucket[0]
	err := c.setValue("k", "v", time.Microsecond)
	common.AssertEquals(nil, err, t)
	time.Sleep(time.Millisecond)
	common.AssertEquals(true, c.setIfNotExist("k", "nv", time.Minute), t)
}

func TestSetInnerOnInitializedLRU(t *testing.T) {
	c := New(segmentCount).cacheBucket[0]
	c.setInner("k", "v", time.Minute)
	v, err := c.getValue("k")
	if err != nil || v != "v" {
		t.Errorf("expected to get value after setInner")
	}
}

func TestSafeDeleteByKeyOnInitializedLRU(t *testing.T) {
	c := New(segmentCount).cacheBucket[0]
	c.setInner("k", "v", time.Minute)
	e := c.elemIndex["k"]
	c.safeDeleteByKey("k", e)
	if _, ok := c.elemIndex["k"]; ok {
		t.Errorf("expected key to be deleted")
	}
}

func TestSafeRemoveOldestOnInitializedLRU(t *testing.T) {
	c := New(segmentCount).cacheBucket[0]
	c.setInner("k", "v", time.Minute)
	c.safeRemoveOldest()
	if _, ok := c.elemIndex["k"]; ok {
		t.Errorf("expected key to be removed")
	}
}

func TestSafeRemoveOldestEmptyList(t *testing.T) {
	c := New(segmentCount).cacheBucket[0]
	c.safeRemoveOldest()
}

func TestSetValueExistingKey(t *testing.T) {
	c := New(segmentCount).cacheBucket[0]
	err := c.setValue("k", "v1", time.Minute)
	common.AssertEquals(nil, err, t)
	err = c.setValue("k", "v2", time.Minute)
	common.AssertEquals(nil, err, t)
	v, err := c.getValue("k")
	common.AssertEquals(nil, err, t)
	common.AssertEquals("v2", v, t)
}

func TestSetInnerOnNilLRU(t *testing.T) {
	var c *lruCache
	c.setInner("k", "v", time.Minute)
}

func TestSafeDeleteByKeyOnNilLRU(t *testing.T) {
	var c *lruCache
	c.safeDeleteByKey("k", nil)
}

func TestSafeRemoveOldestOnNilLRU(t *testing.T) {
	var c *lruCache
	c.safeRemoveOldest()
}

func TestSafeRemoveOldestConvertFailed(t *testing.T) {
	c := New(segmentCount).cacheBucket[0]
	c.mu.Lock()
	c.PushFront("bad")
	c.mu.Unlock()
	c.safeRemoveOldest()
}

func TestEviction(t *testing.T) {
	c := New(segmentCount).cacheBucket[0]
	c.setInner("k1", "v1", time.Duration(neverExpire))
	c.setInner("k2", "v2", time.Duration(neverExpire))
	if _, ok := c.elemIndex["k1"]; ok {
		t.Errorf("expected k1 to be evicted")
	}
}

func TestPaddingElement(t *testing.T) {
	elem := &cacheElement{}
	paddingElement(elem, "val", time.Minute)
	if elem.data != "val" {
		t.Errorf("expected data=val")
	}
	paddingElement(elem, "val2", time.Duration(neverExpire))
	if elem.expireTime != neverExpire {
		t.Errorf("expected neverExpire")
	}
}

func TestIndexOutOfRange(t *testing.T) {
	cache := New(segmentCount)
	cache.segment = 0
	if cache.Set("k", "v", time.Minute) == nil {
		t.Errorf("expected error")
	}
	if _, err := cache.Get("k"); err == nil {
		t.Errorf("expected error")
	}
	cache.Delete("k")
	common.AssertEquals(false, cache.SetIfNotExist("k", "v", time.Minute), t)
	if _, err := cache.IncreaseOne("k", time.Minute); err == nil {
		t.Errorf("expected error")
	}
	if _, err := cache.DecreaseOne("k", time.Minute); err == nil {
		t.Errorf("expected error")
	}
}

func TestConcurrentAccess(t *testing.T) {
	cache := New(segmentCount * 2)
	done := make(chan bool, segmentCount)
	for i := 0; i < segmentCount; i++ {
		go func(idx int) {
			key := fmt.Sprintf("ckey%d", idx)
			err := cache.Set(key, idx, time.Minute)
			_ = err
			_, _ = cache.Get(key)
			_, _ = cache.IncreaseOne(key, time.Minute)
			_, _ = cache.DecreaseOne(key, time.Minute)
			cache.SetIfNotExist(key, idx, time.Minute)
			cache.Delete(key)
			done <- true
		}(i)
	}
	for i := 0; i < segmentCount; i++ {
		<-done
	}
}
