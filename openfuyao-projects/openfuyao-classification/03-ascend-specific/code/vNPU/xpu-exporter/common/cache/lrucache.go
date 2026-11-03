/*
Copyright(C)2020-2023. Huawei Technologies Co.,Ltd. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

	http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
package cache

import (
	"container/list"
	"errors"
	"fmt"
	"math"
	"sync"
	"time"
)

const (
	segmentCount        = 16
	tenYears            = 10 * 365 * 24 * time.Hour
	hashInit     uint32 = 2654435761
	prime32      uint32 = 16777619
	neverExpire  int64  = -1
	countInit    int64  = 1
)

var (
	notInitErr = errors.New("lru cache not init")
	paraErr    = errors.New("parameter error")
)

// cacheElement represents a single entry in the LRU cache
type cacheElement struct {
	key        string
	data       interface{}
	expireTime int64
}

// lruCache implements a thread-safe Least Recently Used cache
type lruCache struct {
	maxSize   int
	elemIndex map[string]*list.Element
	*list.List
	mu sync.Mutex
}

// ConcurrencyLRUCache provides concurrent LRU cache with segmented locking
type ConcurrencyLRUCache struct {
	segment     int
	cacheBucket [segmentCount]*lruCache
}

// index calculates segment index for a key using FNV-1a hash algorithm
func (cl *ConcurrencyLRUCache) index(key string) int {
	var hash = hashInit
	for i := 0; i < len(key); i++ {
		hash *= prime32
		hash ^= uint32(key[i])
	}
	return int(hash & (uint32(cl.segment) - 1))
}

// Set adds or updates a key-value pair with expiration time
func (cl *ConcurrencyLRUCache) Set(key string, value interface{}, expireTime time.Duration) error {
	if cl == nil || cl.cacheBucket[0] == nil {
		return notInitErr
	}
	if expireTime < time.Duration(neverExpire) || expireTime > tenYears {
		return paraErr
	}
	cacheIndex := cl.index(key)
	if cacheIndex < 0 || cacheIndex >= segmentCount {
		return errors.New("index out of valid value")
	}
	return cl.cacheBucket[cacheIndex].setValue(key, value, expireTime)
}

// Get retrieves a value by key, returns error if not found or expired
func (cl *ConcurrencyLRUCache) Get(key string) (interface{}, error) {
	if cl == nil || cl.cacheBucket[0] == nil {
		return nil, notInitErr
	}
	cacheIndex := cl.index(key)
	if cacheIndex < 0 || cacheIndex >= segmentCount {
		return nil, errors.New("index out of valid value")
	}
	return cl.cacheBucket[cacheIndex].getValue(key)
}

// Delete removes a key-value pair from the cache
func (cl *ConcurrencyLRUCache) Delete(key string) {
	if cl == nil || cl.cacheBucket[0] == nil {
		return
	}
	cacheIndex := cl.index(key)
	if cacheIndex < 0 || cacheIndex >= segmentCount {
		return
	}
	cl.cacheBucket[cacheIndex].delValue(key)
}

// SetIfNotExist sets a value only if the key does not already exist
func (cl *ConcurrencyLRUCache) SetIfNotExist(key string, value interface{}, expireTime time.Duration) bool {
	if cl == nil || cl.cacheBucket[0] == nil {
		return false
	}
	if expireTime < time.Duration(neverExpire) || expireTime > tenYears {
		return false
	}
	cacheIndex := cl.index(key)
	if cacheIndex < 0 || cacheIndex >= segmentCount {
		return false
	}
	return cl.cacheBucket[cacheIndex].setIfNotExist(key, value, expireTime)
}

// IncreaseOne increments an integer value by one, initializing if not exists
func (cl *ConcurrencyLRUCache) IncreaseOne(key string, expireTime time.Duration) (int64, error) {
	if err := validate(cl, expireTime); err != nil {
		return 0, err
	}
	cacheIndex := cl.index(key)
	if cacheIndex < 0 || cacheIndex >= segmentCount {
		return 0, errors.New("index out of valid value")
	}
	return cl.cacheBucket[cacheIndex].increment(key, expireTime)
}

// DecreaseOne decrements an integer value by one, initializing if not exists
func (cl *ConcurrencyLRUCache) DecreaseOne(key string, expireTime time.Duration) (int64, error) {
	if err := validate(cl, expireTime); err != nil {
		return 0, err
	}
	cacheIndex := cl.index(key)
	if cacheIndex < 0 || cacheIndex >= segmentCount {
		return 0, errors.New("index out of valid value")
	}
	return cl.cacheBucket[cacheIndex].decrement(key, expireTime)
}

// validate checks if cache is properly initialized and parameters are valid
func validate(cl *ConcurrencyLRUCache, expireTime time.Duration) error {
	if cl == nil || cl.cacheBucket[0] == nil {
		return paraErr
	}
	if expireTime <= 0 && expireTime != time.Duration(neverExpire) {
		return paraErr
	}
	return nil
}

// New creates a new concurrent LRU cache with specified maximum entries
func New(maxEntry int) *ConcurrencyLRUCache {
	if maxEntry <= 0 {
		return nil
	}
	size := (maxEntry + segmentCount - 1) / segmentCount
	var cache [segmentCount]*lruCache
	for i := 0; i < segmentCount; i++ {
		cache[i] = &lruCache{
			maxSize:   size,
			elemIndex: make(map[string]*list.Element, segmentCount),
			List:      list.New(),
			mu:        sync.Mutex{},
		}
	}
	return &ConcurrencyLRUCache{
		segment:     segmentCount,
		cacheBucket: cache,
	}
}

// setValue adds or updates a value in the LRU cache segment
func (c *lruCache) setValue(key string, value interface{}, expireTime time.Duration) error {
	if c == nil || c.elemIndex == nil {
		return errors.New("not initializes")
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	v, ok := c.elemIndex[key]
	if !ok {
		c.setInner(key, value, expireTime)
		return nil
	}
	element, ok := v.Value.(*cacheElement)
	if !ok {
		c.safeDeleteByKey(key, v)
		return errors.New("cacheElement convert failed")
	}
	c.MoveToFront(v)
	paddingElement(element, value, expireTime)
	return nil
}

// getValue retrieves a value from the LRU cache segment
func (c *lruCache) getValue(key string) (interface{}, error) {
	if c == nil || c.elemIndex == nil {
		return nil, errors.New("not initializes")
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	v, ok := c.elemIndex[key]
	if !ok {
		return nil, errors.New("no value found")
	}
	c.MoveToFront(v)
	ele, ok := v.Value.(*cacheElement)
	if !ok {
		c.safeDeleteByKey(key, v)
		return nil, errors.New("cacheElement convert failed")
	}
	if ele.expireTime != neverExpire && time.Now().UnixNano() > ele.expireTime {
		c.safeDeleteByKey(key, v)
		return nil, errors.New("the key was expired")
	}
	return ele.data, nil
}

// delValue removes a key-value pair from the LRU cache segment
func (c *lruCache) delValue(key string) {
	if c == nil || c.elemIndex == nil {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if v, ok := c.elemIndex[key]; ok {
		c.safeDeleteByKey(key, v)
	}
}

// increment increases an integer value in the cache by one
func (c *lruCache) increment(key string, expireTime time.Duration) (int64, error) {
	if c == nil || c.elemIndex == nil {
		return 0, notInitErr
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	v, ok := c.elemIndex[key]
	if !ok {
		c.setInner(key, countInit, expireTime)
		return countInit, nil
	}
	element, ok := v.Value.(*cacheElement)
	if !ok {
		c.safeDeleteByKey(key, v)
		c.setInner(key, countInit, expireTime)
		return countInit, nil
	}
	c.MoveToFront(v)
	if element.expireTime == neverExpire || time.Now().UnixNano() < element.expireTime {
		newValue, ok := element.data.(int64)
		if !ok || newValue == math.MaxInt64 {
			return 0, fmt.Errorf("the cache value is not valid, ok:%v", ok)
		}
		newValue++
		paddingElement(element, newValue, expireTime)
		return newValue, nil
	}
	paddingElement(element, countInit, expireTime)
	return countInit, nil
}

// decrement decreases an integer value in the cache by one
func (c *lruCache) decrement(key string, expireTime time.Duration) (int64, error) {
	if c == nil || c.elemIndex == nil {
		return 0, notInitErr
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	v, ok := c.elemIndex[key]
	if !ok {
		c.setInner(key, int64(0), expireTime)
		return 0, nil
	}
	element, ok := v.Value.(*cacheElement)
	if !ok {
		c.safeDeleteByKey(key, v)
		c.setInner(key, int64(0), expireTime)
		return 0, nil
	}
	c.MoveToFront(v)
	if element.expireTime == neverExpire || time.Now().UnixNano() < element.expireTime {
		newValue, ok := element.data.(int64)
		if !ok || newValue == math.MinInt64 {
			return 0, fmt.Errorf("the cache value is not valid, ok: %v", ok)
		}
		newValue--
		paddingElement(element, newValue, expireTime)
		return newValue, nil
	}
	paddingElement(element, int64(0), expireTime)
	return 0, nil
}

// setIfNotExist sets a value only if the key does not exist in the cache segment
func (c *lruCache) setIfNotExist(key string, value interface{}, expireTime time.Duration) bool {
	if c == nil || c.elemIndex == nil {
		return false
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	v, ok := c.elemIndex[key]
	if !ok {
		c.setInner(key, value, expireTime)
		return true
	}
	ele, ok := v.Value.(*cacheElement)
	if !ok {
		c.safeDeleteByKey(key, v)
		return false
	}
	c.MoveToFront(v)
	if ele.expireTime == neverExpire || time.Now().UnixNano() < ele.expireTime {
		return false
	}
	paddingElement(ele, value, expireTime)
	return true
}

// paddingElement updates cache element with new value and expiration time
func paddingElement(element *cacheElement, value interface{}, expireTime time.Duration) {
	element.data = value
	if expireTime == time.Duration(neverExpire) {
		element.expireTime = neverExpire
		return
	}
	element.expireTime = time.Now().UnixNano() + int64(expireTime)
}

// setInner adds a new element to the LRU cache segment, evicting if needed
func (c *lruCache) setInner(key string, value interface{}, expireTime time.Duration) {
	if c == nil {
		return
	}
	if c.Len()+1 > c.maxSize {
		c.safeRemoveOldest()
	}
	newElem := &cacheElement{
		key:        key,
		data:       value,
		expireTime: neverExpire,
	}
	if expireTime != time.Duration(neverExpire) {
		newElem.expireTime = time.Now().UnixNano() + int64(expireTime)
	}
	e := c.PushFront(newElem)
	c.elemIndex[key] = e
}

// safeDeleteByKey removes an element from the cache segment by key
func (c *lruCache) safeDeleteByKey(key string, v *list.Element) {
	if c == nil {
		return
	}
	c.List.Remove(v)
	delete(c.elemIndex, key)
}

// safeRemoveOldest removes the least recently used element from the cache segment
func (c *lruCache) safeRemoveOldest() {
	if c == nil {
		return
	}
	v := c.List.Back()
	if v == nil {
		return
	}
	c.List.Remove(v)
	ele, ok := v.Value.(*cacheElement)
	if !ok {
		return
	}
	delete(c.elemIndex, ele.key)
}
