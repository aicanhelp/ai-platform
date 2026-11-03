/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *           http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package cache

import "sync"

type Cache struct {
	dict sync.Map
}

func NewCache() *Cache {
	var cache Cache
	return &cache
}

func (c *Cache) Set(key string, value interface{}) {
	c.dict.Store(key, value)
}

func (c *Cache) GetOrSet(key string, value interface{}) (interface{}, bool) {
	return c.dict.LoadOrStore(key, value)
}

func (c *Cache) Get(key string) (interface{}, bool) {
	return c.dict.Load(key)
}

func (c *Cache) Range(f func(key, value interface{}) bool) {
	c.dict.Range(f)
}

func (c *Cache) Delete(key string) {
	c.dict.Delete(key)
}

func (c *Cache) Clear() {
	c.dict.Range(func(key, value interface{}) bool {
		c.dict.Delete(key)
		return true
	})
}

func (c *Cache) GetMustInt(key string) (int, bool) {
	value, ok := c.Get(key)
	res, assert := value.(int)
	if !assert {
		return res, false
	}
	return res, ok
}

func (c *Cache) GetMustString(key string) (string, bool) {
	value, ok := c.Get(key)
	res, assert := value.(string)
	if !assert {
		return res, false
	}
	return res, ok
}

func (c *Cache) GetMustBool(key string) (bool, bool) {
	value, ok := c.Get(key)
	res, assert := value.(bool)
	if !assert {
		return res, false
	}
	return res, ok
}
