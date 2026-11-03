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

package sessions

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"

	"openfuyao/oauth-server/pkg/constants"
)

const testMaxAge = 3600

func TestNewSessionStore(t *testing.T) {
	store := NewSessionStore("test", testMaxAge, []byte("secret-key"))
	assert.NotNil(t, store)
	assert.Equal(t, "test", store.name)
	assert.Equal(t, testMaxAge, store.maxAge)
}

func TestCookieStoreGetEmpty(t *testing.T) {
	store := NewSessionStore("test", testMaxAge, []byte("secret-key"))
	req := httptest.NewRequest(http.MethodGet, "/", nil)
	values := store.Get(req)
	assert.Empty(t, values)
}

func TestCookieStoreGetExpired(t *testing.T) {
	store := NewSessionStore("test", testMaxAge, []byte("secret-key"))
	req := httptest.NewRequest(http.MethodGet, "/", nil)

	session, _ := store.store.New(req, "test")
	session.Values[constants.CookieExpiry] = time.Now().Add(-time.Second).Unix()
	session.Values["key"] = "value"

	rec := httptest.NewRecorder()
	_ = store.store.Save(req, rec, session)

	req = httptest.NewRequest(http.MethodGet, "/", nil)
	req.Header.Set("Cookie", rec.Header().Get("Set-Cookie"))

	values := store.Get(req)
	assert.Empty(t, values)
}

func TestCookieStorePutEmpty(t *testing.T) {
	store := NewSessionStore("test", testMaxAge, []byte("secret-key"))
	rec := httptest.NewRecorder()
	err := store.Put(rec, make(Values))
	assert.NoError(t, err)
}

func TestCookieStorePutWithValues(t *testing.T) {
	store := NewSessionStore("test", testMaxAge, []byte("secret-key"))
	rec := httptest.NewRecorder()
	values := Values{
		"user": "alice",
	}
	err := store.Put(rec, values)
	assert.NoError(t, err)
}

func TestValuesGetString(t *testing.T) {
	v := Values{"name": "alice"}
	name, ok := v.GetString("name")
	assert.True(t, ok)
	assert.Equal(t, "alice", name)

	_, ok = v.GetString("missing")
	assert.False(t, ok)
}

func TestValuesGetInt64(t *testing.T) {
	v := Values{"age": int64(30)}
	age, ok := v.GetInt64("age")
	assert.True(t, ok)
	assert.Equal(t, int64(30), age)

	_, ok = v.GetInt64("missing")
	assert.False(t, ok)
}

func TestValuesGetArrayString(t *testing.T) {
	v := Values{"tags": []string{"go", "test"}}
	tags, ok := v.GetArrayString("tags")
	assert.True(t, ok)
	assert.Equal(t, []string{"go", "test"}, tags)

	_, ok = v.GetArrayString("missing")
	assert.False(t, ok)
}

func TestValuesGetExtras(t *testing.T) {
	extras := map[string][]string{
		"role": {"admin"},
	}
	data, err := json.Marshal(extras)
	assert.NoError(t, err)
	v := Values{constants.UserExtra: data}

	result, ok := v.GetExtras(constants.UserExtra)
	assert.True(t, ok)
	assert.Equal(t, extras, result)
}

func TestValuesGetExtraByKey(t *testing.T) {
	extras := map[string][]string{
		"role": {"admin"},
	}
	data, err := json.Marshal(extras)
	assert.NoError(t, err)
	v := Values{constants.UserExtra: data}

	role, ok := v.GetExtraByKey("role")
	assert.True(t, ok)
	assert.Equal(t, []string{"admin"}, role)

	_, ok = v.GetExtraByKey("missing")
	assert.False(t, ok)
}

func TestValuesSetLoggedIn(t *testing.T) {
	extras := map[string][]string{
		constants.UserFirstLogin: {"true"},
	}
	data, err := json.Marshal(extras)
	assert.NoError(t, err)
	v := Values{constants.UserExtra: data}

	ok := v.SetLoggedIn()
	assert.True(t, ok)

	newExtras, _ := v.GetExtras(constants.UserExtra)
	assert.Equal(t, "false", newExtras[constants.UserFirstLogin][0])
}
