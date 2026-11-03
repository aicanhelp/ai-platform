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

// Package sessions define the store backend for sessions
package sessions

import (
	"encoding/json"
	"net/http"
	"time"

	"github.com/gorilla/sessions"

	"openfuyao/oauth-server/pkg/constants"
	"openfuyao/oauth-server/pkg/zlog"
)

// CookieStore defines the session store structure
type CookieStore struct {
	// name of the cookie used for session data
	name string
	// store of the actual cookie
	store  sessions.Store
	maxAge int
}

// NewSessionStore inits a session store for idp login state
func NewSessionStore(name string, maxAge int, secrets ...[]byte) *CookieStore {
	cookie := sessions.NewCookieStore(secrets...)
	cookie.Options.MaxAge = maxAge
	cookie.Options.HttpOnly = true
	cookie.Options.Secure = true
	cookie.Options.SameSite = http.SameSiteStrictMode
	return &CookieStore{name: name, store: cookie, maxAge: maxAge}
}

// Get fetches the session by request and name
func (s *CookieStore) Get(r *http.Request) Values {
	// always use New to avoid global state
	session, err := s.store.New(r, s.name)

	if err != nil {
		// just log the error so that we can know what is going on
		zlog.LogErrorf("failed to decode secure cookie session %s: %v", s.name, err)

		return make(Values)
	}

	// check expiration
	expiry, ok := Values.GetInt64(session.Values, constants.CookieExpiry)
	if !ok || time.Now().Unix() >= expiry {
		return make(Values)
	}

	return session.Values
}

// Put stores the new cookie value to response writer
func (s *CookieStore) Put(w http.ResponseWriter, v Values) error {
	r := &http.Request{}
	session, err := s.store.New(r, s.name)
	if err != nil {
		return err
	}

	// if v is empty, flush the cookie
	if len(v) == 0 {
		return s.store.Save(r, w, session)
	}

	// store cookie expiration time
	v[constants.CookieExpiry] = time.Now().Add(time.Second * time.Duration(s.maxAge)).Unix()

	// override the values for the session
	session.Values = v

	// write the encoded cookie, the request parameter is ignored
	return s.store.Save(r, w, session)
}

// Values provide interfaces to read string/int data
type Values map[interface{}]interface{}

// GetString fetches string value from the session store
func (v Values) GetString(key string) (string, bool) {
	str, ok := v[key].(string)
	if !ok {
		return "", false
	}
	return str, ok && len(str) != 0
}

// GetInt64 fetches int from the session store
func (v Values) GetInt64(key string) (int64, bool) {
	i, ok := v[key].(int64)
	if !ok {
		return 0, false
	}
	return i, ok && i != 0
}

// GetArrayString fetches array string from the session store
func (v Values) GetArrayString(key string) ([]string, bool) {
	var arrayStr []string
	arrayStr, ok := v[key].([]string)
	if !ok {
		return arrayStr, false
	}
	return arrayStr, ok && arrayStr != nil
}

// GetExtras fetches the extra info from the session store
func (v Values) GetExtras(key string) (map[string][]string, bool) {
	byteExtras, ok := v[key].([]byte)
	if !ok {
		return nil, false
	}
	extras := make(map[string][]string)
	err := json.Unmarshal(byteExtras, &extras)
	return extras, err == nil
}

// GetExtraByKey fetches the specific value in the Extra field
func (v Values) GetExtraByKey(key string) ([]string, bool) {
	extras, ok := v.GetExtras(constants.UserExtra)
	if !ok {
		return nil, ok
	}

	ret, ok := extras[key]
	return ret, ok
}

// SetLoggedIn set the first-login to false in idpLoginState
func (v Values) SetLoggedIn() bool {
	extras, ok := v.GetExtras(constants.UserExtra)
	if !ok {
		return false
	}

	extras[constants.UserFirstLogin][0] = "false"

	// serialize extra (map[string][]string)
	jsonExtra, err := json.Marshal(extras)
	if err != nil {
		zlog.LogErrorf("cannot marshal data, err: %v", err)
		return false
	}
	v[constants.UserExtra] = jsonExtra

	return true
}
