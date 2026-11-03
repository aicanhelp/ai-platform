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

package runtime

import (
	"os"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"

	"openfuyao.com/monitoring-service/pkg/zlog"
)

func TestNewServerDefaultPort(t *testing.T) {
	// When SERVICE_PORT is not set or invalid, use default port 9083
	patches := gomonkey.ApplyFunc(os.Getenv, func(key string) string {
		if key == "SERVICE_PORT" {
			return ""
		}
		return ""
	})
	defer patches.Reset()

	statPatches := gomonkey.ApplyFunc(os.Stat, func(name string) (os.FileInfo, error) {
		// TLS cert does not exist
		return nil, os.ErrNotExist
	})
	defer statPatches.Reset()

	zlogPatches := gomonkey.ApplyFunc(zlog.Warn, func(_ ...interface{}) {})
	defer zlogPatches.Reset()

	s := NewServer()
	assert.NotNil(t, s)
	assert.Equal(t, "0.0.0.0", s.BindAddress)
	assert.Equal(t, defaultServicePort, s.InsecurePort)
	assert.Equal(t, 0, s.SecurePort)
	assert.Empty(t, s.CertFile)
	assert.Empty(t, s.PrivateKeyFile)
	assert.Empty(t, s.RootCAFile)
}

func TestNewServerInvalidPortUsesDefault(t *testing.T) {
	patches := gomonkey.ApplyFunc(os.Getenv, func(key string) string {
		if key == "SERVICE_PORT" {
			return "invalid"
		}
		return ""
	})
	defer patches.Reset()

	statPatches := gomonkey.ApplyFunc(os.Stat, func(name string) (os.FileInfo, error) {
		return nil, os.ErrNotExist
	})
	defer statPatches.Reset()

	zlogPatches := gomonkey.ApplyFunc(zlog.Warn, func(_ ...interface{}) {})
	defer zlogPatches.Reset()

	s := NewServer()
	assert.NotNil(t, s)
	assert.Equal(t, defaultServicePort, s.InsecurePort)
}

func TestNewServerCustomPort(t *testing.T) {
	patches := gomonkey.ApplyFunc(os.Getenv, func(key string) string {
		if key == "SERVICE_PORT" {
			return "9090"
		}
		return ""
	})
	defer patches.Reset()

	statPatches := gomonkey.ApplyFunc(os.Stat, func(name string) (os.FileInfo, error) {
		return nil, os.ErrNotExist
	})
	defer statPatches.Reset()

	s := NewServer()
	assert.NotNil(t, s)
	assert.Equal(t, 9090, s.InsecurePort)
	assert.Equal(t, 0, s.SecurePort)
}

func TestNewServerStatErrorReturnsNil(t *testing.T) {
	patches := gomonkey.ApplyFunc(os.Getenv, func(key string) string {
		if key == "SERVICE_PORT" {
			return "9443"
		}
		return ""
	})
	defer patches.Reset()

	// os.Stat returns an error other than IsNotExist (e.g. permission denied)
	statPatches := gomonkey.ApplyFunc(os.Stat, func(name string) (os.FileInfo, error) {
		return nil, os.ErrPermission
	})
	defer statPatches.Reset()

	// NewServer calls zlog.FormatError when os.Stat fails; mock to avoid nil logger panic
	zlogPatches := gomonkey.ApplyFunc(zlog.FormatError, func(_ string, _ ...interface{}) {})
	defer zlogPatches.Reset()

	s := NewServer()
	assert.Nil(t, s)
}

func TestServerDefaults(t *testing.T) {
	s := &Server{}
	assert.Empty(t, s.BindAddress)
	assert.Equal(t, 0, s.InsecurePort)
	assert.Equal(t, 0, s.SecurePort)
	assert.Empty(t, s.CertFile)
	assert.Empty(t, s.PrivateKeyFile)
	assert.Empty(t, s.RootCAFile)
}
