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

package runtime

import (
	"os"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"
)

const port = 9210

func TestNewServerConfig(t *testing.T) {
	// 执行函数
	cfg := NewServerConfig()

	// 验证默认值
	assert.Equal(t, "0.0.0.0", cfg.BindAddress, "BindAddress 默认值不匹配")
	assert.Equal(t, port, cfg.InsecurePort, "InsecurePort 默认值不匹配")
	assert.Equal(t, 0, cfg.SecurePort, "SecurePort 默认值不匹配")
	assert.Equal(t, "", cfg.CertFile, "CertFile 默认值不匹配")
	assert.Equal(t, "", cfg.PrivateKey, "PrivateKey 默认值不匹配")

	// 验证指针非空
	assert.NotNil(t, cfg, "返回的配置对象不应为 nil")
}

func TestServerConfigValidate(t *testing.T) {
	tests := []struct {
		name    string
		cfg     ServerConfig
		mock    func() *gomonkey.Patches
		wantErr int
	}{
		{
			name: "secure port valid with files",
			cfg:  ServerConfig{SecurePort: 443, CertFile: "cert", PrivateKey: "key"},
			mock: func() *gomonkey.Patches {
				return gomonkey.ApplyFunc(os.Stat, func(string) (os.FileInfo, error) {
					return nil, nil
				})
			},
			wantErr: 0,
		},
		{
			name:    "both ports disabled",
			cfg:     ServerConfig{SecurePort: 0, InsecurePort: 0},
			mock:    nil,
			wantErr: 1,
		},
		{
			name: "cert missing key present",
			cfg:  ServerConfig{SecurePort: 443, CertFile: "cert", PrivateKey: "key"},
			mock: func() *gomonkey.Patches {
				return gomonkey.ApplyFunc(os.Stat, func(p string) (os.FileInfo, error) {
					if p == "cert" {
						return nil, os.ErrNotExist
					}
					return nil, nil
				})
			},
			wantErr: 1,
		},
		{
			name:    "secure file and key missing",
			cfg:     ServerConfig{SecurePort: 443, CertFile: "", PrivateKey: ""},
			mock:    nil,
			wantErr: 2,
		},
		{
			name: "secure port cert not exist",
			cfg:  ServerConfig{SecurePort: 443, CertFile: "cert", PrivateKey: "key"},
			mock: func() *gomonkey.Patches {
				return gomonkey.ApplyFunc(os.Stat, func(string) (os.FileInfo, error) {
					return nil, os.ErrNotExist
				})
			},
			wantErr: 2,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.mock != nil {
				patches := tt.mock()
				defer patches.Reset()
			}
			errs := tt.cfg.Validate()
			assert.Len(t, errs, tt.wantErr)
		})
	}
}
