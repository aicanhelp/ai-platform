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

package clustermanage

import (
	"strings"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/emicklei/go-restful/v3"
	"github.com/stretchr/testify/assert"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/rest"

	"installer-service/pkg/installer"
)

func TestNewWebService(t *testing.T) {
	t.Run("normal case", func(t *testing.T) {
		gv := schema.GroupVersion{Group: "group", Version: "v1"}
		ws := NewWebService(gv)

		assert.NotNil(t, ws)
		assert.Equal(t, "group/v1", ws.RootPath())
	})

	t.Run("empty group version", func(t *testing.T) {
		ws := NewWebService(schema.GroupVersion{})
		assert.NotNil(t, ws)
		assert.Equal(t, "/", ws.RootPath())
	})
}

func TestConfigInstallerWs_AddsWebService(t *testing.T) {
	cases := []struct {
		name  string
		patch func() *gomonkey.Patches
	}{
		{name: "operation-ok", patch: func() *gomonkey.Patches {
			p := gomonkey.NewPatches()
			p.ApplyFunc(installer.NewInstallerOperation, func(_ *rest.Config) (installer.Operation, error) {
				// return a nil operation but no error (we don't invoke it in register)
				return nil, nil
			})
			return p
		}},
		{name: "operation-error", patch: func() *gomonkey.Patches {
			p := gomonkey.NewPatches()
			p.ApplyFunc(installer.NewInstallerOperation, func(_ *rest.Config) (installer.Operation, error) {
				return nil, assert.AnError
			})
			return p
		}},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			patches := tc.patch()
			if patches != nil {
				defer patches.Reset()
			}

			c := restful.NewContainer()
			err := ConfigInstallerWs(c, &rest.Config{})
			assert.NoError(t, err)

			// expect a web service was added for ws groupVersion
			expected := strings.TrimRight(groupVersionWs.String(), "/")
			found := false
			for _, ws := range c.RegisteredWebServices() {
				if ws.RootPath() == expected {
					found = true
					break
				}
			}
			assert.True(t, found, "expected webservice %s to be registered", expected)
		})
	}
}

func TestConfigInstaller(t *testing.T) {
	cases := []struct {
		name    string
		patch   func() *gomonkey.Patches
		wantErr bool
	}{
		{
			name: "operation-ok",
			patch: func() *gomonkey.Patches {
				p := gomonkey.NewPatches()
				p.ApplyFunc(installer.NewInstallerOperation, func(_ *rest.Config) (installer.Operation, error) {
					return nil, nil
				})
				return p
			},
		},
		{
			name: "operation-error",
			patch: func() *gomonkey.Patches {
				p := gomonkey.NewPatches()
				p.ApplyFunc(installer.NewInstallerOperation, func(_ *rest.Config) (installer.Operation, error) {
					return nil, assert.AnError
				})
				return p
			},
			wantErr: true,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			patches := tc.patch()
			defer patches.Reset()

			c := restful.NewContainer()
			err := ConfigInstaller(c, &rest.Config{})
			if tc.wantErr {
				assert.Error(t, err)
				return
			}
			assert.NoError(t, err)

			expected := strings.TrimRight(groupVersion.String(), "/")
			found := false
			for _, ws := range c.RegisteredWebServices() {
				if ws.RootPath() == expected {
					found = true
					assert.NotEmpty(t, ws.Routes())
				}
			}
			assert.True(t, found, "expected webservice %s to be registered", expected)
		})
	}
}
