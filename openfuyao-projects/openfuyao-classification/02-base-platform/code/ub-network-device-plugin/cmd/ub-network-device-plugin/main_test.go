/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of the License at
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package main

import (
	"context"
	"errors"
	"testing"

	ubse "atomgit.com/openeuler/ubs-engine.git/src/sdk/go/engine"
	"github.com/agiledragon/gomonkey/v2"
	"github.com/fsnotify/fsnotify"
	"github.com/stretchr/testify/assert"
	"github.com/urfave/cli/v2"

	"gitcode.com/openFuyao/ub-network-device-plugin/plugin"
	"gitcode.com/openFuyao/ub-network-device-plugin/resource"
	"gitcode.com/openFuyao/ub-network-device-plugin/utils"
	"gitcode.com/openFuyao/ub-network-device-plugin/watch"
)

type MockPlugin struct {
	devices  resource.Devices
	startErr error
	stopErr  error
}

func (m *MockPlugin) Devices() resource.Devices {
	return m.devices
}

func (m *MockPlugin) Start(_ *plugin.StartOpts) error {
	return m.startErr
}

func (m *MockPlugin) Stop() error {
	return m.stopErr
}

func TestStopPlugins(t *testing.T) {
	tests := []struct {
		name    string
		plugins []plugin.Interface
		wantErr bool
	}{
		{
			name:    "empty plugins",
			plugins: []plugin.Interface{},
			wantErr: false,
		},
		{
			name: "single plugin success",
			plugins: []plugin.Interface{
				&MockPlugin{stopErr: nil},
			},
			wantErr: false,
		},
		{
			name: "single plugin error",
			plugins: []plugin.Interface{
				&MockPlugin{stopErr: errors.New("stop error")},
			},
			wantErr: true,
		},
		{
			name: "multiple plugins with errors",
			plugins: []plugin.Interface{
				&MockPlugin{stopErr: errors.New("stop error 1")},
				&MockPlugin{stopErr: errors.New("stop error 2")},
			},
			wantErr: true,
		},
		{
			name: "multiple plugins success",
			plugins: []plugin.Interface{
				&MockPlugin{stopErr: nil},
				&MockPlugin{stopErr: nil},
			},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := stopPlugins(tt.plugins)
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestGetPlugins(t *testing.T) {
	tests := []struct {
		name     string
		setup    func() *gomonkey.Patches
		wantErr  bool
		tearDown func(patches *gomonkey.Patches)
	}{
		{
			name: "plugin.New returns error",
			setup: func() *gomonkey.Patches {
				return gomonkey.ApplyFunc(plugin.New, func(_ context.Context, _ ...plugin.Option) ([]plugin.Interface, error) {
					return nil, errors.New("plugin.New error")
				})
			},
			wantErr: true,
			tearDown: func(patches *gomonkey.Patches) {
				patches.Reset()
			},
		},
		{
			name: "plugin.New returns empty plugins",
			setup: func() *gomonkey.Patches {
				return gomonkey.ApplyFunc(plugin.New, func(_ context.Context, _ ...plugin.Option) ([]plugin.Interface, error) {
					return []plugin.Interface{}, nil
				})
			},
			wantErr: false,
			tearDown: func(patches *gomonkey.Patches) {
				patches.Reset()
			},
		},
		{
			name: "plugin.New returns plugins",
			setup: func() *gomonkey.Patches {
				return gomonkey.ApplyFunc(plugin.New, func(_ context.Context, _ ...plugin.Option) ([]plugin.Interface, error) {
					return []plugin.Interface{&MockPlugin{}}, nil
				})
			},
			wantErr: false,
			tearDown: func(patches *gomonkey.Patches) {
				patches.Reset()
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patches := tt.setup()
			defer tt.tearDown(patches)

			res := &resource.Resources{}
			plugins, err := getPlugins(context.Background(), res)

			if tt.wantErr {
				assert.Error(t, err)
				assert.Nil(t, plugins)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestStartPlugins(t *testing.T) {
	tests := []struct {
		name        string
		setup       func() []*gomonkey.Patches
		wantErr     bool
		wantRestart bool
		tearDown    func(patches []*gomonkey.Patches)
	}{
		{
			name: "InitDefaultResources returns error",
			setup: func() []*gomonkey.Patches {
				p1 := gomonkey.ApplyFunc(resource.InitDefaultResources, func() (*resource.Resources, error) {
					return nil, errors.New("init error")
				})
				return []*gomonkey.Patches{p1}
			},
			wantErr:     true,
			wantRestart: false,
			tearDown: func(patches []*gomonkey.Patches) {
				for _, p := range patches {
					p.Reset()
				}
			},
		},
		{
			name: "getPlugins returns error",
			setup: func() []*gomonkey.Patches {
				p1 := gomonkey.ApplyFunc(resource.InitDefaultResources, func() (*resource.Resources, error) {
					return &resource.Resources{}, nil
				})
				p2 := gomonkey.ApplyFunc(plugin.New, func(_ context.Context, _ ...plugin.Option) ([]plugin.Interface, error) {
					return nil, errors.New("plugin.New error")
				})
				return []*gomonkey.Patches{p1, p2}
			},
			wantErr:     true,
			wantRestart: false,
			tearDown: func(patches []*gomonkey.Patches) {
				for _, p := range patches {
					p.Reset()
				}
			},
		},
		{
			name: "getPlugins returns empty plugins",
			setup: func() []*gomonkey.Patches {
				p1 := gomonkey.ApplyFunc(resource.InitDefaultResources, func() (*resource.Resources, error) {
					return &resource.Resources{}, nil
				})
				p2 := gomonkey.ApplyFunc(plugin.New, func(_ context.Context, _ ...plugin.Option) ([]plugin.Interface, error) {
					return []plugin.Interface{}, nil
				})
				return []*gomonkey.Patches{p1, p2}
			},
			wantErr:     true,
			wantRestart: false,
			tearDown: func(patches []*gomonkey.Patches) {
				for _, p := range patches {
					p.Reset()
				}
			},
		},
		{
			name: "plugin has no devices",
			setup: func() []*gomonkey.Patches {
				p1 := gomonkey.ApplyFunc(resource.InitDefaultResources, func() (*resource.Resources, error) {
					return &resource.Resources{}, nil
				})
				p2 := gomonkey.ApplyFunc(plugin.New, func(_ context.Context, _ ...plugin.Option) ([]plugin.Interface, error) {
					return []plugin.Interface{&MockPlugin{devices: resource.Devices{}}}, nil
				})
				return []*gomonkey.Patches{p1, p2}
			},
			wantErr:     true,
			wantRestart: false,
			tearDown: func(patches []*gomonkey.Patches) {
				for _, p := range patches {
					p.Reset()
				}
			},
		},
		{
			name: "plugin start returns error",
			setup: func() []*gomonkey.Patches {
				p1 := gomonkey.ApplyFunc(resource.InitDefaultResources, func() (*resource.Resources, error) {
					return &resource.Resources{}, nil
				})
				p2 := gomonkey.ApplyFunc(plugin.New, func(_ context.Context, _ ...plugin.Option) ([]plugin.Interface, error) {
					return []plugin.Interface{&MockPlugin{
						devices:  resource.Devices{"dev1": nil},
						startErr: errors.New("start error"),
					}}, nil
				})
				return []*gomonkey.Patches{p1, p2}
			},
			wantErr:     false,
			wantRestart: true,
			tearDown: func(patches []*gomonkey.Patches) {
				for _, p := range patches {
					p.Reset()
				}
			},
		},
		{
			name: "plugin start success",
			setup: func() []*gomonkey.Patches {
				p1 := gomonkey.ApplyFunc(resource.InitDefaultResources, func() (*resource.Resources, error) {
					return &resource.Resources{}, nil
				})
				p2 := gomonkey.ApplyFunc(plugin.New, func(_ context.Context, _ ...plugin.Option) ([]plugin.Interface, error) {
					return []plugin.Interface{&MockPlugin{
						devices:  resource.Devices{"dev1": nil},
						startErr: nil,
					}}, nil
				})
				return []*gomonkey.Patches{p1, p2}
			},
			wantErr:     false,
			wantRestart: false,
			tearDown: func(patches []*gomonkey.Patches) {
				for _, p := range patches {
					p.Reset()
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patches := tt.setup()
			defer tt.tearDown(patches)

			app := cli.NewApp()
			ctx := cli.NewContext(app, nil, nil)
			opts := &plugin.StartOpts{}

			plugins, restart, err := startPlugins(ctx, opts)

			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
			assert.Equal(t, tt.wantRestart, restart)
			_ = plugins
		})
	}
}

func TestStart(t *testing.T) {
	tests := []struct {
		name     string
		setup    func() []*gomonkey.Patches
		wantErr  bool
		tearDown func(patches []*gomonkey.Patches)
	}{
		{
			name: "watch.Files returns error",
			setup: func() []*gomonkey.Patches {
				p1 := gomonkey.ApplyFunc(plugin.StartHealthServer, func(_ string) {})
				p2 := gomonkey.ApplyFunc(watch.Files, func(_ ...string) (*fsnotify.Watcher, error) {
					return nil, errors.New("watch error")
				})
				return []*gomonkey.Patches{p1, p2}
			},
			wantErr: true,
			tearDown: func(patches []*gomonkey.Patches) {
				for _, p := range patches {
					p.Reset()
				}
			},
		},
		{
			name: "EnsureDevicePermission returns error",
			setup: func() []*gomonkey.Patches {
				p1 := gomonkey.ApplyFunc(plugin.StartHealthServer, func(_ string) {})
				p2 := gomonkey.ApplyFunc(watch.Files, func(_ ...string) (*fsnotify.Watcher, error) {
					w, _ := fsnotify.NewWatcher()
					return w, nil
				})
				p3 := gomonkey.ApplyFunc(utils.EnsureDevicePermission, func(_ string, _ int) error {
					return errors.New("permission error")
				})
				return []*gomonkey.Patches{p1, p2, p3}
			},
			wantErr: true,
			tearDown: func(patches []*gomonkey.Patches) {
				for _, p := range patches {
					p.Reset()
				}
			},
		},
		{
			name: "UbsEngineClientInitialize returns error",
			setup: func() []*gomonkey.Patches {
				p1 := gomonkey.ApplyFunc(plugin.StartHealthServer, func(_ string) {})
				p2 := gomonkey.ApplyFunc(watch.Files, func(_ ...string) (*fsnotify.Watcher, error) {
					w, _ := fsnotify.NewWatcher()
					return w, nil
				})
				p3 := gomonkey.ApplyFunc(utils.EnsureDevicePermission, func(_ string, _ int) error {
					return nil
				})
				p4 := gomonkey.ApplyFunc(ubse.UbsEngineClientInitialize, func(_ string) (interface{}, error) {
					return nil, errors.New("engine init error")
				})
				return []*gomonkey.Patches{p1, p2, p3, p4}
			},
			wantErr: true,
			tearDown: func(patches []*gomonkey.Patches) {
				for _, p := range patches {
					p.Reset()
				}
			},
		},
		{
			name: "startPlugins returns error",
			setup: func() []*gomonkey.Patches {
				p1 := gomonkey.ApplyFunc(plugin.StartHealthServer, func(_ string) {})
				p2 := gomonkey.ApplyFunc(watch.Files, func(_ ...string) (*fsnotify.Watcher, error) {
					w, _ := fsnotify.NewWatcher()
					return w, nil
				})
				p3 := gomonkey.ApplyFunc(utils.EnsureDevicePermission, func(_ string, _ int) error {
					return nil
				})
				p4 := gomonkey.ApplyFunc(ubse.UbsEngineClientInitialize, func(_ string) (interface{}, error) {
					return nil, nil
				})
				p5 := gomonkey.ApplyFunc(resource.InitDefaultResources, func() (*resource.Resources, error) {
					return nil, errors.New("init error")
				})
				return []*gomonkey.Patches{p1, p2, p3, p4, p5}
			},
			wantErr: true,
			tearDown: func(patches []*gomonkey.Patches) {
				for _, p := range patches {
					p.Reset()
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patches := tt.setup()
			defer tt.tearDown(patches)

			app := cli.NewApp()
			ctx := cli.NewContext(app, nil, nil)
			opts := &plugin.StartOpts{}

			err := start(ctx, opts)

			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}
