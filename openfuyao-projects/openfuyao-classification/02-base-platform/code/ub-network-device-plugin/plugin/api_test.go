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

package plugin

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/urfave/cli/v2"
)

func TestStartOpts(t *testing.T) {
	opts := &StartOpts{
		Flags:         []cli.Flag{},
		KubeletSocket: "/var/run/kubelet/kubelet.sock",
		SyncInterval:  30 * time.Second,
		HealthPort:    "8088",
	}

	assert.NotNil(t, opts)
	assert.Equal(t, "/var/run/kubelet/kubelet.sock", opts.KubeletSocket)
	assert.Equal(t, 30*time.Second, opts.SyncInterval)
	assert.Equal(t, "8088", opts.HealthPort)
}

func TestInterface(t *testing.T) {
	var _ Interface = (*urmaDevicePlugin)(nil)
}

func TestStartOptsDefaults(t *testing.T) {
	opts := &StartOpts{}
	assert.Nil(t, opts.Flags)
	assert.Empty(t, opts.KubeletSocket)
	assert.Equal(t, time.Duration(0), opts.SyncInterval)
	assert.Empty(t, opts.HealthPort)
}

func TestStartOptsWithFlags(t *testing.T) {
	flags := []cli.Flag{
		&cli.StringFlag{
			Name:  "test",
			Value: "default",
		},
	}
	opts := &StartOpts{
		Flags:         flags,
		KubeletSocket: "/var/run/test.sock",
		SyncInterval:  60 * time.Second,
		HealthPort:    "9090",
	}

	assert.Len(t, opts.Flags, 1)
	assert.Equal(t, "test", opts.Flags[0].Names()[0])
	assert.Equal(t, "/var/run/test.sock", opts.KubeletSocket)
	assert.Equal(t, 60*time.Second, opts.SyncInterval)
	assert.Equal(t, "9090", opts.HealthPort)
}

func TestStartOptsFieldMutation(t *testing.T) {
	opts := &StartOpts{}
	opts.Flags = []cli.Flag{&cli.BoolFlag{Name: "flag1"}}
	opts.KubeletSocket = "/new/socket"
	opts.SyncInterval = 45 * time.Second
	opts.HealthPort = "7070"

	assert.Len(t, opts.Flags, 1)
	assert.Equal(t, "/new/socket", opts.KubeletSocket)
	assert.Equal(t, 45*time.Second, opts.SyncInterval)
	assert.Equal(t, "7070", opts.HealthPort)
}

func TestInterfaceMethods(t *testing.T) {
	mockPlugin := &urmaDevicePlugin{}
	assert.NotNil(t, mockPlugin)

	_ = Interface(mockPlugin)
}

func TestStartOptsMultipleFlags(t *testing.T) {
	flags := []cli.Flag{
		&cli.StringFlag{Name: "socket"},
		&cli.IntFlag{Name: "port"},
		&cli.BoolFlag{Name: "debug"},
	}
	opts := &StartOpts{
		Flags:         flags,
		KubeletSocket: "/var/run/kubelet.sock",
		SyncInterval:  10 * time.Second,
		HealthPort:    "8080",
	}

	assert.Len(t, opts.Flags, 3)
	assert.Equal(t, []string{"socket", "port", "debug"}, []string{
		opts.Flags[0].Names()[0],
		opts.Flags[1].Names()[0],
		opts.Flags[2].Names()[0],
	})
}

func TestStartOptsZeroValues(t *testing.T) {
	opts := StartOpts{}
	assert.Nil(t, opts.Flags)
	assert.Equal(t, "", opts.KubeletSocket)
	assert.Equal(t, time.Duration(0), opts.SyncInterval)
	assert.Equal(t, "", opts.HealthPort)
}
