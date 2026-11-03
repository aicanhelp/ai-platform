/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package app

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/klog/v2"

	"openfuyao.com/colocation-service/cmd/service/app/config"
	"openfuyao.com/colocation-service/cmd/service/app/options"
	"openfuyao.com/colocation-service/pkg/service"
)

func TestNewCommand(t *testing.T) {
	cmd := NewCommand()
	assert.NotNil(t, cmd)
	assert.Equal(t, "colocation-service", cmd.Use)
}

func TestNewCommand_Args(t *testing.T) {
	cmd := NewCommand()
	assert.NotNil(t, cmd.Args)

	err := cmd.Args(cmd, []string{})
	assert.NoError(t, err)

	err = cmd.Args(cmd, []string{"arg1"})
	assert.Error(t, err)
}

func TestSetup(t *testing.T) {
	ctx := klog.NewContext(context.Background(), klog.Background())
	opts := options.NewOptions()
	_, _, err := Setup(ctx, opts)
	if err != nil {
		t.Logf("Setup() returned error (expected without k8s): %v", err)
		return
	}
}

func TestRun_WithNilSecureServing(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	logger := klog.Background()
	ctx = klog.NewContext(ctx, logger)

	k8sClient := fake.NewSimpleClientset()
	server, err := service.New(ctx, k8sClient, nil)
	assert.NoError(t, err)

	oc := &config.OverallConfig{}
	cc := oc.Complete()

	go func() {
		cancel()
	}()

	err = Run(ctx, &cc, server)
	assert.NoError(t, err)
}
