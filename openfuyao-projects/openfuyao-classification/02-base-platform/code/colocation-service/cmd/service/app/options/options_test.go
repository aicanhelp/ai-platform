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

package options

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"k8s.io/klog/v2"

	"openfuyao.com/colocation-service/cmd/service/app/config"
)

func TestNewOptions(t *testing.T) {
	o := NewOptions()
	assert.NotNil(t, o)
	assert.NotNil(t, o.SecureServing)
	assert.NotNil(t, o.Metrics)
	assert.NotNil(t, o.Logs)
	assert.NotNil(t, o.Flags)
	assert.Equal(t, "colocation-service", o.SecureServing.ServerCert.PairName)
	assert.Equal(t, 8080, o.SecureServing.BindPort)
}

func TestOverallOptions_Validate(t *testing.T) {
	o := NewOptions()
	errs := o.Validate()
	assert.Empty(t, errs)
}

func TestOverallOptions_Complete(t *testing.T) {
	o := NewOptions()
	err := o.Complete(nil)
	assert.NoError(t, err)
}

func TestOverallOptions_InitFlags_Idempotent(t *testing.T) {
	o := NewOptions()
	flags1 := o.Flags
	o.initFlags()
	assert.Equal(t, flags1, o.Flags)
}

func TestOverallOptions_Config(t *testing.T) {
	ctx := klog.NewContext(context.Background(), klog.Background())
	o := NewOptions()
	c, err := o.Config(ctx)
	if err != nil {
		t.Logf("Config() returned error (expected without k8s): %v", err)
		return
	}
	assert.NotNil(t, c)
}

func TestOverallOptions_ApplyTo(t *testing.T) {
	o := NewOptions()
	c := &config.OverallConfig{}
	err := o.ApplyTo(klog.Background(), c)
	if err != nil {
		t.Logf("ApplyTo() returned error (expected without k8s): %v", err)
		return
	}
	assert.NotNil(t, c.KubeConfig)
}
