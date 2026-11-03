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

package installer

import (
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/kubernetes"
	k8sfake "k8s.io/client-go/kubernetes/fake"
	"k8s.io/client-go/rest"

	"installer-service/pkg/zlog"
)

func TestNewInstallerOperation(t *testing.T) {
	cfg, dyn, cli := &rest.Config{}, &dynamic.DynamicClient{}, &kubernetes.Clientset{}
	patches := gomonkey.ApplyFunc(zlog.Error, func(args ...interface{}) { //
		// 这是有意为之
	})
	defer patches.Reset()

	tests := []struct {
		name    string
		newDyn  func(*rest.Config) (dynamic.Interface, error)
		newCli  func(*rest.Config) (*kubernetes.Clientset, error)
		wantErr bool
	}{
		{"cli failed", func(_ *rest.Config) (dynamic.Interface, error) { return dyn, nil },
			func(_ *rest.Config) (*kubernetes.Clientset, error) { return nil, assert.AnError }, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := gomonkey.NewPatches()
			if tt.newDyn != nil {
				p.ApplyFunc(dynamic.NewForConfig, tt.newDyn)
			}
			if tt.newCli != nil {
				p.ApplyFunc(kubernetes.NewForConfig, tt.newCli)
			}
			defer p.Reset()

			op, err := NewInstallerOperation(cfg)
			assert.Equal(t, tt.wantErr, err != nil)
			if !tt.wantErr {
				c, _ := op.(*installerClient)
				assert.Equal(t, cfg, c.kubeConfig)
				assert.Equal(t, dyn, c.dynamicClient)
				assert.Equal(t, cli, c.clientset)
			}
		})
	}
}

func TestNewInstallerOperationWithClients(t *testing.T) {
	cs := k8sfake.NewSimpleClientset()
	dyn := &dynamic.DynamicClient{}
	op, err := NewInstallerOperationWithClients(cs, dyn)
	assert.NoError(t, err)

	client, ok := op.(*installerClient)
	assert.True(t, ok)
	assert.Equal(t, cs, client.GetClientSet())
	assert.Equal(t, dyn, client.GetDynamicClient())
}
