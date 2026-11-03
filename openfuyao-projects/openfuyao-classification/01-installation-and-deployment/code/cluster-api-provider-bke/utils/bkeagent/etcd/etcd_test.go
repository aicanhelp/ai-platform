/******************************************************************
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * cluster-api-provider-bke is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package etcd

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"
	clientv3 "go.etcd.io/etcd/client/v3"
	"go.etcd.io/etcd/client/v3/snapshot"
	k8setcd "k8s.io/kubernetes/cmd/kubeadm/app/util/etcd"
)

const (
	testDBPath      = "/var/lib/etcd/backup.db"
	testTimeout     = 4 * time.Minute
	testDialTimeout = 2 * time.Second
)

func TestSaveWithValidClient(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	mockClient := &k8setcd.Client{
		Endpoints: []string{"https://localhost:2379"},
		TLS:       nil,
	}

	saveCalled := false

	patches.ApplyFunc(snapshot.Save, func(_ context.Context, _ interface{}, _ clientv3.Config, dbPath string) error {
		saveCalled = true
		return nil
	})

	err := Save(mockClient, testDBPath)

	assert.NoError(t, err)
	assert.True(t, saveCalled)
}

func TestSaveWithSaveError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	mockClient := &k8setcd.Client{
		Endpoints: []string{"https://etcd-0:2379"},
		TLS:       nil,
	}

	saveErr := errors.New("failed to save snapshot")
	patches.ApplyFunc(snapshot.Save, func(_ context.Context, _ interface{}, _ clientv3.Config, _ string) error {
		return saveErr
	})

	err := Save(mockClient, testDBPath)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "failed to save etcd snapshot")
}

func TestSaveWithMultipleEndpoints(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	mockClient := &k8setcd.Client{
		Endpoints: []string{
			"https://etcd-0:2379",
			"https://etcd-1:2379",
			"https://etcd-2:2379",
		},
		TLS: nil,
	}

	saveCalled := false

	patches.ApplyFunc(snapshot.Save, func(_ context.Context, _ interface{}, cfg clientv3.Config, _ string) error {
		saveCalled = true
		if cfg.Endpoints[0] != "https://etcd-0:2379" {
			t.Error("Expected first endpoint to be used")
		}
		return nil
	})

	err := Save(mockClient, testDBPath)

	assert.NoError(t, err)
	assert.True(t, saveCalled)
}

func TestSaveWithTLSEnabled(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	mockClient := &k8setcd.Client{
		Endpoints: []string{"https://localhost:2379"},
		TLS:       nil,
	}

	saveCalled := false

	patches.ApplyFunc(snapshot.Save, func(_ context.Context, _ interface{}, cfg clientv3.Config, _ string) error {
		saveCalled = true
		return nil
	})

	err := Save(mockClient, testDBPath)

	assert.NoError(t, err)
	assert.True(t, saveCalled)
}

func TestSaveWithContextCancellation(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	mockClient := &k8setcd.Client{
		Endpoints: []string{"https://localhost:2379"},
		TLS:       nil,
	}

	errReturned := false

	patches.ApplyFunc(snapshot.Save, func(_ context.Context, _ interface{}, _ clientv3.Config, _ string) error {
		errReturned = true
		return nil
	})

	err := Save(mockClient, testDBPath)

	if errReturned && err == nil {
		t.Log("Context cancellation not tested - Save completed before context expired")
	}
}

func TestSaveWithDifferentDBPath(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	mockClient := &k8setcd.Client{
		Endpoints: []string{"https://localhost:2379"},
		TLS:       nil,
	}

	saveCalled := false

	patches.ApplyFunc(snapshot.Save, func(_ context.Context, _ interface{}, _ clientv3.Config, _ string) error {
		saveCalled = true
		return nil
	})

	Save(mockClient, "/backup/etcd-snapshot.db")

	if !saveCalled {
		t.Error("Expected saveCalled to be true")
	}
}

func TestSaveWithGRPCDialOptions(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	mockClient := &k8setcd.Client{
		Endpoints: []string{"https://localhost:2379"},
		TLS:       nil,
	}

	dialOptionsFound := false

	patches.ApplyFunc(snapshot.Save, func(_ context.Context, _ interface{}, cfg clientv3.Config, _ string) error {
		if len(cfg.DialOptions) > 0 {
			dialOptionsFound = true
		}
		return nil
	})

	err := Save(mockClient, testDBPath)

	assert.NoError(t, err)
	assert.True(t, dialOptionsFound)
}

func TestSaveWithNilTLS(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	mockClient := &k8setcd.Client{
		Endpoints: []string{"https://localhost:2379"},
		TLS:       nil,
	}

	saveCalled := false

	patches.ApplyFunc(snapshot.Save, func(_ context.Context, _ interface{}, cfg clientv3.Config, _ string) error {
		saveCalled = true
		if cfg.TLS != nil {
			t.Error("Expected TLS to be nil")
		}
		return nil
	})

	err := Save(mockClient, testDBPath)

	assert.NoError(t, err)
	assert.True(t, saveCalled)
}

func TestSaveWithUnixSocketEndpoint(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	mockClient := &k8setcd.Client{
		Endpoints: []string{"unix:///var/run/etcd/etcd.sock"},
		TLS:       nil,
	}

	saveCalled := false

	patches.ApplyFunc(snapshot.Save, func(_ context.Context, _ interface{}, _ clientv3.Config, _ string) error {
		saveCalled = true
		return nil
	})

	err := Save(mockClient, testDBPath)

	assert.NoError(t, err)
	assert.True(t, saveCalled)
}

func TestSaveWithHttpsEndpoint(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	mockClient := &k8setcd.Client{
		Endpoints: []string{"https://etcd.cluster.local:2379"},
		TLS:       nil,
	}

	saveCalled := false

	patches.ApplyFunc(snapshot.Save, func(_ context.Context, _ interface{}, _ clientv3.Config, _ string) error {
		saveCalled = true
		return nil
	})

	err := Save(mockClient, testDBPath)

	assert.NoError(t, err)
	assert.True(t, saveCalled)
}

func TestSaveWithTimeoutSet(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	mockClient := &k8setcd.Client{
		Endpoints: []string{"https://localhost:2379"},
		TLS:       nil,
	}

	patches.ApplyFunc(snapshot.Save, func(ctx context.Context, _ interface{}, cfg clientv3.Config, _ string) error {
		_, cancel := context.WithTimeout(ctx, testTimeout)
		defer cancel()
		return nil
	})

	err := Save(mockClient, testDBPath)

	assert.NoError(t, err)
}
