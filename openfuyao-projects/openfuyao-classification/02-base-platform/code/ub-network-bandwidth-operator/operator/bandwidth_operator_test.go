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

package operator

import (
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"

	"gitcode.com/openFuyao/ub-network-bandwidth-operator/config"
)

type mockBandwidthManager struct {
	SetCalled   bool
	ResetCalled bool
	LastItems   []config.BandwidthItem
	ReturnError bool
	GetCalls    int
}

func (m *mockBandwidthManager) BandwidthSet(items []config.BandwidthItem) error {
	m.SetCalled = true
	m.LastItems = items
	if m.ReturnError {
		return assert.AnError
	}
	m.LastItems = append([]config.BandwidthItem(nil), items...)
	return nil
}

func (m *mockBandwidthManager) BandwidthReset() error {
	m.ResetCalled = true
	if m.ReturnError {
		return assert.AnError
	}
	m.LastItems = nil
	return nil
}

func (m *mockBandwidthManager) BandwidthGet() ([]config.BandwidthItem, error) {
	m.GetCalls++
	if m.ReturnError {
		return nil, assert.AnError
	}
	return append([]config.BandwidthItem(nil), m.LastItems...), nil
}

func createTempConfig(t *testing.T, content string) string {
	t.Helper()
	tmpDir := t.TempDir()
	configPath := filepath.Join(tmpDir, "config.yaml")
	err := os.WriteFile(configPath, []byte(content), 0644)
	assert.NoError(t, err)
	return configPath
}

func waitForCondition(t *testing.T, timeout time.Duration, condition func() bool) {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if condition() {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	assert.True(t, condition())
}

func TestNew(t *testing.T) {
	configPath := createTempConfig(t, `
bandwidths:
- priority: 0
  bandwidth: 25
`)
	mockManager := &mockBandwidthManager{}

	op, err := New(
		WithConfigPath(configPath),
		WithManager(mockManager),
	)
	assert.NoError(t, err)
	assert.NotNil(t, op)
}

func TestNewInvalidConfigPath(t *testing.T) {
	mockManager := &mockBandwidthManager{}

	op, err := New(
		WithConfigPath("/nonexistent/dir/config.yaml"),
		WithManager(mockManager),
	)
	assert.Error(t, err)
	assert.Nil(t, op)
}

func TestOperatorStartStop(t *testing.T) {
	configPath := createTempConfig(t, `
bandwidths:
- priority: 0
  bandwidth: 25
`)
	mockManager := &mockBandwidthManager{}

	op, err := New(
		WithConfigPath(configPath),
		WithManager(mockManager),
	)
	assert.NoError(t, err)

	err = op.Start()
	assert.NoError(t, err)

	time.Sleep(50 * time.Millisecond)

	dummyFile := filepath.Join(filepath.Dir(configPath), "trigger.yaml")
	os.WriteFile(dummyFile, []byte("dummy"), 0644)
	os.Remove(dummyFile)

	time.Sleep(200 * time.Millisecond)

	op.Stop()

	assert.True(t, mockManager.SetCalled)
	assert.GreaterOrEqual(t, mockManager.GetCalls, 1)
	assert.Len(t, mockManager.LastItems, 1)
	assert.Equal(t, uint32(0), mockManager.LastItems[0].Priority)
	assert.Equal(t, uint32(25), mockManager.LastItems[0].Bandwidth)
}

func TestOperatorStopBeforeStart(t *testing.T) {
	configPath := createTempConfig(t, `
bandwidths:
- priority: 0
  bandwidth: 25
`)
	mockManager := &mockBandwidthManager{}

	op, err := New(
		WithConfigPath(configPath),
		WithManager(mockManager),
	)
	assert.NoError(t, err)

	op.Stop()
}

func TestDoneNotClosedOnNormalStop(t *testing.T) {
	configPath := createTempConfig(t, `
bandwidths:
- priority: 0
  bandwidth: 25
`)
	mockManager := &mockBandwidthManager{}

	op, err := New(
		WithConfigPath(configPath),
		WithManager(mockManager),
	)
	assert.NoError(t, err)

	err = op.Start()
	assert.NoError(t, err)

	time.Sleep(50 * time.Millisecond)

	op.Stop()

	select {
	case <-op.Done():
		t.Fatal("Done should not be closed on normal Stop")
	default:
	}
}

func TestFinishStoppedFlag(t *testing.T) {
	configPath := createTempConfig(t, `
bandwidths:
- priority: 0
  bandwidth: 25
`)
	mockManager := &mockBandwidthManager{}

	op, err := New(
		WithConfigPath(configPath),
		WithManager(mockManager),
	)
	assert.NoError(t, err)

	op.Stop()

	bo := op.(*bandwidthOperator)
	bo.finish()

	select {
	case <-bo.doneCh:
		t.Fatal("doneCh should not be closed when stopped is true")
	default:
	}
}

func TestProcessNextItemReconcileError(t *testing.T) {
	configPath := createTempConfig(t, `
bandwidths:
- priority: 0
  bandwidth: 25
`)
	mockManager := &mockBandwidthManager{ReturnError: true}

	op, err := New(
		WithConfigPath(configPath),
		WithManager(mockManager),
	)
	assert.NoError(t, err)

	err = op.Start()
	assert.NoError(t, err)

	time.Sleep(50 * time.Millisecond)

	assert.GreaterOrEqual(t, mockManager.GetCalls, 1)
	dummyFile := filepath.Join(filepath.Dir(configPath), "trigger.yaml")
	os.WriteFile(dummyFile, []byte("dummy"), 0644)
	os.Remove(dummyFile)

	time.Sleep(200 * time.Millisecond)

	op.Stop()
}

func TestWatchConfigEventTriggersReconcile(t *testing.T) {
	configPath := createTempConfig(t, `
bandwidths:
- priority: 0
  bandwidth: 25
`)
	mockManager := &mockBandwidthManager{}

	op, err := New(
		WithConfigPath(configPath),
		WithManager(mockManager),
	)
	assert.NoError(t, err)

	err = op.Start()
	assert.NoError(t, err)

	time.Sleep(50 * time.Millisecond)

	dummyFile := filepath.Join(filepath.Dir(configPath), "trigger.yaml")
	os.WriteFile(dummyFile, []byte("dummy"), 0644)
	os.Remove(dummyFile)

	time.Sleep(200 * time.Millisecond)
	assert.True(t, mockManager.SetCalled)

	newContent := []byte(`
bandwidths:
- priority: 0
  bandwidth: 20
`)
	err = os.WriteFile(configPath, newContent, 0644)
	assert.NoError(t, err)

	dummyFile2 := filepath.Join(filepath.Dir(configPath), "trigger2.yaml")
	os.WriteFile(dummyFile2, []byte("dummy"), 0644)
	os.Remove(dummyFile2)

	time.Sleep(200 * time.Millisecond)

	op.Stop()
}

func TestWatchConfigProjectionRemoveResetsWhenConfigDeleted(t *testing.T) {
	configPath := createTempConfig(t, `
bandwidths:
- priority: 0
  bandwidth: 25
`)
	mockManager := &mockBandwidthManager{}

	op, err := New(
		WithConfigPath(configPath),
		WithManager(mockManager),
	)
	assert.NoError(t, err)

	err = op.Start()
	assert.NoError(t, err)

	waitForCondition(t, time.Second, func() bool {
		return mockManager.SetCalled
	})

	err = os.Remove(configPath)
	assert.NoError(t, err)

	time.Sleep(100 * time.Millisecond)
	assert.False(t, mockManager.ResetCalled)

	projectionFile := filepath.Join(filepath.Dir(configPath), "projection-trigger")
	err = os.WriteFile(projectionFile, []byte("old projection"), 0644)
	assert.NoError(t, err)
	time.Sleep(20 * time.Millisecond)
	err = os.Remove(projectionFile)
	assert.NoError(t, err)

	waitForCondition(t, time.Second, func() bool {
		return mockManager.ResetCalled
	})

	op.Stop()
}

func TestDoneSignalsOnWatcherClose(t *testing.T) {
	configPath := createTempConfig(t, `
bandwidths:
- priority: 0
  bandwidth: 25
`)
	mockManager := &mockBandwidthManager{}

	op, err := New(
		WithConfigPath(configPath),
		WithManager(mockManager),
	)
	assert.NoError(t, err)

	err = op.Start()
	assert.NoError(t, err)

	time.Sleep(50 * time.Millisecond)

	bo := op.(*bandwidthOperator)
	bo.watcher.Close()

	select {
	case <-op.Done():
	case <-time.After(2 * time.Second):
		t.Fatal("Done should be closed when watcher exits unexpectedly")
	}

	op.Stop()
}
