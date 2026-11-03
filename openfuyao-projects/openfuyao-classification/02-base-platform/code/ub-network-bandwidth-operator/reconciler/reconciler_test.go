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

package reconciler

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"gitcode.com/openFuyao/ub-network-bandwidth-operator/config"
)

type mockSDK struct {
	setCalls   [][]config.BandwidthItem
	resetCalls int
	getCalls   int
	setErr     error
	setErrs    []error
	resetErr   error
	getErr     error
	getErrs    []error
	getItems   []config.BandwidthItem
	getItemss  [][]config.BandwidthItem
}

func (m *mockSDK) BandwidthSet(items []config.BandwidthItem) error {
	m.setCalls = append(m.setCalls, items)
	if len(m.setErrs) > 0 {
		err := m.setErrs[0]
		m.setErrs = m.setErrs[1:]
		return err
	}
	return m.setErr
}

func (m *mockSDK) BandwidthReset() error {
	m.resetCalls++
	return m.resetErr
}

func (m *mockSDK) BandwidthGet() ([]config.BandwidthItem, error) {
	m.getCalls++
	if len(m.getErrs) > 0 {
		err := m.getErrs[0]
		m.getErrs = m.getErrs[1:]
		if len(m.getItemss) > 0 {
			items := m.getItemss[0]
			m.getItemss = m.getItemss[1:]
			return items, err
		}
		return m.getItems, err
	}
	if len(m.getItemss) > 0 {
		items := m.getItemss[0]
		m.getItemss = m.getItemss[1:]
		return items, m.getErr
	}
	return m.getItems, m.getErr
}

func writeConfig(t *testing.T, dir, content string) string {
	t.Helper()
	path := filepath.Join(dir, "config.yaml")
	err := os.WriteFile(path, []byte(content), 0644)
	assert.NoError(t, err)
	return path
}

func validConfigYAML(bandwidth uint32) string {
	return fmt.Sprintf(`bandwidths:
  - priority: 0
    bandwidth: %d
`, bandwidth)
}

func TestReconcileCreate(t *testing.T) {
	tmpDir := t.TempDir()
	configPath := writeConfig(t, tmpDir, validConfigYAML(25))

	sdk := &mockSDK{}
	r := NewController(sdk, configPath)

	err := r.Reconcile()
	assert.NoError(t, err)
	assert.Len(t, sdk.setCalls, 1)
	assert.Equal(t, 0, sdk.resetCalls)
	assert.Equal(t, []config.BandwidthItem{{Priority: 0, Bandwidth: 25}}, sdk.setCalls[0])
}

func TestReconcileSkipsWhenRuntimeMatchesConfig(t *testing.T) {
	tmpDir := t.TempDir()
	configPath := writeConfig(t, tmpDir, validConfigYAML(25))

	sdk := &mockSDK{
		getItems: []config.BandwidthItem{
			{Priority: 1, Bandwidth: 50},
			{Priority: 0, Bandwidth: 25},
		},
	}
	r := NewController(sdk, configPath)

	err := r.Reconcile()

	assert.NoError(t, err)
	assert.Equal(t, 1, sdk.getCalls)
	assert.Empty(t, sdk.setCalls)
	assert.Equal(t, 0, sdk.resetCalls)
}

func TestReconcileQueriesRuntimeEveryTime(t *testing.T) {
	tmpDir := t.TempDir()
	configPath := writeConfig(t, tmpDir, validConfigYAML(25))

	sdk := &mockSDK{
		getItemss: [][]config.BandwidthItem{
			{{Priority: 0, Bandwidth: 25}, {Priority: 1, Bandwidth: 50}},
			{{Priority: 0, Bandwidth: 20}, {Priority: 1, Bandwidth: 50}},
		},
	}
	r := NewController(sdk, configPath)

	err := r.Reconcile()
	assert.NoError(t, err)
	assert.Equal(t, 1, sdk.getCalls)
	assert.Empty(t, sdk.setCalls)
	assert.Equal(t, 0, sdk.resetCalls)

	err = r.Reconcile()
	assert.NoError(t, err)
	assert.Equal(t, 2, sdk.getCalls)
	assert.Equal(t, 1, sdk.resetCalls)
	require.Len(t, sdk.setCalls, 1)
	assert.Equal(t, []config.BandwidthItem{{Priority: 0, Bandwidth: 25}}, sdk.setCalls[0])
}

func TestReconcileResetsWhenConfigAbsentAndRuntimeExists(t *testing.T) {
	tmpDir := t.TempDir()
	configPath := filepath.Join(tmpDir, "config.yaml")

	sdk := &mockSDK{
		getItems: []config.BandwidthItem{{Priority: 0, Bandwidth: 25}, {Priority: 1, Bandwidth: 50}},
	}
	r := NewController(sdk, configPath)

	err := r.Reconcile()

	assert.NoError(t, err)
	assert.Equal(t, 1, sdk.getCalls)
	assert.Equal(t, 1, sdk.resetCalls)
}

func TestReconcileReturnsErrorWhenRuntimeQueryFails(t *testing.T) {
	tmpDir := t.TempDir()
	configPath := writeConfig(t, tmpDir, validConfigYAML(25))

	sdk := &mockSDK{getErr: fmt.Errorf("get failed")}
	r := NewController(sdk, configPath)

	err := r.Reconcile()

	assert.Error(t, err)
	if err != nil {
		assert.Contains(t, err.Error(), "query current UBSE config")
	}
	assert.Equal(t, 1, sdk.getCalls)
	assert.Equal(t, 0, sdk.resetCalls)
	assert.Empty(t, sdk.setCalls)
}

func TestReconcileRetriesRuntimeQueryAfterFailure(t *testing.T) {
	tmpDir := t.TempDir()
	configPath := filepath.Join(tmpDir, "config.yaml")

	sdk := &mockSDK{getErr: fmt.Errorf("get failed")}
	r := NewController(sdk, configPath)

	err := r.Reconcile()

	assert.Error(t, err)
	assert.Equal(t, 1, sdk.getCalls)
	assert.Equal(t, 0, sdk.resetCalls)

	sdk.getErr = nil
	sdk.getItems = []config.BandwidthItem{{Priority: 0, Bandwidth: 25}, {Priority: 1, Bandwidth: 50}}

	err = r.Reconcile()

	assert.NoError(t, err)
	assert.Equal(t, 2, sdk.getCalls)
	assert.Equal(t, 1, sdk.resetCalls)
}

func TestReconcileResetsAndRecreatesWhenRuntimeNeedsRebuild(t *testing.T) {
	tmpDir := t.TempDir()
	configPath := writeConfig(t, tmpDir, validConfigYAML(25))

	sdk := &mockSDK{getErr: errResetBeforeApplyRequired}
	r := NewController(sdk, configPath)

	err := r.Reconcile()

	assert.NoError(t, err)
	assert.Equal(t, 1, sdk.getCalls)
	assert.Equal(t, 1, sdk.resetCalls)
	require.Len(t, sdk.setCalls, 1)
	assert.Equal(t, []config.BandwidthItem{{Priority: 0, Bandwidth: 25}}, sdk.setCalls[0])
}

func TestReconcileResetsRuntimeNeedsRebuildWhenConfigAbsent(t *testing.T) {
	tmpDir := t.TempDir()
	configPath := filepath.Join(tmpDir, "config.yaml")

	sdk := &mockSDK{getErr: errResetBeforeApplyRequired}
	r := NewController(sdk, configPath)

	err := r.Reconcile()

	assert.NoError(t, err)
	assert.Equal(t, 1, sdk.getCalls)
	assert.Equal(t, 1, sdk.resetCalls)
	assert.Empty(t, sdk.setCalls)
}

func TestReconcileSkipsInvalidConfigWithoutQueryingRuntime(t *testing.T) {
	tmpDir := t.TempDir()
	configPath := writeConfig(t, tmpDir, `bandwidths:
  - priority: 0
    bandwidth: 60
`)

	sdk := &mockSDK{getErr: errResetBeforeApplyRequired}
	r := NewController(sdk, configPath)

	err := r.Reconcile()

	assert.NoError(t, err)
	assert.Equal(t, 0, sdk.getCalls)
	assert.Equal(t, 0, sdk.resetCalls)
	assert.Empty(t, sdk.setCalls)
}

func TestReconcileReturnsErrorWhenRuntimeRebuildResetFails(t *testing.T) {
	tmpDir := t.TempDir()
	configPath := writeConfig(t, tmpDir, validConfigYAML(25))

	sdk := &mockSDK{
		getErr:   errResetBeforeApplyRequired,
		resetErr: fmt.Errorf("reset failed"),
	}
	r := NewController(sdk, configPath)

	err := r.Reconcile()

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "bandwidth reset")
	assert.Equal(t, 1, sdk.getCalls)
	assert.Equal(t, 1, sdk.resetCalls)
	assert.Empty(t, sdk.setCalls)
}

func TestBandwidthConfigsEqual(t *testing.T) {
	tests := []struct {
		name string
		a    *config.BandwidthConfig
		b    *config.BandwidthConfig
		want bool
	}{
		{
			name: "both nil",
			want: true,
		},
		{
			name: "one nil",
			a:    &config.BandwidthConfig{},
			want: false,
		},
		{
			name: "single priority0 equals runtime with priority1 default",
			a:    &config.BandwidthConfig{Bandwidths: []config.BandwidthItem{{Priority: 0, Bandwidth: 25}}},
			b: &config.BandwidthConfig{Bandwidths: []config.BandwidthItem{
				{Priority: 1, Bandwidth: defaultBandwidth},
				{Priority: 0, Bandwidth: 25},
			}},
			want: true,
		},
		{
			name: "single priority1 equals runtime with priority0 default",
			a:    &config.BandwidthConfig{Bandwidths: []config.BandwidthItem{{Priority: 1, Bandwidth: 20}}},
			b: &config.BandwidthConfig{Bandwidths: []config.BandwidthItem{
				{Priority: 0, Bandwidth: defaultBandwidth},
				{Priority: 1, Bandwidth: 20},
			}},
			want: true,
		},
		{
			name: "two explicit items ignore order",
			a: &config.BandwidthConfig{Bandwidths: []config.BandwidthItem{
				{Priority: 0, Bandwidth: 25},
				{Priority: 1, Bandwidth: 50},
			}},
			b: &config.BandwidthConfig{Bandwidths: []config.BandwidthItem{
				{Priority: 1, Bandwidth: 50},
				{Priority: 0, Bandwidth: 25},
			}},
			want: true,
		},
		{
			name: "default bandwidth differs",
			a:    &config.BandwidthConfig{Bandwidths: []config.BandwidthItem{{Priority: 0, Bandwidth: 25}}},
			b: &config.BandwidthConfig{Bandwidths: []config.BandwidthItem{
				{Priority: 1, Bandwidth: 49},
				{Priority: 0, Bandwidth: 25},
			}},
			want: false,
		},
		{
			name: "bandwidth differs",
			a:    &config.BandwidthConfig{Bandwidths: []config.BandwidthItem{{Priority: 0, Bandwidth: 25}}},
			b:    &config.BandwidthConfig{Bandwidths: []config.BandwidthItem{{Priority: 0, Bandwidth: 20}}},
			want: false,
		},
		{
			name: "priority differs",
			a:    &config.BandwidthConfig{Bandwidths: []config.BandwidthItem{{Priority: 0, Bandwidth: 25}}},
			b:    &config.BandwidthConfig{Bandwidths: []config.BandwidthItem{{Priority: 2, Bandwidth: 25}}},
			want: false,
		},
		{
			name: "duplicate priority is not equal",
			a: &config.BandwidthConfig{Bandwidths: []config.BandwidthItem{
				{Priority: 0, Bandwidth: 20},
				{Priority: 0, Bandwidth: 25},
			}},
			b: &config.BandwidthConfig{Bandwidths: []config.BandwidthItem{
				{Priority: 0, Bandwidth: 25},
				{Priority: 0, Bandwidth: 25},
			}},
			want: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, bandwidthConfigsEqual(tt.a, tt.b))
		})
	}
}

func TestReconcileUpdateResetsPreviousConfig(t *testing.T) {
	tmpDir := t.TempDir()
	configPath := writeConfig(t, tmpDir, validConfigYAML(20))

	sdk := &mockSDK{
		getItemss: [][]config.BandwidthItem{
			nil,
			{{Priority: 0, Bandwidth: 20}, {Priority: 1, Bandwidth: 50}},
		},
	}
	r := NewController(sdk, configPath)

	err := r.Reconcile()
	assert.NoError(t, err)
	assert.Len(t, sdk.setCalls, 1)
	assert.Equal(t, 0, sdk.resetCalls)

	configPath = writeConfig(t, tmpDir, validConfigYAML(25))

	err = r.Reconcile()
	assert.NoError(t, err)
	assert.Equal(t, 1, sdk.resetCalls)
	require.Len(t, sdk.setCalls, 2)
	assert.Equal(t, uint32(25), sdk.setCalls[1][0].Bandwidth)
}

func TestReconcileDeleteResetsPreviousConfig(t *testing.T) {
	tmpDir := t.TempDir()
	configPath := writeConfig(t, tmpDir, validConfigYAML(25))

	sdk := &mockSDK{
		getItemss: [][]config.BandwidthItem{
			nil,
			{{Priority: 0, Bandwidth: 25}, {Priority: 1, Bandwidth: 50}},
		},
	}
	r := NewController(sdk, configPath)

	err := r.Reconcile()
	assert.NoError(t, err)
	assert.Len(t, sdk.setCalls, 1)
	assert.Equal(t, 0, sdk.resetCalls)

	os.Remove(configPath)

	err = r.Reconcile()
	assert.NoError(t, err)
	assert.Equal(t, 1, sdk.resetCalls)
}

func TestReconcileDeleteEmpty(t *testing.T) {
	tmpDir := t.TempDir()
	configPath := writeConfig(t, tmpDir, "")

	sdk := &mockSDK{}
	r := NewController(sdk, configPath)

	err := r.Reconcile()
	assert.NoError(t, err)
	assert.Empty(t, sdk.setCalls)
	assert.Equal(t, 0, sdk.resetCalls)
}

func TestReconcileValidationFailed(t *testing.T) {
	tmpDir := t.TempDir()
	configPath := writeConfig(t, tmpDir, `bandwidths:
  - priority: 0
    bandwidth: 60
`)

	sdk := &mockSDK{}
	r := NewController(sdk, configPath)

	err := r.Reconcile()
	assert.NoError(t, err)
	assert.Empty(t, sdk.setCalls)
	assert.Equal(t, 0, sdk.resetCalls)
}

func TestReconcileParseConfigFailedDoesNotRetry(t *testing.T) {
	tests := []struct {
		name    string
		content string
	}{
		{
			name: "fractional bandwidth",
			content: `bandwidths:
  - priority: 0
    bandwidth: 25.5
`,
		},
		{
			name: "string bandwidth",
			content: `bandwidths:
  - priority: 0
    bandwidth: "25"
`,
		},
		{
			name: "negative bandwidth",
			content: `bandwidths:
  - priority: 0
    bandwidth: -1
`,
		},
		{
			name:    "invalid yaml",
			content: "{{invalid yaml}}",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tmpDir := t.TempDir()
			configPath := writeConfig(t, tmpDir, tt.content)

			sdk := &mockSDK{}
			r := NewController(sdk, configPath)

			err := r.Reconcile()

			assert.NoError(t, err)
			assert.Equal(t, 0, sdk.getCalls)
			assert.Empty(t, sdk.setCalls)
			assert.Equal(t, 0, sdk.resetCalls)
		})
	}
}

func TestReconcileBandwidthSetError(t *testing.T) {
	tmpDir := t.TempDir()
	configPath := writeConfig(t, tmpDir, validConfigYAML(25))

	sdk := &mockSDK{setErr: fmt.Errorf("set failed")}
	r := NewController(sdk, configPath)

	err := r.Reconcile()
	assert.Error(t, err)
}

func TestReconcileConfigExistsResetsAndRetriesSetImmediately(t *testing.T) {
	tmpDir := t.TempDir()
	configPath := writeConfig(t, tmpDir, validConfigYAML(25))

	sdk := &mockSDK{setErrs: []error{errResetBeforeApplyRequired, nil}}
	r := NewController(sdk, configPath)

	err := r.Reconcile()

	assert.NoError(t, err)
	assert.Equal(t, 1, sdk.resetCalls)
	assert.Len(t, sdk.setCalls, 2)
	assert.Equal(t, []config.BandwidthItem{{Priority: 0, Bandwidth: 25}}, sdk.setCalls[1])
}

func TestReconcileConfigExistsReturnsErrorWhenImmediateRetryFails(t *testing.T) {
	tmpDir := t.TempDir()
	configPath := writeConfig(t, tmpDir, validConfigYAML(25))

	retryErr := fmt.Errorf("retry set failed")
	sdk := &mockSDK{setErrs: []error{errResetBeforeApplyRequired, retryErr}}
	r := NewController(sdk, configPath)

	err := r.Reconcile()

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "retry bandwidth set")
	assert.Equal(t, 1, sdk.resetCalls)
	assert.Len(t, sdk.setCalls, 2)
}

func TestReconcileConfigExistsReturnsErrorWhenImmediateResetFails(t *testing.T) {
	tmpDir := t.TempDir()
	configPath := writeConfig(t, tmpDir, validConfigYAML(25))

	sdk := &mockSDK{
		setErr:   errResetBeforeApplyRequired,
		resetErr: fmt.Errorf("reset failed"),
	}
	r := NewController(sdk, configPath)

	err := r.Reconcile()

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "bandwidth reset")
	assert.Equal(t, 1, sdk.resetCalls)
	assert.Len(t, sdk.setCalls, 1)
}

func TestReconcileBandwidthResetError(t *testing.T) {
	tmpDir := t.TempDir()
	configPath := writeConfig(t, tmpDir, validConfigYAML(25))

	sdk := &mockSDK{
		getItemss: [][]config.BandwidthItem{
			nil,
			{{Priority: 0, Bandwidth: 25}, {Priority: 1, Bandwidth: 50}},
		},
	}
	r := NewController(sdk, configPath)

	err := r.Reconcile()
	assert.NoError(t, err)

	sdk.resetErr = fmt.Errorf("reset failed")
	configPath = writeConfig(t, tmpDir, validConfigYAML(20))

	err = r.Reconcile()
	assert.Error(t, err)
	assert.Equal(t, 1, sdk.resetCalls)
}

func TestReconcileConfigDeletedNoPreviousState(t *testing.T) {
	tmpDir := t.TempDir()
	configPath := filepath.Join(tmpDir, "config.yaml")
	os.Remove(configPath)

	sdk := &mockSDK{}
	r := NewController(sdk, configPath)

	err := r.Reconcile()
	assert.NoError(t, err)
	assert.Empty(t, sdk.setCalls)
	assert.Equal(t, 0, sdk.resetCalls)
}
