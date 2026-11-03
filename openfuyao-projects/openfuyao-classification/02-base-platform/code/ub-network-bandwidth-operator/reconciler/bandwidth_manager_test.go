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
	"errors"
	"testing"

	"github.com/stretchr/testify/assert"

	"gitcode.com/openFuyao/ub-network-bandwidth-operator/config"
	"gitcode.com/openFuyao/ub-network-bandwidth-operator/sdk/ubse"
)

func TestNewBandwidthManager(t *testing.T) {
	bm := NewBandwidthManager()
	assert.NotNil(t, bm)
}

func TestBandwidthManagerImplementsInterface(t *testing.T) {
	var bm BandwidthManager = NewBandwidthManager()
	assert.NotNil(t, bm)
}

func TestBuildSDKBandwidthItemsAddsDefaultWithoutUnitConversion(t *testing.T) {
	bm := NewBandwidthManager()

	tests := []struct {
		name  string
		items []config.BandwidthItem
		want  []ubse.BandwidthItem
	}{
		{
			name:  "single priority0 fills priority1 default",
			items: []config.BandwidthItem{{Priority: 0, Bandwidth: 25}},
			want:  []ubse.BandwidthItem{{Priority: 0, Bandwidth: 25}, {Priority: 1, Bandwidth: 50}},
		},
		{
			name:  "single priority1 fills priority0 default",
			items: []config.BandwidthItem{{Priority: 1, Bandwidth: 20}},
			want:  []ubse.BandwidthItem{{Priority: 1, Bandwidth: 20}, {Priority: 0, Bandwidth: 50}},
		},
		{
			name:  "two items pass through without reordering",
			items: []config.BandwidthItem{{Priority: 1, Bandwidth: 40}, {Priority: 0, Bandwidth: 10}},
			want:  []ubse.BandwidthItem{{Priority: 1, Bandwidth: 40}, {Priority: 0, Bandwidth: 10}},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := bm.buildSDKBandwidthItems(tt.items)
			assert.Equal(t, tt.want, got)
		})
	}
}

func TestBandwidthManagerSetReturnsDaemonErrorAfterPacking(t *testing.T) {
	manager := NewBandwidthManager()

	err := manager.BandwidthSet([]config.BandwidthItem{{Priority: 0, Bandwidth: 25}})

	assert.Error(t, err)
}

func TestBandwidthConfigExistsErrorWrapsDaemonError(t *testing.T) {
	err := wrapBandwidthSetError(&ubse.DaemonError{
		Code: ubse.BandwidthConfigExistsCode,
	})

	assert.True(t, errors.Is(err, errResetBeforeApplyRequired))
	assert.True(t, ubse.IsDaemonErrorCode(err, ubse.BandwidthConfigExistsCode))
	assert.Contains(t, err.Error(), "ubse daemon returned error: 11101")
}

func TestBandwidthManagerSetTreatsExistingMatchingConfigAsSuccess(t *testing.T) {
	oldSet, oldGet := ubseBandwidthSet, ubseBandwidthGet
	defer func() {
		ubseBandwidthSet = oldSet
		ubseBandwidthGet = oldGet
	}()
	ubseBandwidthSet = func(_ []ubse.BandwidthItem) error {
		return &ubse.DaemonError{Code: ubse.BandwidthConfigExistsCode}
	}
	ubseBandwidthGet = func() ([]ubse.BandwidthItem, error) {
		return []ubse.BandwidthItem{{Priority: 1, Bandwidth: 50}, {Priority: 0, Bandwidth: 25}}, nil
	}
	manager := NewBandwidthManager()

	err := manager.BandwidthSet([]config.BandwidthItem{{Priority: 0, Bandwidth: 25}})

	assert.NoError(t, err)
}

func TestBandwidthManagerSetRequiresResetWhenExistingConfigDiffers(t *testing.T) {
	oldSet, oldGet := ubseBandwidthSet, ubseBandwidthGet
	defer func() {
		ubseBandwidthSet = oldSet
		ubseBandwidthGet = oldGet
	}()
	ubseBandwidthSet = func(_ []ubse.BandwidthItem) error {
		return &ubse.DaemonError{Code: ubse.BandwidthConfigExistsCode}
	}
	ubseBandwidthGet = func() ([]ubse.BandwidthItem, error) {
		return []ubse.BandwidthItem{{Priority: 1, Bandwidth: 50}, {Priority: 0, Bandwidth: 20}}, nil
	}
	manager := NewBandwidthManager()

	err := manager.BandwidthSet([]config.BandwidthItem{{Priority: 0, Bandwidth: 25}})

	assert.Error(t, err)
	assert.True(t, errors.Is(err, errResetBeforeApplyRequired))
}

func TestBandwidthManagerSetRequiresResetWhenExistingConfigQueryFails(t *testing.T) {
	oldSet, oldGet := ubseBandwidthSet, ubseBandwidthGet
	defer func() {
		ubseBandwidthSet = oldSet
		ubseBandwidthGet = oldGet
	}()
	ubseBandwidthSet = func(_ []ubse.BandwidthItem) error {
		return &ubse.DaemonError{Code: ubse.BandwidthConfigExistsCode}
	}
	ubseBandwidthGet = func() ([]ubse.BandwidthItem, error) {
		return nil, errors.New("query failed")
	}
	manager := NewBandwidthManager()

	err := manager.BandwidthSet([]config.BandwidthItem{{Priority: 0, Bandwidth: 25}})

	assert.Error(t, err)
	assert.True(t, errors.Is(err, errResetBeforeApplyRequired))
}

func TestBandwidthManagerGetConvertsSDKItems(t *testing.T) {
	oldGet := ubseBandwidthGet
	defer func() {
		ubseBandwidthGet = oldGet
	}()
	ubseBandwidthGet = func() ([]ubse.BandwidthItem, error) {
		return []ubse.BandwidthItem{{Priority: 1, Bandwidth: 50}, {Priority: 0, Bandwidth: 25}}, nil
	}
	manager := NewBandwidthManager()

	items, err := manager.BandwidthGet()

	assert.NoError(t, err)
	assert.Equal(t, []config.BandwidthItem{{Priority: 1, Bandwidth: 50}, {Priority: 0, Bandwidth: 25}}, items)
}

func TestBandwidthManagerGetReturnsQueryError(t *testing.T) {
	oldGet := ubseBandwidthGet
	defer func() {
		ubseBandwidthGet = oldGet
	}()
	ubseBandwidthGet = func() ([]ubse.BandwidthItem, error) {
		return nil, errors.New("query failed")
	}
	manager := NewBandwidthManager()

	items, err := manager.BandwidthGet()

	assert.Error(t, err)
	assert.Nil(t, items)
}

func TestBandwidthManagerGetRequiresResetForOVSResidualErrors(t *testing.T) {
	tests := []struct {
		name string
		code uint32
	}{
		{name: "ovs interface missing", code: ubse.BandwidthGetOVSInterfaceMissingCode},
		{name: "ovs interface invalid", code: ubse.BandwidthGetOVSInterfaceInvalidCode},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			oldGet := ubseBandwidthGet
			defer func() {
				ubseBandwidthGet = oldGet
			}()
			ubseBandwidthGet = func() ([]ubse.BandwidthItem, error) {
				return nil, &ubse.DaemonError{Code: tt.code}
			}
			manager := NewBandwidthManager()

			items, err := manager.BandwidthGet()

			assert.Error(t, err)
			assert.Nil(t, items)
			assert.True(t, errors.Is(err, errResetBeforeApplyRequired))
			assert.True(t, ubse.IsDaemonErrorCode(err, tt.code))
		})
	}
}

func TestBandwidthItemsEqual(t *testing.T) {
	tests := []struct {
		name string
		a    []ubse.BandwidthItem
		b    []ubse.BandwidthItem
		want bool
	}{
		{
			name: "same items ignore order",
			a:    []ubse.BandwidthItem{{Priority: 0, Bandwidth: 25}, {Priority: 1, Bandwidth: 50}},
			b:    []ubse.BandwidthItem{{Priority: 1, Bandwidth: 50}, {Priority: 0, Bandwidth: 25}},
			want: true,
		},
		{
			name: "length differs",
			a:    []ubse.BandwidthItem{{Priority: 0, Bandwidth: 25}},
			b:    []ubse.BandwidthItem{{Priority: 0, Bandwidth: 25}, {Priority: 1, Bandwidth: 50}},
			want: false,
		},
		{
			name: "priority missing",
			a:    []ubse.BandwidthItem{{Priority: 0, Bandwidth: 25}},
			b:    []ubse.BandwidthItem{{Priority: 1, Bandwidth: 25}},
			want: false,
		},
		{
			name: "bandwidth differs",
			a:    []ubse.BandwidthItem{{Priority: 0, Bandwidth: 25}},
			b:    []ubse.BandwidthItem{{Priority: 0, Bandwidth: 20}},
			want: false,
		},
		{
			name: "duplicate priority is not equal",
			a:    []ubse.BandwidthItem{{Priority: 0, Bandwidth: 20}, {Priority: 0, Bandwidth: 25}},
			b:    []ubse.BandwidthItem{{Priority: 0, Bandwidth: 25}, {Priority: 0, Bandwidth: 25}},
			want: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, bandwidthItemsEqual(tt.a, tt.b))
		})
	}
}

func TestBandwidthManagerResetReturnsDaemonErrorAfterPacking(t *testing.T) {
	manager := NewBandwidthManager()

	err := manager.BandwidthReset()

	assert.Error(t, err)
}
