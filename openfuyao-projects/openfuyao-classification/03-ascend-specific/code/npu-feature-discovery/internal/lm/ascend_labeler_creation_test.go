// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// openFuyao is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan PSL v2.
// You may obtain a copy of Mulan PSL v2 at:
//
//	http://license.coscl.org.cn/MulanPSL2
//
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
// MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
// See the Mulan PSL v2 for more details.

package lm

import (
	"errors"
	"reflect"
	"strings"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/urfave/cli/v2"

	"openfuyao.com/npu-feature-discovery/internal/lm/common"
)

func TestNewLabelersSuccess(t *testing.T) {
	wantLabeler := common.Labels{"npu.present": "true"}
	patch := gomonkey.ApplyFunc(NewAscendLabeler, func(*cli.Context) (common.Labeler, error) {
		return wantLabeler, nil
	})
	defer patch.Reset()

	got, err := NewLabelers(nil)
	if err != nil {
		t.Fatalf("NewLabelers() err = %v", err)
	}
	if got == nil {
		t.Fatal("NewLabelers() got nil labeler")
	}
	if !reflect.DeepEqual(got.Labels(), wantLabeler.Labels()) {
		t.Errorf("NewLabelers() Labels() = %v, want %v", got.Labels(), wantLabeler.Labels())
	}
}

func TestNewLabelersAscendLabelerError(t *testing.T) {
	wantErr := errors.New("ascend labeler failed")
	patch := gomonkey.ApplyFunc(NewAscendLabeler, func(*cli.Context) (common.Labeler, error) {
		return nil, wantErr
	})
	defer patch.Reset()

	got, err := NewLabelers(nil)
	if err == nil {
		t.Fatal("NewLabelers() expected error")
	}
	if got != nil {
		t.Errorf("NewLabelers() = %v, want nil", got)
	}
	if !strings.Contains(err.Error(), "error creating Ascend labeler:") {
		t.Errorf("NewLabelers() err = %v", err)
	}
	if !strings.Contains(err.Error(), wantErr.Error()) {
		t.Errorf("NewLabelers() err should wrap original: %v", err)
	}
}
