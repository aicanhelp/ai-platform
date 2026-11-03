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

package main

import (
	"errors"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"
	"github.com/urfave/cli/v2"

	"gitcode.com/openFuyao/ub-network-bandwidth-operator/config"
	"gitcode.com/openFuyao/ub-network-bandwidth-operator/operator"
)

type mockOperator struct {
	startErr error
	stopped  bool
}

func (m *mockOperator) Start() error {
	return m.startErr
}

func (m *mockOperator) Stop() {
	m.stopped = true
}

func TestStart_WatchFilesError(t *testing.T) {
	patches := gomonkey.ApplyFunc(operator.StartHealthServer, func(_ string) {
		// Health server startup is outside this error-path test.
	})
	defer patches.Reset()

	patches.ApplyFunc(config.NewWatcher, func(_ string) (*config.Watcher, error) {
		return nil, errors.New("watch error")
	})

	app := cli.NewApp()
	ctx := cli.NewContext(app, nil, nil)

	err := start(ctx)
	assert.Error(t, err)
}

func TestStart_OperatorNewError(t *testing.T) {
	patches := gomonkey.ApplyFunc(operator.StartHealthServer, func(_ string) {
		// Health server startup is outside this constructor error-path test.
	})
	defer patches.Reset()

	patches.ApplyFunc(operator.New, func(_ ...operator.Option) (operator.Interface, error) {
		return nil, errors.New("operator new error")
	})

	app := cli.NewApp()
	ctx := cli.NewContext(app, nil, nil)

	err := start(ctx)
	assert.Error(t, err)
}
