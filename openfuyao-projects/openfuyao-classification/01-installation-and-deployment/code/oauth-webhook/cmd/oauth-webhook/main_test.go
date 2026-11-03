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
	"os"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/spf13/cobra"
	"github.com/stretchr/testify/require"
	"k8s.io/component-base/cli"

	"openfuyao/oauth-webhook/cmd/oauth-webhook/app"
)

func TestMain(t *testing.T) {
	var (
		runCalled    bool
		exitCalled   bool
		exitCode     int
		receivedCmd  *cobra.Command
		expectedCode = 7
	)

	patches := gomonkey.NewPatches()
	patches.ApplyFunc(app.NewOAuthWebhookCommand, func() *cobra.Command {
		return &cobra.Command{Use: "test-cmd"}
	})
	patches.ApplyFunc(cli.Run, func(cmd *cobra.Command) int {
		runCalled = true
		receivedCmd = cmd
		return expectedCode
	})
	patches.ApplyFunc(os.Exit, func(code int) {
		exitCalled = true
		exitCode = code
	})
	defer patches.Reset()

	main()

	require.True(t, runCalled)
	require.True(t, exitCalled)
	require.Equal(t, expectedCode, exitCode)
	require.NotNil(t, receivedCmd)
	require.Equal(t, "test-cmd", receivedCmd.Use)
}
