/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
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
	"k8s.io/component-base/cli"

	"openfuyao/oauth-server/cmd/oauth-server/app"
	"openfuyao/oauth-server/pkg/zlog"
)

func main() {
	cmd := app.NewOAuthServerCommand()
	code := cli.Run(cmd)
	if code != 0 {
		zlog.LogFatalf("Application exited with error code: %d", code)
	}
}
