/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *           http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package common

import (
	"os"
	"path/filepath"
	"strings"

	"installer/pkg/constants"
	"installer/pkg/zlog"
)

// InitLog is init log
func InitLog(debug bool) {
	if !debug {
		return
	}
	logCfg := &zlog.LogConfig{
		Level:       constants.LogLevelDebug,
		EncoderType: "console",
		Path:        zlog.DefaultLogPath,
		FileName:    "installer.log",
		MaxSize:     20,
		MaxBackups:  0,
		MaxAge:      7,
		LocalTime:   false,
		Compress:    true,
		OutMod:      constants.LogOutModThree,
	}
	exePath, err := os.Executable()
	if err == nil {
		serviceName := strings.TrimSuffix(filepath.Base(exePath), filepath.Ext(filepath.Base(exePath)))
		logCfg.Path = filepath.Join(zlog.DefaultLogPath, serviceName)
	}
	zlog.Logger = zlog.GetLogger(logCfg)
}
