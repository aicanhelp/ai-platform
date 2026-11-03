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

package zlog

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
)

func TestLogWriteSyncerMirrorsStdoutAndFile(t *testing.T) {
	modes := []string{"both", "file", " BOTH ", ""}
	for _, mode := range modes {
		t.Run(mode, func(t *testing.T) {
			dir := t.TempDir()
			var stdout bytes.Buffer
			conf := &LogConfig{
				Level:       "info",
				EncoderType: "console",
				Path:        dir,
				FileName:    "root.log",
				MaxSize:     10,
				OutMod:      mode,
			}
			logger := zap.New(zapcore.NewCore(
				getEncoder(conf),
				logWriteSyncer(conf, zapcore.AddSync(&stdout)),
				zapcore.InfoLevel,
			)).Sugar()
			logger.Info("dual-output-marker")
			_ = logger.Sync()

			if !strings.Contains(stdout.String(), "dual-output-marker") {
				t.Fatalf("stdout = %q, want marker", stdout.String())
			}
			body, err := os.ReadFile(filepath.Join(dir, "root.log"))
			if err != nil {
				t.Fatalf("read log file: %v", err)
			}
			if !strings.Contains(string(body), "dual-output-marker") {
				t.Fatalf("file = %q, want marker", string(body))
			}
		})
	}
}
