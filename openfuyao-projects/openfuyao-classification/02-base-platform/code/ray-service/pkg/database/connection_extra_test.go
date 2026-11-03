/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package database

import (
	"io"
	"os"
	"path/filepath"
	"testing"

	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
	"gorm.io/gorm"

	"openfuyao-rayservice/pkg/cmd/config"
	dbruntime "openfuyao-rayservice/pkg/database/runtime"
	"openfuyao-rayservice/pkg/zlog"
)

// newPanicFatalLogger returns a logger whose Fatal methods panic instead of
// calling os.Exit, so fatal code paths become testable.
func newPanicFatalLogger() *zap.SugaredLogger {
	core := zapcore.NewCore(
		zapcore.NewConsoleEncoder(zap.NewProductionEncoderConfig()),
		zapcore.AddSync(io.Discard),
		zapcore.InfoLevel,
	)
	return zap.New(core, zap.WithFatalHook(zapcore.WriteThenPanic)).Sugar()
}

// closeDBConnection closes the underlying connection pool so that the db
// file is released (required on Windows before removing temp dirs).
func closeDBConnection(t *testing.T, db *gorm.DB) {
	t.Helper()
	sqlDB, err := db.DB()
	if err != nil {
		t.Fatalf("failed to get sql db: %v", err)
	}
	if err := sqlDB.Close(); err != nil {
		t.Fatalf("failed to close sql db: %v", err)
	}
}

func TestConnectRemovesExistingDbFile(t *testing.T) {
	dir := t.TempDir()
	dbFile := filepath.Join(dir, "test-remove.db")

	// create a valid db file first so that the removal branch is exercised
	first := Connect(&config.RunConfig{
		DbConfig: &dbruntime.DbConfig{DbFilePath: dbFile, RemoveExistDb: false},
	})
	if first == nil {
		t.Fatalf("Connect() = nil, want a valid db connection")
	}
	closeDBConnection(t, first)

	second := Connect(&config.RunConfig{
		DbConfig: &dbruntime.DbConfig{DbFilePath: dbFile, RemoveExistDb: true},
	})
	if second == nil {
		t.Fatalf("Connect() = nil, want a valid db connection")
	}
	closeDBConnection(t, second)

	if _, err := os.Stat(dbFile); err != nil {
		t.Errorf("db file should exist after connect, got error: %v", err)
	}
}

func TestConnectKeepsExistingDbFile(t *testing.T) {
	dir := t.TempDir()
	dbFile := filepath.Join(dir, "test-keep.db")

	first := Connect(&config.RunConfig{
		DbConfig: &dbruntime.DbConfig{DbFilePath: dbFile, RemoveExistDb: false},
	})
	if first == nil {
		t.Fatalf("Connect() = nil, want a valid db connection")
	}
	closeDBConnection(t, first)

	second := Connect(&config.RunConfig{
		DbConfig: &dbruntime.DbConfig{DbFilePath: dbFile, RemoveExistDb: false},
	})
	if second == nil {
		t.Fatalf("Connect() = nil, want a valid db connection")
	}
	closeDBConnection(t, second)
}

func TestConnectFatalOnInvalidDbFile(t *testing.T) {
	dir := t.TempDir()
	// a non-sqlite file cannot be opened by the driver, which hits the fatal path
	dbFile := filepath.Join(dir, "invalid.db")
	if err := os.WriteFile(dbFile, []byte("not a sqlite file"), 0600); err != nil {
		t.Fatalf("failed to write invalid db file: %v", err)
	}

	oldLogger := zlog.Logger
	zlog.Logger = newPanicFatalLogger()
	defer func() { zlog.Logger = oldLogger }()

	defer func() {
		if r := recover(); r == nil {
			t.Errorf("Connect() should panic (fatal) when the db file cannot be opened")
		}
	}()

	Connect(&config.RunConfig{
		DbConfig: &dbruntime.DbConfig{DbFilePath: dbFile, RemoveExistDb: false},
	})
}

func TestConnectFatalOnInvalidDbPath(t *testing.T) {
	dir := t.TempDir()
	// make the db path a non-empty directory: os.Remove fails (warn branch is
	// covered) and the sqlite driver cannot open it, which hits the fatal path.
	dbPath := filepath.Join(dir, "invalid-dir.db")
	innerDir := filepath.Join(dbPath, "inner")
	if err := os.MkdirAll(innerDir, 0750); err != nil {
		t.Fatalf("failed to create invalid db path dir: %v", err)
	}
	if err := os.WriteFile(filepath.Join(innerDir, "keep.txt"), []byte("keep"), 0600); err != nil {
		t.Fatalf("failed to create file inside invalid db path: %v", err)
	}

	oldLogger := zlog.Logger
	zlog.Logger = newPanicFatalLogger()
	defer func() { zlog.Logger = oldLogger }()

	defer func() {
		if r := recover(); r == nil {
			t.Errorf("Connect() should panic (fatal) when the db path cannot be opened")
		}
	}()

	Connect(&config.RunConfig{
		DbConfig: &dbruntime.DbConfig{DbFilePath: dbPath, RemoveExistDb: true},
	})
}
