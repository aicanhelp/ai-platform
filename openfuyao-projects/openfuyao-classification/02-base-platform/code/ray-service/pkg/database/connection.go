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

// Package zlog provides a set of utilities for logging configurations and operations.
// It utilizes the zap logging library to configure and manage log outputs efficiently.
// This package supports different log output modes including console, file, and a combination of both.
// It includes configuration setup for logging levels, formats, and output destinations using
// the lumberjack library for log file rotation and management.
package database

import (
	"os"

	"gorm.io/driver/sqlite"
	"gorm.io/gorm"
	"gorm.io/gorm/logger"

	"openfuyao-rayservice/pkg/cmd/config"
	"openfuyao-rayservice/pkg/zlog"
)

// Connect returns a new database connection
func Connect(cfg *config.RunConfig) *gorm.DB {
	if cfg.DbConfig.RemoveExistDb {
		err := os.Remove(cfg.DbConfig.DbFilePath)
		if err != nil && !os.IsNotExist(err) {
			zlog.Warnf("Failed to remove existing database file: %v", err)
		}
	}

	db, err := gorm.Open(sqlite.Open(cfg.DbConfig.DbFilePath), &gorm.Config{
		Logger: logger.Default.LogMode(logger.Info),
	})

	if err != nil {
		zlog.FormatFatal("Create DB Connection ERROR! %v", err)
		return nil
	}
	db1, err := db.DB()
	if err != nil {
		zlog.FormatFatal("Connect DB Connection ERROR! %v", err)
		return nil
	}

	err = db1.Ping()
	if err != nil {
		zlog.FormatFatal("Ping DB ERROR! %v", err)
		return nil
	}

	zlog.Info("connected to database successfully on DbFilePath:" + cfg.DbConfig.DbFilePath)

	return db
}
