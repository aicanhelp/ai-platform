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

package backup

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/pkg/errors"

	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/executor/exec"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/job/builtin/plugin"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/log"
)

const (
	Name = "Backup"
	// RwxRxRx is the permission of the directory
	RwxRxRx = 0755
)

var (
	defaultBackupSaveTo = filepath.Join(utils.Workspace, "backup")
)

type BackupPlugin struct {
	exec exec.Executor
}

func New(exec exec.Executor) *BackupPlugin {
	return &BackupPlugin{
		exec: exec,
	}
}

func (c *BackupPlugin) Name() string {
	return Name
}

func (c *BackupPlugin) Param() map[string]plugin.PluginParam {
	return map[string]plugin.PluginParam{
		"backupDirs": {
			Key:         "backupDirs",
			Value:       "",
			Required:    true,
			Default:     "",
			Description: "backup dirs, split by ',' ",
		},
		"backupFiles": {
			Key:         "backupFiles",
			Value:       "",
			Required:    false,
			Default:     "",
			Description: "backup files, split by ',' ",
		},
		"saveTo": {
			Key:         "saveTo",
			Value:       "",
			Required:    false,
			Default:     defaultBackupSaveTo,
			Description: "dirs to save backup",
		},
	}
}

// Execute is the entry point of the plugin
// backupDir backs up a single directory
func (c *BackupPlugin) backupDir(dir, saveTo string) error {
	if dir == "" {
		return nil
	}
	if !filepath.IsAbs(dir) {
		return errors.Errorf("backupDirs must be absolute path, but got %q", dir)
	}
	if !utils.IsDir(dir) {
		log.Warnf("backup dir %q is not a dir, skip backup", dir)
		return nil
	}
	if !utils.Exists(dir) {
		log.Warnf("backup dir %q not exists, skip backup", dir)
		return nil
	}
	srcDir := filepath.Join(dir, "*")
	targetDir := filepath.Join(saveTo, dir)
	if err := os.MkdirAll(targetDir, RwxRxRx); err != nil {
		return errors.Wrapf(err, "create backup dir %q failed", targetDir)
	}
	log.Infof("backup dir %q to %q", dir, targetDir)
	if out, err := c.exec.ExecuteCommandWithOutput("sh", "-c", fmt.Sprintf("cp -rf %s %s", srcDir, targetDir)); err != nil {
		return errors.Wrapf(err, "backup dir %q failed, out: %s", dir, out)
	}
	return nil
}

// backupFile backs up a single file
func (c *BackupPlugin) backupFile(file, saveTo string) error {
	if file == "" {
		return nil
	}
	if !filepath.IsAbs(file) {
		return errors.Errorf("backupFiles must be absolute path, but got %q", file)
	}
	if !utils.Exists(file) {
		log.Warnf("backup file %q not exists, skip backup", file)
		return nil
	}
	targetFile := filepath.Join(saveTo, file)
	targetFileDir := filepath.Dir(targetFile)
	if err := os.MkdirAll(targetFileDir, RwxRxRx); err != nil {
		return errors.Wrapf(err, "create backup file dir %q failed", targetFileDir)
	}
	log.Infof("backup file %q to %q", file, targetFile)
	if err := c.exec.ExecuteCommand("sh", "-c", fmt.Sprintf("cp -f %s %s", file, targetFile)); err != nil {
		return errors.Wrapf(err, "backup file %q failed", file)
	}
	return nil
}

func (c *BackupPlugin) Execute(ctx context.Context, commands []string) ([]string, error) {
	cc := *c
	cc.exec = exec.Bound(c.exec, ctx)
	c = &cc

	commandMap, err := plugin.ParseCommands(c, commands)
	if err != nil {
		return nil, err
	}
	saveTo := commandMap["saveTo"]

	tBackupDirs := commandMap["backupDirs"]
	tBackupFiles := commandMap["backupFiles"]
	if tBackupDirs == "" && tBackupFiles == "" {
		log.Warnf("backupDirs and backupFiles is empty, skip backup")
		return nil, nil
	}

	backupDirs := strings.Split(tBackupDirs, ",")
	backupFiles := strings.Split(tBackupFiles, ",")
	if (backupFiles == nil || len(backupFiles) == 0) || (backupDirs == nil || len(backupDirs) == 0) {
		log.Warnf("backupDirs or backupFiles is empty, skip backup")
		return nil, nil
	}

	if !filepath.IsAbs(saveTo) {
		return nil, errors.Errorf("saveTo must be absolute path, but got %q", saveTo)
	}
	// create saveTo dir
	if !utils.Exists(saveTo) || !utils.IsDir(saveTo) {
		if err := os.MkdirAll(saveTo, RwxRxRx); err != nil {
			return nil, errors.Wrapf(err, "create saveTo dir %q failed", saveTo)
		}
	}
	// backup dirs
	for _, dir := range backupDirs {
		if err := c.backupDir(dir, saveTo); err != nil {
			return nil, err
		}
	}

	for _, file := range backupFiles {
		if err := c.backupFile(file, saveTo); err != nil {
			return nil, err
		}
	}

	return nil, nil
}
