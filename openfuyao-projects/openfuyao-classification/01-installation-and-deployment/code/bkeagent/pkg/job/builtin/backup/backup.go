package backup

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/pkg/errors"

	"gopkg.openfuyao.cn/bkeagent/pkg/executor/exec"
	"gopkg.openfuyao.cn/bkeagent/pkg/job/builtin/plugin"
	"gopkg.openfuyao.cn/bkeagent/utils"
	"gopkg.openfuyao.cn/bkeagent/utils/log"
)

const (
	Name = "Backup"
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
func (c *BackupPlugin) Execute(commands []string) ([]string, error) {
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
		if err := os.MkdirAll(saveTo, 0755); err != nil {
			return nil, errors.Wrapf(err, "create saveTo dir %q failed", saveTo)
		}
	}
	// backup dirs
	for _, dir := range backupDirs {
		if dir == "" {
			continue
		}
		if !filepath.IsAbs(dir) {
			return nil, errors.Errorf("backupDirs must be absolute path, but got %q", dir)
		}
		if !utils.IsDir(dir) {
			log.Warnf("backup dir %q is not a dir, skip backup", dir)
			continue
		}
		if !utils.Exists(dir) {
			log.Warnf("backup dir %q not exists, skip backup", dir)
			continue
		}
		srcDir := filepath.Join(dir, "*")
		// backup dir use cp
		// todo compress dir
		targetDir := filepath.Join(saveTo, dir)
		if err := os.MkdirAll(targetDir, 0755); err != nil {
			return nil, errors.Wrapf(err, "create backup dir %q failed", targetDir)
		}
		log.Infof("backup dir %q to %q", dir, targetDir)
		if out, err := c.exec.ExecuteCommandWithOutput("sh", "-c", fmt.Sprintf("cp -rf %s %s", srcDir, targetDir)); err != nil {
			return nil, errors.Wrapf(err, "backup dir %q failed, out: %s", dir, out)
		}
	}

	for _, file := range backupFiles {
		if file == "" {
			continue
		}
		if !filepath.IsAbs(file) {
			return nil, errors.Errorf("backupFiles must be absolute path, but got %q", file)
		}
		if !utils.Exists(file) {
			log.Warnf("backup file %q not exists, skip backup", file)
			continue
		}
		// backup file use cp
		targetFile := filepath.Join(saveTo, file)
		targetFileDir := filepath.Dir(targetFile)
		if err := os.MkdirAll(targetFileDir, 0755); err != nil {
			return nil, errors.Wrapf(err, "create backup file dir %q failed", targetFileDir)
		}
		log.Infof("backup file %q to %q", file, targetFile)
		if err := c.exec.ExecuteCommand("sh", "-c", fmt.Sprintf("cp -f %s %s", file, targetFile)); err != nil {
			return nil, errors.Wrapf(err, "backup file %q failed", file)
		}
	}

	return nil, nil
}
