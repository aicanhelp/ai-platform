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

package kubelet

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/executor/exec"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/log"
)

const defaultDeliveredFileBasePath = "/etc/kubernetes"

func (kp *kubeletPlugin) processKubeletFiles(files []confv1beta1.FileSpec, config map[string]string) error {
	if len(files) == 0 {
		return nil
	}

	basePath := strings.TrimSpace(config["fileBasePath"])
	if basePath == "" {
		basePath = defaultDeliveredFileBasePath
	}

	for i, file := range files {
		if err := kp.writeDeliveredFile(file, basePath, config); err != nil {
			return fmt.Errorf("write delivered file[%d] %q: %w", i, file.Path, err)
		}
	}
	log.Infof("processed %d delivered file(s) from KubeletConfig", len(files))
	return nil
}

func (kp *kubeletPlugin) processKubeletCommands(commands []confv1beta1.CommandSpec, config map[string]string) error {
	if len(commands) == 0 {
		return nil
	}

	for i, command := range commands {
		if err := kp.executeDeliveredCommand(command); err != nil {
			return fmt.Errorf("execute delivered command[%d] %q: %w", i, command.Command, err)
		}
	}
	log.Infof("processed %d delivered command(s) from KubeletConfig", len(commands))
	return nil
}

func (kp *kubeletPlugin) writeDeliveredFile(file confv1beta1.FileSpec, basePath string, config map[string]string) error {
	filePath := strings.TrimSpace(file.Path)
	if filePath == "" {
		return fmt.Errorf("file path is required")
	}

	resolvedPath, err := resolveDeliveredFilePath(filePath, basePath)
	if err != nil {
		return err
	}

	content, err := substituteDeliveredContent(file.Content, config, kp.exec)
	if err != nil {
		return fmt.Errorf("substitute file content: %w", err)
	}

	if err := os.MkdirAll(filepath.Dir(resolvedPath), utils.RwxRxRx); err != nil {
		return fmt.Errorf("create parent directory: %w", err)
	}

	fileMode, err := parseFilePermissions(file.Permissions)
	if err != nil {
		return err
	}

	if err := os.WriteFile(resolvedPath, []byte(content), fileMode); err != nil {
		return fmt.Errorf("write file: %w", err)
	}

	// File content is written before chown. If chown fails, the file remains on disk with
	// the mode from WriteFile but may have an incorrect owner; the error is returned to
	// the caller without rolling back the write.
	if err := applyFileOwner(kp.exec, file.Owner, resolvedPath); err != nil {
		return err
	}

	log.Infof("wrote delivered file to %s", resolvedPath)
	return nil
}

func (kp *kubeletPlugin) executeDeliveredCommand(command confv1beta1.CommandSpec) error {
	commandLine := strings.TrimSpace(command.Command)
	if commandLine == "" {
		return fmt.Errorf("command is required")
	}

	execCommand, execArgs := deliveredCommandInvocation(commandLine, command.Args)
	workingDir := strings.TrimSpace(command.WorkingDir)

	var (
		out string
		err error
	)
	if workingDir != "" {
		out, err = kp.exec.ExecuteCommandWithCombinedOutputInDir(workingDir, execCommand, execArgs...)
	} else {
		out, err = kp.exec.ExecuteCommandWithCombinedOutput(execCommand, execArgs...)
	}
	if err != nil {
		return fmt.Errorf("command failed: %w, out: %s", err, out)
	}

	if strings.TrimSpace(out) != "" {
		log.Debugf("delivered command output: %s", strings.TrimSpace(out))
	}
	return nil
}

// deliveredCommandInvocation selects how to run a delivered command.
// When args is non-empty, commandLine is used as the executable and args as its arguments.
// Otherwise commandLine is passed to "sh -c" so multiline shell scripts (if/then, pipelines,
// etc.) run as a single script without line-joining or cd prepending.
func deliveredCommandInvocation(commandLine string, args []string) (string, []string) {
	if len(args) > 0 {
		return commandLine, args
	}
	return "sh", []string{"-c", commandLine}
}

// resolveDeliveredFilePath resolves the on-disk path for a delivered file.
// Relative paths are joined under basePath; path traversal ("..") is rejected.
// Absolute paths are accepted as-is after filepath.Clean and are intentionally not
// restricted to basePath or a whitelist directory, so authors may target arbitrary
// system paths (e.g. /etc/kubernetes/...). Ensure KubeletConfig CR content is trusted.
func resolveDeliveredFilePath(filePath, basePath string) (string, error) {
	filePath = strings.TrimSpace(filePath)
	if filePath == "" {
		return "", fmt.Errorf("file path is required")
	}

	if filepath.IsAbs(filePath) {
		cleaned := filepath.Clean(filePath)
		return cleaned, nil
	}

	basePath = filepath.Clean(basePath)
	resolved := filepath.Clean(filepath.Join(basePath, filePath))
	rel, err := filepath.Rel(basePath, resolved)
	if err != nil {
		return "", fmt.Errorf("invalid file path %q: %w", filePath, err)
	}
	if rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
		return "", fmt.Errorf("invalid file path %q: path traversal not allowed", filePath)
	}
	return resolved, nil
}

// parseFilePermissions parses an optional octal permission string (e.g. "0644", "0755")
// into os.FileMode. An empty string defaults to RwRR.
func parseFilePermissions(permissions string) (os.FileMode, error) {
	permissions = strings.TrimSpace(permissions)
	if permissions == "" {
		return RwRR, nil
	}

	value, err := strconv.ParseUint(permissions, 8, 32)
	if err != nil {
		return 0, fmt.Errorf("invalid file permissions %q: %w", permissions, err)
	}
	return os.FileMode(value), nil
}

func substituteDeliveredContent(content string, config map[string]string, executor exec.Executor) (string, error) {
	if config["enableVariableSubstitution"] != "true" {
		return content, nil
	}

	substitutor := &VariableSubstitutor{
		config: config,
		exec:   executor,
	}
	return substitutor.Substitute(content)
}

func applyFileOwner(executor exec.Executor, owner, filePath string) error {
	owner = strings.TrimSpace(owner)
	if owner == "" {
		return nil
	}

	out, err := executor.ExecuteCommandWithCombinedOutput("chown", owner, filePath)
	if err != nil {
		return fmt.Errorf("chown %s %s failed: %w, out: %s", owner, filePath, err, out)
	}
	return nil
}
