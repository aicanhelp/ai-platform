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

package selfupdate

import (
	"context"
	_ "embed"
	"fmt"
	"os"
	"path"
	"strings"

	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/executor/exec"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/job/builtin/plugin"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils"
	agentdownload "gopkg.openfuyao.cn/cluster-api-provider-bke/utils/bkeagent/download"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/log"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/version"
)

const (
	Name            = "SelfUpdate"
	agentBinaryName = "bkeagent"
	DefaultAgentUrl = "http://http.bocloud.k8s:40080/files/bkeagent-latest-linux-{.arch}"
	// restartDelaySeconds gives controller time to persist command status before service stop.
	restartDelaySeconds = 3
	// RwxRxRx is the permission of the directory
	RwxRxRx = 0755
)

var restartScript = path.Join(utils.AgentScripts, "update.sh")

//go:embed update.sh
var updateScript string

type UpdatePlugin struct {
	exec exec.Executor
}

func New(exec exec.Executor) plugin.Plugin {
	return &UpdatePlugin{
		exec: exec,
	}
}

func (u UpdatePlugin) Name() string {
	return Name
}

func (u UpdatePlugin) Param() map[string]plugin.PluginParam {
	return map[string]plugin.PluginParam{
		"agentUrl": {
			Key:         "agentUrl",
			Value:       DefaultAgentUrl,
			Required:    false,
			Default:     DefaultAgentUrl,
			Description: "Agent download url used to replace existing BKEAgent binary",
		},
	}
}

func (u UpdatePlugin) Execute(ctx context.Context, commands []string) ([]string, error) {
	u.exec = exec.Bound(u.exec, ctx)

	runtimeParam, err := plugin.ParseCommands(u, commands)
	if err != nil {
		return nil, err
	}

	agentURL := strings.TrimSpace(runtimeParam["agentUrl"])
	if agentURL == "" {
		return nil, fmt.Errorf("agentUrl is required")
	}
	binPath, err := downloadAgentBinary(agentURL)
	if err != nil {
		return nil, err
	}

	if !u.NeedUpdate(binPath) {
		log.Infof("The agent is up to date, skip!")
		return nil, nil
	}

	if !utils.Exists(utils.AgentScripts) {
		if err := os.MkdirAll(utils.AgentScripts, RwxRxRx); err != nil {
			return nil, err
		}
	}

	if !utils.Exists(restartScript) {
		if err := os.WriteFile(restartScript, []byte(updateScript), RwxRxRx); err != nil {
			return nil, err
		}
	}

	// Delay stop/restart a bit, so command status can be persisted before this process exits.
	updateCmd := fmt.Sprintf("nohup /bin/sh -c 'sleep %d; %s %s' >/dev/null 2>&1 &",
		restartDelaySeconds, restartScript, binPath)
	if err := u.exec.ExecuteCommand("/bin/sh", "-c", updateCmd); err != nil {
		return nil, err
	}
	return nil, nil
}

func (u UpdatePlugin) NeedUpdate(binPath string) bool {
	if !utils.Exists(binPath) {
		return true
	}
	out, err := u.exec.ExecuteCommandWithCombinedOutput("/bin/sh", "-c",
		fmt.Sprintf("chmod +x %s && %s -v", binPath, binPath))
	if err != nil || strings.TrimSpace(out) == "" {
		return true
	}
	downloadGitCommitID := gitCommitFromVersionOutput(out)
	if downloadGitCommitID == "" {
		return true
	}
	return version.GitCommitID != downloadGitCommitID
}

// gitCommitFromVersionOutput parses GitCommitId from `bkeagent -v` (PrintVersion) output.
// A single-line value without ":" is accepted for backward compatibility with legacy callers.
func gitCommitFromVersionOutput(out string) string {
	for _, line := range strings.Split(out, "\n") {
		if idx := strings.Index(line, "GitCommitId:"); idx >= 0 {
			return strings.TrimSpace(line[idx+len("GitCommitId:"):])
		}
	}
	trimmed := strings.TrimSpace(out)
	if trimmed != "" && !strings.Contains(trimmed, "\n") && !strings.Contains(trimmed, ":") {
		return trimmed
	}
	return ""
}

// downloadAgentBinary downloads the agent from agentUrl into AgentBin and names it bkeagent.
func downloadAgentBinary(agentURL string) (string, error) {
	if err := agentdownload.ExecDownload(agentURL, utils.AgentBin, agentBinaryName, "0755"); err != nil {
		return "", fmt.Errorf("download agent from %q: %w", agentURL, err)
	}
	binPath := path.Join(utils.AgentBin, agentBinaryName)
	if err := os.Chmod(binPath, RwxRxRx); err != nil {
		return "", fmt.Errorf("chmod %s: %w", binPath, err)
	}
	log.Infof("downloaded bkeagent binary to %s", binPath)
	return binPath, nil
}
