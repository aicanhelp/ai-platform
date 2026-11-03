/******************************************************************
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package phaseutil

import (
	"fmt"
	"os"
	"regexp"
	"strings"

	"github.com/pkg/errors"

	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/log"
)

// BKEAgentServiceFilePermission is the file permission for rendered bkeagent.service.
const BKEAgentServiceFilePermission = 0644

// AgentLogLevelCustomExtraKey is the customExtra key used to set bkeagent business log level.
// Example: customExtra.agentLogLevel: DEBUG
const AgentLogLevelCustomExtraKey = "agentLogLevel"

// NormalizeAgentLogLevel returns the canonical ologger level for customExtra.agentLogLevel.
// Empty or non-allowlisted values return ("", false).
func NormalizeAgentLogLevel(level string) (string, bool) {
	return log.NormalizeAllowedLevel(level)
}

// RenderBKEAgentServiceContent applies cluster NTPServer, AgentHealthPort and optional
// customExtra.agentLogLevel to service unit bytes.
// HTTP-downloaded units and /bkeagent.service.tmpl share the same --ntpserver= / --health-port= placeholders.
// --log-level is only injected when agentLogLevel is non-empty, so default units stay compatible with older binaries.
func RenderBKEAgentServiceContent(bkeCluster *bkev1beta1.BKECluster, raw []byte) []byte {
	ntpServer := ""
	healthPort := ""
	logLevel := ""
	if bkeCluster != nil && bkeCluster.Spec.ClusterConfig != nil {
		ntpServer = bkeCluster.Spec.ClusterConfig.Cluster.NTPServer
		healthPort = bkeCluster.Spec.ClusterConfig.Cluster.AgentHealthPort
		if bkeCluster.Spec.ClusterConfig.CustomExtra != nil {
			logLevel = bkeCluster.Spec.ClusterConfig.CustomExtra[AgentLogLevelCustomExtraKey]
		}
	}

	content := strings.ReplaceAll(string(raw), "--ntpserver=", fmt.Sprintf("--ntpserver=%s", ntpServer))
	content = strings.ReplaceAll(content, "--health-port=", fmt.Sprintf("--health-port=%s", healthPort))
	content = applyAgentLogLevel(content, logLevel)
	return []byte(content)
}

// applyAgentLogLevel injects --log-level=<level> into ExecStart when level is set.
// Existing --log-level flags/placeholders are stripped first, then the new flag is appended when non-empty.
// When level is empty or not in the ologger allowlist, --log-level is removed so default units
// stay compatible with older binaries and untrusted customExtra values are not written into ExecStart.
func applyAgentLogLevel(content, logLevel string) string {
	content = stripLogLevelFlag(content)
	canonical, ok := log.NormalizeAllowedLevel(logLevel)
	if !ok {
		return content
	}
	return appendFlagToExecStart(content, fmt.Sprintf("--log-level=%s", canonical))
}

var logLevelFlagPattern = regexp.MustCompile(`\s*--log-level=\S*`)

func stripLogLevelFlag(content string) string {
	return logLevelFlagPattern.ReplaceAllString(content, "")
}

// appendFlagToExecStart appends flagValue to the first ExecStart= directive.
// systemd line continuations ending with '\' are followed to the last physical
// line of that ExecStart block so multi-line units stay valid.
func appendFlagToExecStart(content, flagValue string) string {
	lines := strings.Split(content, "\n")
	start := -1
	for i, line := range lines {
		if strings.HasPrefix(strings.TrimSpace(line), "ExecStart=") {
			start = i
			break
		}
	}
	if start < 0 {
		return content
	}

	end := start
	for end < len(lines)-1 && endsWithLineContinuation(lines[end]) {
		end++
	}

	line := strings.TrimRight(lines[end], " \t\r")
	if strings.HasSuffix(line, "\\") {
		line = strings.TrimRight(strings.TrimSuffix(line, "\\"), " \t")
	}
	lines[end] = line + " " + flagValue
	return strings.Join(lines, "\n")
}

func endsWithLineContinuation(line string) bool {
	return strings.HasSuffix(strings.TrimRight(line, " \t\r"), "\\")
}

// WriteRenderedBKEAgentServiceFile renders service content and writes to destPath.
func WriteRenderedBKEAgentServiceFile(bkeCluster *bkev1beta1.BKECluster, destPath string, raw []byte) error {
	if err := os.WriteFile(destPath, RenderBKEAgentServiceContent(bkeCluster, raw), BKEAgentServiceFilePermission); err != nil {
		return errors.Wrap(err, "write bkeagent.service")
	}
	return nil
}

// RenderBKEAgentServiceFile renders /bkeagent.service.tmpl like EnsureBKEAgent.prepareServiceFile.
func RenderBKEAgentServiceFile(bkeCluster *bkev1beta1.BKECluster, servicePath string) error {
	file, err := os.ReadFile("/bkeagent.service.tmpl")
	if err != nil {
		return errors.Wrap(err, "read /bkeagent.service.tmpl")
	}
	return WriteRenderedBKEAgentServiceFile(bkeCluster, servicePath, file)
}
