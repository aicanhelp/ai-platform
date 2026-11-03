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

package scriptutil

import (
	"fmt"
	"net"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"time"

	"github.com/pkg/errors"

	bkenet "gopkg.openfuyao.cn/cluster-api-provider-bke/common/utils/net"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/log"
)

// ScriptConfig 脚本配置
type ScriptConfig struct {
	ScriptName string            `json:"scriptName"`
	Order      int               `json:"order"`
	Params     map[string]string `json:"params,omitempty"`
}

// GetCurrentNodeIP 获取当前节点IP
func GetCurrentNodeIP() (string, error) {
	ips, err := bkenet.GetAllInterfaceIP()
	if err != nil {
		log.Warnf("get interface IPs failed: %v", err)
		return "", errors.Wrapf(err, "failed to get interface IPs")
	}

	for _, ipStr := range ips {
		tmpIP, _, err := net.ParseCIDR(ipStr)
		if err != nil {
			continue
		}
		ipStr := tmpIP.String()
		if ipStr == "127.0.0.1" || ipStr == "::1" {
			continue
		}
		log.Infof("auto-detected nodeIP=%s", ipStr)
		return ipStr, nil
	}

	log.Warnf("no valid node IP found")
	return "", errors.New("no valid node IP found")
}

// SanitizeFileName 清理文件名，将特殊字符替换为下划线
func SanitizeFileName(value string) string {
	replacer := strings.NewReplacer(":", "_", "/", "_", "\\", "_", " ", "_")
	return replacer.Replace(value)
}

// PreviewScript 预览脚本内容，如果超过最大长度则截断
func PreviewScript(script string, max int) string {
	if max <= 0 {
		return ""
	}
	if len(script) <= max {
		return script
	}
	return script[:max] + " ... (truncated)"
}

// RenderScriptWithParams 参数渲染：将参数值替换到脚本模板中
func RenderScriptWithParams(scriptContent string, params map[string]string) string {
	renderedScript := scriptContent
	re := regexp.MustCompile(`\$\{([a-zA-Z_][a-zA-Z0-9_]*)\}`)

	renderedScript = re.ReplaceAllStringFunc(renderedScript, func(match string) string {
		// 提取参数名（去掉 ${}）
		paramName := match[2 : len(match)-1]
		// 如果参数存在，替换为参数值
		if value, ok := params[paramName]; ok {
			return value
		}
		// 如果参数不存在，保留原样
		return match
	})

	return renderedScript
}

func WriteRenderedScriptToDisk(scriptStoreDir, scriptName, nodeIP, renderedScript string) (string, error) {
	if err := os.MkdirAll(scriptStoreDir, 0755); err != nil {
		return "", errors.Wrapf(err, "failed to create script dir: %s", scriptStoreDir)
	}

	safeNodeIP := SanitizeFileName(nodeIP)
	baseName := scriptName
	ext := ""
	if idx := strings.LastIndex(scriptName, "."); idx > 0 {
		baseName = scriptName[:idx]
		ext = scriptName[idx:]
	}

	fileName := fmt.Sprintf("%s-%s-%d%s", baseName, safeNodeIP, time.Now().UnixNano(), ext)
	scriptPath := filepath.Join(scriptStoreDir, fileName)

	if err := os.WriteFile(scriptPath, []byte(renderedScript), 0600); err != nil {
		return "", errors.Wrapf(err, "failed to write script file: %s", scriptPath)
	}
	// 最小可执行权限：仅拥有者可读可执行
	if err := os.Chmod(scriptPath, 0500); err != nil {
		return "", errors.Wrapf(err, "failed to chmod script file: %s", scriptPath)
	}

	return scriptPath, nil
}
