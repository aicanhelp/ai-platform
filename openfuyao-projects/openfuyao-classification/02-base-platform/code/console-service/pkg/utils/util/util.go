/*
 *
 *  * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 *  * openFuyao is licensed under Mulan PSL v2.
 *  * You can use this software according to the terms and conditions of the Mulan PSL v2.
 *  * You may obtain a copy of Mulan PSL v2 at:
 *  *          http://license.coscl.org.cn/MulanPSL2
 *  * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 *  * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 *  * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 *  * See the Mulan PSL v2 for more details.
 *
 */

/*
Package util include console-service level util function
*/
package util

import (
	"fmt"
	"os"
	"strings"
	"time"

	"k8s.io/client-go/kubernetes"

	"console-service/pkg/constant"
	"console-service/pkg/server/config"
	"console-service/pkg/utils/k8sutil"
	"console-service/pkg/zlog"
)

// ClearByte clear byte slice by setting every index to zero
func ClearByte(value []byte) {
	for i := range value {
		value[i] = 0
	}
}

const (
	oauthServerHost    = "oauth-server-host"
	consoleServiceHost = "console-service-host"
	consoleWebsiteHost = "console-website-host"
	alertHost          = "alert-host"
	monitoringHost     = "monitoring-host"
	webTerminalHost    = "webterminal-host"
	marketplaceHost    = "marketplace-host"
	pluginHost         = "plugin-management-host"
	applicationHost    = "application-management-host"
	userManagementHost = "user-management-host"
	insecureSkipVerify = "insecure-skip-verify"
	serverName         = "server-name"
)

var configDir = "/etc/console-service/openfuyao-config"

const (
	configAPIMaxAttempts = 5
	configAPIRetryBase   = 200 * time.Millisecond
)

// GetConsoleServiceConfig returns the console-service used to verify the apiserver. Prefers the service-account
// bundle mounted by kubelet (no apiserver call), then kube-root-ca ConfigMap with retries.
func GetConsoleServiceConfig(c kubernetes.Interface) (*config.ConsoleServiceConfig, error) {
	cfg, err := loadConfigFromPodFiles()
	if err == nil {
		return cfg, nil
	}
	zlog.Warnf("failed to load console-service config from %s, fallback to console-service-config ConfigMap: %v", configDir, err)

	var lastErr error
	for attempt := 0; attempt < configAPIMaxAttempts; attempt++ {
		if attempt > 0 {
			time.Sleep(configAPIRetryBase * time.Duration(1<<attempt))
		}
		configMap, err := k8sutil.GetConfigMap(c, constant.ConsoleServiceConfigmap,
			constant.ConsoleServiceDefaultNamespace)
		if err != nil {
			lastErr = err
			continue
		}
		return parseConfig(configMap.Data), nil
	}

	zlog.Warnf("failed to load console-service config from console-service-config ConfigMap after max retries %d: %v", configAPIMaxAttempts, lastErr)
	return nil, lastErr
}

// loadConfigFromPodFiles returns config parsed from configDir when at least one key file exists with content.
func loadConfigFromPodFiles() (*config.ConsoleServiceConfig, error) {
	entries, err := os.ReadDir(configDir)
	if err != nil {
		return nil, err
	}
	data := make(map[string]string)
	for _, entry := range entries {
		if entry.IsDir() || entry.Name() == "..data" {
			continue
		}
		filePath := configDir + "/" + entry.Name()
		content, err := os.ReadFile(filePath)
		if err != nil {
			zlog.Warnf("failed to read console-service config %s: %v", filePath, err)
			continue
		}
		v := strings.TrimSpace(string(content))
		if v == "" {
			continue
		}
		data[entry.Name()] = v
	}
	if len(data) == 0 {
		return nil, fmt.Errorf("no console-service config available (mount %s empty)", configDir)
	}
	return parseConfig(data), nil
}

func parseConfig(consoleServiceConfigMap map[string]string) *config.ConsoleServiceConfig {
	return &config.ConsoleServiceConfig{
		OAuthServerHost:    consoleServiceConfigMap[oauthServerHost],
		ConsoleServerHost:  consoleServiceConfigMap[consoleServiceHost],
		ConsoleWebsiteHost: consoleServiceConfigMap[consoleWebsiteHost],
		AlertHost:          consoleServiceConfigMap[alertHost],
		MonitoringHost:     consoleServiceConfigMap[monitoringHost],
		WebTerminalHost:    consoleServiceConfigMap[webTerminalHost],
		PluginHost:         consoleServiceConfigMap[pluginHost],
		ApplicationHost:    consoleServiceConfigMap[applicationHost],
		MarketPlaceHost:    consoleServiceConfigMap[marketplaceHost],
		UserManagementHost: consoleServiceConfigMap[userManagementHost],
		InsecureSkipVerify: consoleServiceConfigMap[insecureSkipVerify],
		ServerName:         consoleServiceConfigMap[serverName],
	}
}
