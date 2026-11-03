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

// Package config defines console-service-config
package config

// ConsoleServiceConfig console-service config read from console service configmap.data
type ConsoleServiceConfig struct {
	OAuthServerHost    string `json:"oauth-server-host"`
	ConsoleServerHost  string `json:"console-server-host"`
	ConsoleWebsiteHost string `json:"console-website-host"`
	AlertHost          string `json:"alert-host"`
	MarketPlaceHost    string `json:"marketplace-host"`
	PluginHost         string `json:"plugin-management-host"`
	ApplicationHost    string `json:"application-management-host"`
	UserManagementHost string `json:"user-management-host"`
	MonitoringHost     string `json:"monitoring-host"`
	WebTerminalHost    string `json:"webterminal-host"`
	InsecureSkipVerify string `json:"insecure-skip-verify"`
	ServerName         string `json:"server-name"`
}
