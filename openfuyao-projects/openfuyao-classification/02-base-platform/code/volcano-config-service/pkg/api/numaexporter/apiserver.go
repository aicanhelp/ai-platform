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

// Package numaexporter provides functionality to initialize and start the HTTP server for numa_exporter application.
package numaexporter

import (
	"net/http"

	"github.com/alecthomas/kingpin/v2"
	"github.com/prometheus/common/promlog"
	"github.com/prometheus/common/version"
	"github.com/prometheus/exporter-toolkit/web"
	"github.com/prometheus/exporter-toolkit/web/kingpinflag"

	"volcano-config-service/pkg/cmd/metrics"
	"volcano-config-service/pkg/zlog"
)

// InitServer is the main entry point of the application to start the server.
func InitServer() {
	var (
		metricsPath = kingpin.Flag(
			"web.metrics-path",
			"Path under which to expose metrics.",
		).Default("/metrics").String()
		maxConcurrentScrapes = kingpin.Flag(
			"web.max-concurrent-scrapes",
			"Maximum number of concurrent scrape requests. Use 0 to disable.").Default("20").Int()
		toolkitFlags = kingpinflag.AddFlags(kingpin.CommandLine, ":9201")
	)

	kingpin.Version(version.Print("numa_exporter"))
	kingpin.HelpFlag.Short('h')
	kingpin.Parse()

	zlog.FormatInfo("Starting numa_exporter. Version: %s", version.Info())

	promlogConfig := &promlog.Config{}
	logger := promlog.New(promlogConfig)

	h, err := metrics.NewHandler(*maxConcurrentScrapes, logger)
	if err != nil {
		zlog.FormatError("Create handler failed. Error: %v", err)
		return
	}

	http.Handle(*metricsPath, h)

	server := &http.Server{}
	if err := web.ListenAndServe(server, toolkitFlags, logger); err != nil {
		zlog.FormatError("Web server startup failed. Error: %v", err)
		return
	}
}
