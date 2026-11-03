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

// Package metrics provides an HTTP handler for handling Prometheus metrics.
// It includes a `Handler` struct that implements the `http.Handler` interface
// and is responsible for creating and handling metrics. The metrics handler uses the Prometheus
// client library to collect and expose metrics data.
package metrics

import (
	"fmt"
	stdlog "log"
	"net/http"

	"github.com/go-kit/log"
	"github.com/go-kit/log/level"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"github.com/prometheus/common/version"

	"volcano-config-service/pkg/collector"
)

// Handler provides an HTTP handler for serving Prometheus metrics.
type Handler struct {
	metricsRegistry *prometheus.Registry
	metricsHandler  http.Handler
	maxRequests     int
	logger          log.Logger
}

// NewHandler creates a new `Handler` instance and initializes the metrics handler.
func NewHandler(maxRequests int, logger log.Logger) (*Handler, error) {
	h := &Handler{
		metricsRegistry: prometheus.NewRegistry(),
		maxRequests:     maxRequests,
		logger:          logger,
	}

	if err := h.createMetricsHandler(); err != nil {
		return nil, fmt.Errorf("couldn't create metrics handler: %s", err)
	}

	return h, nil
}

func (h *Handler) createMetricsHandler() error {
	nc, err := collector.NewNumaCollector()
	if err != nil {
		return fmt.Errorf("couldn't create collector: %s", err)
	}

	h.metricsRegistry.MustRegister(version.NewCollector("numa_exporter"))
	if err := h.metricsRegistry.Register(nc); err != nil {
		return fmt.Errorf("couldn't register numa collector: %s", err)
	}

	h.metricsHandler = promhttp.HandlerFor(
		h.metricsRegistry,
		promhttp.HandlerOpts{
			ErrorLog:            stdlog.New(log.NewStdlibAdapter(level.Error(h.logger)), "", 0),
			ErrorHandling:       promhttp.ContinueOnError,
			MaxRequestsInFlight: h.maxRequests,
		},
	)

	return nil
}

// ServeHTTP implements the http.Handler interface to serve HTTP requests.
// It delegates the handling to the metricsHandler.
func (h *Handler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	h.metricsHandler.ServeHTTP(w, r)
}
