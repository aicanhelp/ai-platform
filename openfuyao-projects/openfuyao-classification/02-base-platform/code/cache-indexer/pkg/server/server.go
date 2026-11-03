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

package server

import (
	"context"
	"errors"
	"net/http"

	"github.com/go-logr/logr"

	"gitcode.com/openFuyao/cache-indexer/pkg/config"
	"gitcode.com/openFuyao/cache-indexer/pkg/score"
)

// HTTPServer wraps a net/http.Server with a logr-aware lifecycle.
type HTTPServer struct {
	cfg config.HTTPConfig
	log logr.Logger
	srv *http.Server
}

// New builds the HTTP server. The score service is injected so tests can
// swap in a stub; S1 wires the no-op implementation.
func New(cfg config.HTTPConfig, log logr.Logger, scoreSvc score.Service) *HTTPServer {
	cfg = normalizeHTTPConfig(cfg)
	mux := newMux(log, scoreSvc, cfg)
	return &HTTPServer{
		cfg: cfg,
		log: log,
		srv: &http.Server{
			Addr:              cfg.Addr,
			Handler:           mux,
			ReadHeaderTimeout: cfg.ReadHeaderTimeout,
			ReadTimeout:       cfg.ReadTimeout,
			WriteTimeout:      cfg.WriteTimeout,
			IdleTimeout:       cfg.IdleTimeout,
			MaxHeaderBytes:    cfg.MaxHeaderBytes,
		},
	}
}

func normalizeHTTPConfig(cfg config.HTTPConfig) config.HTTPConfig {
	if cfg.Addr == "" {
		cfg.Addr = config.DefaultHTTPAddr
	}
	if cfg.ShutdownTimeout <= 0 {
		cfg.ShutdownTimeout = config.DefaultHTTPShutdownTimeout
	}
	if cfg.ReadHeaderTimeout <= 0 {
		cfg.ReadHeaderTimeout = config.DefaultHTTPReadHeaderTimeout
	}
	if cfg.ReadTimeout <= 0 {
		cfg.ReadTimeout = config.DefaultHTTPReadTimeout
	}
	if cfg.WriteTimeout <= 0 {
		cfg.WriteTimeout = config.DefaultHTTPWriteTimeout
	}
	if cfg.IdleTimeout <= 0 {
		cfg.IdleTimeout = config.DefaultHTTPIdleTimeout
	}
	if cfg.MaxHeaderBytes <= 0 {
		cfg.MaxHeaderBytes = config.DefaultHTTPMaxHeaderBytes
	}
	if cfg.MaxHitRateBodyBytes <= 0 {
		cfg.MaxHitRateBodyBytes = config.DefaultHTTPMaxHitRateBodyBytes
	}
	return cfg
}

// Addr returns the bind address actually configured.
func (s *HTTPServer) Addr() string { return s.cfg.Addr }

// Start runs the server in the calling goroutine and returns when the
// listener stops; callers usually wrap it in `go s.Start()`.
func (s *HTTPServer) Start() error {
	if err := s.srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
		return err
	}
	return nil
}

// Shutdown gracefully terminates the server, bounded by ctx.
func (s *HTTPServer) Shutdown(ctx context.Context) error {
	return s.srv.Shutdown(ctx)
}
