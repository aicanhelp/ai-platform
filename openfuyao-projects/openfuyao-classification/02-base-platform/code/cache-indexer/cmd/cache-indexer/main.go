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

// Command cache-indexer is the entrypoint binary. It wires config, logging,
// the HTTP server, and the discovery / ingest planes.
package main

import (
	"context"
	"fmt"
	"os"
	"os/signal"
	"strings"
	"syscall"

	"github.com/go-logr/logr"
	"k8s.io/client-go/kubernetes"

	"gitcode.com/openFuyao/cache-indexer/pkg/blockkey"
	"gitcode.com/openFuyao/cache-indexer/pkg/config"
	"gitcode.com/openFuyao/cache-indexer/pkg/discovery"
	indexl1 "gitcode.com/openFuyao/cache-indexer/pkg/index/l1"
	indexl3 "gitcode.com/openFuyao/cache-indexer/pkg/index/l3"
	ingestl1 "gitcode.com/openFuyao/cache-indexer/pkg/ingest/l1"
	ingestl3 "gitcode.com/openFuyao/cache-indexer/pkg/ingest/l3"
	"gitcode.com/openFuyao/cache-indexer/pkg/logging"
	"gitcode.com/openFuyao/cache-indexer/pkg/score"
	"gitcode.com/openFuyao/cache-indexer/pkg/server"
)

func main() {
	os.Exit(run())
}

type appDeps struct {
	log       logr.Logger
	l1Indexer indexl1.Indexer
	l3Indexer indexl3.Indexer
	registry  discovery.Registry
}

type discoveryPlane struct {
	manager discovery.Manager
	l1Sub   ingestl1.SubscriptionManager
	l3Sub   ingestl3.MooncakePollTargetUpdater
}

func run() int {
	cfg := config.Load()

	log, err := logging.New(cfg.Log.Level)
	if err != nil {
		// write to stderr and exit when logger init failed
		if _, writeErr := os.Stderr.WriteString("failed to init logger: " + err.Error() + "\n"); writeErr != nil {
			return 1
		}
		return 1
	}
	if err := runWithLogger(cfg, log.WithName("cache-indexer")); err != nil {
		return 1
	}
	return 0
}

func runWithLogger(cfg *config.Settings, log logr.Logger) error {
	log.Info("cache-indexer starting")

	deps := newAppDeps(log)
	scoreSvc := score.New(
		log.WithName("score"),
		newBuilder(log, cfg.BlockKey),
		deps.l1Indexer,
		deps.l3Indexer,
		deps.registry,
	)
	srv := server.New(cfg.HTTP, log.WithName("http"), scoreSvc)

	rootCtx, rootCancel := context.WithCancel(context.Background())
	plane, err := startDiscoveryPlane(rootCtx, cfg.Discovery, cfg.BlockKey, cfg.Ingest, deps)
	if err != nil {
		log.Error(err, "discovery is required; aborting startup")
		rootCancel()
		return err
	}
	defer func() {
		plane.stop()
		rootCancel()
	}()

	serveErr := startHTTPServer(srv)

	// Keep this message stable for deployment smoke tests.
	log.Info("server started", "addr", srv.Addr())
	if err := waitForShutdown(log, serveErr); err != nil {
		return err
	}

	shutdownCtx, cancel := context.WithTimeout(context.Background(), cfg.HTTP.ShutdownTimeout)
	defer cancel()
	if err := srv.Shutdown(shutdownCtx); err != nil {
		log.Error(err, "http server shutdown error")
	}
	log.Info("server stopped")
	return nil
}

func newAppDeps(log logr.Logger) appDeps {
	return appDeps{
		log:       log,
		l1Indexer: indexl1.New(log.WithName("index-l1")),
		l3Indexer: indexl3.New(log.WithName("index-l3")),
		registry:  discovery.NewRegistry(log.WithName("discovery-registry")),
	}
}

func newBuilder(log logr.Logger, cfg config.BlockKeyConfig) blockkey.Builder {
	builderCfg, err := newBlockKeyConfig(cfg)
	if err != nil {
		log.Error(err, "builder disabled; L1 hit rate stays 0")
		return nil
	}
	builder, err := blockkey.New(log.WithName("blockkey"), builderCfg)
	if err != nil {
		log.Error(err, "builder disabled; L1 hit rate stays 0")
		return nil
	}
	return builder
}

func newBlockKeyConfig(cfg config.BlockKeyConfig) (blockkey.Config, error) {
	algo, err := parseHashAlgo(cfg.PrefixCachingHashAlgo)
	if err != nil {
		return blockkey.Config{}, err
	}
	return blockkey.Config{
		PythonHashSeed: cfg.PythonHashSeed,
		Algo:           algo,
		EmitInt64:      cfg.UseIntBlockHashes,
	}, nil
}

func parseHashAlgo(value string) (blockkey.HashAlgo, error) {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case "", "sha256_cbor":
		return blockkey.HashSHA256CBOR, nil
	case "xxhash_cbor":
		return blockkey.HashXXHash3CBOR, nil
	default:
		return blockkey.HashSHA256CBOR, fmt.Errorf("unsupported prefix caching hash algo %q", value)
	}
}

func startHTTPServer(srv *server.HTTPServer) <-chan error {
	serveErr := make(chan error, 1)
	go func() {
		serveErr <- srv.Start()
	}()
	return serveErr
}

func waitForShutdown(log logr.Logger, serveErr <-chan error) error {
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	defer signal.Stop(sigCh)

	select {
	case sig := <-sigCh:
		log.Info("received signal, shutting down", "signal", sig.String())
		return nil
	case err := <-serveErr:
		if err != nil {
			log.Error(err, "http server exited unexpectedly")
			return err
		}
		return nil
	}
}

func startDiscoveryPlane(
	rootCtx context.Context,
	dcfg config.DiscoveryConfig,
	bcfg config.BlockKeyConfig,
	icfg config.IngestConfig,
	deps appDeps,
) (*discoveryPlane, error) {
	if dcfg.Namespace == "" {
		return nil, fmt.Errorf("discovery namespace is empty (set POD_NAMESPACE)")
	}

	client, err := newKubeClient()
	if err != nil {
		return nil, fmt.Errorf("build Kubernetes client: %w", err)
	}
	plane := newDiscoveryPlane(dcfg, bcfg, icfg, client, deps)
	if err := plane.manager.Start(rootCtx); err != nil {
		plane.stop()
		return nil, fmt.Errorf("start discovery manager: %w", err)
	}
	return plane, nil
}

func newDiscoveryPlane(
	dcfg config.DiscoveryConfig,
	bcfg config.BlockKeyConfig,
	icfg config.IngestConfig,
	client kubernetes.Interface,
	deps appDeps,
) *discoveryPlane {
	l1Sub := ingestl1.NewWithConfig(deps.log.WithName("ingest-l1"), deps.l1Indexer, icfg.L1)
	l3Sub := ingestl3.NewWithConfig(deps.log.WithName("ingest-l3"), deps.l3Indexer, ingestl3.PollerConfig{
		Scheme:       icfg.L3.Scheme,
		PollInterval: icfg.L3.PollInterval,
		HTTPTimeout:  icfg.L3.HTTPTimeout,
		EmitInt64:    bcfg.UseIntBlockHashes,
	})
	subscriber := newSubscriberAdapter(deps.log.WithName("subscriber"), l1Sub, l3Sub)
	rec := discovery.NewReconciler(deps.log.WithName("discovery-reconciler"), subscriber)
	return &discoveryPlane{
		manager: discovery.NewManager(dcfg, deps.log.WithName("discovery"), client, rec, deps.registry),
		l1Sub:   l1Sub,
		l3Sub:   l3Sub,
	}
}

func (p *discoveryPlane) stop() {
	if p == nil {
		return
	}
	if p.manager != nil {
		_ = p.manager.Stop()
	}
	if p.l1Sub != nil {
		p.l1Sub.Stop()
	}
	if p.l3Sub != nil {
		p.l3Sub.Stop()
	}
}
