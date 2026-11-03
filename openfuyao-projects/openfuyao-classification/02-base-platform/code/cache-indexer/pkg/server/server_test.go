package server

import (
	"context"
	"errors"
	"net"
	"net/http"
	"testing"
	"time"

	"github.com/go-logr/logr/testr"

	"gitcode.com/openFuyao/cache-indexer/pkg/config"
)

const (
	testServeAddr    = "127.0.0.1:0"
	testReadyTimeout = 2 * time.Second
	testPollDelay    = 20 * time.Millisecond
)

func TestNormalizeHTTPConfig_FillsDefaults(t *testing.T) {
	cfg := normalizeHTTPConfig(config.HTTPConfig{})
	if cfg.Addr != config.DefaultHTTPAddr {
		t.Fatalf("Addr=%q want=%q", cfg.Addr, config.DefaultHTTPAddr)
	}
	if cfg.MaxHeaderBytes != config.DefaultHTTPMaxHeaderBytes {
		t.Fatalf("MaxHeaderBytes=%d want=%d", cfg.MaxHeaderBytes, config.DefaultHTTPMaxHeaderBytes)
	}
	if cfg.MaxHitRateBodyBytes != config.DefaultHTTPMaxHitRateBodyBytes {
		t.Fatalf("MaxHitRateBodyBytes=%d want=%d", cfg.MaxHitRateBodyBytes, config.DefaultHTTPMaxHitRateBodyBytes)
	}
}

func TestNew_UsesNormalizedAddr(t *testing.T) {
	srv := New(config.HTTPConfig{}, testr.New(t), &stubScore{})
	if srv.Addr() != config.DefaultHTTPAddr {
		t.Fatalf("Addr=%q want=%q", srv.Addr(), config.DefaultHTTPAddr)
	}
}

func TestShutdown_StopsServeLoop(t *testing.T) {
	ln, err := net.Listen("tcp", testServeAddr)
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	defer ln.Close()

	srv := New(config.HTTPConfig{Addr: ln.Addr().String()}, testr.New(t), &stubScore{})
	serveErr := make(chan error, 1)
	go func() {
		serveErr <- srv.srv.Serve(ln)
	}()

	waitServerReady(t, srv.Addr())

	ctx, cancel := context.WithTimeout(context.Background(), testReadyTimeout)
	defer cancel()
	if err := srv.Shutdown(ctx); err != nil {
		t.Fatalf("Shutdown: %v", err)
	}

	err = <-serveErr
	if err != nil && !errors.Is(err, http.ErrServerClosed) {
		t.Fatalf("Serve error: %v", err)
	}
}

func waitServerReady(t *testing.T, addr string) {
	t.Helper()
	deadline := time.Now().Add(testReadyTimeout)
	for time.Now().Before(deadline) {
		conn, err := net.DialTimeout("tcp", addr, testPollDelay)
		if err == nil {
			_ = conn.Close()
			return
		}
		time.Sleep(testPollDelay)
	}
	t.Fatalf("server %s not ready", addr)
}
