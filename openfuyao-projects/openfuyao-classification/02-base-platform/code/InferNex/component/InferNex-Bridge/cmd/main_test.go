package main

import (
	"crypto/tls"
	"flag"
	"os"
	"testing"
)

func TestTLSOptsFromHTTP2(t *testing.T) {
	t.Parallel()
	if opts := tlsOptsFromHTTP2(true); opts != nil {
		t.Fatalf("expected nil tls opts when http2 enabled, got %v", opts)
	}
	opts := tlsOptsFromHTTP2(false)
	if len(opts) != 1 {
		t.Fatalf("expected one tls opt when http2 disabled, got %d", len(opts))
	}
	cfg := &tls.Config{}
	opts[0](cfg)
	if len(cfg.NextProtos) != 1 || cfg.NextProtos[0] != "http/1.1" {
		t.Fatalf("expected http/1.1 only, got %v", cfg.NextProtos)
	}
}

func TestNewMetricsOptions(t *testing.T) {
	t.Parallel()
	f := cliFlags{
		metricsAddr:     ":8443",
		secureMetrics:   true,
		metricsCertPath: "/certs",
		metricsCertName: "tls.crt",
		metricsCertKey:  "tls.key",
	}
	opts := newMetricsOptions(f, nil)
	if opts.BindAddress != ":8443" || !opts.SecureServing {
		t.Fatalf("unexpected metrics options: %+v", opts)
	}
	if opts.CertDir != "/certs" || opts.CertName != "tls.crt" || opts.KeyName != "tls.key" {
		t.Fatalf("expected cert options set, got %+v", opts)
	}
	if opts.FilterProvider == nil {
		t.Fatal("expected authn/authz filter provider when secure metrics enabled")
	}
}

func TestControllerTemplateNamespace(t *testing.T) {
	t.Run("prefer INFERNEX_TEMPLATE_NAMESPACE", func(t *testing.T) {
		t.Setenv("INFERNEX_TEMPLATE_NAMESPACE", "infernex-ns")
		t.Setenv("POD_NAMESPACE", "pod-ns")
		if got := controllerTemplateNamespace(); got != "infernex-ns" {
			t.Fatalf("expected infernex-ns, got %q", got)
		}
	})
	t.Run("fallback POD_NAMESPACE", func(t *testing.T) {
		t.Setenv("INFERNEX_TEMPLATE_NAMESPACE", "")
		t.Setenv("POD_NAMESPACE", "pod-ns")
		if got := controllerTemplateNamespace(); got != "pod-ns" {
			t.Fatalf("expected pod-ns, got %q", got)
		}
	})
}

func TestParseCLI_DefaultAndOverrides(t *testing.T) {
	t.Run("defaults", func(t *testing.T) {
		oldFS := flag.CommandLine
		oldArgs := os.Args
		t.Cleanup(func() {
			flag.CommandLine = oldFS
			os.Args = oldArgs
		})
		flag.CommandLine = flag.NewFlagSet("test-defaults", flag.ContinueOnError)
		os.Args = []string{"cmd.test"}
		f := parseCLI()
		if f.metricsAddr != "0" || f.probeAddr != ":8081" || !f.secureMetrics || f.enableHTTP2 {
			t.Fatalf("unexpected default flags: %+v", f)
		}
	})

	t.Run("overrides", func(t *testing.T) {
		oldFS := flag.CommandLine
		oldArgs := os.Args
		t.Cleanup(func() {
			flag.CommandLine = oldFS
			os.Args = oldArgs
		})
		flag.CommandLine = flag.NewFlagSet("test-overrides", flag.ContinueOnError)
		os.Args = []string{
			"cmd.test",
			"--metrics-bind-address=:8443",
			"--health-probe-bind-address=:18081",
			"--leader-elect=true",
			"--metrics-secure=false",
			"--webhook-cert-path=/tmp/webhook",
			"--metrics-cert-path=/tmp/metrics",
			"--enable-http2=true",
		}
		f := parseCLI()
		if f.metricsAddr != ":8443" || f.probeAddr != ":18081" || !f.enableLeaderElection || f.secureMetrics || !f.enableHTTP2 {
			t.Fatalf("unexpected parsed flags: %+v", f)
		}
		if f.webhookCertPath != "/tmp/webhook" || f.metricsCertPath != "/tmp/metrics" {
			t.Fatalf("expected cert paths parsed: %+v", f)
		}
	})
}

func TestNewWebhookServer(t *testing.T) {
	t.Parallel()
	srv := newWebhookServer(cliFlags{}, nil)
	if srv == nil {
		t.Fatal("expected webhook server instance")
	}
	srv = newWebhookServer(cliFlags{
		webhookCertPath: "/tmp/certs",
		webhookCertName: "tls.crt",
		webhookCertKey:  "tls.key",
	}, nil)
	if srv == nil {
		t.Fatal("expected webhook server instance with cert config")
	}
}

func TestNewMetricsOptions_Insecure(t *testing.T) {
	t.Parallel()
	opts := newMetricsOptions(cliFlags{metricsAddr: ":8080", secureMetrics: false}, nil)
	if opts.SecureServing || opts.FilterProvider != nil {
		t.Fatalf("expected insecure metrics options, got %+v", opts)
	}
}

func TestDisableHTTP2TLSOpt(t *testing.T) {
	t.Parallel()
	cfg := &tls.Config{}
	disableHTTP2TLSOpt()(cfg)
	if len(cfg.NextProtos) != 1 || cfg.NextProtos[0] != "http/1.1" {
		t.Fatalf("expected http/1.1 only, got %v", cfg.NextProtos)
	}
}

