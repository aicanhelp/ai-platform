package main

import (
	"os"
	"path/filepath"
	"testing"

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/rest"
	igwapiv1 "sigs.k8s.io/gateway-api-inference-extension/api/v1"
	gwapiv1 "sigs.k8s.io/gateway-api/apis/v1"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/envtest"

	infernexv1alpha1 "gitcode.com/openFuyao/InferNex/api/v1alpha1"
)

func testManagerOptions(f cliFlags) ctrl.Options {
	tlsOpts := tlsOptsFromHTTP2(f.enableHTTP2)
	return ctrl.Options{
		Scheme:                 scheme,
		Metrics:                newMetricsOptions(f, tlsOpts),
		WebhookServer:          newWebhookServer(f, tlsOpts),
		HealthProbeBindAddress: f.probeAddr,
		LeaderElection:         f.enableLeaderElection,
		LeaderElectionID:       leaderElectionID,
	}
}

func newTestManager(cfg *rest.Config, f cliFlags) (ctrl.Manager, error) {
	return ctrl.NewManager(cfg, testManagerOptions(f))
}

func TestManagerOptions(t *testing.T) {
	t.Parallel()
	opts := testManagerOptions(cliFlags{
		metricsAddr:          "0",
		probeAddr:            ":8081",
		enableLeaderElection: true,
		enableHTTP2:          true,
	})
	if opts.Scheme == nil || opts.Metrics.BindAddress != "0" || opts.HealthProbeBindAddress != ":8081" {
		t.Fatalf("unexpected manager options: %+v", opts)
	}
	if !opts.LeaderElection || opts.LeaderElectionID != leaderElectionID {
		t.Fatalf("expected leader election configured, got election=%v id=%q", opts.LeaderElection, opts.LeaderElectionID)
	}
	if opts.WebhookServer == nil {
		t.Fatal("expected webhook server configured")
	}
}

func TestSchemeRegistersCoreTypes(t *testing.T) {
	t.Parallel()
	cases := []struct {
		gv   schema.GroupVersion
		kind string
	}{
		{infernexv1alpha1.GroupVersion, "InferNexService"},
		{infernexv1alpha1.GroupVersion, "InferNexServiceConfig"},
		{schema.GroupVersion{Group: gwapiv1.GroupName, Version: gwapiv1.GroupVersion.Version}, "Gateway"},
		{schema.GroupVersion{Group: igwapiv1.GroupName, Version: igwapiv1.GroupVersion.Version}, "InferencePool"},
	}
	for _, tc := range cases {
		if !scheme.Recognizes(tc.gv.WithKind(tc.kind)) {
			t.Fatalf("scheme missing %s", tc.kind)
		}
	}
}

func TestManagerBootstrapWithEnvtest(t *testing.T) {
	binDir := firstEnvTestBinaryDir()
	if binDir == "" {
		t.Skip("envtest binaries not found; run make setup-envtest")
	}
	absBinDir, err := filepath.Abs(binDir)
	if err != nil {
		t.Fatalf("resolve envtest binary dir: %v", err)
	}
	// envtest prefers KUBEBUILDER_ASSETS over BinaryAssetsDirectory; normalize both.
	t.Setenv("KUBEBUILDER_ASSETS", absBinDir)

	testEnv := &envtest.Environment{
		BinaryAssetsDirectory: absBinDir,
		CRDDirectoryPaths:     []string{filepath.Join("..", "config", "crd", "bases")},
		ErrorIfCRDPathMissing:   true,
	}
	cfg, err := testEnv.Start()
	if err != nil {
		t.Fatalf("envtest start: %v", err)
	}
	t.Cleanup(func() {
		_ = testEnv.Stop()
	})

	flags := cliFlags{metricsAddr: "0", probeAddr: "0", secureMetrics: false}
	t.Setenv("INFERNEX_TEMPLATE_NAMESPACE", "infernex-system")
	t.Setenv("ENABLE_WEBHOOKS", "true")

	mgr, err := newTestManager(cfg, flags)
	if err != nil {
		t.Fatalf("newTestManager error: %v", err)
	}
	if err := installProbes(mgr); err != nil {
		t.Fatalf("installProbes error: %v", err)
	}
	if err := registerControllersAndWebhooks(mgr); err != nil {
		t.Fatalf("registerControllersAndWebhooks error: %v", err)
	}
}

func firstEnvTestBinaryDir() string {
	if assets := os.Getenv("KUBEBUILDER_ASSETS"); assets != "" {
		if _, err := os.Stat(filepath.Join(assets, "etcd")); err == nil {
			return assets
		}
	}
	return findEnvTestBinaryDir(filepath.Join("..", "bin", "k8s"))
}

func findEnvTestBinaryDir(basePath string) string {
	entries, err := os.ReadDir(basePath)
	if err != nil {
		return ""
	}
	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		candidate := filepath.Join(basePath, entry.Name())
		if _, err := os.Stat(filepath.Join(candidate, "etcd")); err == nil {
			return candidate
		}
		if nested := findEnvTestBinaryDir(candidate); nested != "" {
			return nested
		}
	}
	return ""
}
