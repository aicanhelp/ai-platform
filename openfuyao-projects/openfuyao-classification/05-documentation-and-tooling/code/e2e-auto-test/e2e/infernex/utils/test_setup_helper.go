package utils

import (
	"context"
	"fmt"
	"time"

	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/k8s"
)

// HermesTestEnv holds shared test environment state for Hermes Router tests.
// GatewayURL is resolved lazily on first inference request and stores the full
// inference endpoint URL, including the request path.
type HermesTestEnv struct {
	Client     *k8s.K8SClient
	Ctx        context.Context
	Cancel     context.CancelFunc
	HelmConfig map[string]*ReleaseConfig
	Namespace  string
	GatewayURL string
}

// SetupHermesRouterTest initializes a HermesTestEnv with k8s client, helm config,
// and namespace. The inference gateway URL is resolved lazily after deployment,
// when a test first sends traffic. The caller should use DeferCleanup(env.Cancel)
// to release the context.
func SetupHermesRouterTest(timeout time.Duration) (*HermesTestEnv, error) {
	return setupHermesRouterTest(timeout, false)
}

// SetupHermesRouterTestWithoutGateway is kept as a compatibility alias for tests
// that only validate deployment behavior and never send inference traffic.
func SetupHermesRouterTestWithoutGateway(timeout time.Duration) (*HermesTestEnv, error) {
	return setupHermesRouterTest(timeout, false)
}

func setupHermesRouterTest(timeout time.Duration, resolveGateway bool) (*HermesTestEnv, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)

	k8sClient, err := k8s.NewK8SClientFromLocalKubeconfig()
	if err != nil {
		cancel()
		return nil, fmt.Errorf("create k8s client failed: %v", err)
	}

	helmConfig, err := LoadHelmConfig()
	if err != nil {
		cancel()
		return nil, fmt.Errorf("load helm config failed: %v", err)
	}

	cfg, ok := helmConfig["infernex"]
	if !ok || cfg == nil {
		cancel()
		return nil, fmt.Errorf("helm config missing infernex entry")
	}

	err = k8sClient.CreateNamespaceIfNotExists(ctx, cfg.Namespace)
	if err != nil {
		cancel()
		return nil, fmt.Errorf("create namespace failed: %v", err)
	}

	err = CleanupLegacyReleaseConflictsForInfernex(ctx, k8sClient, cfg.ReleaseName, cfg.Namespace)
	if err != nil {
		cancel()
		return nil, fmt.Errorf("cleanup legacy conflicts failed: %v", err)
	}

	gatewayURL := ""
	if resolveGateway {
		gatewayURL, err = ResolveGatewayURL(ctx, k8sClient, []string{cfg.Namespace, "ai-inference"})
		if err != nil {
			cancel()
			return nil, fmt.Errorf("resolve gateway url failed: %v", err)
		}
	}

	return &HermesTestEnv{
		Client:     k8sClient,
		Ctx:        ctx,
		Cancel:     cancel,
		HelmConfig: helmConfig,
		Namespace:  cfg.Namespace,
		GatewayURL: gatewayURL,
	}, nil
}
