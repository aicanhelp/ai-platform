/******************************************************************
 * Copyright (c) 2024 Bocloud Technologies Co., Ltd.
 * cluster-api-provider-bke is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package main

import (
	"context"
	"crypto/tls"
	"flag"
	"fmt"
	"net"
	"net/http"
	"net/http/pprof"
	"os"
	"strings"
	"time"

	"github.com/pkg/errors"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/kubernetes"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	_ "k8s.io/client-go/plugin/pkg/client/auth"
	"k8s.io/client-go/tools/record"
	"k8s.io/client-go/util/workqueue"
	clusterv1 "sigs.k8s.io/cluster-api/api/v1beta1"
	bootstrapv1 "sigs.k8s.io/cluster-api/bootstrap/kubeadm/api/v1beta1"
	"sigs.k8s.io/cluster-api/controllers/remote"
	controlv1beta1 "sigs.k8s.io/cluster-api/controlplane/kubeadm/api/v1beta1"
	clusterexpv1 "sigs.k8s.io/cluster-api/exp/api/v1beta1"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/controller"
	"sigs.k8s.io/controller-runtime/pkg/healthz"
	"sigs.k8s.io/controller-runtime/pkg/log/zap"

	agentv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkeagent/v1beta1"
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	configv1alpha1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/v1alpha1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/initialize/versions"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/common/security"
	commonutils "gopkg.openfuyao.cn/cluster-api-provider-bke/common/utils"
	capbkecontrollers "gopkg.openfuyao.cn/cluster-api-provider-bke/controllers/capbke"
	clusterversioncontrollers "gopkg.openfuyao.cn/cluster-api-provider-bke/controllers/clusterversion"
	releaseimagecontrollers "gopkg.openfuyao.cn/cluster-api-provider-bke/controllers/releaseimage"
	upgradepathcontrollers "gopkg.openfuyao.cn/cluster-api-provider-bke/controllers/upgradepath"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/command"
	pkgkube "gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/kube"
	bkemetrics "gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/metrics"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/oci"
	releasemanifest "gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/release/manifest"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/statusmanage"
	pathstore "gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/upgradepath"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/config"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/nodeutil"
	scriptshelper "gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/scriptshelper"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/log"
	v "gopkg.openfuyao.cn/cluster-api-provider-bke/version"
	capbkewebhooks "gopkg.openfuyao.cn/cluster-api-provider-bke/webhooks/capbke"
)

const (
	// EventBroadcasterBurstSize is the burst size for the event broadcaster
	EventBroadcasterBurstSize = 10000
	// EventBroadcasterLRUCacheSize is the LRU cache size for the event broadcaster
	EventBroadcasterLRUCacheSize = 4096 * 2
	// EventBroadcasterMaxIntervalInSeconds is the max interval in seconds for the event broadcaster
	EventBroadcasterMaxIntervalInSeconds = 30
)

const (
	// FastSlowRateLimiterSlowDuration is the slow duration for the fast-slow rate limiter (60 seconds)
	FastSlowRateLimiterSlowDuration = 60 * time.Second
	// FastSlowRateLimiterFastDuration is the fast duration for the fast-slow rate limiter (2 seconds)
	FastSlowRateLimiterFastDuration = 2 * time.Second
	// FastSlowRateLimiterRetryCount is the retry count for the fast-slow rate limiter (10 retries)
	FastSlowRateLimiterRetryCount = 10
)

const (
	// securityInitRetryInterval is the retry interval for the eager password
	// encryption key load at startup.
	securityInitRetryInterval = 2 * time.Second
	// securityInitTimeout bounds the eager password encryption key load:
	// transient failures during the management-cluster bootstrap window (the
	// service VIP is unreachable until the CNI is ready) are retried within
	// this budget; persistent failures then exit for visibility.
	securityInitTimeout = 5 * time.Minute
)

var scheme = runtime.NewScheme()

func setupLogger() *log.Logger {
	return log.With("component", "setup")
}

func init() {
	utilruntime.Must(clientgoscheme.AddToScheme(scheme))
	utilruntime.Must(clusterv1.AddToScheme(scheme))
	utilruntime.Must(clusterexpv1.AddToScheme(scheme))
	utilruntime.Must(agentv1beta1.AddToScheme(scheme))
	utilruntime.Must(controlv1beta1.AddToScheme(scheme))
	utilruntime.Must(bootstrapv1.AddToScheme(scheme))

	utilruntime.Must(bkev1beta1.AddToScheme(scheme))
	utilruntime.Must(configv1alpha1.AddToScheme(scheme))
	//+kubebuilder:scaffold:scheme
}

func logStartupConfig() {
	// Keep the startup message but avoid mutating time.Local in init/main.
	// Mutating time.Local can race with logger background goroutines.
	if _, err := time.LoadLocation("Asia/Shanghai"); err == nil {
		setupLogger().Info("Set timezone to Asia/Shanghai")
	}
	statusmanage.LogInitConfig()
}

func main() {
	if _, err := versions.Load(); err != nil {
		setupLogger().Errorf("failed to load embedded versions manifest: %v", err)
		os.Exit(1)
	}
	setupLogger().Infof(
		"Loaded versions manifest: manifestVersion=%s, kubernetes=%s, etcd=%s, containerd=%s, openFuyao=%s, etcdImageTag=%s, pauseImageTag=%s",
		versions.ManifestVersion(),
		versions.KubernetesVersion(),
		versions.EtcdVersion(),
		versions.ContainerdVersion(),
		versions.OpenFuyaoVersion(),
		versions.EtcdImageTag(),
		versions.PauseImageTag(),
	)
	logStartupConfig()
	setupLogger().Info("Starting the BKE Cluster API Provider")
	printVersionInfo()

	printManifestsBuildInfo()

	mgr, tracker := createManager()
	ctx := ctrl.SetupSignalHandler()

	setupSecurity(ctx, mgr)

	setupControllers(ctx, mgr, tracker)
	setupWebhooks(mgr)

	// TTL janitor: purge finished commands after spec.TTLSecondsAfterFinished
	// and strip finalizers from terminating ones (agent cannot do either).
	if err := mgr.Add(&command.CommandTTLJanitor{Client: mgr.GetClient()}); err != nil {
		setupLogger().Errorf("unable to add command ttl janitor: %v", err)
		os.Exit(1)
	}

	// Setup health checks based on probe scheme
	if config.ProbeScheme == "https" {
		// Start independent HTTPS health check server
		if err := startHTTPSHealthServer(ctx, mgr); err != nil {
			setupLogger().Errorf("unable to start HTTPS health check server: %v", err)
			os.Exit(1)
		}
	} else {
		// Use default HTTP health check server
		setupHealthChecks(mgr)
	}

	registerMetric(mgr)
	registerProfiler(mgr)

	setupLogger().Info("starting manager")
	if err := mgr.Start(ctx); err != nil {
		setupLogger().Errorf("problem running manager: %v", err)
		os.Exit(1)
	}
}

// printVersionInfo prints version information
func printVersionInfo() {
	setupLogger().Infof("🤯 Version: %s", v.Version)
	setupLogger().Infof("🤔 GitCommitId: %s", v.GitCommitID)
	setupLogger().Infof("👉 Architecture: %s", v.Architecture)
	setupLogger().Infof("⏲ BuildTime: %s", v.BuildTime)
}

// printManifestsBuildInfo prints manifests build information
func printManifestsBuildInfo() {
	if manifestInfo, err := commonutils.GetManifestsBuildInfo(); err == nil {
		for _, v := range manifestInfo {
			setupLogger().Info(v)
		}
	} else {
		setupLogger().Infof("(ignore) Failed to get manifests build info: %v", err)
	}
}

// createManager creates and configures the manager
func createManager() (ctrl.Manager, *remote.ClusterCacheTracker) {
	config.ConfigurationFlag()

	// ologger initialization via OLOGGER_CONFIG env var (set in Pod spec by ConfigMap mount)
	// Config path: /etc/openFuyao/ologger/ologger.yaml (default ologger path)
	// Config values: see config/ologger/ologger.yaml in project root
	// Use default console encoder since log.Encoder is no longer available
	opts := zap.Options{
		Development: os.Getenv("DEBUG") == "true",
	}
	opts.BindFlags(flag.CommandLine)

	flag.Parse()
	config.ResolveClientConfig()
	ctrl.SetLogger(zap.New(zap.UseFlagOptions(&opts)))

	// Configure health probe based on scheme
	healthProbeAddr := config.ProbeAddr
	if config.ProbeScheme == "https" {
		// Disable default HTTP health check server when using HTTPS
		healthProbeAddr = "0"
	}

	mgr, err := ctrl.NewManager(pkgkube.ApplyThrottlingConfig(ctrl.GetConfigOrDie()), ctrl.Options{
		Scheme:                 scheme,
		MetricsBindAddress:     config.MetricsAddr,
		Port:                   config.WebhookPort,
		Host:                   config.WebhookHost,
		HealthProbeBindAddress: healthProbeAddr,
		LeaderElection:         config.EnableLeaderElection,
		LeaderElectionID:       "e2b5373a.bocloud.com",
		CertDir:                config.WebhookCertDir,
		EventBroadcaster: record.NewBroadcasterWithCorrelatorOptions(record.CorrelatorOptions{
			KeyFunc:              EventAggregatorByMessageFunc,
			BurstSize:            EventBroadcasterBurstSize,
			LRUCacheSize:         EventBroadcasterLRUCacheSize,
			MaxIntervalInSeconds: EventBroadcasterMaxIntervalInSeconds,
		}),
	})
	if err != nil {
		setupLogger().Errorf("unable to start manager: %v", err)
		os.Exit(1)
	}

	if err := scriptshelper.CreateScriptsConfigMaps(mgr.GetClient()); err != nil {
		setupLogger().Errorf("unable to create scripts configmaps: %v", err)
	}

	tracker, err := remote.NewClusterCacheTracker(
		mgr,
		remote.ClusterCacheTrackerOptions{
			Indexes: remote.DefaultIndexes,
		},
	)
	if err != nil {
		setupLogger().Errorf("unable to create cluster cache tracker: %v", err)
		os.Exit(1)
	}

	return mgr, tracker
}

// retrySecurityInit retries the eager key load to ride out the
// management-cluster bootstrap window: the pod reaches the apiserver via the
// service VIP, which stays unreachable until the CNI is ready, so transient
// failures (connection refused, discovery errors) are expected there.
// Permanent failures (invalid Secret data, errors.Is ErrInvalidKeySecret)
// fail fast; the overall timeout guards against persistent outages.
func retrySecurityInit(ctx context.Context, provider *security.SecretKeyProvider) error {
	return wait.PollUntilContextTimeout(ctx, securityInitRetryInterval, securityInitTimeout, true,
		func(ctx context.Context) (bool, error) {
			_, err := provider.Key(ctx)
			if err == nil {
				return true, nil
			}
			if errors.Is(err, security.ErrInvalidKeySecret) {
				return false, err
			}
			setupLogger().Warnf("failed to initialize password encryption key, retrying: %v", err)
			return false, nil
		})
}

// setupSecurity initializes the BKENode password encryption key:
// the key is eagerly loaded at startup (fail fast); when the Secret does not
// exist, the guard detects key loss (emitting warnings) and a new key is
// generated. Any load failure exits the process so kubelet restarts it and
// the problem becomes visible.
func setupSecurity(ctx context.Context, mgr ctrl.Manager) {
	get, create := security.CtrlRuntimeSecretIO(
		mgr.GetAPIReader(), mgr.GetClient(),
		security.DefaultSecretNamespace, security.DefaultSecretName)

	provider, err := security.NewSecretKeyProvider(security.SecretKeyProviderConfig{
		GetSecret:    get,
		CreateSecret: create,
		Guard: security.CtrlRuntimeBKENodeGuard(
			mgr.GetAPIReader(), mgr.GetEventRecorderFor("bke-password-key")),
	})
	if err != nil {
		setupLogger().Errorf("unable to create password key provider: %v", err)
		os.Exit(1)
	}
	security.SetKeyProvider(provider)

	if err := retrySecurityInit(ctx, provider); err != nil {
		setupLogger().Errorf("failed to initialize password encryption key: %v", err)
		os.Exit(1)
	}
	setupLogger().Info("password encryption key initialized")

	// Watch the key Secret so external changes are followed without a
	// restart: accidental deletion self-heals from the cached key and an
	// external rotation refreshes the cache.
	clientset, err := kubernetes.NewForConfig(mgr.GetConfig())
	if err != nil {
		setupLogger().Errorf("unable to create clientset for key secret watcher: %v", err)
		os.Exit(1)
	}
	if err := mgr.Add(security.NewSecretWatchRunnable(provider, clientset)); err != nil {
		setupLogger().Errorf("unable to add key secret watcher: %v", err)
		os.Exit(1)
	}
}

// setupControllers sets up the controllers
func setupControllers(ctx context.Context, mgr ctrl.Manager, tracker *remote.ClusterCacheTracker) {
	ociClient := newOCIClient()
	releaseStore := releasemanifest.NewStore(releasemanifest.ReleaseCacheDir(), releasemanifest.OCIPuller{Client: ociClient}, nil)
	setupBKEControllers(ctx, mgr, tracker, releaseStore)
	setupUpgradeControllers(mgr, ociClient, releaseStore)
}

func setupBKEControllers(ctx context.Context, mgr ctrl.Manager, tracker *remote.ClusterCacheTracker, releaseStore *releasemanifest.Store) {
	if err := (&capbkecontrollers.BKEClusterReconciler{
		Client:       mgr.GetClient(),
		Cache:        mgr.GetCache(),
		Scheme:       mgr.GetScheme(),
		Recorder:     mgr.GetEventRecorderFor("bke-cluster"),
		RestConfig:   mgr.GetConfig(),
		Tracker:      tracker,
		ReleaseStore: releaseStore,
	}).SetupWithManager(ctx, mgr, concurrency(config.BkeClusterConcurrency)); err != nil {
		setupLogger().Errorf("unable to create controller BKECluster: %v", err)
		os.Exit(1)
	}
	if err := (&capbkecontrollers.BKEMachineReconciler{
		Client:   mgr.GetClient(),
		Scheme:   mgr.GetScheme(),
		Recorder: mgr.GetEventRecorderFor("bke-machine"),
	}).SetupWithManager(mgr, concurrency(config.BkeMachineConcurrency)); err != nil {
		setupLogger().Errorf("unable to create controller BKEMachine: %v", err)
		os.Exit(1)
	}
}

func setupUpgradeControllers(mgr ctrl.Manager, ociClient *oci.Client, releaseStore *releasemanifest.Store) {
	upgradePathService := pathstore.NewService()
	if err := (&clusterversioncontrollers.ClusterVersionReconciler{
		Client:      mgr.GetClient(),
		Scheme:      mgr.GetScheme(),
		Recorder:    mgr.GetEventRecorderFor("cluster-version"),
		PathService: upgradePathService,
		OCIClient:   ociClient,
	}).SetupWithManager(mgr); err != nil {
		setupLogger().Errorf("unable to create controller ClusterVersion: %v", err)
		os.Exit(1)
	}
	if err := (&releaseimagecontrollers.ReleaseImageReconciler{

		Client:    mgr.GetClient(),
		Scheme:    mgr.GetScheme(),
		OCIClient: ociClient,
		Store:     releaseStore,
	}).SetupWithManager(mgr); err != nil {
		setupLogger().Errorf("unable to create controller ReleaseImage: %v", err)
		os.Exit(1)
	}

	if config.EnableOCIDigestMonitor {
		setupLogger().Info("OCI digest monitor enabled",
			"checkIntervalSeconds", config.OCIDigestCheckInterval,
			"ociRef", pathstore.DefaultUpgradePathOCIRef())
		if err := mgr.Add(pathstore.NewDigestMonitorRunnable(
			mgr.GetClient(),
			ociClient,
			time.Duration(config.OCIDigestCheckInterval)*time.Second,
		)); err != nil {
			setupLogger().Errorf("unable to add digest monitor runnable: %v", err)
			os.Exit(1)
		}
	} else {
		setupLogger().Info("OCI digest monitor disabled")
	}
	setupUpgradePathController(mgr, upgradePathService)
}

func setupUpgradePathController(mgr ctrl.Manager, service pathstore.Loader) {
	if err := (&upgradepathcontrollers.UpgradePathReconciler{
		Client:      mgr.GetClient(),
		Scheme:      mgr.GetScheme(),
		PathService: service,
	}).SetupWithManager(mgr); err != nil {
		setupLogger().Errorf("unable to create controller UpgradePath: %v", err)
		os.Exit(1)
	}
}

func newOCIClient() *oci.Client {
	return oci.NewClientFromConfig(oci.ClientConfig{
		InsecureSkipTLSVerify: config.OCIRegistryInsecure,
		Username:              config.OCIRegistryUsername,
		Password:              config.OCIRegistryPassword,
	})
}

// setupWebhooks sets up the webhooks
func setupWebhooks(mgr ctrl.Manager) {
	if err := (&capbkewebhooks.BKECluster{
		Client:      mgr.GetClient(),
		NodeFetcher: nodeutil.NewNodeFetcher(mgr.GetClient()),
		APIReader:   mgr.GetAPIReader(),
	}).SetupWebhookWithManager(mgr); err != nil {
		setupLogger().Errorf("unable to create webhook BKECluster: %v", err)
		os.Exit(1)
	}
	if err := (&capbkewebhooks.UpgradePath{
		Client:    mgr.GetClient(),
		APIReader: mgr.GetAPIReader(),
	}).SetupWebhookWithManager(mgr); err != nil {
		setupLogger().Errorf("unable to create webhook UpgradePath: %v", err)
		os.Exit(1)
	}
	if err := (&capbkewebhooks.BKENode{
		Client: mgr.GetClient(),
	}).SetupWebhookWithManager(mgr); err != nil {
		setupLogger().Errorf("unable to create webhook BKENode: %v", err)
		os.Exit(1)
	}
}

// setupHealthChecks sets up health and ready checks for HTTP probe
func setupHealthChecks(mgr ctrl.Manager) {
	if err := mgr.AddHealthzCheck("healthz", healthz.Ping); err != nil {
		setupLogger().Errorf("unable to set up health check: %v", err)
		os.Exit(1)
	}
	if err := mgr.AddReadyzCheck("readyz", healthz.Ping); err != nil {
		setupLogger().Errorf("unable to set up ready check: %v", err)
		os.Exit(1)
	}
}

// validateTLSCertificates checks if TLS certificate files exist
func validateTLSCertificates(certPath, keyPath string) error {
	if _, err := os.Stat(certPath); os.IsNotExist(err) {
		return fmt.Errorf("TLS certificate not found at %s", certPath)
	}
	if _, err := os.Stat(keyPath); os.IsNotExist(err) {
		return fmt.Errorf("TLS key not found at %s", keyPath)
	}
	return nil
}

// loadTLSConfig loads TLS certificate and creates TLS configuration
func loadTLSConfig(certPath, keyPath string) (*tls.Config, error) {
	cert, err := tls.LoadX509KeyPair(certPath, keyPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load TLS certificate: %w", err)
	}
	return &tls.Config{
		Certificates: []tls.Certificate{cert},
		MinVersion:   tls.VersionTLS12,
	}, nil
}

// setupHealthEndpoints sets up health check endpoints on the mux
func setupHealthEndpoints(mux *http.ServeMux) {
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/healthz" {
			http.NotFound(w, r)
			return
		}
		w.WriteHeader(http.StatusOK)
		_, err := w.Write([]byte("ok"))
		if err != nil {
			return
		}
	})

	mux.HandleFunc("/readyz", func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/readyz" {
			http.NotFound(w, r)
			return
		}
		w.WriteHeader(http.StatusOK)
		_, err := w.Write([]byte("ok"))
		if err != nil {
			return
		}
	})
}

// startHTTPSListener starts the HTTPS server listener in a goroutine
func startHTTPSListener(server *http.Server, tlsConfig *tls.Config, certPath, keyPath string) {
	setupLogger().Info("starting HTTPS health check server", "address", server.Addr, "cert", certPath, "key", keyPath)
	ln, err := net.Listen("tcp", server.Addr)
	if err != nil {
		setupLogger().Errorf("failed to listen on HTTPS health check port: %v", err)
		return
	}
	tlsListener := tls.NewListener(ln, tlsConfig)
	if err = server.Serve(tlsListener); err != nil && !errors.Is(err, http.ErrServerClosed) {
		setupLogger().Errorf("HTTPS health check server error: %v", err)
	}
}

// setupGracefulShutdown sets up graceful shutdown for the HTTPS server
func setupGracefulShutdown(ctx context.Context, server *http.Server) {
	go func() {
		<-ctx.Done()
		setupLogger().Info("shutting down HTTPS health check server")
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		if err := server.Shutdown(shutdownCtx); err != nil {
			setupLogger().Errorf("error shutting down HTTPS health check server: %v", err)
		}
	}()
}

// startHTTPSHealthServer starts an independent HTTPS health check server
// using TLS certificate from /etc/kubernetes/tls-server.crt and tls-server.key
// Note: Uses port 9444 by default to avoid conflict with webhook server (port 9443)
func startHTTPSHealthServer(ctx context.Context, mgr ctrl.Manager) error {
	const (
		certPath = "/etc/kubernetes/tls-server.crt"
		keyPath  = "/etc/kubernetes/tls-server.key"
	)
	port := config.ProbePort
	if port <= 0 {
		port = 9444 // Default port for HTTPS health check server
	}

	if err := validateTLSCertificates(certPath, keyPath); err != nil {
		return err
	}

	tlsConfig, err := loadTLSConfig(certPath, keyPath)
	if err != nil {
		return err
	}

	mux := http.NewServeMux()
	setupHealthEndpoints(mux)

	server := &http.Server{
		Addr:      fmt.Sprintf(":%d", port),
		Handler:   mux,
		TLSConfig: tlsConfig,
	}

	go startHTTPSListener(server, tlsConfig, certPath, keyPath)
	setupGracefulShutdown(ctx, server)

	return nil
}

// EventAggregatorByMessageFunc aggregates events by exact match on event.Source, event.InvolvedObject, event.Type,
// event.Reason, event.ReportingController and event.ReportingInstance
func EventAggregatorByMessageFunc(event *corev1.Event) (string, string) {
	return strings.Join([]string{
		event.Source.Component,
		event.Source.Host,
		event.InvolvedObject.Kind,
		event.InvolvedObject.Namespace,
		event.InvolvedObject.Name,
		string(event.InvolvedObject.UID),
		event.InvolvedObject.APIVersion,
		event.Type,
		event.Message,
		event.ReportingController,
		event.ReportingInstance,
	},
		""), event.Message
}

func concurrency(c int) controller.Options {
	recoverPanic := true
	return controller.Options{
		MaxConcurrentReconciles: c,
		RecoverPanic:            &recoverPanic,
		RateLimiter: workqueue.NewItemFastSlowRateLimiter(
			FastSlowRateLimiterFastDuration,
			FastSlowRateLimiterSlowDuration,
			FastSlowRateLimiterRetryCount),
	}
}

func registerMetric(mgr ctrl.Manager) {
	if config.MetricsAddr == "0" {
		return
	}

	if err := mgr.AddMetricsExtraHandler("/export", bkemetrics.DefaultRegister().HttpExportFunc()); err != nil {
		setupLogger().Errorf("unable to set up extra metrics handler: %v", err)
		os.Exit(1)
	}
	if err := mgr.AddMetricsExtraHandler("/cluster", bkemetrics.DefaultRegister().HttpClusterFunc()); err != nil {
		setupLogger().Errorf("unable to set up extra metrics handler: %v", err)
		os.Exit(1)
	}
}

func registerProfiler(m ctrl.Manager) {
	if os.Getenv("DEBUG") != "true" && config.MetricsAddr == "0" {
		return
	}

	endpoints := map[string]http.HandlerFunc{
		"/debug/pprof/":        pprof.Index,
		"/debug/pprof/cmdline": pprof.Cmdline,
		"/debug/pprof/profile": pprof.Profile,
		"/debug/pprof/symbol":  pprof.Symbol,
		"/debug/pprof/trace":   pprof.Trace,
	}

	for path, handler := range endpoints {
		err := m.AddMetricsExtraHandler(path, handler)
		if err != nil {
			setupLogger().Errorf("unable to set up pprof handler: %v", err)
			os.Exit(1)
		}
	}
}
