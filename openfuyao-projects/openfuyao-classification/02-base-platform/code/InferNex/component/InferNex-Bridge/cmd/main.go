/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

// Package main starts the InferNex Bridge controller manager.
package main

// Import all Kubernetes client auth plugins (e.g. Azure, GCP, OIDC, etc.)
// to ensure that exec-entrypoint and run can make use of them.
import (
	"crypto/tls"
	"flag"
	"os"

	"k8s.io/apimachinery/pkg/runtime"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	_ "k8s.io/client-go/plugin/pkg/client/auth"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/healthz"
	"sigs.k8s.io/controller-runtime/pkg/log/zap"
	"sigs.k8s.io/controller-runtime/pkg/metrics/filters"
	metricsserver "sigs.k8s.io/controller-runtime/pkg/metrics/server"
	"sigs.k8s.io/controller-runtime/pkg/webhook"
	igwapiv1 "sigs.k8s.io/gateway-api-inference-extension/api/v1"
	gwapiv1 "sigs.k8s.io/gateway-api/apis/v1"
	lwsv1 "sigs.k8s.io/lws/api/leaderworkerset/v1"

	infernexv1alpha1 "gitcode.com/openFuyao/InferNex/api/v1alpha1"
	"gitcode.com/openFuyao/InferNex/internal/bootstrap"
	"gitcode.com/openFuyao/InferNex/internal/controller"
	webhookv1alpha1 "gitcode.com/openFuyao/InferNex/internal/webhook/v1alpha1"
	// +kubebuilder:scaffold:imports
)

const leaderElectionID = "85d958d4.infernex.io"

var (
	scheme   = runtime.NewScheme()
	setupLog = ctrl.Log.WithName("setup")
)

func init() {
	utilruntime.Must(clientgoscheme.AddToScheme(scheme))

	utilruntime.Must(infernexv1alpha1.AddToScheme(scheme))
	utilruntime.Must(gwapiv1.AddToScheme(scheme))
	utilruntime.Must(igwapiv1.AddToScheme(scheme))
	utilruntime.Must(lwsv1.AddToScheme(scheme))
	// +kubebuilder:scaffold:scheme
}

// cliFlags holds parsed command-line options for the manager.
type cliFlags struct {
	metricsAddr          string
	metricsCertPath      string
	metricsCertName      string
	metricsCertKey       string
	webhookCertPath      string
	webhookCertName      string
	webhookCertKey       string
	enableLeaderElection bool
	probeAddr            string
	secureMetrics        bool
	enableHTTP2          bool
	zapOpts              zap.Options
}

func parseCLI() cliFlags {
	var f cliFlags
	f.zapOpts = zap.Options{Development: true}
	f.zapOpts.BindFlags(flag.CommandLine)

	flag.StringVar(&f.metricsAddr, "metrics-bind-address", "0", "The address the metrics endpoint binds to. "+
		"Use :8443 for HTTPS or :8080 for HTTP, or leave as 0 to disable the metrics service.")
	flag.StringVar(&f.probeAddr, "health-probe-bind-address", ":8081", "The address the probe endpoint binds to.")
	flag.BoolVar(&f.enableLeaderElection, "leader-elect", false,
		"Enable leader election for controller manager. "+
			"Enabling this will ensure there is only one active controller manager.")
	flag.BoolVar(&f.secureMetrics, "metrics-secure", true,
		"If set, the metrics endpoint is served securely via HTTPS. Use --metrics-secure=false to use HTTP instead.")
	flag.StringVar(&f.webhookCertPath, "webhook-cert-path", "", "The directory that contains the webhook certificate.")
	flag.StringVar(&f.webhookCertName, "webhook-cert-name", "tls.crt", "The name of the webhook certificate file.")
	flag.StringVar(&f.webhookCertKey, "webhook-cert-key", "tls.key", "The name of the webhook key file.")
	flag.StringVar(&f.metricsCertPath, "metrics-cert-path", "",
		"The directory that contains the metrics server certificate.")
	flag.StringVar(&f.metricsCertName, "metrics-cert-name", "tls.crt", "The name of the metrics server certificate file.")
	flag.StringVar(&f.metricsCertKey, "metrics-cert-key", "tls.key", "The name of the metrics server key file.")
	flag.BoolVar(&f.enableHTTP2, "enable-http2", false,
		"If set, HTTP/2 will be enabled for the metrics and webhook servers")
	flag.Parse()
	return f
}

func disableHTTP2TLSOpt() func(*tls.Config) {
	// Disable HTTP/2 by default (Stream Cancellation / Rapid Reset); see GHSA-qppj-fm5r-hxr3, GHSA-4374-p667-p6c8.
	return func(c *tls.Config) {
		setupLog.Info("disabling http/2")
		c.NextProtos = []string{"http/1.1"}
	}
}

func tlsOptsFromHTTP2(enableHTTP2 bool) []func(*tls.Config) {
	if enableHTTP2 {
		return nil
	}
	return []func(*tls.Config){disableHTTP2TLSOpt()}
}

func newWebhookServer(f cliFlags, tlsOpts []func(*tls.Config)) webhook.Server {
	opts := webhook.Options{TLSOpts: tlsOpts}
	if len(f.webhookCertPath) > 0 {
		setupLog.Info("Initializing webhook certificate watcher using provided certificates",
			"webhook-cert-path", f.webhookCertPath, "webhook-cert-name", f.webhookCertName, "webhook-cert-key", f.webhookCertKey)
		opts.CertDir = f.webhookCertPath
		opts.CertName = f.webhookCertName
		opts.KeyName = f.webhookCertKey
	}
	return webhook.NewServer(opts)
}

func newMetricsOptions(f cliFlags, tlsOpts []func(*tls.Config)) metricsserver.Options {
	o := metricsserver.Options{
		BindAddress:   f.metricsAddr,
		SecureServing: f.secureMetrics,
		TLSOpts:       tlsOpts,
	}
	if f.secureMetrics {
		o.FilterProvider = filters.WithAuthenticationAndAuthorization
	}
	if len(f.metricsCertPath) > 0 {
		setupLog.Info("Initializing metrics certificate watcher using provided certificates",
			"metrics-cert-path", f.metricsCertPath, "metrics-cert-name", f.metricsCertName, "metrics-cert-key", f.metricsCertKey)
		o.CertDir = f.metricsCertPath
		o.CertName = f.metricsCertName
		o.KeyName = f.metricsCertKey
	}
	return o
}

func newManager(f cliFlags) (ctrl.Manager, error) {
	tlsOpts := tlsOptsFromHTTP2(f.enableHTTP2)
	return ctrl.NewManager(ctrl.GetConfigOrDie(), ctrl.Options{
		Scheme:                 scheme,
		Metrics:                newMetricsOptions(f, tlsOpts),
		WebhookServer:          newWebhookServer(f, tlsOpts),
		HealthProbeBindAddress: f.probeAddr,
		LeaderElection:         f.enableLeaderElection,
		LeaderElectionID:       leaderElectionID,
	})
}

func registerControllersAndWebhooks(mgr ctrl.Manager) error {
	if err := (&controller.InferNexServiceReconciler{
		Client:            mgr.GetClient(),
		Scheme:            mgr.GetScheme(),
		TemplateNamespace: controllerTemplateNamespace(),
	}).SetupWithManager(mgr); err != nil {
		setupLog.Error(err, "unable to create controller", "controller", "InferNexService")
		return err
	}
	if os.Getenv("ENABLE_WEBHOOKS") != "false" {
		webhookv1alpha1.SetupLLMInferenceServiceWebhooksWithManager(mgr)
		webhookv1alpha1.SetupInferNexServiceValidationWebhookWithManager(mgr)
	}
	// +kubebuilder:scaffold:builder
	return nil
}

func installProbes(mgr ctrl.Manager) error {
	if err := mgr.AddHealthzCheck("healthz", healthz.Ping); err != nil {
		setupLog.Error(err, "unable to set up health check")
		return err
	}
	if err := mgr.AddReadyzCheck("readyz", healthz.Ping); err != nil {
		setupLog.Error(err, "unable to set up ready check")
		return err
	}
	return nil
}

func main() {
	f := parseCLI()
	ctrl.SetLogger(zap.New(zap.UseFlagOptions(&f.zapOpts)))

	mgr, err := newManager(f)
	if err != nil {
		setupLog.Error(err, "unable to start manager")
		os.Exit(1)
	}
	if err := registerControllersAndWebhooks(mgr); err != nil {
		os.Exit(1)
	}
	if err := installProbes(mgr); err != nil {
		os.Exit(1)
	}
	if err := mgr.Add(bootstrap.DefaultTemplatesRunnable(mgr.GetClient(), controllerTemplateNamespace())); err != nil {
		setupLog.Error(err, "unable to register default templates bootstrap")
		os.Exit(1)
	}
	setupLog.Info("starting manager")
	if err := mgr.Start(ctrl.SetupSignalHandler()); err != nil {
		setupLog.Error(err, "problem running manager")
		os.Exit(1)
	}
}

func controllerTemplateNamespace() string {
	if ns := os.Getenv("INFERNEX_TEMPLATE_NAMESPACE"); ns != "" {
		return ns
	}
	return os.Getenv("POD_NAMESPACE")
}
