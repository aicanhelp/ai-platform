/*
Copyright 2022.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package main

import (
	"flag"
	"os"
	_ "time/tzdata"

	// Import all Kubernetes client auth plugins (e.g. Azure, GCP, OIDC, etc.)
	// to ensure that exec-entrypoint and run can make use of them.
	"k8s.io/apimachinery/pkg/runtime"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	_ "k8s.io/client-go/plugin/pkg/client/auth"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/log/zap"

	agentv1beta1 "gopkg.openfuyao.cn/bkeagent/api/v1beta1"
	"gopkg.openfuyao.cn/bkeagent/controllers"
	"gopkg.openfuyao.cn/bkeagent/pkg/job"
	"gopkg.openfuyao.cn/bkeagent/utils"
	"gopkg.openfuyao.cn/bkeagent/utils/log"
	"gopkg.openfuyao.cn/bkeagent/utils/option"
	v "gopkg.openfuyao.cn/bkeagent/version"
	//+kubebuilder:scaffold:imports
)

var (
	scheme   = runtime.NewScheme()
	setupLog = ctrl.Log.WithName("setup")
)

func init() {
	utilruntime.Must(clientgoscheme.AddToScheme(scheme))
	utilruntime.Must(agentv1beta1.AddToScheme(scheme))
	//+kubebuilder:scaffold:scheme
}

func main() {
	var ntpServer string
	var ntpcron string
	var showVersion bool

	flag.StringVar(&ntpServer, "ntpserver", "", "The ntp server address.")
	flag.StringVar(&ntpcron, "ntpcron", "0 */1 * * *", "The sync time cron.")
	flag.BoolVar(&showVersion, "v", false, "Show version and build information.")

	flag.StringVar(&option.Platform, "platform", "", "The platform of the target cluster.")
	flag.StringVar(&option.Version, "version", "", "The version of the target cluster.")

	opts := zap.Options{
		Development: os.Getenv("DEBUG") == "true",
	}
	opts.BindFlags(flag.CommandLine)
	flag.Parse()

	if showVersion {
		v.PrintVersion()
		os.Exit(0)
	}

	// Print version
	v.LogPrintVersion()

	ctrl.SetLogger(zap.New(zap.UseFlagOptions(&opts)))

	//	In the target cluster, check and install the CRD
	if err := enableCrdHasInstalled(); err != nil {
		log.Errorf("The CRD cannot be installed in the target cluster, %s", err.Error())
		return
	}

	ctx := ctrl.SetupSignalHandler()
	mgr, err := ctrl.NewManager(ctrl.GetConfigOrDie(), ctrl.Options{
		Scheme:             scheme,
		MetricsBindAddress: "0",
		LeaderElection:     false,
	})
	if err != nil {
		setupLog.Error(err, "unable to start manager")
		os.Exit(1)
	}

	j, err := job.NewJob(mgr.GetClient())
	if err != nil {
		log.Fatal(err)
	}

	hostName := utils.HostName()

	log.Infof("BKEAgent node hostName: %s", hostName)
	log.Infof("BKEAgent listening cluster: %s", mgr.GetConfig().Host)

	if err = (&controllers.CommandReconciler{
		Client:    mgr.GetClient(),
		APIReader: mgr.GetAPIReader(),
		Scheme:    mgr.GetScheme(),
		Job:       j,
		NodeName:  hostName,
		Ctx:       ctx,
	}).SetupWithManager(mgr); err != nil {
		setupLog.Error(err, "unable to create controller", "controller", "Command")
		os.Exit(1)
	}
	//+kubebuilder:scaffold:builder

	setupLog.Info("starting manager")
	if err := mgr.Start(ctx); err != nil {
		setupLog.Error(err, "problem running manager")
		os.Exit(1)
	}
}
