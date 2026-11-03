/*
 *
 *  * Copyright (c) 2024 China Unicom Digital Technology Co., Ltd.
 *  * openFuyao is licensed under Mulan PSL v2.
 *  * You can use this software according to the terms and conditions of the Mulan PSL v2.
 *  * You may obtain a copy of Mulan PSL v2 at:
 *  *          http://license.coscl.org.cn/MulanPSL2
 *  * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 *  * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 *  * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 *  * See the Mulan PSL v2 for more details.
 *
 */

package main

import (
	"context"
	"flag"
	"math/rand"
	"net/http"
	"time"

	"k8s.io/apimachinery/pkg/util/wait"
	_ "k8s.io/client-go/plugin/pkg/client/auth"
	"k8s.io/client-go/tools/leaderelection/resourcelock"
	"k8s.io/klog/v2"
	"k8s.io/klog/v2/klogr"
	"sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/cache"
	"sigs.k8s.io/controller-runtime/pkg/config"
	"sigs.k8s.io/controller-runtime/pkg/healthz"
	"sigs.k8s.io/controller-runtime/pkg/metrics/server"

	"openfuyao.com/colocation-management/cmd/colocation-manager/apps"
	"openfuyao.com/colocation-management/pkg/colocation-manager/aggregate"
	"openfuyao.com/colocation-management/pkg/colocation-manager/checkpoint"
	"openfuyao.com/colocation-management/pkg/colocation-manager/controller"
	"openfuyao.com/colocation-management/pkg/colocation-manager/recommend"
	"openfuyao.com/colocation-management/pkg/colocation-manager/webhook"
	"openfuyao.com/colocation-management/pkg/colocation-overquota-agent/report"
	"openfuyao.com/colocation-management/pkg/common"
	"openfuyao.com/colocation-management/pkg/utils"
)

func main() {
	configuration := apps.NewConfiguration()
	configuration.InitFlags(flag.CommandLine)
	klog.InitFlags(nil)
	flag.Parse()
	rand.Seed(time.Now().UnixNano())
	controllerruntime.SetLogger(klogr.New())
	configuration.PrintAndExitIfRequested()

	go wait.Forever(klog.Flush, common.KlogFlushInterval)
	defer klog.Flush()

	if configuration.EnablePprof {
		go func() {
			if err := http.ListenAndServe(configuration.PprofAddr, nil); err != nil {
				klog.Fatalf("unable to start pprof. %v", err)
			}
		}()
	}

	ctx := controllerruntime.SetupSignalHandler()
	clusterState := aggregate.NewClusterState(ctx, configuration)

	mgr := setupControllerManager(configuration)

	go func() {
		if !mgr.GetCache().WaitForCacheSync(ctx) {
			klog.Fatalf("main: sync controller-manger cache failed.")
		}
		klog.V(common.VerboseDebugLog).Infof("main: sync controller-manger cache done")
		utils.NotifySyncDone()
	}()

	setupControllers(ctx, mgr, clusterState, configuration)

	startBackgroundService(ctx, mgr, clusterState, configuration)

	klog.Info("Starting controller runtime manager")
	if err := mgr.Start(ctx); err != nil {
		klog.Fatalf("fail to start controller runtime manager. %v", err)
	}
}

// setupControllerManager setup controller manager
func setupControllerManager(configuration *apps.Configuration) controllerruntime.Manager {
	mgrCfg := controllerruntime.GetConfigOrDie()
	configuration.SetRestConfig(mgrCfg)
	mgrCfg.UserAgent = "colocation-manager"

	mgrOpt := controllerruntime.Options{
		Scheme:                     apps.GetScheme(),
		Metrics:                    server.Options{BindAddress: configuration.MetricsAddr},
		HealthProbeBindAddress:     configuration.HealthProbeAddr,
		LeaderElection:             configuration.EnableLeaderElection,
		LeaderElectionID:           "colocation-manager",
		LeaderElectionNamespace:    configuration.Namespace,
		LeaderElectionResourceLock: resourcelock.LeasesResourceLock,
		Cache:                      cache.Options{SyncPeriod: &configuration.CacheSyncPeriod},
		Controller:                 config.Controller{MaxConcurrentReconciles: 10},
	}

	mgr, err := controllerruntime.NewManager(controllerruntime.GetConfigOrDie(), mgrOpt)
	if err != nil {
		klog.Fatalf("unable to start manager. %v", err)
	}
	if err := mgr.AddHealthzCheck("healthz", healthz.Ping); err != nil {
		klog.Fatalf("unable to set up health check. %v", err)
	}
	if err := mgr.AddReadyzCheck("readyz", healthz.Ping); err != nil {
		klog.Fatalf("unable to set up ready check. %v", err)
	}
	return mgr
}

// setupControllers setup all controllers
func setupControllers(ctx context.Context, mgr controllerruntime.Manager,
	clusterState *aggregate.ClusterState, configuration *apps.Configuration) {
	if err := controller.NewPodReconciler(ctx, mgr, clusterState, configuration.RefreshPodsInterval,
		configuration.AggregateStateGCInterval).Run(); err != nil {
		klog.Fatalf("fail to setup Pod controller. %v", err)
	}
	if err := controller.NewNodeReconciler(ctx, mgr, clusterState,
		configuration.RefreshNodesInterval).Run(); err != nil {
		klog.Fatalf("fail to setup Node controller. %v", err)
	}
	if err := controller.NewContainerCheckpointReconciler(ctx, mgr, clusterState).SetupWithManager(); err != nil {
		klog.Fatalf("fail to setup Checkpoint controller. %v", err)
	}
}

// startBackgroundService start all background services
func startBackgroundService(ctx context.Context, mgr controllerruntime.Manager,
	clusterState *aggregate.ClusterState, configuration *apps.Configuration) {
	ckptLoaderWriter := checkpoint.NewLoaderWriter(ctx, mgr, configuration, clusterState)
	go ckptLoaderWriter.Run()

	usageCollector := report.NewUsageCollector(configuration, clusterState)
	go usageCollector.Run(ctx)

	recommender := recommend.NewResourceRecommender(ctx, mgr, configuration, clusterState)
	go recommender.Run()

	whsvr := webhook.NewWebhookServer(ctx, configuration)
	go whsvr.Run()
}
