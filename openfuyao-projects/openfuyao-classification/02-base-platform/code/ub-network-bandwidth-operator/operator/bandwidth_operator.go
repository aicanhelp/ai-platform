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

package operator

import (
	"fmt"
	"sync"
	"time"

	"github.com/fsnotify/fsnotify"
	"golang.org/x/time/rate"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/klog/v2"

	"gitcode.com/openFuyao/ub-network-bandwidth-operator/config"
	"gitcode.com/openFuyao/ub-network-bandwidth-operator/reconciler"
)

const (
	key                 = "bandwidth-config"
	initialRetryDelay   = time.Second
	maxRetryDelay       = 30 * time.Second
	reconcileQueueQPS   = 10
	reconcileQueueBurst = 100
)

type bandwidthOperator struct {
	opts       *StartOpts
	watcher    *config.Watcher
	queue      workqueue.TypedRateLimitingInterface[string]
	reconciler *reconciler.Controller
	stopCh     chan struct{}
	doneCh     chan struct{}
	doneOnce   sync.Once
	stopped    bool
}

func New(opts ...Option) (Interface, error) {
	startOpts := defaultStartOpts()
	for _, o := range opts {
		o(startOpts)
	}

	watcher, err := config.NewWatcher(startOpts.ConfigPath)
	if err != nil {
		return nil, fmt.Errorf("create config watcher: %w", err)
	}

	rateLimiter := workqueue.NewTypedMaxOfRateLimiter[string](
		workqueue.NewTypedItemExponentialFailureRateLimiter[string](initialRetryDelay, maxRetryDelay),
		&workqueue.TypedBucketRateLimiter[string]{Limiter: rate.NewLimiter(reconcileQueueQPS, reconcileQueueBurst)},
	)

	queue := workqueue.NewTypedRateLimitingQueueWithConfig(rateLimiter, workqueue.TypedRateLimitingQueueConfig[string]{
		Name: "bandwidth-config",
	})

	r := reconciler.NewController(startOpts.Manager, startOpts.ConfigPath)

	return &bandwidthOperator{
		opts:       startOpts,
		watcher:    watcher,
		queue:      queue,
		reconciler: r,
		stopCh:     make(chan struct{}),
		doneCh:     make(chan struct{}),
	}, nil
}

func (o *bandwidthOperator) Start() error {
	klog.Info("starting bandwidth operator")

	o.queue.Add(key)

	go func() {
		o.runWorker()
		o.finish()
	}()
	go func() {
		o.watchConfig()
		o.finish()
	}()

	return nil
}

func (o *bandwidthOperator) finish() {
	if o.stopped {
		return
	}
	o.doneOnce.Do(func() {
		klog.Warning("operator exited unexpectedly, signaling done")
		close(o.doneCh)
	})
}

func (o *bandwidthOperator) Done() <-chan struct{} {
	return o.doneCh
}

func (o *bandwidthOperator) Stop() {
	klog.Info("stopping bandwidth operator")
	o.stopped = true
	close(o.stopCh)
	o.queue.ShutDown()
	o.watcher.Close()
}

func (o *bandwidthOperator) runWorker() {
	for o.processNextItem() {
	}
}

func (o *bandwidthOperator) processNextItem() bool {
	item, shutdown := o.queue.Get()
	if shutdown {
		return false
	}
	defer o.queue.Done(item)

	if err := o.reconciler.Reconcile(); err != nil {
		klog.Errorf("reconcile failed: %v, will retry", err)
		o.queue.AddRateLimited(item)
		return true
	}
	o.queue.Forget(item)
	klog.Infof("reconciled bandwidth configuration end")
	return true
}

func (o *bandwidthOperator) watchConfig() {
	for {
		select {
		case <-o.stopCh:
			return
		case event, ok := <-o.watcher.Events():
			if !ok {
				return
			}
			if event.Op&fsnotify.Remove == fsnotify.Remove {
				if event.Name == o.opts.ConfigPath {
					continue
				}
				klog.Infof("received config file changed")
				o.queue.Add(key)
			}
		case err, ok := <-o.watcher.Errors():
			if !ok {
				return
			}
			klog.Errorf("config watcher error: %v", err)
		}
	}
}
