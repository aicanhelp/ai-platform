// Copyright(c) Huawei Technologies Co., Ltd. 2025-2025. All right reserved.

package timer

import (
	"context"
	"sync"
	"time"

	"k8s.io/klog/v2"
)

const defaultPeriod = 30 * time.Second
const defaultTimeOut = 5 * time.Minute

type ConditionFunc func(ctx context.Context) error

// RetryableRunner is used to run a function. if call failed, retry it at intervals until timeout
// if there is a goroutine running already, stop it, wait it finished and then run the function
type RetryableRunner struct {
	cancelFunc context.CancelFunc

	startLock sync.Mutex
	taskLock  sync.Mutex
}

func (r *RetryableRunner) Run(ctx context.Context, condition ConditionFunc) {
	r.RunUtil(ctx, defaultTimeOut, defaultPeriod, condition)
}

func (r *RetryableRunner) RunUtil(ctx context.Context, timeout time.Duration, period time.Duration, condition ConditionFunc) {
	r.startLock.Lock()
	defer r.startLock.Unlock()

	if r.cancelFunc != nil {
		r.cancelFunc()
	}

	timeoutCxt, cancelFunc := context.WithTimeout(context.Background(), timeout)
	// store cancel func in struct, then next caller would cancel it
	r.cancelFunc = cancelFunc

	go r.poll(ctx, period, timeoutCxt, condition)
}

func (r *RetryableRunner) poll(ctx context.Context, period time.Duration, timeoutCtx context.Context, condition ConditionFunc) {
	r.taskLock.Lock()
	defer r.taskLock.Unlock()

	if err := condition(ctx); err == nil {
		return
	}

	ticker := time.NewTicker(period)
	defer ticker.Stop()

	var err error
loop:
	for {
		select {
		case <-ctx.Done():
			break loop
		case <-ticker.C:
			if err = condition(ctx); err == nil {
				break loop
			}
		case <-timeoutCtx.Done():
			err = condition(ctx)
			break loop
		}
	}

	if err != nil {
		klog.Warning(err)
	}
}
