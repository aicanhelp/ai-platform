package mco

import (
	"context"
	"fmt"
	"time"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	. "gitcode.com/openFuyao/e2e-auto-test/e2e/many-core-orchestrator/utils"
)

type cleanupWaiter func(context.Context) error

func UninstallReleaseAndWait(ctx context.Context, waiters ...cleanupWaiter) error {
	if err := HelmUninstallLocal(ctx, MCOHelmConfig); err != nil {
		return err
	}
	if err := WaitHelmReleaseDeleted(ctx, MCOHelmConfig, 120*time.Second, 3*time.Second); err != nil {
		return err
	}
	for _, waiter := range waiters {
		if err := waiter(ctx); err != nil {
			return err
		}
	}
	return WaitReleaseNamespacesStable(ctx)
}

func UninstallExistingReleaseAndWait(ctx context.Context, waiters ...cleanupWaiter) error {
	exists, err := HelmReleaseExists(ctx, MCOHelmConfig)
	if err != nil {
		return err
	}
	if !exists {
		return nil
	}
	return UninstallReleaseAndWait(ctx, waiters...)
}

func WaitKataDeployResourcesDeleted(ctx context.Context) error {
	deadline := time.Now().Add(120 * time.Second)
	var last map[string]bool
	for time.Now().Before(deadline) {
		resources, err := GetKataDeployResources(k8sClient, ctx)
		if err == nil && len(resources) == 0 {
			return nil
		}
		last = resources
		if waitCleanupPoll(ctx, 2*time.Second) {
			return fmt.Errorf("context canceled waiting for kata resources deleted: %w", ctx.Err())
		}
	}
	return fmt.Errorf("kata resources not deleted within 120s: %v", last)
}

func WaitSchedulerResourcesDeleted(ctx context.Context) error {
	deadline := time.Now().Add(120 * time.Second)
	var last map[string]bool
	for time.Now().Before(deadline) {
		resources, err := GetSchedulerResources(k8sClient, ctx)
		if err == nil && len(resources) == 0 {
			return nil
		}
		last = resources
		if waitCleanupPoll(ctx, 2*time.Second) {
			return fmt.Errorf("context canceled waiting for scheduler resources deleted: %w", ctx.Err())
		}
	}
	return fmt.Errorf("scheduler resources not deleted within 120s: %v", last)
}

func WaitReleaseNamespacesStable(ctx context.Context) error {
	return WaitNamespacesStable(ctx, 120*time.Second, 2*time.Second, MCODeployNamespace, MCOVolcanoNamespace)
}

func WaitNamespacesStable(ctx context.Context, timeout, poll time.Duration, namespaces ...string) error {
	deadline := time.Now().Add(timeout)
	stableCount := 0
	var lastStates []string
	var lastErr error

	for time.Now().Before(deadline) {
		allStable := true
		states := make([]string, 0, len(namespaces))
		for _, namespace := range namespaces {
			ns, err := k8sClient.Clientset.CoreV1().Namespaces().Get(ctx, namespace, metav1.GetOptions{})
			switch {
			case apierrors.IsNotFound(err):
				states = append(states, namespace+"=absent")
			case err != nil:
				allStable = false
				lastErr = err
				states = append(states, namespace+"=unknown")
			case ns.DeletionTimestamp != nil:
				allStable = false
				states = append(states, namespace+"=terminating")
			default:
				states = append(states, namespace+"=active")
			}
		}
		lastStates = states
		if allStable {
			stableCount++
			if stableCount >= 3 {
				return nil
			}
		} else {
			stableCount = 0
		}
		if waitCleanupPoll(ctx, poll) {
			return fmt.Errorf("context canceled waiting for namespaces stable: %w", ctx.Err())
		}
	}
	if lastErr != nil {
		return fmt.Errorf("namespaces %v not stable within %v; last states: %v: %w", namespaces, timeout, lastStates, lastErr)
	}
	return fmt.Errorf("namespaces %v not stable within %v; last states: %v", namespaces, timeout, lastStates)
}

func waitCleanupPoll(ctx context.Context, d time.Duration) bool {
	timer := time.NewTimer(d)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return true
	case <-timer.C:
		return false
	}
}
