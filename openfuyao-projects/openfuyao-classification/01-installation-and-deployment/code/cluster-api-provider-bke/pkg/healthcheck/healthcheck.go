/******************************************************************
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package healthcheck

import (
	"context"
	"fmt"
	"os/exec"
	"time"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/client-go/kubernetes"
)

// CheckType is the health check kind supported by Run.
type CheckType string

const (
	CheckTypePodReady      CheckType = "PodReady"
	CheckTypeEndpointReady CheckType = "EndpointReady"
	CheckTypeCustom        CheckType = "Custom"
)

const (
	defaultTimeout  = 5 * time.Minute
	defaultInterval = 2 * time.Second
)

// Spec configures health checks for Yaml/Helm Installer Apply.
type Spec struct {
	// Enabled skips Run entirely when false.
	Enabled bool
	// Timeout bounds the overall retry window; 0 uses defaultTimeout.
	Timeout time.Duration
	// Interval is the poll period between attempts; 0 uses defaultInterval.
	Interval time.Duration
	Checks   []Check
	// CommandRunner overrides custom-check execution (tests); nil uses exec.CommandContext.
	CommandRunner CommandRunner
}

// Check is one health probe.
type Check struct {
	Type CheckType

	// Namespace applies to PodReady / EndpointReady.
	Namespace string

	// LabelSelector is required for PodReady (Kubernetes label selector string).
	LabelSelector string
	// MinReady is the minimum Ready pods for PodReady.
	// When <= 0, all matched pods must be Ready (and at least one must match).
	MinReady int

	// ServiceName is required for EndpointReady.
	ServiceName string
	// Port optionally restricts EndpointReady to a specific port; 0 means any port.
	Port int

	// Command is argv for Custom (exit code 0 = pass). First element is the binary.
	Command []string
}

// CommandRunner runs custom health check commands.
type CommandRunner interface {
	Run(ctx context.Context, name string, args ...string) error
}

type execCommandRunner struct{}

func (execCommandRunner) Run(ctx context.Context, name string, args ...string) error {
	cmd := exec.CommandContext(ctx, name, args...)
	return cmd.Run()
}

// Run executes all checks until success or timeout.
func Run(ctx context.Context, client kubernetes.Interface, hc Spec) error {
	if !hc.Enabled || len(hc.Checks) == 0 {
		return nil
	}
	timeout := hc.Timeout
	if timeout <= 0 {
		timeout = defaultTimeout
	}
	interval := hc.Interval
	if interval <= 0 {
		interval = defaultInterval
	}
	runner := hc.CommandRunner
	if runner == nil {
		runner = execCommandRunner{}
	}

	deadlineCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	var lastErr error
	for {
		lastErr = runOnce(deadlineCtx, client, runner, hc.Checks)
		if lastErr == nil {
			return nil
		}
		timer := time.NewTimer(interval)
		select {
		case <-deadlineCtx.Done():
			timer.Stop()
			if lastErr != nil {
				return fmt.Errorf("healthcheck timed out: %w", lastErr)
			}
			return fmt.Errorf("healthcheck timed out: %w", deadlineCtx.Err())
		case <-timer.C:
		}
	}
}

func runOnce(ctx context.Context, client kubernetes.Interface, runner CommandRunner, checks []Check) error {
	for i, check := range checks {
		if err := ctx.Err(); err != nil {
			return err
		}
		var err error
		switch check.Type {
		case CheckTypePodReady:
			err = checkPodReady(ctx, client, check)
		case CheckTypeEndpointReady:
			err = checkEndpointReady(ctx, client, check)
		case CheckTypeCustom:
			err = checkCustom(ctx, runner, check)
		default:
			err = fmt.Errorf("unsupported health check type %q", check.Type)
		}
		if err != nil {
			return fmt.Errorf("check[%d] %s: %w", i, check.Type, err)
		}
	}
	return nil
}

func checkPodReady(ctx context.Context, client kubernetes.Interface, check Check) error {
	if client == nil {
		return fmt.Errorf("kubernetes client is nil")
	}
	if check.LabelSelector == "" {
		return fmt.Errorf("labelSelector is required")
	}
	selector, err := labels.Parse(check.LabelSelector)
	if err != nil {
		return fmt.Errorf("parse labelSelector: %w", err)
	}
	list, err := client.CoreV1().Pods(check.Namespace).List(ctx, metav1.ListOptions{
		LabelSelector: selector.String(),
	})
	if err != nil {
		return err
	}
	matched := len(list.Items)
	if matched == 0 {
		return fmt.Errorf("no pods matched labelSelector %q", check.LabelSelector)
	}
	ready := 0
	for i := range list.Items {
		if isPodReady(&list.Items[i]) {
			ready++
		}
	}
	// MinReady <= 0: all matched pods must be Ready (CRD / YAMLSpec contract).
	if check.MinReady <= 0 {
		if ready < matched {
			return fmt.Errorf("ready pods %d < matched %d (require all Ready)", ready, matched)
		}
		return nil
	}
	if ready < check.MinReady {
		return fmt.Errorf("ready pods %d < minReady %d (matched %d)", ready, check.MinReady, matched)
	}
	return nil
}

func isPodReady(pod *corev1.Pod) bool {
	if pod == nil {
		return false
	}
	for _, c := range pod.Status.Conditions {
		if c.Type == corev1.PodReady && c.Status == corev1.ConditionTrue {
			return true
		}
	}
	return false
}

func checkEndpointReady(ctx context.Context, client kubernetes.Interface, check Check) error {
	if client == nil {
		return fmt.Errorf("kubernetes client is nil")
	}
	if check.ServiceName == "" {
		return fmt.Errorf("serviceName is required")
	}
	ep, err := client.CoreV1().Endpoints(check.Namespace).Get(ctx, check.ServiceName, metav1.GetOptions{})
	if err != nil {
		return err
	}
	for _, subset := range ep.Subsets {
		if len(subset.Addresses) == 0 {
			continue
		}
		if check.Port <= 0 || subsetHasPort(subset, int32(check.Port)) {
			return nil
		}
	}
	if check.Port > 0 {
		return fmt.Errorf("service %s/%s has no ready endpoint addresses for port %d",
			check.Namespace, check.ServiceName, check.Port)
	}
	return fmt.Errorf("service %s/%s has no ready endpoint addresses", check.Namespace, check.ServiceName)
}

func subsetHasPort(subset corev1.EndpointSubset, port int32) bool {
	for _, p := range subset.Ports {
		if p.Port == port {
			return true
		}
	}
	return false
}

func checkCustom(ctx context.Context, runner CommandRunner, check Check) error {
	if len(check.Command) == 0 {
		return fmt.Errorf("command is required")
	}
	if runner == nil {
		return fmt.Errorf("command runner is nil")
	}
	name := check.Command[0]
	var args []string
	if len(check.Command) > 1 {
		args = check.Command[1:]
	}
	if err := runner.Run(ctx, name, args...); err != nil {
		return fmt.Errorf("custom command %v: %w", check.Command, err)
	}
	return nil
}
