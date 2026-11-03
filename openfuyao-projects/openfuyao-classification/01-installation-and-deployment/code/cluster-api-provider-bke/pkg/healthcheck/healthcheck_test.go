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
	"testing"
	"time"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/kubernetes/fake"
)

type recordingRunner struct {
	calls int
	err   error
}

func (r *recordingRunner) Run(context.Context, string, ...string) error {
	r.calls++
	return r.err
}

func TestRun_DisabledOrEmpty(t *testing.T) {
	if err := Run(context.Background(), nil, Spec{Enabled: false, Checks: []Check{{Type: CheckTypeCustom}}}); err != nil {
		t.Fatal(err)
	}
	if err := Run(context.Background(), nil, Spec{Enabled: true}); err != nil {
		t.Fatal(err)
	}
}

func TestRun_PodReadySuccessAndMinReady(t *testing.T) {
	scheme := runtime.NewScheme()
	_ = corev1.AddToScheme(scheme)
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "p1",
			Namespace: "ns",
			Labels:    map[string]string{"app": "x"},
		},
		Status: corev1.PodStatus{
			Conditions: []corev1.PodCondition{{Type: corev1.PodReady, Status: corev1.ConditionTrue}},
		},
	}
	client := fake.NewSimpleClientset(pod)
	err := Run(context.Background(), client, Spec{
		Enabled:  true,
		Timeout:  time.Second,
		Interval: 10 * time.Millisecond,
		Checks: []Check{{
			Type:          CheckTypePodReady,
			Namespace:     "ns",
			LabelSelector: "app=x",
			MinReady:      1,
		}},
	})
	if err != nil {
		t.Fatalf("expected success: %v", err)
	}

	err = Run(context.Background(), client, Spec{
		Enabled:  true,
		Timeout:  50 * time.Millisecond,
		Interval: 10 * time.Millisecond,
		Checks: []Check{{
			Type:          CheckTypePodReady,
			Namespace:     "ns",
			LabelSelector: "app=x",
			MinReady:      2,
		}},
	})
	if err == nil {
		t.Fatal("expected timeout when minReady not met")
	}
}

func TestRun_PodReadyMinReadyZeroRequiresAll(t *testing.T) {
	ready := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: "p1", Namespace: "ns", Labels: map[string]string{"app": "x"},
		},
		Status: corev1.PodStatus{
			Conditions: []corev1.PodCondition{{Type: corev1.PodReady, Status: corev1.ConditionTrue}},
		},
	}
	notReady := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: "p2", Namespace: "ns", Labels: map[string]string{"app": "x"},
		},
		Status: corev1.PodStatus{
			Conditions: []corev1.PodCondition{{Type: corev1.PodReady, Status: corev1.ConditionFalse}},
		},
	}

	okClient := fake.NewSimpleClientset(ready)
	if err := Run(context.Background(), okClient, Spec{
		Enabled:  true,
		Timeout:  time.Second,
		Interval: 10 * time.Millisecond,
		Checks: []Check{{
			Type:          CheckTypePodReady,
			Namespace:     "ns",
			LabelSelector: "app=x",
			MinReady:      0,
		}},
	}); err != nil {
		t.Fatalf("minReady=0 with all ready: %v", err)
	}

	failClient := fake.NewSimpleClientset(ready, notReady)
	if err := Run(context.Background(), failClient, Spec{
		Enabled:  true,
		Timeout:  40 * time.Millisecond,
		Interval: 10 * time.Millisecond,
		Checks: []Check{{
			Type:          CheckTypePodReady,
			Namespace:     "ns",
			LabelSelector: "app=x",
			MinReady:      0,
		}},
	}); err == nil {
		t.Fatal("expected failure when minReady=0 and not all pods ready")
	}
}

func TestRun_EndpointReady(t *testing.T) {
	ep := &corev1.Endpoints{
		ObjectMeta: metav1.ObjectMeta{Name: "svc", Namespace: "ns"},
		Subsets: []corev1.EndpointSubset{{
			Addresses: []corev1.EndpointAddress{{IP: "10.0.0.1"}},
			Ports:     []corev1.EndpointPort{{Port: 80}},
		}},
	}
	client := fake.NewSimpleClientset(ep)
	if err := Run(context.Background(), client, Spec{
		Enabled: true,
		Timeout: time.Second,
		Checks: []Check{{
			Type:        CheckTypeEndpointReady,
			Namespace:   "ns",
			ServiceName: "svc",
		}},
	}); err != nil {
		t.Fatalf("expected endpoint ready: %v", err)
	}

	if err := Run(context.Background(), client, Spec{
		Enabled: true,
		Timeout: time.Second,
		Checks: []Check{{
			Type:        CheckTypeEndpointReady,
			Namespace:   "ns",
			ServiceName: "svc",
			Port:        80,
		}},
	}); err != nil {
		t.Fatalf("expected endpoint ready on port 80: %v", err)
	}

	if err := Run(context.Background(), client, Spec{
		Enabled:  true,
		Timeout:  40 * time.Millisecond,
		Interval: 10 * time.Millisecond,
		Checks: []Check{{
			Type:        CheckTypeEndpointReady,
			Namespace:   "ns",
			ServiceName: "svc",
			Port:        443,
		}},
	}); err == nil {
		t.Fatal("expected timeout when required port is missing")
	}

	empty := fake.NewSimpleClientset(&corev1.Endpoints{
		ObjectMeta: metav1.ObjectMeta{Name: "svc", Namespace: "ns"},
	})
	if err := Run(context.Background(), empty, Spec{
		Enabled:  true,
		Timeout:  40 * time.Millisecond,
		Interval: 10 * time.Millisecond,
		Checks: []Check{{
			Type:        CheckTypeEndpointReady,
			Namespace:   "ns",
			ServiceName: "svc",
		}},
	}); err == nil {
		t.Fatal("expected timeout for empty endpoints")
	}
}

func TestRun_CustomPassAndFail(t *testing.T) {
	ok := &recordingRunner{}
	if err := Run(context.Background(), nil, Spec{
		Enabled:       true,
		Timeout:       time.Second,
		CommandRunner: ok,
		Checks:        []Check{{Type: CheckTypeCustom, Command: []string{"true"}}},
	}); err != nil {
		t.Fatal(err)
	}
	if ok.calls != 1 {
		t.Fatalf("calls=%d", ok.calls)
	}

	fail := &recordingRunner{err: fmt.Errorf("exit 1")}
	if err := Run(context.Background(), nil, Spec{
		Enabled:       true,
		Timeout:       40 * time.Millisecond,
		Interval:      10 * time.Millisecond,
		CommandRunner: fail,
		Checks:        []Check{{Type: CheckTypeCustom, Command: []string{"false"}}},
	}); err == nil {
		t.Fatal("expected custom failure timeout")
	}
	if fail.calls < 1 {
		t.Fatal("expected retries")
	}
}

func TestRun_UnsupportedType(t *testing.T) {
	err := Run(context.Background(), nil, Spec{
		Enabled:  true,
		Timeout:  30 * time.Millisecond,
		Interval: 5 * time.Millisecond,
		Checks:   []Check{{Type: "Unknown"}},
	})
	if err == nil {
		t.Fatal("expected unsupported type error")
	}
}
