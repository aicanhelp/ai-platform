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
	"strings"
	"testing"
	"time"

	cvv1alpha1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/v1alpha1"
)

func TestFromCRD_DisabledSkipsChecks(t *testing.T) {
	spec, err := FromCRD(&cvv1alpha1.HealthCheckSpec{
		Enabled: false,
		Timeout: "not-a-duration",
		Checks: []cvv1alpha1.HealthCheckItemSpec{{
			Type: string(CheckTypeCustom),
		}},
	})
	if err != nil {
		t.Fatalf("disabled convert: %v", err)
	}
	if spec.Enabled || len(spec.Checks) != 0 {
		t.Fatalf("expected empty disabled spec, got %#v", spec)
	}
}

func TestFromCRD_AllTypes(t *testing.T) {
	spec, err := FromCRD(&cvv1alpha1.HealthCheckSpec{
		Enabled:  true,
		Timeout:  "5m",
		Interval: "2s",
		Checks: []cvv1alpha1.HealthCheckItemSpec{
			{
				Type: string(CheckTypePodReady),
				PodReady: &cvv1alpha1.PodReadyCheckSpec{
					Namespace:     "kube-system",
					LabelSelector: "k8s-app=kube-dns",
					MinReady:      2,
				},
			},
			{
				Type: string(CheckTypeEndpointReady),
				EndpointReady: &cvv1alpha1.EndpointReadyCheckSpec{
					Namespace:   "kube-system",
					ServiceName: "kube-dns",
					Port:        53,
				},
			},
			{
				Type:   string(CheckTypeCustom),
				Custom: &cvv1alpha1.CustomCheckSpec{Command: "curl -sf http://127.0.0.1/healthz"},
			},
		},
	})
	if err != nil {
		t.Fatalf("convert: %v", err)
	}
	if spec.Timeout != 5*time.Minute || spec.Interval != 2*time.Second {
		t.Fatalf("unexpected durations: timeout=%v interval=%v", spec.Timeout, spec.Interval)
	}
	if len(spec.Checks) != 3 {
		t.Fatalf("expected 3 checks, got %d", len(spec.Checks))
	}
	if spec.Checks[0].Type != CheckTypePodReady ||
		spec.Checks[0].Namespace != "kube-system" ||
		spec.Checks[0].LabelSelector != "k8s-app=kube-dns" ||
		spec.Checks[0].MinReady != 2 {
		t.Fatalf("unexpected podReady check: %#v", spec.Checks[0])
	}
	if spec.Checks[1].Type != CheckTypeEndpointReady ||
		spec.Checks[1].ServiceName != "kube-dns" ||
		spec.Checks[1].Port != 53 {
		t.Fatalf("unexpected endpointReady check: %#v", spec.Checks[1])
	}
	wantCmd := []string{"/bin/sh", "-c", "curl -sf http://127.0.0.1/healthz"}
	if spec.Checks[2].Type != CheckTypeCustom ||
		len(spec.Checks[2].Command) != 3 ||
		spec.Checks[2].Command[0] != wantCmd[0] ||
		spec.Checks[2].Command[1] != wantCmd[1] ||
		spec.Checks[2].Command[2] != wantCmd[2] {
		t.Fatalf("unexpected custom check: %#v", spec.Checks[2])
	}
}

func TestFromCRD_InvalidTimeout(t *testing.T) {
	_, err := FromCRD(&cvv1alpha1.HealthCheckSpec{
		Enabled: true,
		Timeout: "bogus",
	})
	if err == nil || !strings.Contains(err.Error(), "healthCheck.timeout") {
		t.Fatalf("expected timeout parse error, got %v", err)
	}
}

func TestFromCRD_MissingNestedConfig(t *testing.T) {
	_, err := FromCRD(&cvv1alpha1.HealthCheckSpec{
		Enabled: true,
		Checks: []cvv1alpha1.HealthCheckItemSpec{{
			Type: string(CheckTypePodReady),
		}},
	})
	if err == nil || !strings.Contains(err.Error(), "checks[0]") || !strings.Contains(err.Error(), "podReady") {
		t.Fatalf("expected missing podReady error, got %v", err)
	}
}
