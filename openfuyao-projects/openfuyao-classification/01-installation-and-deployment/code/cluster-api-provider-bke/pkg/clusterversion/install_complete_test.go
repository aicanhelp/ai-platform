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

package clusterversion

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	cvv1alpha1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/v1alpha1"
)

func TestNeedsInstallCompletion(t *testing.T) {
	tests := []struct {
		name string
		cv   *cvv1alpha1.ClusterVersion
		want bool
	}{
		{
			name: "installing",
			cv: &cvv1alpha1.ClusterVersion{
				Spec:   cvv1alpha1.ClusterVersionSpec{DesiredVersion: "v2.6.0"},
				Status: cvv1alpha1.ClusterVersionStatus{Phase: cvv1alpha1.ClusterVersionPhaseInstalling},
			},
			want: true,
		},
		{
			name: "empty current",
			cv: &cvv1alpha1.ClusterVersion{
				Spec:   cvv1alpha1.ClusterVersionSpec{DesiredVersion: "v2.6.0"},
				Status: cvv1alpha1.ClusterVersionStatus{Phase: cvv1alpha1.ClusterVersionPhasePending},
			},
			want: true,
		},
		{
			// Non-Upgrading phases still need install completion patch per current logic.
			name: "ready_not_upgrading",
			cv: &cvv1alpha1.ClusterVersion{
				Spec: cvv1alpha1.ClusterVersionSpec{DesiredVersion: "v2.6.0"},
				Status: cvv1alpha1.ClusterVersionStatus{
					CurrentVersion: "v2.6.0",
					Phase:          cvv1alpha1.ClusterVersionPhaseReady,
				},
			},
			want: true,
		},
		{
			name: "upgrading",
			cv: &cvv1alpha1.ClusterVersion{
				Spec: cvv1alpha1.ClusterVersionSpec{DesiredVersion: "v2.7.0"},
				Status: cvv1alpha1.ClusterVersionStatus{
					CurrentVersion: "v2.6.0",
					Phase:          cvv1alpha1.ClusterVersionPhaseUpgrading,
				},
			},
			want: false,
		},
		{
			name: "ready_pending_hop",
			cv: &cvv1alpha1.ClusterVersion{
				Spec: cvv1alpha1.ClusterVersionSpec{DesiredVersion: "v2.7.0"},
				Status: cvv1alpha1.ClusterVersionStatus{
					CurrentVersion: "v2.6.0",
					Phase:          cvv1alpha1.ClusterVersionPhaseReady,
				},
			},
			want: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := NeedsInstallCompletion(tt.cv); got != tt.want {
				t.Fatalf("NeedsInstallCompletion() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestApplyInstallCompleteStatus(t *testing.T) {
	cv := &cvv1alpha1.ClusterVersion{
		ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"},
		Spec:       cvv1alpha1.ClusterVersionSpec{DesiredVersion: "v2.6.0"},
		Status:     cvv1alpha1.ClusterVersionStatus{Phase: cvv1alpha1.ClusterVersionPhaseInstalling},
	}
	ApplyInstallCompleteStatus(cv, "v2.6.0")
	if cv.Status.CurrentVersion != "v2.6.0" {
		t.Fatalf("current %q", cv.Status.CurrentVersion)
	}
	if cv.Status.Phase != cvv1alpha1.ClusterVersionPhaseReady {
		t.Fatalf("phase %q", cv.Status.Phase)
	}
	if len(cv.Status.Conditions) != 1 || cv.Status.Conditions[0].Reason != "InstallComplete" {
		t.Fatalf("conditions %+v", cv.Status.Conditions)
	}
}
