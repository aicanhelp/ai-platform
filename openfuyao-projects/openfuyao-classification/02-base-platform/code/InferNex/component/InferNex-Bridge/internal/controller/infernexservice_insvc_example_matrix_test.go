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

package controller

import (
	"testing"

	"k8s.io/utils/ptr"

	infernexv1alpha1 "gitcode.com/openFuyao/InferNex/api/v1alpha1"
)

// Mirrors config/examples/insvc matrix (ag-01–03, pd-01–05).
func TestInsvcExampleMatrixWorkloadKind(t *testing.T) {
	t.Parallel()

	cases := []struct {
		id       string
		workload *infernexv1alpha1.InferenceEngineWorkloadSpec
		wantKind string
		wantSize int32
	}{
		{"ag-01", &infernexv1alpha1.InferenceEngineWorkloadSpec{DataParallelSize: ptr.To(int32(1)), DataParallelSizeLocal: ptr.To(int32(1)), Template: testTemplate("e:v1")}, workloadKindDeployment, 1},
		{"ag-02", &infernexv1alpha1.InferenceEngineWorkloadSpec{DataParallelSize: ptr.To(int32(1)), DataParallelSizeLocal: ptr.To(int32(1)), Template: testTemplate("e:v1")}, workloadKindDeployment, 1},
		{"ag-03", &infernexv1alpha1.InferenceEngineWorkloadSpec{DataParallelSize: ptr.To(int32(2)), DataParallelSizeLocal: ptr.To(int32(1)), Template: testTemplate("e:v1")}, workloadKindLeaderWorkerSet, 2},
		{"pd-04-prefill", &infernexv1alpha1.InferenceEngineWorkloadSpec{DataParallelSize: ptr.To(int32(2)), DataParallelSizeLocal: ptr.To(int32(2)), Template: testTemplate("e:v1")}, workloadKindDeployment, 1},
		{"pd-05-decode", &infernexv1alpha1.InferenceEngineWorkloadSpec{DataParallelSize: ptr.To(int32(2)), DataParallelSizeLocal: ptr.To(int32(1)), Template: testTemplate("e:v1")}, workloadKindLeaderWorkerSet, 2},
	}
	for _, tc := range cases {
		t.Run(tc.id, func(t *testing.T) {
			plan, err := buildEngineWorkloadPlan("engine-aggregate", tc.workload)
			if err != nil {
				t.Fatalf("buildEngineWorkloadPlan: %v", err)
			}
			if plan.WorkloadKind != tc.wantKind {
				t.Fatalf("kind = %q, want %q", plan.WorkloadKind, tc.wantKind)
			}
			if plan.GroupSize != tc.wantSize {
				t.Fatalf("groupSize = %d, want %d", plan.GroupSize, tc.wantSize)
			}
		})
	}
}
