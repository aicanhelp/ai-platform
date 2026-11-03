/*
* Copyright (c) 2024 China Unicom Digital Technology Co., Ltd.
* openFuyao is licensed under Mulan PSL v2.
* You can use this software according to the terms and conditions of the Mulan PSL v2.
* You may obtain a copy of Mulan PSL v2 at:
*          http://license.coscl.org.cn/MulanPSL2
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
* EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
* MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
* See the Mulan PSL v2 for more details.
* Author: YuXiang Guo
* Date: 2024-11-25
 */

package core

import (
	"encoding/json"
	"reflect"
	"testing"

	"openfuyao.com/colocation-management/pkg/common"
	"openfuyao.com/colocation-management/pkg/utils"
)

func TestConvertNRIRawPod2PodInfo(t *testing.T) {

	_, configuration, marshal := SetupTest(t)

	pod := createTestPod(marshal)

	info := PodInfo{}
	info.ConvertNRIRawPod2PodInfo(&pod)

	expectinfo := PodInfo{
		PodRequest: PodRequest{
			PodMeta: PodSandboxMetadata{
				Name:      "pod1",
				UID:       "pod1",
				Namespace: "default",
			},
			PodSandboxConfig: utils.PodSandboxConfig{
				Annotations: map[string]string{
					common.ExtenderResourceConfigurationAnnotation: string(marshal),
				},
				Labels: map[string]string{},
				Linux: &utils.LinuxPodSandboxConfig{
					CgroupParent: "kubepods.slice/kubepods-podxxxxxx.slice",
				},
			},
			ExtendedResourceConfiguration: &configuration,
		},
	}

	if !reflect.DeepEqual(info.PodRequest.PodMeta, expectinfo.PodRequest.PodMeta) {
		t.Errorf("Expected  to be %v, got %v", expectinfo.PodRequest.PodMeta, info.PodRequest.PodMeta)
	}

	if !reflect.DeepEqual(info.PodRequest.PodSandboxConfig, expectinfo.PodRequest.PodSandboxConfig) {
		t.Errorf("Expected  to be %v, got %v", expectinfo.PodRequest.PodSandboxConfig, info.PodRequest.PodSandboxConfig)
	}
	expectedJSON, err := json.Marshal(expectinfo.PodRequest.ExtendedResourceConfiguration)
	if err != nil {
		t.Errorf("Error marshaling expected JSON: %v", err)
	}
	actualJSON, err := json.Marshal(info.PodRequest.ExtendedResourceConfiguration)
	if err != nil {
		t.Errorf("Error marshaling actual JSON: %v", err)
	}

	if string(expectedJSON) != string(actualJSON) {
		t.Errorf("Expected ExtendedResourceConfiguration to be %s, got %s",
			string(expectedJSON), string(actualJSON))
	}

}

func TestPodAdmit(t *testing.T) {
	tests := []struct {
		name     string
		podInfo  *PodInfo
		expected AdmitResult
	}{
		{
			name: "empty annotations",
			podInfo: &PodInfo{
				PodRequest: PodRequest{
					PodSandboxConfig: utils.PodSandboxConfig{
						Annotations: map[string]string{},
					},
				},
			},
			expected: CheckAnnotationAdmit(map[string]string{}),
		},
		{
			name: "with annotations",
			podInfo: &PodInfo{
				PodRequest: PodRequest{
					PodSandboxConfig: utils.PodSandboxConfig{
						Annotations: map[string]string{
							"key1": "value1",
							"key2": "value2"},
					},
				},
			},
			expected: CheckAnnotationAdmit(map[string]string{
				"key1": "value1",
				"key2": "value2",
			}),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := tt.podInfo.Admit()

			if !reflect.DeepEqual(result, tt.expected) {
				t.Errorf("Expected %v, got %v", tt.expected, result)
			}
		})
	}
}
