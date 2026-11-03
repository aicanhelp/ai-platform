/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of the License at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package scriptutil

import (
	"context"
	"testing"

	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

func TestLoadBatchConfig(t *testing.T) {
	tests := []struct {
		name       string
		configType BatchConfigType
		nodeIP     string
		objects    []client.Object
		wantName   string
		wantErr    bool
	}{
		{
			name:       "postprocess batch config found",
			configType: PostprocessBatchConfig,
			nodeIP:     "10.0.0.1",
			objects: []client.Object{
				newBatchMapping("postprocess", `{"10.0.0.1":"001"}`),
				newBatchConfig("postprocess", "001"),
			},
			wantName: "postprocess-config-batch-001",
		},
		{
			name:       "preprocess batch config found",
			configType: PreprocessBatchConfig,
			nodeIP:     "10.0.0.1",
			objects: []client.Object{
				newBatchMapping("preprocess", `{"10.0.0.1":"002"}`),
				newBatchConfig("preprocess", "002"),
			},
			wantName: "preprocess-config-batch-002",
		},
		{
			name:       "mapping config does not exist",
			configType: PostprocessBatchConfig,
			nodeIP:     "10.0.0.1",
		},
		{
			name:       "node is not mapped",
			configType: PostprocessBatchConfig,
			nodeIP:     "10.0.0.1",
			objects: []client.Object{
				newBatchMapping("postprocess", `{"10.0.0.2":"001"}`),
			},
		},
		{
			name:       "batch config does not exist",
			configType: PostprocessBatchConfig,
			nodeIP:     "10.0.0.1",
			objects: []client.Object{
				newBatchMapping("postprocess", `{"10.0.0.1":"001"}`),
			},
		},
		{
			name:       "mapping is invalid",
			configType: PostprocessBatchConfig,
			nodeIP:     "10.0.0.1",
			objects: []client.Object{
				newBatchMapping("postprocess", `{bad json`),
			},
			wantErr: true,
		},
		{
			name:       "config type is invalid",
			configType: BatchConfigType("invalid"),
			nodeIP:     "10.0.0.1",
			wantErr:    true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			scheme := runtime.NewScheme()
			require.NoError(t, corev1.AddToScheme(scheme))
			c := fake.NewClientBuilder().WithScheme(scheme).WithObjects(tt.objects...).Build()

			config, err := LoadBatchConfig(context.Background(), c, tt.configType, tt.nodeIP)
			if tt.wantErr {
				require.Error(t, err)
				return
			}

			require.NoError(t, err)
			if tt.wantName == "" {
				require.Nil(t, config)
				return
			}
			require.Equal(t, tt.wantName, config.Name)
		})
	}
}

func newBatchMapping(configType, mapping string) *corev1.ConfigMap {
	return &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: scriptConfigNamespace,
			Name:      configType + "-node-batch-mapping",
		},
		Data: map[string]string{"mapping.json": mapping},
	}
}

func newBatchConfig(configType, batchID string) *corev1.ConfigMap {
	return &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: scriptConfigNamespace,
			Name:      configType + "-config-batch-" + batchID,
		},
	}
}
