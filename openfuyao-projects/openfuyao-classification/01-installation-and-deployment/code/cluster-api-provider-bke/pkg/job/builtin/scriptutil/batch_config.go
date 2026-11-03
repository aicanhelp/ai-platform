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
	"fmt"

	"github.com/pkg/errors"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/util/json"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

const scriptConfigNamespace = "user-system"

// BatchConfigType identifies the script-processing stage that owns a batch config.
type BatchConfigType string

const (
	// PreprocessBatchConfig identifies batch configuration for preprocess scripts.
	PreprocessBatchConfig BatchConfigType = "preprocess"
	// PostprocessBatchConfig identifies batch configuration for postprocess scripts.
	PostprocessBatchConfig BatchConfigType = "postprocess"
)

// LoadBatchConfig finds the batch config assigned to nodeIP. A nil ConfigMap
// without an error means that the node has no batch config.
func LoadBatchConfig(ctx context.Context, c client.Client, configType BatchConfigType, nodeIP string) (*corev1.ConfigMap, error) {
	if configType != PreprocessBatchConfig && configType != PostprocessBatchConfig {
		return nil, errors.Errorf("unsupported script config type %q", configType)
	}

	mappingKey := client.ObjectKey{
		Namespace: scriptConfigNamespace,
		Name:      fmt.Sprintf("%s-node-batch-mapping", configType),
	}
	mappingCM := &corev1.ConfigMap{}
	if err := c.Get(ctx, mappingKey, mappingCM); err != nil {
		if apierrors.IsNotFound(err) {
			return nil, nil
		}
		return nil, errors.Wrapf(err, "get batch mapping ConfigMap %s", mappingKey.Name)
	}

	var mapping map[string]string
	if err := json.Unmarshal([]byte(mappingCM.Data["mapping.json"]), &mapping); err != nil {
		return nil, errors.Wrapf(err, "parse batch mapping ConfigMap %s", mappingKey.Name)
	}

	batchID, ok := mapping[nodeIP]
	if !ok {
		return nil, nil
	}

	configKey := client.ObjectKey{
		Namespace: scriptConfigNamespace,
		Name:      fmt.Sprintf("%s-config-batch-%s", configType, batchID),
	}
	configCM := &corev1.ConfigMap{}
	if err := c.Get(ctx, configKey, configCM); err != nil {
		if apierrors.IsNotFound(err) {
			return nil, nil
		}
		return nil, errors.Wrapf(err, "get batch config ConfigMap %s", configKey.Name)
	}

	return configCM, nil
}
