/******************************************************************
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package common

import (
	"encoding/json"
	"fmt"
	"os"

	"installer/pkg/fuyao/v1beta1/manifests"
	"installer/pkg/fuyao/v1beta1/metadata"
)

// UnmarshalJSON 统一json解析
func UnmarshalJSON[T *manifests.OsVersionMetaData | *manifests.ManifestMetaData | *metadata.MetaData | *metadata.Version](data T, filePath string) error {
	jsonData, err := os.ReadFile(filePath)
	if err != nil {
		return fmt.Errorf("Failed to read JSON file: %v\n", err)
	}

	err = json.Unmarshal(jsonData, data)
	if err != nil {
		return fmt.Errorf("Failed to unmarshal JSON: %v\n", err)
	}

	return nil
}
