/******************************************************************
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * cluster-api-provider-bke is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package patchutil

import (
	jsonpatch "github.com/evanphx/json-patch"
	jsonpatchv2 "gomodules.xyz/jsonpatch/v2"
	"k8s.io/apimachinery/pkg/util/json"
)

func Diff(old, new interface{}) (jsonpatch.Patch, error) {
	oldBytes, err := json.Marshal(old)
	if err != nil {
		return nil, err
	}
	newBytes, err := json.Marshal(new)
	if err != nil {
		return nil, err
	}

	patches, err := jsonpatchv2.CreatePatch(oldBytes, newBytes)
	if err != nil {
		return nil, err
	}

	if patches == nil || len(patches) == 0 {
		return nil, nil
	}

	pBytes, err := json.Marshal(patches)
	if err != nil {
		return nil, err
	}

	decodePatch, err := jsonpatch.DecodePatch(pBytes)
	if err != nil {
		return nil, err
	}
	return decodePatch, nil
}

func GetDiffPaths(patch jsonpatch.Patch) []string {
	var diffpaths []string
	var errors []error

	for _, diff := range patch {
		path, err := diff.Path()
		if err != nil {
			continue
		}
		diffpaths = append(diffpaths, path)
	}
	if len(errors) > 0 {
		return nil
	}
	return diffpaths
}
