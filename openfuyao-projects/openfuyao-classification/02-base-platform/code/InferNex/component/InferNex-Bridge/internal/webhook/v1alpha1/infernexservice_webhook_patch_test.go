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
package v1alpha1

import (
	"encoding/json"
	"testing"
)

func TestApplyInferNexPatchesToLLMInferenceServiceConfig_decodeTemplate(t *testing.T) {
	raw := `{
  "apiVersion": "serving.kserve.io/v1alpha1",
  "kind": "LLMInferenceServiceConfig",
  "spec": {
    "template": {
      "containers": [{
        "name": "main",
        "securityContext": {
          "capabilities": { "drop": ["ALL"] }
        }
      }],
      "initContainers": [{
        "name": "llm-d-routing-sidecar",
        "image": "sidecar:1"
      }]
    }
  }
}`
	var obj map[string]interface{}
	if err := json.Unmarshal([]byte(raw), &obj); err != nil {
		t.Fatal(err)
	}
	applyInferNexPatchesToLLMInferenceServiceConfig(obj)

	spec, ok := obj["spec"].(map[string]interface{})
	if !ok {
		t.Fatalf("spec should be an object, got %T", obj["spec"])
	}
	tpl, ok := spec["template"].(map[string]interface{})
	if !ok {
		t.Fatalf("template should be an object, got %T", spec["template"])
	}
	containers, ok := tpl["containers"].([]interface{})
	if !ok || len(containers) == 0 {
		t.Fatalf("containers should be non-empty array, got %T", tpl["containers"])
	}
	main, ok := containers[0].(map[string]interface{})
	if !ok {
		t.Fatalf("containers[0] should be object, got %T", containers[0])
	}
	sc, ok := main["securityContext"].(map[string]interface{})
	if !ok {
		t.Fatalf("securityContext should be object, got %T", main["securityContext"])
	}
	caps, ok := sc["capabilities"].(map[string]interface{})
	if !ok {
		t.Fatalf("capabilities should be preserved, got %T", sc["capabilities"])
	}
	drop, ok := caps["drop"].([]interface{})
	if !ok || len(drop) != 1 || drop[0] != "ALL" {
		t.Fatalf("expected drop ALL preserved (not stripped), got %v", drop)
	}

	inits, ok := tpl["initContainers"].([]interface{})
	if !ok {
		t.Fatalf("initContainers should be array, got %T", tpl["initContainers"])
	}
	if len(inits) != 0 {
		t.Fatalf("expected initContainers cleared, got len %d", len(inits))
	}
}
