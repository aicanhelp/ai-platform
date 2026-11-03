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

package upgrade

import "testing"

func TestDeclarativeUpgradeCatalog(t *testing.T) {
	var manifestCount, inlineCount int
	for _, spec := range DeclarativeUpgradeCatalog() {
		switch spec.Mode {
		case UpgradeExecutionManifest:
			manifestCount++
			if spec.ManifestPath == "" {
				t.Fatalf("manifest spec %q missing path", spec.Name)
			}
		case UpgradeExecutionInline:
			inlineCount++
			if spec.InlineHandler == "" {
				t.Fatalf("inline spec %q missing handler", spec.Name)
			}
		}
	}
	if manifestCount != 3 {
		t.Fatalf("expected 3 manifest components, got %d", manifestCount)
	}
	if inlineCount != 7 {
		t.Fatalf("expected 7 inline handlers, got %d", inlineCount)
	}
}

func TestManifestComponentManifestPath(t *testing.T) {
	got := ManifestComponentManifestPath(ComponentProvider, ComponentManifestVersion)
	want := "provider/v1.0.0/component.yaml"
	if got != want {
		t.Fatalf("got %q want %q", got, want)
	}
}

func TestInlineUpgradeHandlers(t *testing.T) {
	handlers := InlineUpgradeHandlers()
	if len(handlers) != 7 {
		t.Fatalf("expected 7 handlers, got %d", len(handlers))
	}
}
