/******************************************************************
 * Copyright (c) 2026 Bocloud Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

// Package preflight hosts module-level architecture tests that enforce the
// dependency boundary between the SDK and its host (bkeadm). The SDK module
// gopkg.openfuyao.cn/preflight MUST NOT import any package from
// gopkg.openfuyao.cn/bkeadm/... — this keeps the SDK host-agnostic so it can
// be moved to its own repository later by deleting the local replace.
package preflight

import (
	"os/exec"
	"strings"
	"testing"
)

// forbiddenImportPrefixes lists the host-side module roots the SDK must never
// import. Adding any of these would re-couple the SDK to a specific host.
var forbiddenImportPrefixes = []string{
	"gopkg.openfuyao.cn/bkeadm/",
}

// TestNoHostImports freezes the dependency boundary: no package in the
// preflight module may transitively import anything under bkeadm. This is the
// single guard that makes the SDK relocatable by deleting the local replace.
//
// It shells out to `go list -deps ./...` (the same tool the build uses) and
// asserts the flattened dependency set contains no forbidden prefix. The test
// runs inside the nested module's own go.mod context.
func TestNoHostImports(t *testing.T) {
	out, err := exec.Command("go", "list", "-deps", "./...").CombinedOutput()
	if err != nil {
		t.Fatalf("go list -deps ./... failed: %v\n%s", err, out)
	}
	var leaks []string
	for _, line := range strings.Split(string(out), "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		for _, prefix := range forbiddenImportPrefixes {
			if strings.HasPrefix(line, prefix) {
				leaks = append(leaks, line)
			}
		}
	}
	if len(leaks) > 0 {
		t.Fatalf("SDK module imports host packages (forbidden, breaks relocatability):\n  %s\n"+
			"Remove the import or move the host-specific code into bkeadm's preflightbridge.",
			strings.Join(leaks, "\n  "))
	}
}

// TestModulePathIsStable freezes the module path so a future repository move
// does not silently rename it. The path gopkg.openfuyao.cn/preflight is the
// future standalone module name.
func TestModulePathIsStable(t *testing.T) {
	out, err := exec.Command("go", "list", "-m").CombinedOutput()
	if err != nil {
		t.Fatalf("go list -m failed: %v\n%s", err, out)
	}
	got := strings.TrimSpace(string(out))
	if got != "gopkg.openfuyao.cn/preflight" {
		t.Fatalf("module path = %q, want gopkg.openfuyao.cn/preflight", got)
	}
}
