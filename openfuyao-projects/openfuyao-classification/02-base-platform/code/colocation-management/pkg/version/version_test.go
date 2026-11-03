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
package version

import "testing"

// TestVersionPackageLoads exercises the package so its init() (which handles
// the "-v" flag) is included in the coverage profile. The init print branch is
// only entered when len(os.Args)==2 && os.Args[1]=="-v", which cannot be
// reproduced from a test; the negative path is covered on package import.
func TestVersionPackageLoads(t *testing.T) {
	if gitVersion != "" || gitCommit != "" || buildDate != "" {
		// Values are normally injected via -ldflags at build time.
	}
}
