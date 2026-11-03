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

// Package dataplane contains the flux-sandbox data plane e2e specs: the
// flux-sandbox-agent HTTP API, sandbox lifecycle, reconcile behavior, and
// sandbox-watcher orphan cleanup. The E2B runtime and the control plane are
// out of scope.
package dataplane

import (
	"log"
	"testing"

	ginkgo "github.com/onsi/ginkgo/v2"
	gomega "github.com/onsi/gomega"

	"gitcode.com/openFuyao/e2e-auto-test/e2e/flux-sandbox/framework"
	fenv "gitcode.com/openFuyao/e2e-auto-test/e2e/framework/env"
)

func init() {
	// 与本仓库其他 Suite 保持一致：从 e2e 根目录加载 .env（已导出的环境变量优先）。
	if err := fenv.LoadEnv(".env"); err != nil {
		log.Fatalf("Failed to load .env file: %v", err)
	}
}

// TestDataplane is the entry point Ginkgo uses to discover and run every
// spec in this suite.
func TestDataplane(t *testing.T) {
	gomega.RegisterFailHandler(ginkgo.Fail)
	ginkgo.RunSpecs(t, "Dataplane E2E Suite")
}

var _ = ginkgo.BeforeSuite(func() {
	e, err := framework.Setup()
	gomega.Expect(err).NotTo(gomega.HaveOccurred(), "E2E setup failed")
	env = *e

	// Start every spec from a clean slate: no leftover sandboxes from
	// previous runs, so full capacity is available.
	cleanSlate()
})

var _ = ginkgo.AfterSuite(func() {
	env.Teardown()
})
