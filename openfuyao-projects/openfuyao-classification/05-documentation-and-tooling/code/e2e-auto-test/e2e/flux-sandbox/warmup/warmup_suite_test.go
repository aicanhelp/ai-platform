//go:build e2e

/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You can obtain a copy of PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package warmup

import (
	"log"
	"testing"

	ginkgo "github.com/onsi/ginkgo/v2"
	gomega "github.com/onsi/gomega"

	fenv "gitcode.com/openFuyao/e2e-auto-test/e2e/framework/env"
)

func init() {
	// 与本仓库其他 Suite 保持一致：从 e2e 根目录加载 .env（已导出的环境变量优先）。
	if err := fenv.LoadEnv(".env"); err != nil {
		log.Fatalf("Failed to load .env file: %v", err)
	}
}

// TestWarmup is the entry point Ginkgo uses to discover and run every warm
// spec in this suite.
func TestWarmup(t *testing.T) {
	gomega.RegisterFailHandler(ginkgo.Fail)
	ginkgo.RunSpecs(t, "Warm E2E Suite")
}

var _ = ginkgo.BeforeSuite(func() {
	setupWarmEnv(ginkgo.GinkgoT())
})

var _ = ginkgo.AfterSuite(func() {
	teardownWarmEnv()
})
