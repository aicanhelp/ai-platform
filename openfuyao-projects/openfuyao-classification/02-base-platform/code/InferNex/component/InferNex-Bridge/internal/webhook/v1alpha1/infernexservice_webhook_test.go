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
	ginkgo "github.com/onsi/ginkgo/v2"
	gomega "github.com/onsi/gomega"

	infernexv1alpha1 "gitcode.com/openFuyao/InferNex/api/v1alpha1"
)

var _ = ginkgo.Describe("InferNexService Webhook", func() {
	var (
		obj *infernexv1alpha1.InferNexService
	)

	ginkgo.BeforeEach(func() {
		obj = &infernexv1alpha1.InferNexService{}
		gomega.Expect(obj).NotTo(gomega.BeNil(), "Expected obj to be initialized")
	})

	ginkgo.It("initializes webhook test object", func() {
		gomega.Expect(obj).NotTo(gomega.BeNil())
	})
})
