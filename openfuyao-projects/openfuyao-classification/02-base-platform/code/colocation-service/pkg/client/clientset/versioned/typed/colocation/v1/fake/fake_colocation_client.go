/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package fake

import (
	rest "k8s.io/client-go/rest"
	testing "k8s.io/client-go/testing"

	v1 "openfuyao.com/colocation-service/pkg/client/clientset/versioned/typed/colocation/v1"
)

type FakeColocationV1 struct {
	*testing.Fake
}

func (c *FakeColocationV1) ColocationWorkloadMatchRules(namespace string) v1.ColocationWorkloadMatchRuleInterface {
	return &FakeColocationWorkloadMatchRules{Fake: c, ns: namespace}
}

// RESTClient returns a RESTClient that is used to communicate
// with API server by this client implementation.
func (c *FakeColocationV1) RESTClient() rest.Interface {
	var ret *rest.RESTClient
	return ret
}
