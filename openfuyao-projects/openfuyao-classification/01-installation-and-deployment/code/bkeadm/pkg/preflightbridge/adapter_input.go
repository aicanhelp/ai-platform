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

package preflightbridge

import (
	"gopkg.openfuyao.cn/preflight/adapter/openfuyao/initadapter"
	"gopkg.openfuyao.cn/preflight/api"
)

// AdapterInput is the BKE host DTO. OpenFuyaoAdapter converts it to the
// generic SDK ResolveResult before Checker.Check is called.
type AdapterInput struct {
	Operation     api.Operation
	Init          *initadapter.Input
	ClusterCreate *ClusterCreateInput
}

// ClusterCreateInput is a sanitized BKE host projection. It contains no
// credentials and is never exposed by the generic SDK API.
type ClusterCreateInput struct {
	ClusterYAML []byte
	NodesYAML   []byte
	Workspace   string
}
