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

// Package apis defines dependency-free data types shared across planes.
// Query, index, ingest, and discovery all use this model.
package apis

// BlockHash is the canonical hex ID of one KV block.
// Build it only through vLLM, pkg/blockkey, or normalization paths.
type BlockHash string

// TransportEndpoint is the Mooncake client host:port used as the L3 index key.
// It is not a vLLM Pod IP; discovery.Registry resolves that mapping.
type TransportEndpoint string
