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

// Package tokenizedprompt defines the shared PrepareData contract for request tokenization.
// The DAG marker lives here while the runtime tokenized prompt payload remains on LLMRequest.
package tokenizedprompt

// TokenizedPromptKey is an exported constant used by Hermes Router.
const TokenizedPromptKey = "TokenizedPromptKey"

// Info is the comparable DAG marker for tokenized prompt dependencies.
type Info struct{}
