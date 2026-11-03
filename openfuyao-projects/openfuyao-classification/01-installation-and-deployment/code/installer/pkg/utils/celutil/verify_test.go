/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *           http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package celutil

import (
	"fmt"
	"reflect"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/google/cel-go/cel"
	"github.com/stretchr/testify/assert"
)

// TestVerifyParams tests the VerifyParams function.
func TestVerifyParamsProgramError(t *testing.T) {
	expression := "a > b"
	params := map[string]interface{}{"a": 5, "b": 3}
	opts := cel.Declarations()

	patches := gomonkey.ApplyFunc(cel.NewEnv, func(opts ...cel.EnvOption) (*cel.Env, error) {
		return &cel.Env{}, nil
	})

	defer patches.Reset()

	patches.ApplyMethod(reflect.TypeOf(&cel.Env{}), "Compile", func(_ *cel.Env, expr string) (*cel.Ast, *cel.Issues) {
		return &cel.Ast{}, nil
	})

	patches.ApplyMethod(reflect.TypeOf(&cel.Env{}), "Program", func(_ *cel.Env, ast *cel.Ast) (cel.Program, error) {
		return nil, fmt.Errorf("program error")
	})

	result, err := VerifyParams(expression, params, opts)
	assert.Error(t, err)
	assert.False(t, result)
}
