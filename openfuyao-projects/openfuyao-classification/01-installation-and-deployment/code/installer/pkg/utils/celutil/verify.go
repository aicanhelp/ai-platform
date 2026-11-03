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

	"github.com/google/cel-go/cel"
)

// VerifyParams is verified input parameters
func VerifyParams(expression string, params map[string]interface{}, opts cel.EnvOption) (bool, error) {
	// 创建 Cel 环境
	env, err := cel.NewEnv(opts)
	if err != nil {
		return false, fmt.Errorf("rrror creating Cel environment: %v", err)
	}

	// 解析 Cel 表达式
	ast, iss := env.Compile(expression)
	if iss.Err() != nil {
		return false, fmt.Errorf("error compiling expression: %v", iss.Err())
	}

	// 创建 Cel 评估器
	prg, err := env.Program(ast)
	if err != nil {
		return false, fmt.Errorf("error creating program: %v", err)
	}

	// 评估 Cel 表达式
	out, _, err := prg.Eval(params)
	if err != nil {
		return false, fmt.Errorf("error evaluating program: %v", err)
	}

	result, ok := out.Value().(bool)
	if !ok {
		return false, fmt.Errorf("error: cel eval out is not a boolean value")
	}

	return result, nil
}
