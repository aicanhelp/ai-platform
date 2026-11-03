/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package httputil

import (
	"testing"

	"github.com/stretchr/testify/assert"

	"installer-service/pkg/constant"
)

func TestGetDefaultSuccessResponseJson(t *testing.T) {
	t.Run("should return new instance each time", func(t *testing.T) {
		// 多次调用函数
		resp1 := GetDefaultSuccessResponseJson()
		resp2 := GetDefaultSuccessResponseJson()

		// 验证是不同的实例
		assert.NotSame(t, resp1, resp2, "Should return new instance each time")

		// 修改第一个实例
		resp1.Message = "modified"

		// 验证第二个实例未受影响
		assert.Equal(t, "success", resp2.Message, "Second instance should not be modified")
	})

	t.Run("should have expected constant values", func(t *testing.T) {
		// 调用函数
		// 验证常量值
		if constant.Success != 0 {
			t.Logf("constant.Success = %d (expected non-zero value)", constant.Success)
		}
		assert.NotEmpty(t, constant.Success, "constant.Success should not be empty")
	})
}

func TestGetDefaultClientFailureResponseJson(t *testing.T) {
	t.Run("should return default client failure response", func(t *testing.T) {
		resp := GetDefaultClientFailureResponseJson()
		assert.NotNil(t, resp)
		assert.Equal(t, int32(constant.ClientError), resp.Code)
		assert.Equal(t, "bad request", resp.Message)
		assert.Nil(t, resp.Data)
	})
}

func TestGetDefaultServerFailureResponseJson(t *testing.T) {
	t.Run("should return default failure response", func(t *testing.T) {
		// 调用函数
		resp := GetDefaultServerFailureResponseJson()

		// 验证结果
		assert.NotNil(t, resp, "Response should not be nil")
		assert.Equal(t, int32(constant.ServerError), resp.Code, "Code should be constant.ServerError")
		assert.Equal(t, "remote server busy", resp.Message, "Message should be 'remote server busy'")
		assert.Nil(t, resp.Data, "Data should be nil")
	})

	t.Run("should return new instance each time", func(t *testing.T) {
		// 多次调用函数
		resp1 := GetDefaultServerFailureResponseJson()
		resp2 := GetDefaultServerFailureResponseJson()

		// 验证是不同的实例
		assert.NotSame(t, resp1, resp2, "Should return new instance each time")

		// 修改第一个实例
		resp1.Message = "modified"

		// 验证第二个实例未受影响
		assert.Equal(t, "remote server busy", resp2.Message, "Second instance should not be modified")
	})

	t.Run("should have expected constant values", func(t *testing.T) {
		// 验证常量值
		assert.NotEmpty(t, constant.ServerError, "constant.ServerError should not be empty")
	})
}

func TestGetParamsEmptyErrorResponseJson(t *testing.T) {
	t.Run("should return default parameters empty error response", func(t *testing.T) {
		// 调用函数
		resp := GetParamsEmptyErrorResponseJson()

		// 验证结果
		assert.NotNil(t, resp, "Response should not be nil")
		assert.Equal(t, int32(constant.ClientError), resp.Code, "Code should be constant.ClientError")
		assert.Equal(t, "parameters not found", resp.Message, "Message should be 'parameters not found'")
		assert.Nil(t, resp.Data, "Data should be nil")
	})

	t.Run("should return new instance each time", func(t *testing.T) {
		// 多次调用函数
		resp1 := GetParamsEmptyErrorResponseJson()
		resp2 := GetParamsEmptyErrorResponseJson()

		// 验证是不同的实例
		assert.NotSame(t, resp1, resp2, "Should return new instance each time")

		// 修改第一个实例
		resp1.Message = "modified"

		// 验证第二个实例未受影响
		assert.Equal(t, "parameters not found", resp2.Message, "Second instance should not be modified")
	})
}

func TestGetResponseJson(t *testing.T) {
	t.Run("should return correct response struct with valid inputs", func(t *testing.T) {
		// 定义测试输入
		code := int32(200)
		msg := "success"
		data := map[string]interface{}{"key": "value"}

		// 调用函数
		resp := GetResponseJson(code, msg, data)

		// 验证结果
		assert.NotNil(t, resp, "Response should not be nil")
		assert.Equal(t, code, resp.Code, "Code should match the input")
		assert.Equal(t, msg, resp.Message, "Message should match the input")
		assert.Equal(t, data, resp.Data, "Data should match the input")
	})

	t.Run("should handle empty message and nil data", func(t *testing.T) {
		// 定义测试输入
		code := int32(400)
		msg := ""
		var data any = nil // 使用接口类型

		// 调用函数
		resp := GetResponseJson(code, msg, data)

		// 验证结果
		assert.NotNil(t, resp, "Response should not be nil")
		assert.Equal(t, code, resp.Code, "Code should match the input")
		assert.Equal(t, msg, resp.Message, "Message should match the input")
		assert.Nil(t, resp.Data, "Data should be nil")
	})

	t.Run("should handle negative error code with nil data", func(t *testing.T) {
		// 定义测试输入
		code := int32(-1)
		msg := "error occurred"
		var data any = nil // 使用接口类型

		// 调用函数
		resp := GetResponseJson(code, msg, data)

		// 验证结果
		assert.NotNil(t, resp, "Response should not be nil")
		assert.Equal(t, code, resp.Code, "Code should match the input")
		assert.Equal(t, msg, resp.Message, "Message should match the input")
		assert.Nil(t, resp.Data, "Data should be nil")
	})
}
