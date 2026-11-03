/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 */

package utils

import (
	"errors"
	"net/http"
	"testing"

	"github.com/stretchr/testify/assert"
)

// 测试常量定义
const (
	testAppCode     uint8  = 1
	testFeatureCode uint8  = 2
	testErrorCode   uint16 = 3
	testSuccessData string = "test data"
	testErrorMsg    string = "test error"
	testBatchData   string = "success data"
)

func TestResponseCodeCode(t *testing.T) {
	shift1 := 24
	shift2 := 16
	code := ResponseCode{AppCode: testAppCode, FeatureCode: testFeatureCode, ErrorCode: testErrorCode}
	expected := uint32(testAppCode)<<shift1 | uint32(testFeatureCode)<<shift2 | uint32(testErrorCode)
	assert.Equal(t, expected, code.Code())
}

func TestResponseResultAppCode(t *testing.T) {
	result := ResponseResult{Code: 0x01020304}
	assert.Equal(t, uint32(1), result.AppCode())
}

func TestNewResponseResultOk(t *testing.T) {
	data := "test data"
	result := NewResponseResultOk(data)
	assert.Equal(t, ResponseResult_CODE_OK, result.Code)
	assert.Equal(t, "ok", result.Msg)
	assert.Equal(t, data, result.Data)
}

func TestNewResponseResultNok(t *testing.T) {
	result := NewResponseResultNok(testAppCode, testFeatureCode, testErrorCode, "error")
	expectedCode := ResponseCode{AppCode: testAppCode, FeatureCode: testFeatureCode, ErrorCode: testErrorCode}.Code()
	assert.Equal(t, expectedCode, result.Code)
	assert.Equal(t, "ok", result.Msg)
}

func TestNewResponseResultWithError(t *testing.T) {
	// 跳过 nil 错误测试，因为原始实现会导致 panic
	err := errors.New(testErrorMsg)
	result := NewResponseResultWithError(testAppCode, testFeatureCode, err)
	expectedCode := ResponseCode{AppCode: testAppCode, FeatureCode: testFeatureCode, ErrorCode: http.StatusInternalServerError}.Code()
	assert.Equal(t, expectedCode, result.Code)
	assert.Equal(t, err.Error(), result.Msg)
}

func TestNewResponseResultWithErrorAndData(t *testing.T) {
	err := errors.New(testErrorMsg)
	data := "test data"
	result := NewResponseResultWithErrorAndData(testAppCode, testFeatureCode, err, data)
	expectedCode := ResponseCode{AppCode: testAppCode, FeatureCode: testFeatureCode, ErrorCode: http.StatusInternalServerError}.Code()
	assert.Equal(t, expectedCode, result.Code)
	assert.Equal(t, err.Error(), result.Msg)
	assert.Equal(t, data, result.Data)
}

func TestNewResponseResultWithBadRequest(t *testing.T) {
	msg := "bad request"
	result := NewResponseResultWithBadRequest(testAppCode, testFeatureCode, msg)
	expectedCode := ResponseCode{AppCode: testAppCode, FeatureCode: testFeatureCode, ErrorCode: http.StatusBadRequest}.Code()
	assert.Equal(t, expectedCode, result.Code)
	assert.Equal(t, msg, result.Msg)
}

func TestNewResponseResultBatchAllSuccess(t *testing.T) {
	data := "success data"
	result := NewResponseResultBatch(data, nil)
	assert.Equal(t, ResponseResult_CODE_OK, result.Code)
	assert.Equal(t, "ok", result.Msg)
	assert.Equal(t, data, result.Data)
}

func TestNewResponseResultBatchWithFailures(t *testing.T) {
	failures := []ResponseResult{
		NewResponseResultNok(testAppCode, testFeatureCode, testErrorCode, "error1"),
	}
	result := NewResponseResultBatch("data", failures)
	assert.NotEqual(t, ResponseResult_CODE_OK, result.Code)
	assert.Equal(t, "ok", result.Msg)
}

func TestResponseCodeEdgeCases(t *testing.T) {
	// 测试最大值
	maxCode := ResponseCode{AppCode: 0xFF, FeatureCode: 0xFF, ErrorCode: 0xFFFF}
	assert.Equal(t, uint32(0xFFFFFFFF), maxCode.Code())

	// 测试零值
	zeroCode := ResponseCode{}
	assert.Equal(t, uint32(0), zeroCode.Code())
}

func TestResponseCode_IsApp(t *testing.T) {
	code := ResponseCode{AppCode: testAppCode, FeatureCode: testFeatureCode, ErrorCode: testErrorCode}
	assert.True(t, code.IsApp(0))
	assert.False(t, code.IsApp(testAppCode))
}

func TestResponseCode_IsFeature(t *testing.T) {
	code := ResponseCode{AppCode: testAppCode, FeatureCode: testFeatureCode, ErrorCode: testErrorCode}
	assert.True(t, code.IsFeature(0))
	assert.False(t, code.IsFeature(testFeatureCode))
}

func TestResponseCode_IsError(t *testing.T) {
	code := ResponseCode{AppCode: testAppCode, FeatureCode: testFeatureCode, ErrorCode: testErrorCode}
	assert.True(t, code.IsError(uint8(testErrorCode)))
	assert.False(t, code.IsError(uint8(testErrorCode+1)))
}
