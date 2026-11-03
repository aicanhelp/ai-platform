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

package utils

import "net/http"

const (
	// ResponseResult_CODE_OK 正常返回结果
	ResponseResult_CODE_OK uint32 = 0

	// response code 位移常量
	responseAppCodeShift     = 24
	responseFeatureCodeShift = 16
)

type ResponseCode struct {
	AppCode     uint8  `json:"appCode"`
	FeatureCode uint8  `json:"featureCode"`
	ErrorCode   uint16 `json:"errorCode"`
}

func (code ResponseCode) Code() uint32 {
	codeUint32 := uint32(code.AppCode)<<responseAppCodeShift | uint32(code.FeatureCode)<<responseFeatureCodeShift | uint32(code.ErrorCode)
	return codeUint32
}

func (code ResponseCode) IsApp(appCode uint8) bool {
	appCodeUint32 := uint32(appCode) << responseAppCodeShift
	responseCode := code.Code()
	return appCodeUint32&responseCode == uint32(appCode)
}

func (code ResponseCode) IsFeature(featureCode uint8) bool {
	featureCodeUint32 := uint32(featureCode) << responseFeatureCodeShift
	responseCode := code.Code()
	return featureCodeUint32&responseCode == uint32(featureCode)
}

func (code ResponseCode) IsError(errorCode uint8) bool {
	errorCodeUint32 := uint32(errorCode)
	responseCode := code.Code()
	return errorCodeUint32&responseCode == uint32(errorCode)
}

type ResponseResult struct {
	Code uint32 `json:"code"`
	Msg  string `json:"message"`
	Data any    `json:"data"`
}

func (result ResponseResult) AppCode() uint32 {
	return result.Code >> responseAppCodeShift
}

func NewResponseResultOk(data any) ResponseResult {
	return ResponseResult{
		Code: ResponseResult_CODE_OK,
		Msg:  "ok",
		Data: data,
	}
}

func NewResponseResultNok(appCode uint8, featureCode uint8, errorCode uint16, message string) ResponseResult {
	return ResponseResult{
		Code: ResponseCode{AppCode: appCode, FeatureCode: featureCode, ErrorCode: errorCode}.Code(),
		Msg:  "ok",
	}
}

func NewResponseResultWithErrorAndData(appCode uint8, featureCode uint8, err error, data any) ResponseResult {
	return ResponseResult{
		Code: ResponseCode{AppCode: appCode, FeatureCode: featureCode, ErrorCode: http.StatusInternalServerError}.Code(),
		Msg:  err.Error(),
		Data: data,
	}
}

func NewResponseResultWithError(appCode uint8, featureCode uint8, err error) ResponseResult {
	return ResponseResult{
		Code: ResponseCode{AppCode: appCode, FeatureCode: featureCode, ErrorCode: http.StatusInternalServerError}.Code(),
		Msg:  err.Error(),
	}
}

func NewResponseResultWithBadRequest(appCode uint8, featureCode uint8, msg string) ResponseResult {
	return ResponseResult{
		Code: ResponseCode{AppCode: appCode, FeatureCode: featureCode, ErrorCode: http.StatusBadRequest}.Code(),
		Msg:  msg,
	}
}

type ResponseResultBatch struct {
	Successed any              `json:"successed"`
	Failed    []ResponseResult `json:"failed"`
}

func NewResponseResultBatch(successedData any, failedResult []ResponseResult) ResponseResult {
	if len(failedResult) == 0 {
		return ResponseResult{
			Code: ResponseResult_CODE_OK,
			Msg:  "ok",
			Data: successedData,
		}
	}
	responseResultBatch := ResponseResultBatch{
		Successed: successedData,
		Failed:    failedResult,
	}
	appCode := failedResult[0].AppCode()
	return ResponseResult{
		Code: appCode,
		Msg:  "ok",
		Data: responseResultBatch,
	}
}
