/*
 *
 *  * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 *  * openFuyao is licensed under Mulan PSL v2.
 *  * You can use this software according to the terms and conditions of the Mulan PSL v2.
 *  * You may obtain a copy of Mulan PSL v2 at:
 *  *          http://license.coscl.org.cn/MulanPSL2
 *  * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 *  * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 *  * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 *  * See the Mulan PSL v2 for more details.
 *
 */

package v1beta1

import (
	"fmt"
	"net/http"
	"strconv"
	"time"

	"github.com/emicklei/go-restful/v3"

	"console-service/pkg/constant"
	"console-service/pkg/utils/httputil"
)

const (
	mapKeyOffset            = "offset"
	mapKeyCurrentServerTime = "currentServerTime"
	secondToHour            = 3600
	hourToMin               = 60
	decimalBase             = 10
)

// Handler Component handler
type Handler struct{}

// NewHandler Component handler
func NewHandler() *Handler {
	return &Handler{}
}

func (h *Handler) getCurrentTimeOffset(request *restful.Request, response *restful.Response) {
	currentServerTime := time.Now()
	_, offset := currentServerTime.Zone()

	sign := "+"
	if offset < 0 {
		sign = "-"
		offset = -offset
	}

	hours := offset / secondToHour
	minutes := (offset % secondToHour) / hourToMin
	offsetString := fmt.Sprintf("UTC%s%02d:%02d", sign, hours, minutes)
	data := make(map[string]string)
	data[mapKeyOffset] = offsetString
	data[mapKeyCurrentServerTime] = strconv.FormatInt(currentServerTime.Unix(), decimalBase)
	result := &httputil.ResponseJson{
		Code: constant.Success,
		Msg:  "success",
		Data: data,
	}
	_ = response.WriteHeaderAndEntity(http.StatusOK, result)
}
