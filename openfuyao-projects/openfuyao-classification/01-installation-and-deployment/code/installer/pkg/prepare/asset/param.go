/******************************************************************
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package asset

import (
	"fmt"
	"os"
	"path/filepath"
	"text/template"

	"github.com/pkg/errors"

	"installer/pkg/common"
	"installer/pkg/constants"
	"installer/pkg/fuyao/v1beta1/base"
	fuyaov1beta1 "installer/pkg/fuyao/v1beta1/cluster"
	v1beta1values "installer/pkg/fuyao/v1beta1/values"
	"installer/pkg/zlog"
)

var (
	valuesFile = "default-values.yaml"
)

// ParamsMap 获取组件参数map
// component 组件对象
// destPath 解压后的安装包路径
func ParamsMap(component fuyaov1beta1.Component, destPath string, AdvancedParams []base.Param) map[string]interface{} {
	var params = make(map[string]interface{})

	// 组件自定义参数
	for _, param := range ParamMerge(component, destPath) {
		params[param.Name] = param.Value
	}

	// 高级参数
	for _, param := range AdvancedParams {
		params[param.Name] = param.Value
	}

	return params
}

// ParamMerge 合并组件参数
// component 组件对象
// destPath 解压后的安装包路径
func ParamMerge(component fuyaov1beta1.Component, destPath string) []base.Param {
	values := component.Param
	valuesPath := filepath.Join(destPath, component.Group, component.Name, valuesFile)

	_, err := os.Stat(valuesPath)
	if err != nil && os.IsNotExist(err) {
		return []base.Param{}
	} else if err != nil {
		zlog.Fatalf("error file stat", err)
	} else {
		zlog.Info("start merge param")
	}

	defaultValues := &v1beta1values.Values{}
	if err := common.UnmarshalYaml[*v1beta1values.Values](defaultValues, valuesPath); err != nil {
		zlog.Fatalf("error:", err)
	}

	var newValues []base.Param
	for _, defaultValue := range defaultValues.Spec.Params {
		isSkip := false
		for _, value := range values {
			if defaultValue.Name == value.Name {
				newValues = append(newValues, value)
				isSkip = true
				break
			}
		}
		if !isSkip {
			newValues = append(newValues, defaultValue)
		}
	}
	return newValues
}

// RenderComponentParams 渲染组件参数
// component 组件对象
// filePath 目标文件路径
// data 渲染的数据
func RenderComponentParams(component fuyaov1beta1.Component, filePath string, data map[string]interface{}) {
	tmpl, err := template.ParseFiles(filePath)
	zlog.Info("Parse file path is ", filePath)
	if err != nil {
		zlog.Fatalf("Error parsing shell file, component is ", component.Name, err)
	}

	// 创建输出文件
	file, err := os.Create(filePath)
	if err != nil {
		zlog.Fatalf("Error create shell file, component is ", component.Name, err)
	}
	defer file.Close()

	err = file.Chmod(constants.RwxRxRx)
	if err != nil {
		zlog.Fatalf(fmt.Sprintf("failed to set file permission: %v", err))
	}

	// 使用模板对象将数据写入文件
	err = tmpl.Execute(file, data)
	if err != nil {
		zlog.Fatalf(fmt.Sprintf("Failed write data: %v", err))
	}
}

// RenderComponentParam 渲染组件参数
// filePath 目标文件路径
// data 渲染的数据
func RenderComponentParam(filePath string, data map[string]interface{}) error {
	tmpl, err := template.ParseFiles(filePath)
	zlog.Info("Parse file path is ", filePath)
	if err != nil {
		return errors.New(fmt.Sprintf("Error parsing shell file：%v\n", err))
	}

	// 创建输出文件
	file, err := os.Create(filePath)
	if err != nil {
		return errors.New(fmt.Sprintf("Error create shell file：%v\n", err))
	}
	defer file.Close()
	if err := file.Chmod(constants.RwxXX); err != nil {
		return fmt.Errorf("failed to set file permission: %v", err)
	}

	// 使用模板对象将数据写入文件
	err = tmpl.Execute(file, data)
	if err != nil {
		return errors.New(fmt.Sprintf("Failed to render param：%v\n", err))
	}
	return nil
}
