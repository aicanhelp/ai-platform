/*
Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
*/

// conf package provides configuration loading functions for oschecktool
package conf

import (
	"fmt"
	"os"
	"path"
	"path/filepath"
	"regexp"
	"strings"

	"gopkg.in/yaml.v3"

	"openfuyao.com/oscheck/internal/utils"
)

var (
	reParam          = regexp.MustCompile(`\s*{{\s*([a-zA-Z0-9_\-]+)\s*}}\s*`)
	reParamNameIndex = 1
)

// LoadConf 初始化配置，实际就是初始化全局变量
func LoadConf() (*CheckConfig, error) {
	confPath := path.Join(utils.GetStartDir(), "conf")
	if _, err := os.Stat(confPath); os.IsNotExist(err) {
		return nil, fmt.Errorf("conf directory does not exist at path: %s", confPath)
	}
	config, err := loadConfFromPath(confPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load conf from path: %s, error: %s", confPath, err.Error())
	}
	utils.GetLogger("").Printf("Configuration loaded successfully from path: %s", confPath)
	return config, nil
}

// loadConfFromPath 加载配置文件
func loadConfFromPath(confPath string) (*CheckConfig, error) {
	checkItems, err := loadCheckItems(path.Join(confPath, "check-items"))
	if err != nil {
		return nil, fmt.Errorf("failed to load check items from path: %s, error: %s", confPath, err.Error())
	}
	checkSets, err := loadCheckSets(path.Join(confPath, "check-sets"), checkItems)
	if err != nil {
		return nil, err
	}

	return &CheckConfig{
		CheckItems: checkItems,
		CheckSets:  checkSets,
	}, nil
}

// CheckConfig 定义了检查配置的结构, 从conf目录加载
type CheckConfig struct {
	CheckItems map[string]CheckItem `yaml:"checkItems"`
	CheckSets  map[string]CheckSet  `yaml:"checkSets"`
}

// CheckItem 定义了单个检查项的结构
type CheckItem struct {
	Name     string           `yaml:"name" json:"name"`
	Params   []CheckItemParam `yaml:"params" json:"params"`
	Kind     string           `yaml:"kind" json:"kind"`
	Doc      string           `yaml:"doc" json:"doc"`
	Spec     interface{}      `yaml:"spec" json:"spec"`
	FilePath string           `yaml:"-" json:"-"`
}

// FillParamValues 填充检查项的参数值
// 检查项的参数值从paramValues中获取，paramValues的key是参数名，value是参数值
// 如果参数值为空，且参数有默认值，则使用默认值
// 如果参数值为空，且参数没有默认值，则使用空字符串
func (item *CheckItem) FillParamValues(paramValues map[string][]string) {
	// 没有参数可替换，则忽略
	if len(item.Params) == 0 {
		return
	}
	paramMap := make(map[string]CheckItemParam, 0)
	for _, param := range item.Params {
		if param.DefaultValue == nil {
			if param.Multi {
				param.DefaultValue = []string{}
			} else {
				param.DefaultValue = ""
			}
		}
		paramMap[param.Name] = param
	}
	walkAndFillParams(item.Spec, paramMap, paramValues)
}

func walkAndFillParams(item any, paramMap map[string]CheckItemParam, paramValues map[string][]string) {
	switch v := item.(type) {
	case map[string]any:
		// 递归处理嵌套的 map
		for key, value := range v {
			switch value := value.(type) {
			case map[string]any, []any:
				walkAndFillParams(value, paramMap, paramValues)
			case string:
				paramValue, ok := getStringParamValue(value, paramMap, paramValues)
				if !ok {
					continue
				}
				v[key] = paramValue
			default:
				// 其他类型的参数，直接返回
				v[key] = value
			}
		}
	case []any:
		// 递归处理嵌套的数组
		for _, subItem := range v {
			walkAndFillParams(subItem, paramMap, paramValues)
		}
	default:
		return
	}
}

func getStringParamValue(value string, pMap map[string]CheckItemParam, pValues map[string][]string) (any, bool) {
	paramNames := reParam.FindStringSubmatch(value)
	if len(paramNames) == 0 {
		return "", false
	}
	paramName := paramNames[reParamNameIndex]
	paramInfo, ok := pMap[paramName]
	if !ok {
		return "", false
	}
	paramValue := paramInfo.DefaultValue
	rawValues, exist := pValues[paramName]
	if exist {
		if paramInfo.Multi {
			paramValue = rawValues

		} else {
			paramValue = rawValues[0]
		}
	}
	return paramValue, true
}

// CheckItemParam 定义了检查项的参数结构
// 检查项的参数可以在检查项的spec中使用，参数的格式是{{paramName}}
// 检查项的参数可以有默认值，默认值可以是字符串，也可以是字符串数组
// 检查项的参数可以是多值参数，多值参数在检查项的spec中使用时，需要用数组表示
type CheckItemParam struct {
	Name         string      `yaml:"name" json:"name"`
	Desc         string      `yaml:"desc" json:"desc"`
	DefaultValue interface{} `yaml:"defaultValue" json:"defaultValue"`
	Multi        bool        `yaml:"multi" json:"multi"`
}

// CheckSet 定义了检查集的结构
// 检查集包含了多个检查项，检查项可以在检查集中引用
// 检查集可以引用其他检查集，被引用的检查集会被合并到当前检查集中
// 检查集可以引用检查项，被引用的检查项会被合并到当前检查集中
type CheckSet struct {
	Name      string               `yaml:"name"`
	Desc      string               `yaml:"desc"`
	Include   []string             `yaml:"include"`
	ItemNames []string             `yaml:"items"`
	FilePath  string               `yaml:"-"`
	Items     map[string]CheckItem `yaml:"-"`
}

func loadCheckSets(path string, checkItems map[string]CheckItem) (map[string]CheckSet, error) {
	checkSets := make(map[string]CheckSet)
	entries, err := os.ReadDir(path)
	hasErr := false
	if err != nil {
		utils.PromptMsg("Failed to read path: %s, because: %s", path, err.Error())
		return nil, err
	}
	for _, entry := range entries {
		if entry.IsDir() || !(strings.HasSuffix(entry.Name(), ".yaml") || strings.HasSuffix(entry.Name(), ".yml")) {
			continue
		}
		fullPath := filepath.Join(path, entry.Name())
		checkSet, err := loadCheckSetFromFile(fullPath, checkItems)
		if err != nil {
			utils.GetDefaultLogger().Printf("Failed to read checkset file: %s, error: %s", fullPath, err.Error())
			hasErr = true
			continue
		}
		if existCheckSet, exist := checkSets[checkSet.Name]; exist {
			utils.PromptMsg("Duplicate names of Check Set: %s, %s", existCheckSet.FilePath, checkSet.FilePath)
			hasErr = true
			continue
		}
		checkSets[checkSet.Name] = checkSet
	}

	for name, checkset := range checkSets {
		for _, includeName := range checkset.Include {
			includeCheckSet, exist := checkSets[includeName]
			if !exist {
				utils.PromptMsg(`Check Set "%s" referenced  by %s is not exist`, includeName, checkset.FilePath)
				hasErr = true
			}
			for name, item := range includeCheckSet.Items {
				checkset.Items[name] = item
			}
		}
		checkSets[name] = checkset
	}

	if hasErr {
		return nil, fmt.Errorf("failed to load check sets")
	}
	return checkSets, nil
}

func loadCheckItems(path string) (map[string]CheckItem, error) {
	checkItems := make(map[string]CheckItem)

	entries, err := os.ReadDir(path)
	if err != nil {
		utils.PromptMsg("Failed to read path: %s, because: %s", path, err.Error())
		return checkItems, err
	}
	hasErr := false
	for _, entry := range entries {
		if entry.IsDir() || !(strings.HasSuffix(entry.Name(), ".yaml") || strings.HasSuffix(entry.Name(), ".yml")) {
			continue
		}
		fullPath := filepath.Join(path, entry.Name())
		checkItem, err := loadCheckItemFromFile(fullPath)
		if err != nil {
			utils.PromptMsg("Failed to read check item file: %s, error: %s", fullPath, err.Error())
			hasErr = true
			continue
		}
		existCheckItem, exist := checkItems[checkItem.Name]
		if exist {
			utils.PromptMsg("Duplicate names of check item: %s, %s:", existCheckItem.FilePath, checkItem.FilePath)
			hasErr = true
			continue
		}
		checkItems[checkItem.Name] = checkItem
	}
	if hasErr {
		return checkItems, fmt.Errorf("failed to load check items")
	}
	return checkItems, nil

}

func loadCheckSetFromFile(path string, checkItems map[string]CheckItem) (CheckSet, error) {
	var checkSet CheckSet

	content, err := os.ReadFile(path)
	if err != nil {
		utils.PromptMsg("Failed to load checkset from file: %s, error: %s", path, err.Error())
		return checkSet, fmt.Errorf("failed to load checkset from file: %s", path)
	}

	err = yaml.Unmarshal(content, &checkSet)
	if err != nil {
		utils.PromptMsg("Failed to parse checkset file: %s, error: %s", path, err.Error())
		return checkSet, fmt.Errorf("format error: %s", path)
	}
	checkSet.Name = strings.TrimSpace(checkSet.Name)
	checkSet.Desc = strings.TrimSpace(checkSet.Desc)
	checkSet.FilePath = path
	if checkSet.Name == "" {
		utils.PromptMsg("CheckSet name is empty in file: %s", path)
		return checkSet, fmt.Errorf("checkset name is empty")
	}

	items := make(map[string]CheckItem, len(checkSet.ItemNames))
	errItems := make([]string, 0)
	for _, itemName := range checkSet.ItemNames {
		item, exist := checkItems[itemName]
		if exist {
			items[itemName] = item
		} else {
			errItems = append(errItems, itemName)
		}
	}
	checkSet.Items = items
	if len(errItems) > 0 {
		utils.PromptMsg("CheckSet %s references items that do not exist: %v", checkSet.Name, errItems)
		return checkSet, fmt.Errorf("the following check items cannot be found: %v", errItems)
	}

	return checkSet, nil
}

func loadCheckItemFromFile(path string) (CheckItem, error) {
	content, err := os.ReadFile(path)
	if err != nil {
		return CheckItem{}, err
	}
	var checkItem CheckItem
	err = yaml.Unmarshal(content, &checkItem)
	if err != nil {
		utils.GetDefaultLogger().Printf("Failed to unmarshal file: %s, error: %s", path, err)
		return checkItem, fmt.Errorf("format error")
	}
	checkItem.FilePath = path
	checkItem.Name = strings.TrimSpace(checkItem.Name)
	checkItem.Kind = strings.TrimSpace(checkItem.Kind)
	if checkItem.Name == "" || checkItem.Kind == "" {
		utils.GetDefaultLogger().Printf("CheckItem name or kind is empty in file: %s", path)
		return checkItem, fmt.Errorf("checkitem name or kind is empty")
	}

	return checkItem, nil
}
