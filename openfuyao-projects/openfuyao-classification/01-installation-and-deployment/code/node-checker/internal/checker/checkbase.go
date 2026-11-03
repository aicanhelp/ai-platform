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

// check  package provides check functions for oschecktool
package checker

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"openfuyao.com/oscheck/internal/conf"
	"openfuyao.com/oscheck/internal/utils"
)

// ResultType 检查结果类型, 也就是合法、不合法、错误
type ResultType int

const (
	// ResultValid 检查结果类型, 合法
	ResultValid ResultType = iota
	// ResultInvalid 检查结果类型, 不合法
	ResultInvalid
	// ResultError 检查结果类型, 错误
	ResultError
)

// Context 检查上下文
type Context struct {
	RootPath string
}

// MarshalText 将检查结果序列化，实际上是为了在Yaml中输出字符串
func (t ResultType) MarshalText() ([]byte, error) {
	var text string
	switch t {
	case ResultError:
		text = "error"
	case ResultInvalid:
		text = "failed"
	case ResultValid:
		text = "ok"
	default:
		return []byte{}, fmt.Errorf("failed to marshal %v for CheckRstType", t)
	}
	return []byte(text), nil
}

// UnmarshalText 将检查结果反序列化，实际上是为了从Yaml中读取字符串
func (t *ResultType) UnmarshalText(text []byte) error {
	var value ResultType
	switch string(text) {
	case "error":
		value = ResultError
	case "failed":
		value = ResultInvalid
	case "ok":
		value = ResultValid
	default:
		return fmt.Errorf("failed to parse %s for CheckRstType", text)

	}
	*t = value
	return nil
}

const (
	matchLast  = "last"
	matchFirst = "first"
	matchOnce  = "once"
)

// SubItemParamBase 检查子项参数基类
type SubItemParamBase struct {
	// Name 参数名称
	Name string `json:"name"`
	// Key 参数对应的键名，主要用于kvChecker
	Key string `json:"key"`
	// Expect 期望值
	Expect interface{} `json:"expect"`
	// Type 参数类型, 目前支持string、int、intRange、versionRange、regex
	Type string `json:"type"`
	// Doc 子检查项说明
	Doc string `json:"doc"`
	// Default 默认值
	Default interface{} `json:"default"`
	// MatchType 匹配类型，目前支持last、first、once
	MatchType string `json:"matchType"`
}

// Init 初始化子检查项参数
func (p *SubItemParamBase) Init() error {
	if !checkParamFormat(*p) {
		return fmt.Errorf("param format error")
	}
	return nil
}

func getForJsonNumber(maybeNumber interface{}) interface{} {
	jsoNum, ok := maybeNumber.(json.Number)
	if ok {
		i64v, err := jsoNum.Int64()
		if err == nil {
			return i64v
		} else {
			f64v, err := jsoNum.Float64()
			if err == nil {
				return f64v
			}
		}

	}
	if str, ok := maybeNumber.(string); ok {
		if num, err := utils.ParseInt10Base(str); err == nil {
			return num
		}
	}
	return maybeNumber

}

// UnmarshalJSON 自定义反序列化，主要是为了保证Expect、Default能够尽量转换为指定的格式，matchType能够有过一个默认值
func (p *SubItemParamBase) UnmarshalJSON(b []byte) error {
	// 复写此方法主要是为了保证Expect、Default能够尽量转换为指定的格式，matchType能够有过一个默认值
	type Alias SubItemParamBase
	tmp := Alias{}
	decoder := json.NewDecoder(bytes.NewReader(b))
	decoder.UseNumber()
	if err := decoder.Decode(&tmp); err != nil {
		return err
	}
	p.Name = tmp.Name
	p.Expect = getForJsonNumber(tmp.Expect)
	p.Type = tmp.Type
	p.Doc = tmp.Doc
	p.Key = tmp.Key
	p.MatchType = strings.TrimSpace(tmp.MatchType)
	if p.MatchType == "" {
		p.MatchType = matchLast
	}
	if tmp.Default == nil {
		p.Default = ""
	} else {
		p.Default = getForJsonNumber(tmp.Default)
	}
	return nil
}

// ParamCheckError 检查参数错误，用于能够提示一些具体的错误信息
type ParamCheckError struct {
	ErrField  []string
	Msg       string
	FormatErr bool
}

// Error 实现error接口
func (e ParamCheckError) Error() string {
	return e.Msg
}

func checkParamFormat(param SubItemParamBase) bool {
	rslt := len(param.Name) > 0
	rslt = rslt && param.Type != ""
	rslt = rslt && (param.MatchType == matchLast || param.MatchType == matchFirst || param.MatchType == matchOnce)
	return rslt
}

// Result 检查结果
type Result struct {
	HostName  string     `yaml:"hostName" json:"hostName"`
	CheckSet  string     `yaml:"checkSet" json:"checkSet"`
	Result    ResultType `yaml:"result" json:"result"`
	StartTime time.Time  `yaml:"startTime" json:"startTime"`
	EndTime   time.Time  `yaml:"endTime" json:"endTime"`
	Items     []ItemRslt `yaml:"items,omitempty" json:"items,omitempty"`
}

// ItemRslt 检查项结果
type ItemRslt struct {
	Key      string        `yaml:"key" json:"key"`
	Result   ResultType    `yaml:"result" json:"result"`
	Doc      string        `yaml:"doc" json:"doc"`
	SubItems []SubItemRslt `yaml:"subItems,omitempty" json:"subItems,omitempty"`
}

// SubItemRslt 检查子项结果
type SubItemRslt struct {
	Key    string     `yaml:"key" json:"key"`
	Expect string     `yaml:"expect" json:"expect"`
	Real   string     `yaml:"real" json:"real"`
	Result ResultType `yaml:"result" json:"result"`
	Doc    string     `yaml:"doc" json:"doc"`
}

// Checker 检查器接口
type Checker interface {

	// Init 初始化Checker, ctx为上下文，item为配置
	// 返回ParamCheckError表示参数错误，返回nil則初始化成功
	Init(ctx Context, item conf.CheckItem) *ParamCheckError

	// Check 检查
	// 返回ItemRslt表示检查结果
	Check() ItemRslt
}

var checkFactories map[string]func() Checker = make(map[string]func() Checker)

// RegisterCheckerFactory 注册检查器工厂
func RegisterCheckerFactory(checkerType string, factory func() Checker) {
	checkFactories[checkerType] = factory
}

// GetChecker 获取检查器实例
func GetChecker(checkerType string) (Checker, error) {
	factory, ok := checkFactories[checkerType]
	if !ok {
		return nil, fmt.Errorf("checker type %s not found", checkerType)
	}
	checker := factory()
	if checker == nil {
		return nil, fmt.Errorf("checker type %s factory returned nil", checkerType)
	}
	return checker, nil
}

func matchValue(matchType string, checker ValueChecker, values []string) bool {
	switch matchType {
	case matchFirst:
		return checker.Check(values[0])
	case matchLast:
		return checker.Check(values[len(values)-1])
	case matchOnce:
		for _, value := range values {
			if checker.Check(value) {
				return true
			}
		}
	// 不会走到这个分支
	default:
		return false
	}
	return false
}

func convertSpec2Special(checkItem conf.CheckItem, spec interface{}) error {
	content, err := json.Marshal(checkItem.Spec)
	if err != nil {
		return err
	}
	err = json.Unmarshal(content, spec)
	if err != nil {
		return err
	}
	return nil
}
