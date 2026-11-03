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
	"fmt"
	"math"
	"regexp"
	"strings"

	"openfuyao.com/oscheck/internal/utils"
)

var (
	regexIntRange          *regexp.Regexp = regexp.MustCompile(`^([\[(])\s*(\d*)\s*,\s*(\d*)\s*([\])])$`)
	versionRange           *regexp.Regexp = regexp.MustCompile(`^([\[(])\s*([\d.]*)\s*,\s*([\d.]*)\s*([\])])$`)
	regSpace               *regexp.Regexp = regexp.MustCompile(`\s+`)
	rangeLeftBracketIndex                 = 1
	rangeRightBracketIndex                = 4
	rangeLeftValueIndex                   = 2
	rangeRightValueIndex                  = 3

	// emptyChecker 恒定返回成功
	emptyChecker = EmptyValueChecker{}
)

// ValueChecker 用于检查特定的Value是否符合expect中的标记
type ValueChecker interface {

	// Init 使用expect初始化这个ValueCheker，如果出现错误则返回error
	Init(expect interface{}) error

	// Check 检查value是否符合预期，成功返回true，失败返回false，如果出现异常则返回error
	Check(value string) bool
}

// GetValueChecker 根据valueType获取对应类型的ValueCheker，并使用expect对其进行初始化
func GetValueChecker(valueType string, expect interface{}) (ValueChecker, error) {
	var checker ValueChecker
	valueType = strings.ToLower(strings.TrimSpace(valueType))
	switch valueType {
	case "int", "integer":
		checker = &IntChecker{}
	case "int-range":
		checker = &IntRangeChecker{}
	case "regex", "re":
		checker = &ReChecker{}
	case "version-range":
		checker = &VersionRangeChecker{}
	case "string", "str":
		checker = &StringChecker{}
	case "":
		checker = emptyChecker
	default:
		return nil, fmt.Errorf("unknown type: <%s>", valueType)
	}
	err := checker.Init(expect)
	return checker, err
}

// IntChecker 用来检查整型值
type IntChecker struct {
	expect int64
}

// Init 使用expect初始化这个IntChecker，如果出现错误则返回error
func (c *IntChecker) Init(expect interface{}) error {
	switch v := expect.(type) {
	case string:
		value, err := utils.ParseInt10Base(v)
		c.expect = value
		return err
	case int64:
		c.expect = v
	case int:
		c.expect = int64(v)
	case int16:
		c.expect = int64(v)
	case int32:
		c.expect = int64(v)
	default:
		return fmt.Errorf("%v cannot convert to int64", expect)
	}
	return nil
}

// Check 检查value是否符合预期，成功返回true，失败返回false，如果出现异常则返回error
func (c IntChecker) Check(value string) bool {
	iv, err := utils.ParseInt10Base(value)
	if err != nil {
		return false
	}
	return iv == c.expect
}

// IntRangeChecker 用来检查整型范围值
type IntRangeChecker struct {
	left      int64
	right     int64
	leftOpen  bool
	rightOpen bool
}

// Init 使用expect初始化这个IntRangeChecker，如果出现错误则返回error
func (c *IntRangeChecker) Init(expect interface{}) error {
	expectStr := fmt.Sprintf("%v", expect)
	match := regexIntRange.FindStringSubmatch(expectStr)
	if len(match) == 0 {
		return fmt.Errorf("failed to parse expect string: %s", expect)
	}
	leftBracket := match[rangeLeftBracketIndex]
	rightBracket := match[rangeRightBracketIndex]
	minStr := match[rangeLeftValueIndex]
	maxStr := match[rangeRightValueIndex]

	if leftBracket == "(" {
		c.leftOpen = true
	}
	if rightBracket == ")" {
		c.rightOpen = true
	}

	if len(minStr) == 0 {
		c.left = math.MinInt64
		c.leftOpen = true
	} else {
		var err error
		c.left, err = utils.ParseInt10Base(minStr)
		if err != nil {
			return err
		}
	}

	if len(maxStr) == 0 {
		c.right = math.MaxInt64
		c.rightOpen = true
	} else {
		var err error
		c.right, err = utils.ParseInt10Base(maxStr)
		if err != nil {
			return err
		}
	}
	if c.left > c.right {
		return fmt.Errorf("%s: minimum > maximum", expect)
	}
	return nil
}

// Check 检查value是否符合预期，成功返回true，失败返回false，如果出现异常则返回error
func (c *IntRangeChecker) Check(value string) bool {
	iv, err := utils.ParseInt10Base(value)

	// 转换失败，作检查失败论，而非报错，因为可能输入的内容格式不符合要求
	if err != nil {
		return false
	}
	rslt := true
	if c.leftOpen {
		rslt = rslt && (iv > c.left)
	} else {
		rslt = rslt && (iv >= c.left)
	}
	if c.rightOpen {
		rslt = rslt && (iv < c.right)
	} else {
		rslt = rslt && (iv <= c.right)
	}
	return rslt
}

// StringChecker 用来检查字符串值
type StringChecker struct {
	expect string
}

// Init 使用expect初始化这个StringChecker，如果出现错误则返回error
func (c *StringChecker) Init(expect interface{}) error {
	expectStr := fmt.Sprintf("%v", expect)
	// 统一格式化字符串，避免因为格式问题出错
	c.expect = strings.TrimSpace(regSpace.ReplaceAllString(expectStr, " "))
	return nil
}

// Check 检查value是否符合预期，成功返回true，失败返回false，如果出现异常则返回error
func (c StringChecker) Check(value string) bool {
	value = strings.TrimSpace(regSpace.ReplaceAllString(value, " "))
	return value == c.expect
}

// EmptyValueChecker 恒定返回成功的ValueChecker
type EmptyValueChecker struct {
}

// Init 使用expect初始化这个EmptyValueChecker，如果出现错误则返回error
func (c EmptyValueChecker) Init(expect interface{}) error {
	if expect != nil {
		return fmt.Errorf("expect is not empty")
	}
	return nil
}

// Check 恒定返回true
func (c EmptyValueChecker) Check(value string) bool {
	return true
}

// ReChecker 使用正则表达式值进行检查
type ReChecker struct {
	expect *regexp.Regexp
}

// Init 使用expect初始化这个ReChecker，如果出现错误则返回error
func (c *ReChecker) Init(expect interface{}) error {
	expectStr := strings.TrimSpace(fmt.Sprintf("%v", expect))

	if len(expectStr) == 0 {
		return fmt.Errorf("expect string is empty")
	}
	re, err := regexp.Compile(expectStr)
	if err != nil {
		return err
	}
	c.expect = re
	return nil
}

// Check 检查value是否符合预期，成功返回true，失败返回false，如果出现异常则返回error
func (c ReChecker) Check(value string) bool {
	return c.expect.MatchString(strings.TrimSpace(value))
}

// VersionRangeChecker 用来检查版本号范围值
type VersionRangeChecker struct {
	left      []int64
	right     []int64
	leftOpen  bool
	rightOpen bool
}

// Init 使用expect初始化这个VersionRangeChecker，如果出现错误则返回error
func (c *VersionRangeChecker) Init(expect interface{}) error {

	expectStr := strings.TrimSpace(fmt.Sprintf("%v", expect))
	if len(expectStr) == 0 {
		return fmt.Errorf("expect is empty")
	}
	match := versionRange.FindStringSubmatch(expectStr)
	if len(match) == 0 {
		return fmt.Errorf("failed to parse expect: <%s>", expect)
	}

	leftBracket := match[rangeLeftBracketIndex]
	rightBracket := match[rangeRightBracketIndex]
	minStr := match[rangeLeftValueIndex]
	maxStr := match[rangeRightValueIndex]

	if leftBracket == "(" {
		c.leftOpen = true
	}
	if rightBracket == ")" {
		c.rightOpen = true
	}

	if len(minStr) == 0 {
		c.leftOpen = true
		c.left = []int64{}
	} else {
		version, err := parseVersion(minStr)
		if err != nil {
			return err
		}
		c.left = version
	}
	if len(maxStr) == 0 {
		c.rightOpen = true
		c.right = []int64{}
	} else {
		version, err := parseVersion(maxStr)
		if err != nil {
			return err
		}
		c.right = version
	}
	if compareVersion(c.left, c.right) > 0 {
		return fmt.Errorf("minimum > maximum : <%s>", expect)
	}
	return nil
}

// Check 检查value是否符合预期，成功返回true，失败返回false，如果出现异常则返回error
func (c VersionRangeChecker) Check(value string) bool {
	version, err := parseVersion(value)
	if err != nil {
		return false
	}
	rslt := true

	if len(c.left) > 0 {
		if c.leftOpen {
			rslt = rslt && compareVersion(c.left, version) < 0
		} else {
			rslt = rslt && compareVersion(c.left, version) <= 0
		}
	}
	if len(c.right) > 0 {
		if c.rightOpen {
			rslt = rslt && compareVersion(version, c.right) < 0
		} else {
			rslt = rslt && compareVersion(version, c.right) <= 0
		}
	}
	return rslt

}

func parseVersion(vs string) ([]int64, error) {
	vs = strings.TrimSpace(vs)
	if len(vs) == 0 {
		return []int64{}, nil
	}
	versions := strings.Split(vs, ".")
	rst := make([]int64, 0, len(versions))
	for _, v := range versions {
		if len(v) == 0 {
			return nil, fmt.Errorf("error version format: <%s>", vs)
		}
		iv, err := utils.ParseInt10Base(v)
		if err != nil {
			return nil, fmt.Errorf("error version format: <%s>", vs)
		}
		rst = append(rst, iv)
	}
	return rst, nil
}

// compareVersion 比较版本号，left == right 返回0， left>right 返回1， left < right返回-1
// 如果长度不相等，则只比较长度相等的部分，长度不相等的部分认为相等
func compareVersion(left []int64, right []int64) int {
	minLen := min(len(left), len(right))
	for i := 0; i < minLen; i++ {
		if left[i] > right[i] {
			return 1
		} else if left[i] < right[i] {
			return -1
		}
	}
	return 0
}
