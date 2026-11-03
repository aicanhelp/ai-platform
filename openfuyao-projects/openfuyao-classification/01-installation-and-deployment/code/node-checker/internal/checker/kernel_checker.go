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
	"bufio"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"

	"openfuyao.com/oscheck/internal/conf"
	"openfuyao.com/oscheck/internal/utils"
)

const (
	procModulesPath   = "/proc/modules"
	sysModulePath     = "/sys/module"
	moduleStatusIndex = 5
)

// KernelModuleSubItemParam defines the parameters for each sub-checker of the kernel module checker
type KernelModuleSubItemParam struct {
	Detail     []SubItemParamBase `json:"detail"`
	Doc        string             `json:"doc"`
	Name       string             `json:"name"`
	ModuleName string             `json:"moduleName"`
}

// kernelModuleChecker implements the Checker interface for checking kernel modules
type kernelModuleChecker struct {
	ctx         Context
	subCheckers []KernelModuleSubChecker
	conf        conf.CheckItem
	logger      *log.Logger
}

// KernelModuleSubChecker defines a sub-checker for a specific kernel module
type KernelModuleSubChecker struct {
	param          KernelModuleSubItemParam
	detailCheckers []kernelModuleDetailSubChecker
}

// kernelModuleDetailSubChecker checks a specific detail of a kernel module
type kernelModuleDetailSubChecker struct {
	detailPath   string
	expect       interface{}
	valueChecker ValueChecker
}

// Init initializes the kernel module checker with the given context and configuration
func (c *kernelModuleChecker) Init(ctx Context, config conf.CheckItem) *ParamCheckError {
	c.ctx = ctx
	c.conf = config
	c.logger = utils.GetLogger("KernelModuleChecker")
	paramCheckErr := ParamCheckError{
		ErrField:  []string{},
		Msg:       "",
		FormatErr: false,
	}
	subCheckerParams := make([]KernelModuleSubItemParam, 0)
	err := convertSpec2Special(config, &subCheckerParams)
	if err != nil {
		paramCheckErr.FormatErr = true
		paramCheckErr.Msg = "Format Error"
		c.logger.Printf("Failed to Init KenerlModuleChecker,  config file: <%s>, because <%s>",
			config.FilePath, err.Error())
		return &paramCheckErr
	}

	for _, subCheckerParam := range subCheckerParams {
		if len(subCheckerParam.Name) == 0 {
			errField := fmt.Sprintf("spec.[name=%s].name", subCheckerParam.Name)
			paramCheckErr.ErrField = append(paramCheckErr.ErrField, errField)
			c.logger.Printf("Failed to Init KernelModuleChecker, config file: <%s>, field: <%s>",
				c.conf.FilePath, errField)
			continue
		}
		detailSubCheckers := make([]kernelModuleDetailSubChecker, 0, len(subCheckerParam.Detail))
		for _, detailItem := range subCheckerParam.Detail {
			detailSubChecker := c.getDetailSubChecker(detailItem, subCheckerParam, &paramCheckErr)
			if detailSubChecker != nil {
				detailSubCheckers = append(detailSubCheckers, kernelModuleDetailSubChecker{
					detailPath:   detailItem.Name,
					expect:       detailItem.Expect,
					valueChecker: detailSubChecker,
				})
			}
		}

		kmSubChecker := KernelModuleSubChecker{
			param:          subCheckerParam,
			detailCheckers: detailSubCheckers,
		}
		c.subCheckers = append(c.subCheckers, kmSubChecker)
	}
	if len(paramCheckErr.ErrField) > 0 {
		return &paramCheckErr
	}
	return nil
}

func (c *kernelModuleChecker) getDetailSubChecker(detailItem SubItemParamBase,
	subCheckerParam KernelModuleSubItemParam, paramCheckErr *ParamCheckError) ValueChecker {
	if !checkParamFormat(detailItem) {
		errField := fmt.Sprintf("spec.[name=%s].detail.[name=%s]", subCheckerParam.Name, detailItem.Name)
		paramCheckErr.ErrField = append(paramCheckErr.ErrField, errField)
		c.logger.Printf("Failed to Init KernelModuleChecker, config file: <%s>, field: <%s>",
			c.conf.FilePath, errField)
		return nil

	}
	vc, err := GetValueChecker(detailItem.Type, detailItem.Expect)
	if err != nil {
		errField := fmt.Sprintf("spec.[name=%s].detail.[name=%s]", subCheckerParam.Name, detailItem.Name)
		paramCheckErr.ErrField = append(paramCheckErr.ErrField, errField)
		c.logger.Printf("Failed to Init KernelModuleChecker, config file: <%s>, field: <%s>, because: %s",
			c.conf.FilePath, errField, err.Error())
		return nil
	}
	return vc
}

// Check performs the kernel module checks and returns the results
func (c *kernelModuleChecker) Check() ItemRslt {
	rslt := ItemRslt{
		Key:      c.conf.Name,
		Result:   ResultValid,
		Doc:      c.conf.Doc,
		SubItems: make([]SubItemRslt, 0, len(c.subCheckers)),
	}
	kms, err := c.getKernelModulesLoaded(c.ctx.RootPath)
	if err != nil {
		rslt.Result = ResultError
		return rslt
	}

	for _, subChecker := range c.subCheckers {
		subItemRslt := c.execSubChecker(subChecker, kms)
		if subItemRslt.Result == ResultInvalid {
			rslt.Result = ResultInvalid
		}
		rslt.SubItems = append(rslt.SubItems, subItemRslt)
	}
	return rslt
}

func (c *kernelModuleChecker) execSubChecker(subChecker KernelModuleSubChecker,
	kernelModuleLoaded map[string]interface{}) SubItemRslt {
	realMsg := make([]string, 0)
	expectMsg := []string{"Loaded"}
	subItemRslt := SubItemRslt{
		Key:    subChecker.param.Name,
		Result: ResultValid,
		Doc:    subChecker.param.Doc,
	}
	name := subChecker.param.ModuleName
	if len(name) == 0 {
		name = subChecker.param.Name
	}
	_, ok := kernelModuleLoaded[name]
	if !ok {
		subItemRslt.Expect = "Loaded"
		subItemRslt.Real = "Not Loaded"
		subItemRslt.Result = ResultInvalid
		return subItemRslt
	}
	realMsg = append(realMsg, "Loaded")

	for _, detailChecker := range subChecker.detailCheckers {
		expectMsg = append(expectMsg, fmt.Sprintf("%s=%v", detailChecker.detailPath, detailChecker.expect))
		content, err := c.getKernelModuleDetail(c.ctx.RootPath, name, detailChecker.detailPath)
		if err != nil {
			realMsg = append(realMsg, fmt.Sprintf("%s=Read Failed", detailChecker.detailPath))
			subItemRslt.Result = ResultInvalid
			continue
		}
		checkRslt := detailChecker.valueChecker.Check(content)
		realMsg = append(realMsg, fmt.Sprintf("%s=%v", detailChecker.detailPath, content))
		if !checkRslt {
			subItemRslt.Result = ResultInvalid
		}
	}
	subItemRslt.Expect = strings.Join(expectMsg, "\n")
	subItemRslt.Real = strings.Join(realMsg, "\n")
	return subItemRslt
}

func (c *kernelModuleChecker) getKernelModulesLoaded(rootPath string) (map[string]interface{}, error) {

	rslt := make(map[string]interface{})
	// // 通過/proc/modules來检索模块
	moduleFile, err := os.Open(filepath.Join(rootPath, procModulesPath))
	if err != nil {
		c.logger.Printf("Failed to read /proc/modules,  because: <%s> ", err.Error())
		return rslt, fmt.Errorf("failed to read module from /proc/modules")
	}
	defer utils.CloseAll(c.logger, moduleFile)
	scanner := bufio.NewScanner(moduleFile)
	for scanner.Scan() {
		if fields := strings.Fields(scanner.Text()); len(fields) >= moduleStatusIndex+1 {
			moduleName := fields[0]
			status := fields[moduleStatusIndex]
			if status == "Live" {
				rslt[moduleName] = struct{}{}
			}
		}
	}
	// 補充通過/sys/module目錄查詢到的信息，实际上这个目录中有所有内核模块的信息，但也包括一些残留的信息,暂时忽略这些信息
	moduleEntries, err := os.ReadDir(filepath.Join(rootPath, sysModulePath))
	if err != nil {
		c.logger.Printf("Failed to read /sys/module, because: <%s>", err.Error())
		return rslt, fmt.Errorf("failed to read module from /sys/module/")
	}
	for _, moduleEntry := range moduleEntries {
		if moduleEntry.IsDir() {
			rslt[moduleEntry.Name()] = struct{}{}
		}
	}
	return rslt, nil

}

func (c *kernelModuleChecker) getKernelModuleDetail(rootPath, module, detailPath string) (string, error) {
	fullPath := filepath.Join(rootPath, "/sys/module", module, detailPath)
	bytes, err := os.ReadFile(fullPath)
	if err != nil {
		c.logger.Printf("Failed to read file: <%s>, because: %s", fullPath, err.Error())
		return "", fmt.Errorf("failed read file: <%s>", detailPath)
	}
	return string(bytes), nil
}

func init() {
	RegisterCheckerFactory(
		"kernel-module-checker",
		func() Checker {
			return &kernelModuleChecker{}
		},
	)
	RegisterCheckerFactory(
		"kernel-module",
		func() Checker {
			return &kernelModuleChecker{}
		},
	)
}
