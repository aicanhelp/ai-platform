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
	"encoding/json"
	"fmt"
	"log"
	"reflect"
	"strings"
	"time"

	"openfuyao.com/oscheck/internal/conf"
	"openfuyao.com/oscheck/internal/utils"
)

func init() {
	RegisterCheckerFactory(
		"command",
		func() Checker {
			return &commandChecker{}
		},
	)
	RegisterCheckerFactory(
		"command-checker",
		func() Checker {
			return &commandChecker{}
		},
	)
}

// CommandSubItemParam defines the parameters for a command sub-item checker
type CommandSubItemParam struct {
	SubItemParamBase
	Command        string `json:"command"`
	TimeoutSeconds int64  `json:"timeoutSeconds"`
}

// ConfCommandChecker defines the configuration structure for command checkers
type ConfCommandChecker struct {
	Items []CommandSubItemParam `json:"subItems"`
}

// UnmarshalJSON custom unmarshals a CommandSubItemParam from JSON
func (p *CommandSubItemParam) UnmarshalJSON(b []byte) error {
	var raw map[string]interface{}
	if err := json.Unmarshal(b, &raw); err != nil {
		return err
	}

	if err := json.Unmarshal(b, &p.SubItemParamBase); err != nil {
		return err
	}
	if v, ok := raw["command"]; ok {
		switch v := v.(type) {
		case string:
			p.Command = v
		default:
			return &json.InvalidUnmarshalError{Type: reflect.TypeOf(v)}
		}
	}

	if v, ok := raw["timeoutSeconds"]; ok {
		switch val := v.(type) {
		case string:
			timeout, err := utils.ParseInt10Base(val)
			if err == nil {
				p.TimeoutSeconds = timeout
			}
		case float64:
			p.TimeoutSeconds = int64(val)
		default:
			p.TimeoutSeconds = utils.DefaultCmdTimeoutSeconds
		}
	} else {
		p.TimeoutSeconds = utils.DefaultCmdTimeoutSeconds
	}
	return nil
}

type commandSubChecker struct {
	param   CommandSubItemParam
	checker ValueChecker
}

type commandChecker struct {
	ctx         Context
	subCheckers []commandSubChecker
	config      conf.CheckItem
	logger      *log.Logger
}

// Init initializes the command checker with the given context and configuration
func (c *commandChecker) Init(ctx Context, config conf.CheckItem) *ParamCheckError {
	c.ctx = ctx
	c.config = config
	c.logger = utils.GetLogger("CommandChecker")

	checkErr := ParamCheckError{
		ErrField:  make([]string, 0),
		Msg:       "",
		FormatErr: false,
	}
	var subItems []CommandSubItemParam
	err := convertSpec2Special(config, &subItems)
	if err != nil {
		checkErr.FormatErr = true
		checkErr.Msg = "Format Error"
		return &checkErr
	}
	for _, subItem := range subItems {
		if len(subItem.Command) <= 0 {
			checkErr.ErrField = append(checkErr.ErrField, fmt.Sprintf("spec.subItems.[name=%s].command", subItem.Name))
			c.logger.Printf("Formated check error, config file: <%s>, field: <%s> should not be empty",
				config.FilePath, fmt.Sprintf("spec.subItems.[name=%s].command", subItem.Name))
			continue
		}

		vc, err := GetValueChecker(subItem.Type, subItem.Expect)
		if err != nil {
			checkErr.ErrField = append(checkErr.ErrField, fmt.Sprintf("spec.subItems.[name=%s].command", subItem.Name))
			c.logger.Printf("Formated check error, config file: <%s>, failed to get value checker for field: <%s>, because: %s",
				config.FilePath, fmt.Sprintf("spec.subItems.[name=%s].command", subItem.Name), err.Error())
		}
		subChecker := commandSubChecker{
			param:   subItem,
			checker: vc,
		}
		c.subCheckers = append(c.subCheckers, subChecker)

	}
	if len(checkErr.ErrField) > 0 {
		return &checkErr
	}
	return nil
}

// Check performs the command checks and returns the results
func (c *commandChecker) Check() ItemRslt {

	hasInvalid := false

	rslt := ItemRslt{
		Key:      c.config.Name,
		Result:   ResultValid,
		Doc:      c.config.Doc,
		SubItems: make([]SubItemRslt, 0, len(c.subCheckers)),
	}

	for _, subChecker := range c.subCheckers {
		subItemRslt := c.execSubCheck(subChecker)

		if subItemRslt.Result == ResultInvalid {
			rslt.Result = ResultInvalid
		}
		rslt.SubItems = append(rslt.SubItems, subItemRslt)
	}
	if hasInvalid {
		rslt.Result = ResultInvalid
	}
	return rslt
}

func (c *commandChecker) execSubCheck(subChecker commandSubChecker) SubItemRslt {
	subItemRslt := SubItemRslt{
		Key:    subChecker.param.Name,
		Expect: fmt.Sprintf("%v", subChecker.param.Expect),
		Result: ResultValid,
		Doc:    subChecker.param.Doc,
	}
	// 如果使用的是emptyChecker，意味着不需要对命令输出进行比对，那么直接使用返回值进行判断
	if subChecker.checker == emptyChecker && subItemRslt.Expect == "" {
		subItemRslt.Expect = "exit 0"
	}
	cmd := &utils.Command{
		Command:       "sh",
		Args:          append([]string{"-c"}, subChecker.param.Command),
		Timeout:       time.Duration(subChecker.param.TimeoutSeconds * int64(time.Second)),
		RootPath:      c.ctx.RootPath,
		CombineOutput: true,
		Logger:        c.logger,
	}
	cmdRslt, err := cmd.Exec()

	if err != nil {
		c.logger.Printf("Failed to execute command <%s> for config file: <%s>, error: <%s>, cmdOut: %s",
			subChecker.param.Command, c.config.FilePath, err.Error(), cmdRslt.Stdout)

		subItemRslt.Result = ResultInvalid
		subItemRslt.Real = "Failed to execute command"

	} else if subChecker.checker == emptyChecker {
		subItemRslt.Real = fmt.Sprintf("exit %d", cmdRslt.ExitCode)
		if cmdRslt.ExitCode != 0 {
			subItemRslt.Result = ResultInvalid
		}

	} else {
		subItemRslt.Real = strings.TrimSpace(cmdRslt.Stdout)
		matchRslt := matchValue(subChecker.param.MatchType, subChecker.checker, strings.Split(subItemRslt.Real, "\n"))
		if !matchRslt {
			subItemRslt.Result = ResultInvalid
		}
	}
	return subItemRslt
}
