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
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"time"

	"openfuyao.com/oscheck/internal/conf"
	"openfuyao.com/oscheck/internal/utils"
)

var (
	defaultKVSplitter = regexp.MustCompile(`[ \t]+`)
)

const (
	kvSize = 2
)

// ConfKVChecker defines the configuration structure for KV checkers
type ConfKVChecker struct {
	Source       ConfCheckSource    `json:"source"`
	ConfKvParser ConfKVParser       `json:"kvParser"`
	SubItems     []SubItemParamBase `json:"subItems"`
	confFile     string
}

// ConfCheckSource defines the source of key-value pairs, which can be files or command outputs
type ConfCheckSource struct {
	FilePaths  []string `json:"file"`
	Commands   []string `json:"command"`
	SoVersions []string `json:"soversions"`
}

// ConfKVParser defines the parameters for parsing key-value pairs
type ConfKVParser struct {
	ReSplitter string `json:"reSplitter"`
	ReGroup    string `json:"reGroup"`
	// ReIgnore 忽略满足此条件的正则表达式，默认是以#开始的行, 也就是符合如下规则^\w#.*$
	ReIgnore string `json:"reIgnore"`
	// keyCaseSensitive 大小写是否敏感，默认为false
	KeyCaseSensitive bool `json:"keyCaseSensitive"`
}

// kvChecker implements the Checker interface for checking key-value pairs
type kvChecker struct {
	ctx         Context
	conf        conf.CheckItem
	kvParser    KVParser
	spec        ConfKVChecker
	subCheckers []kvSubChecker
	logger      *log.Logger
}

// KVParser defines the interface for parsing key-value pairs from a line of text
type KVParser interface {
	parse(line string) []string
}

// Init initializes the kvChecker with the given context and configuration
func (c *kvChecker) Init(ctx Context, conf conf.CheckItem) *ParamCheckError {
	c.ctx = ctx
	c.conf = conf
	c.logger = utils.GetLogger("KVChecker")

	hasErr := false
	checkErr := ParamCheckError{
		ErrField: make([]string, 0),
	}
	spec := ConfKVChecker{}
	err := convertSpec2Special(conf, &spec)
	if err != nil {
		c.logger.Printf("Failed to parse config : %s, because: %s", conf.FilePath, err.Error())
		unmarshallErr, ok := err.(*json.UnmarshalTypeError)
		if ok {
			checkErr.ErrField = append(checkErr.ErrField, fmt.Sprintf("spec.%s", unmarshallErr.Field))
		}
		checkErr.FormatErr = true
		checkErr.Msg = "Format error"
		return &checkErr
	}
	c.checkKvCheckerSource(&spec, &checkErr)

	kvparser, err := c.getKvParser(spec.ConfKvParser)
	if err != nil {
		c.logger.Printf("Failed to parse config : %s, because: %s", conf.FilePath, err.Error())
		hasErr = true
		checkErr.FormatErr = true
		checkErr.ErrField = append(checkErr.ErrField, "spec.KvParser")
	} else {
		c.kvParser = kvparser
	}

	c.spec = spec
	hasErr = c.initSubCheckers(checkErr)
	if hasErr {
		return &checkErr
	}
	return nil
}

func (c *kvChecker) initSubCheckers(checkErr ParamCheckError) bool {
	hasErr := false
	for _, param := range c.spec.SubItems {
		param.Name = strings.TrimSpace(param.Name)
		if len(param.Name) == 0 {
			checkErr.ErrField = append(checkErr.ErrField, fmt.Sprintf("spec.subItems.[name=%s]", param.Name))
			c.logger.Printf("Failed to init checker from file: %s, because subItem name is empty", c.conf.FilePath)
			hasErr = true
			continue
		}
		valueChecker, err := GetValueChecker(param.Type, param.Expect)
		if err != nil {
			c.logger.Printf("Failed to init checker from file: %s, init valuce checker failed for: %s, because: %s",
				c.conf.FilePath, param.Name, err.Error())
			hasErr = true
			checkErr.ErrField = append(checkErr.ErrField, fmt.Sprintf("spec.subItems.[name=%s]", param.Name))
			continue
		}
		subChecker := &kvSubChecker{
			param:   param,
			checker: valueChecker,
		}
		c.subCheckers = append(c.subCheckers, *subChecker)
	}
	return hasErr
}

func (c *kvChecker) getSource() (map[string][]string, error) {
	rslt := make(map[string][]string)
	lineProcessor := func(source, line string) {
		line = strings.TrimSpace(line)
		values := c.kvParser.parse(line)

		if len(values) != kvSize {
			c.logger.Printf("Failed to parse line, source: <%v>, raw line data: <%s>,parse result: <%s>",
				source, line, values)
			return
		}
		key := strings.TrimSpace(values[0])
		if !c.spec.ConfKvParser.KeyCaseSensitive {
			key = strings.ToUpper(key)
		}
		curValues, exist := rslt[key]
		if !exist {
			curValues = make([]string, 0, 1)
		}
		rslt[key] = append(curValues, values[1])
	}
	if len(c.spec.Source.FilePaths) > 0 {
		err := c.getKvFromFiles(c.ctx.RootPath, c.spec.Source.FilePaths, lineProcessor)
		return rslt, err
	} else if len(c.spec.Source.Commands) > 0 {
		err := c.getKvFromCommand(c.ctx.RootPath, c.spec.Source.Commands, lineProcessor)
		return rslt, err
	}
	return rslt, fmt.Errorf("failed to get source, because there is no source")

}

func (c *kvChecker) getKvFromFiles(rootPath string, filePaths []string, lineprocessor func(string, string)) error {
	for _, filePath := range filePaths {
		originFilePath := filePath
		filePath = filepath.Join(rootPath, filePath)
		var fileNames []string
		if strings.Contains(filePath, "*") {
			var err error = nil
			fileNames, err = filepath.Glob(filePath)
			if err != nil {
				return fmt.Errorf("failed to parse file for path: %s, because: %s", originFilePath, err.Error())
			}
		} else {
			fileNames = []string{filePath}
		}

		for _, fileName := range fileNames {
			file, err := os.Open(fileName)
			if err != nil {
				return fmt.Errorf("failed to read file for path: %s, because: %s", originFilePath, err.Error())
			}
			sc := bufio.NewScanner(file)
			for sc.Scan() {
				line := sc.Text()
				lineprocessor(fileName, line)
			}
			utils.CloseAll(c.logger, file)
		}
	}
	return nil
}

func (c *kvChecker) getKvFromCommand(rootPath string, commands []string, lineprocessor func(string, string)) error {
	for _, command := range commands {
		if strings.Contains(command, "$") {
			command = strings.ReplaceAll(command, "$", "\\$")
		}
		exec := utils.Command{
			Command:       "sh",
			Args:          []string{"-c", command},
			CombineOutput: true,
			Timeout:       utils.DefaultCmdTimeoutSeconds * time.Second,
			RootPath:      rootPath,
			Logger:        c.logger,
		}
		cmdRslt, err := exec.Exec()
		if err != nil {
			return err
		}
		msg := cmdRslt.Stdout
		exitCode := cmdRslt.ExitCode
		c.logger.Printf("Execute command: <%s>, exit code: <%d>, output: <%s>", command, exitCode, msg)
		sc := bufio.NewScanner(strings.NewReader(msg))
		for sc.Scan() {
			line := sc.Text()
			lineprocessor(command, line)
		}
	}
	return nil
}

// Check performs the key-value checking based on the initialized configuration
func (c *kvChecker) Check() ItemRslt {
	hasErr := false
	hasInvalid := false

	rslt := ItemRslt{
		Key:      c.conf.Name,
		Result:   ResultValid,
		Doc:      c.conf.Doc,
		SubItems: make([]SubItemRslt, 0, len(c.subCheckers)),
	}
	source, err := c.getSource()
	if err != nil {
		c.logger.Printf("Failed to get source for check file: <%s>, because: %s", c.spec.confFile, err.Error())
		rslt.Result = ResultError
		return rslt
	}

	for _, subchecker := range c.subCheckers {
		valuesToCheck := c.getValuesToCheck(subchecker, source)
		matchRslt := matchValue(subchecker.param.MatchType, subchecker.checker, valuesToCheck)
		rsltValue := ResultValid
		if !matchRslt {
			hasInvalid = true
			rsltValue = ResultInvalid
		}
		subItemRslt := SubItemRslt{
			Key:    subchecker.param.Name,
			Expect: fmt.Sprintf("%v", subchecker.param.Expect),
			Real:   strings.Join(valuesToCheck, ";"),
			Result: rsltValue,
			Doc:    subchecker.param.Doc,
		}
		rslt.SubItems = append(rslt.SubItems, subItemRslt)
	}

	if hasErr {
		rslt.Result = ResultError
	} else if hasInvalid {
		rslt.Result = ResultInvalid
	}

	return rslt

}

func (c *kvChecker) getValuesToCheck(subchecker kvSubChecker, source map[string][]string) []string {
	key := subchecker.param.Key
	valuesToCheck := make([]string, 0, 1)
	var keys []string
	if len(key) > 0 {
		keys = strings.Split(key, "\n")
	} else {
		keys = []string{subchecker.param.Name}
	}
	for _, key := range keys {
		if !c.spec.ConfKvParser.KeyCaseSensitive {
			key = strings.ToUpper(key)
		}
		values, ok := source[key]
		if ok && len(values) > 0 {
			valuesToCheck = append(valuesToCheck, values...)
		}
	}
	if len(valuesToCheck) == 0 {
		valuesToCheck = append(valuesToCheck, fmt.Sprintf("%v", subchecker.param.Default))
	}
	return valuesToCheck
}

func (c *kvChecker) checkKvCheckerSource(conf *ConfKVChecker, checkErr *ParamCheckError) {
	if len(conf.Source.Commands) == 0 && len(conf.Source.FilePaths) == 0 {
		c.logger.Printf("spec.source is empty: %v", conf.confFile)
		checkErr.ErrField = append(checkErr.ErrField, "spec.source")
		checkErr.FormatErr = true
		return
	}

	if len(conf.Source.FilePaths) > 0 {
		formatedSourceFiles := make([]string, 0, len(conf.Source.FilePaths))
		confBasePath := filepath.Dir(conf.confFile)
		for _, filePath := range conf.Source.FilePaths {
			if filepath.IsAbs(filePath) {
				formatedSourceFiles = append(formatedSourceFiles, filePath)
			} else {
				formatedSourceFiles = append(formatedSourceFiles, filepath.Join(confBasePath, filePath))
			}
		}
		conf.Source.FilePaths = formatedSourceFiles
	}
}

func (c *kvChecker) getKvParser(conf ConfKVParser) (KVParser, error) {
	var reSplitter, reGroupper, reIgnore *regexp.Regexp
	var kvParser KVParser
	var err error
	if len(conf.ReGroup) > 0 {
		reGroupper, err = regexp.Compile(conf.ReGroup)
		if err != nil {
			c.logger.Printf("Failed to compile reGroup: %s, because: %s", conf.ReGroup, err.Error())
			return kvParser, err
		}
	}

	if len(conf.ReSplitter) > 0 {
		reSplitter, err = regexp.Compile(conf.ReSplitter)
		if err != nil {
			c.logger.Printf("Failed to compile reSplitter: %s, because: %s", conf.ReSplitter, err.Error())
			return kvParser, err
		}

	}

	if len(conf.ReIgnore) > 0 {
		reIgnore, err = regexp.Compile(conf.ReIgnore)
		if err != nil {
			c.logger.Printf("Failed to compile reIgnore: %s, because: %s", conf.ReIgnore, err.Error())
			return kvParser, err
		}
	}
	kvParser = &reKvParser{
		reSplitter:    reSplitter,
		reGroupper:    reGroupper,
		reIgnore:      reIgnore,
		caseSensitive: conf.KeyCaseSensitive,
	}
	return kvParser, nil
}

type kvSubChecker struct {
	param   SubItemParamBase
	checker ValueChecker
}

type reKvParser struct {
	reSplitter    *regexp.Regexp
	reGroupper    *regexp.Regexp
	reIgnore      *regexp.Regexp
	caseSensitive bool
}

func (p *reKvParser) parse(line string) []string {
	line = strings.TrimSpace(line)
	if p.reIgnore != nil && p.reIgnore.MatchString(line) {
		return []string{}
	}

	if p.reGroupper != nil {
		rslt := p.reGroupper.FindStringSubmatch(line)
		if len(rslt) > 1 {
			if len(rslt) > kvSize {
				// 如果有多个分组，则使用中划线连接前面的分组作为key，最后一个分组作为value
				return append([]string{strings.Join(rslt[1:len(rslt)-1], "-")}, rslt[len(rslt)-1])
			}
			return rslt[1:]
		}
		return []string{}
	}
	if p.reSplitter == nil {
		p.reSplitter = defaultKVSplitter
	}

	rslt := p.reSplitter.Split(line, kvSize)
	if len(rslt) > 0 {
		return rslt
	}
	return []string{}
}

func init() {
	RegisterCheckerFactory(
		"kv",
		func() Checker {
			return &kvChecker{}
		},
	)
	RegisterCheckerFactory(
		"kv-checker",
		func() Checker {
			return &kvChecker{}
		},
	)
}
