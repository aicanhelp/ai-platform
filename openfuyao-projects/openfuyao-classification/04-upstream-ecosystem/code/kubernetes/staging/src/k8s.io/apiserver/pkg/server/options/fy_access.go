/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package options

import (
	"fmt"
	"math"
	"os"

	"github.com/spf13/pflag"
	"k8s.io/apiserver/pkg/server"
	"k8s.io/component-base/fylogs"
)

const (
	minFileSize     = 5   // minimum size of single file: 5MB
	maxFileSize     = 100 // maximum size of single file: 100MB
	noLimit         = 0
	twiceOfFileSize = 2

	defaultFilePerm = 0644

	defaultEnableRotate    = true
	defaultEnableCompress  = true
	defaultCompressPerm    = 0440
	defaultMaxLogFileSize  = 20  // default size of single file: 20MB
	defaultTotalSizeLimit  = 150 // default size of total file: 150MB
	defaultTotalCountLimit = 80

	defaultIgnoreEnospc = true

	defaultCheckLogDeletedPeriod = 0 // It is measured in seconds
)

// FYAccessLogOptions define some attributes for access log
type FYAccessLogOptions struct {
	FYAccessLogPath        string
	FYAccessLogPermissions uint32

	FYAccessLogEnableRotate        bool
	FYAccessLogEnableCompress      bool
	FYAccessLogCompressPermissions uint32
	FYAccessLogMaxSize             int64
	FYAccessLogTotalSize           int64
	FYAccessLogTotalCount          int

	FYAccessLogIgnoreEnospc bool

	FYAccessLogCheckLogDeletedPeriod int
}

// NewFYAccessLogOptions returns a new FYAccessLogOptions
func NewFYAccessLogOptions() *FYAccessLogOptions {
	return &FYAccessLogOptions{
		FYAccessLogPermissions: defaultFilePerm,

		FYAccessLogEnableRotate:        defaultEnableRotate,
		FYAccessLogEnableCompress:      defaultEnableCompress,
		FYAccessLogCompressPermissions: defaultCompressPerm,
		FYAccessLogMaxSize:             defaultMaxLogFileSize,
		FYAccessLogTotalSize:           defaultTotalSizeLimit,
		FYAccessLogTotalCount:          defaultTotalCountLimit,

		FYAccessLogIgnoreEnospc: defaultIgnoreEnospc,

		FYAccessLogCheckLogDeletedPeriod: defaultCheckLogDeletedPeriod,
	}
}

// ApplyTo Apply the FYAccessLogOptions configuration to the server configuration
func (o *FYAccessLogOptions) ApplyTo(c *server.Config) error {
	if o == nil || len(o.FYAccessLogPath) == 0 {
		return nil
	}

	if c == nil {
		return fmt.Errorf("server config must be non-nil")
	}

	accessLogFile, getAccessLogFileErr := o.getAccessLogFile()
	if getAccessLogFileErr != nil {
		return fmt.Errorf("get access log file err: %w", getAccessLogFileErr)
	}

	writerWrap, wrapFileErr := o.toWrapFileConfig().NewWrapFile(accessLogFile)
	if wrapFileErr != nil {
		return fmt.Errorf("wrap file output err: %w", wrapFileErr)
	}

	c.FYAccessLog = server.FYAccessLogInfo{AccessLogWriter: writerWrap}
	return nil
}

// AddFlags is used to add some flags to log
func (o *FYAccessLogOptions) AddFlags(fs *pflag.FlagSet) {
	if o == nil {
		return
	}

	fs.StringVar(&o.FYAccessLogPath, "fy-access-log-path", o.FYAccessLogPath,
		"If set, all requests coming to the apiserver will be logged to this file.")
	fs.Uint32Var(&o.FYAccessLogPermissions, "fy-access-log-permissions", o.FYAccessLogPermissions,
		"If non-empty, change the log file permission to this value. "+
			"Default permission value is 0644, the recommended permission by huawei is 640.")

	fs.BoolVar(&o.FYAccessLogEnableRotate, "fy-access-log-enable-rotate", o.FYAccessLogEnableRotate,
		"Enable rotate access log, default value is true")
	fs.BoolVar(&o.FYAccessLogEnableCompress, "fy-access-log-enable-compress", o.FYAccessLogEnableCompress,
		"Enable compress log file, need enable rotate first. Default is enabled with value false")
	fs.Uint32Var(&o.FYAccessLogCompressPermissions, "fy-access-log-compress-permissions", o.FYAccessLogCompressPermissions,
		"Compress log file permission. Default permission value is 0440, the recommended permission by huawei is 0440.")
	fs.Int64Var(&o.FYAccessLogMaxSize, "fy-access-log-max-size", o.FYAccessLogMaxSize,
		"Max size of each one log file, it need enable rotate first; The unit is MB, value range are [5MB, 100MB]")
	fs.Int64Var(&o.FYAccessLogTotalSize, "fy-access-log-total-size", o.FYAccessLogTotalSize,
		"Max total size of all access log file, it need enable rotate first; The unit is MB,"+
			" the value should not be negative, 0 means no limit. Default value is 0")
	fs.IntVar(&o.FYAccessLogTotalCount, "fy-access-log-total-count", o.FYAccessLogTotalCount,
		"Max log file count limit, it need enable rotate first."+
			" The value should not be negative, 0 means no limit. Default value is 0")

	fs.BoolVar(&o.FYAccessLogIgnoreEnospc, "fy-access-log-ignore-enospc", o.FYAccessLogIgnoreEnospc,
		"When the disk space is full, the process does not exit but does not record logs."+
			" If the disk space is restored, the system records logs again and records the time when the fault occurs."+
			" Default is enable with value true")
	fs.IntVar(
		&o.FYAccessLogCheckLogDeletedPeriod,
		"fy-access-log-check-log-deleted-period",
		o.FYAccessLogCheckLogDeletedPeriod,
		"Periodically checks whether files are deleted unexpectedly, it need enable rotate first."+
			" The unit is second (e.g. 5 second). Default value is 0 that means do not check periodically")

}

// Validate checks invalid config combination
func (o *FYAccessLogOptions) Validate() []error {
	if !o.enabled() {
		return nil
	}

	var allErrors []error
	if err := o.validateRotateOptions(); err != nil {
		allErrors = append(allErrors, err...)
	}

	if err := o.validateCheckLogDeletedOptions(); err != nil {
		allErrors = append(allErrors, err...)
	}

	return allErrors
}

func (o *FYAccessLogOptions) validateRotateOptions() []error {
	var errors []error
	if !o.FYAccessLogEnableRotate {
		return nil
	}

	if o.FYAccessLogMaxSize < minFileSize || o.FYAccessLogMaxSize > maxFileSize {
		errors = append(
			errors,
			fmt.Errorf("invalid access log max size %d , the range is [%dMB, %dMB]",
				o.FYAccessLogMaxSize,
				minFileSize,
				maxFileSize,
			))
	}

	if o.FYAccessLogTotalCount <= noLimit {
		errors = append(
			errors,
			fmt.Errorf("invalid access log total count %d , it must be a positive number",
				o.FYAccessLogTotalCount,
			))
	}

	if o.FYAccessLogTotalSize <= (twiceOfFileSize * int64(math.Max(float64(o.FYAccessLogMaxSize), float64(minFileSize)))) {
		errors = append(
			errors,
			fmt.Errorf("error param '%s', the total size(%dMB) must twice greater than each file size(%dMB)",
				"fy-access-log-total-size",
				o.FYAccessLogTotalSize,
				int64(math.Max(float64(o.FYAccessLogMaxSize), float64(minFileSize)))),
		)
	}

	return errors
}

func (o *FYAccessLogOptions) validateCheckLogDeletedOptions() []error {
	var errors []error
	if o.FYAccessLogCheckLogDeletedPeriod < 0 {
		errors = append(
			errors,
			fmt.Errorf("invalid access log check log deleted period %d ,"+
				" it must be greater than or equal to 0", o.FYAccessLogCheckLogDeletedPeriod),
		)
	}
	return errors
}

func (o *FYAccessLogOptions) toWrapFileConfig() *fylogs.WrapFileConfig {
	return o.getWrapFileConfig()
}

func (o FYAccessLogOptions) getWrapFileConfig() *fylogs.WrapFileConfig {
	wrapFileConfig := fylogs.NewWrapFileConfig()
	wrapFileConfig.FilePath = o.FYAccessLogPath
	wrapFileConfig.FilePermissions = o.FYAccessLogPermissions

	wrapFileConfig.EnableRotate = o.FYAccessLogEnableRotate
	wrapFileConfig.EnableCompress = o.FYAccessLogEnableCompress
	wrapFileConfig.CompressPerm = o.FYAccessLogCompressPermissions
	wrapFileConfig.EachFileMaxSize = o.FYAccessLogMaxSize
	wrapFileConfig.TotalSizeLimit = o.FYAccessLogTotalSize
	wrapFileConfig.TotalCountLimit = o.FYAccessLogTotalCount

	wrapFileConfig.IgnoreEnospc = o.FYAccessLogIgnoreEnospc

	wrapFileConfig.CheckLogDeletedPeriod = o.FYAccessLogCheckLogDeletedPeriod
	return wrapFileConfig
}

// Check whether the access log backend is enabled based on the options.
func (o *FYAccessLogOptions) enabled() bool {
	return o != nil && len(o.FYAccessLogPath) != 0
}

func (o *FYAccessLogOptions) getAccessLogFile() (*os.File, error) {
	accessLogFile, openFileErr := os.OpenFile(
		o.FYAccessLogPath,
		os.O_APPEND|os.O_WRONLY|os.O_CREATE,
		os.FileMode(o.FYAccessLogPermissions),
	)
	if openFileErr != nil {
		return nil, fmt.Errorf("failed to create log file %v: %w", accessLogFile, openFileErr)
	}
	return accessLogFile, nil
}
