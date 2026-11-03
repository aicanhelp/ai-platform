/*
Copyright 2018 The Kubernetes Authors.
Copyright (c) 2025 Huawei Technologies Co., Ltd.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// Package main
// 通过日志拦截, 实现轮转, 磁盘恢复功能
package main

import (
	"flag"
	"fmt"
	"io"
	"log"
	"math"
	"os"
	"os/exec"
	"os/signal"
	"strings"
	"syscall"

	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/component-base/fylogs"
)

const (
	twiceOfFileSize = 2 // 合法的日志文件总大小应大于单个文件大小的两倍

	defaultFilePerm = 0644

	defaultEnableRotate   = false
	defaultEnableCompress = true
	defaultCompressPerm   = 0440
	defaultFileSize       = 20  // 默认单个文件大小: 20MB
	defaultTotalSize      = 150 // 默认最大文件占用: 150MB
	defaultLogCount       = 10

	defaultIgnoreEnospc = false

	minFileSize = 5   // 单个文件允许配置的最小值: 5MB
	maxFileSize = 100 // 单个文件允许配置的最大值: 100MB

	noLimit  = 0
	notCheck = 0
)

var (
	logFilePath    = flag.String("log-file", "", "If non-empty, save stdout to this file")
	alsoToStdOut   = flag.Bool("also-stdout", false, "useful with log-file, log to standard output as well as the log file")
	redirectStderr = flag.Bool("redirect-stderr", true, "treat stderr same as stdout")

	// FY: permission specifies the log file permission
	permission = flag.Int("fy-file-permission", defaultFilePerm, fmt.Sprintf("If non-empty, change the log file permission to this value. Default permission value is %o, the recommanded permission by fuyao is 640.", defaultFilePerm))

	compressPermission = flag.Int("fy-compress-permission", defaultCompressPerm, fmt.Sprintf("Compress log file permission. Default permission value is %o, the recommended permission by fuyao is %o", defaultCompressPerm, defaultCompressPerm))

	// FY: ignoreEnospc will ignore ENOSPC
	ignoreEnospc = flag.Bool("fy-ignore-nospace-error", false, "When the disk space is full, the process does not exist but does record logs. If the disk space is restored, the system records logs again and records the time when the fault occurs. Default is unable with value false")

	enableRotate = flag.Bool("fy-rotate-enable", false, "Enable rotate log, default value is false")

	eachFileSize = flag.Int64("fy-each-file-size", defaultFileSize, fmt.Sprintf("Max size of each one log file, it need enable rotate first; The unit is MB, value range are [%dMB, %dMB]", minFileSize, maxFileSize))

	totalSizeLimit = flag.Int64("fy-total-size-limit", defaultTotalSize, fmt.Sprintf("Max total size of all log file, it need enable rotate first; The unit is MB, the value should not be negative, value %d means no limit.", noLimit))

	totalCountLimit = flag.Int("fy-total-count-limit", defaultLogCount, fmt.Sprintf("Max log file count limit, it need enable rotate first. The value should not be negative, value %d means no limit.", noLimit))

	enableCompress = flag.Bool("fy-compress-enable", true, "Enable compress log file, need enable rotate first. Default is enabled with value true")

	checkLogDeletePeriod = flag.Int("fy-check-log-deleted-period", notCheck, "Periodically chacks whether files are deleted unexpectedly, it need enable rotate first. The unit is second (e.g. 5 second). Default value is 0 that means do not check periodically")

	enableDaemonSubprocess = flag.Bool("fy-enable-daemon-subprocess", false, "Enable to daemon subprocess. After the subprocess exist, kube-log-runner restarts the process. Default is enabled with value false")
)

func main() {
	flag.Parse()

	if err := FYConfigureAndRun(); err != nil {
		log.Fatal(err)
	}
}

func (h *FYKubeLogRunnerOptions) getFlags() {
	h.FYFilePermission = uint32(*permission)

	h.FYRotateEnable = *enableRotate
	h.FYCompressEnable = *enableCompress
	h.FYCompressPermission = uint32(*compressPermission)
	h.FYEachFileSize = *eachFileSize
	h.FYTotalSizeLimit = *totalSizeLimit
	h.FYTotalCountLimit = *totalCountLimit

	h.FYAccessLogIgnoreEnospc = *ignoreEnospc

	h.FYCheckLogDeletePeriod = *checkLogDeletePeriod
}

// FYKubeLogRunnerOptions config options for fuyao kube-log-runner
type FYKubeLogRunnerOptions struct {
	logFilePath      string
	FYFilePermission uint32

	FYRotateEnable       bool
	FYCompressEnable     bool
	FYCompressPermission uint32
	FYEachFileSize       int64
	FYTotalSizeLimit     int64
	FYTotalCountLimit    int

	FYAccessLogIgnoreEnospc bool

	FYCheckLogDeletePeriod int
}

func newFYKubeLogRunnerOptions() *FYKubeLogRunnerOptions {
	return &FYKubeLogRunnerOptions{
		logFilePath:      *logFilePath,
		FYFilePermission: defaultFilePerm,

		FYRotateEnable:       defaultEnableRotate,
		FYCompressEnable:     defaultEnableCompress,
		FYCompressPermission: defaultCompressPerm,
		FYEachFileSize:       defaultFileSize,
		FYTotalSizeLimit:     defaultTotalSize,
		FYTotalCountLimit:    defaultLogCount,

		FYAccessLogIgnoreEnospc: defaultIgnoreEnospc,

		FYCheckLogDeletePeriod: notCheck,
	}
}

func (o *FYKubeLogRunnerOptions) validate() []error {
	var errors []error
	if !o.FYRotateEnable {
		return nil
	}

	if o.FYEachFileSize < minFileSize || o.FYEachFileSize > maxFileSize {
		errors = append(
			errors,
			fmt.Errorf("invalid each log file size %d, the range is [%dMB, %dMB]", o.FYEachFileSize, minFileSize, maxFileSize),
		)
	}

	if o.FYTotalCountLimit <= noLimit {
		errors = append(errors, fmt.Errorf("invalid total count limit %d, it must be a postive number", o.FYTotalCountLimit))
	}

	if o.FYTotalSizeLimit < (twiceOfFileSize * int64(math.Max(float64(o.FYEachFileSize), float64(minFileSize)))) {
		errors = append(
			errors,
			fmt.Errorf("error param '%s', the total size(%dMB) must twice greater than each file size(%dMB)",
				"fy-total-size-limit", o.FYTotalSizeLimit, int64(math.Max(float64(o.FYEachFileSize), float64(minFileSize)))),
		)
	}

	return errors
}

func tryWrapFileOutput(LogFile *os.File) (io.Writer, error) {
	fyKubeLogRunnerOptions := newFYKubeLogRunnerOptions()
	fyKubeLogRunnerOptions.getFlags()

	if errors := fyKubeLogRunnerOptions.validate(); errors != nil {
		return nil, fmt.Errorf("get wrap file config err: %w", utilerrors.NewAggregate(errors))
	}

	writeWrap, wrapFileErr := fyKubeLogRunnerOptions.getWrapFileConfig().NewWrapFile(LogFile)
	if wrapFileErr != nil {
		return nil, fmt.Errorf("wrap file output err: %w", wrapFileErr)
	}

	return writeWrap, nil
}

func (h *FYKubeLogRunnerOptions) getWrapFileConfig() *fylogs.WrapFileConfig {
	wrapFileConfig := fylogs.NewWrapFileConfig()

	wrapFileConfig.FilePath = *logFilePath
	wrapFileConfig.FilePermissions = h.FYFilePermission

	wrapFileConfig.EnableRotate = h.FYRotateEnable
	wrapFileConfig.EnableCompress = h.FYCompressEnable
	wrapFileConfig.CompressPerm = h.FYCompressPermission
	wrapFileConfig.EachFileMaxSize = h.FYEachFileSize
	wrapFileConfig.TotalSizeLimit = h.FYTotalSizeLimit
	wrapFileConfig.TotalCountLimit = h.FYTotalCountLimit

	wrapFileConfig.IgnoreEnospc = h.FYAccessLogIgnoreEnospc

	wrapFileConfig.CheckLogDeletedPeriod = h.FYCheckLogDeletePeriod

	return wrapFileConfig
}

func startCmd(exe string, exeArgs []string, outputStream io.Writer, errStream io.Writer) error {
	for {
		cmd := exec.Command(exe, exeArgs...)
		cmd.Stdout = outputStream
		cmd.Stderr = errStream

		log.Printf("Running command: \n%v", cmdInfo(cmd))
		err := cmd.Start()
		if err != nil {
			return fmt.Errorf("starting command: %w", err)
		}
		cha := make(chan struct{})
		// Handle signals and shutdown process gracefully
		go fySetupSigHandler(cmd.Process, cha)
		if err := cmd.Wait(); err != nil {
			log.Printf("running command: %v", err)
		}
		close(cha)
	}
}

func fySetupSigHandler(process *os.Process, sign chan struct{}) {
	// terminationSignals are signals that cause the program to exit in the
	// supported platforms (linux, darwin, windows).
	terminationSignals := []os.Signal{syscall.SIGHUP, syscall.SIGINT, syscall.SIGTERM, syscall.SIGQUIT}

	c := make(chan os.Signal, 1)
	signal.Notify(c, terminationSignals...)

	// Block until a signal is received
	log.Println("Now listening for interrupts")
	select {
	case s := <-c:
		log.Printf("Got signal: %v. Sending down to process (PID %v)", s, process.Pid)
		if err := process.Signal(s); err != nil {
			log.Fatalf("Failed to signal process: %v", err)
		}
		log.Printf("Signaled process %v successfully", process.Pid)
	case <-sign:
		return
	}
}

func FYConfigureAndRun() error {
	var (
		outputStream io.Writer = os.Stdout
		errStream    io.Writer = os.Stderr
	)

	args := flag.Args()
	if len(args) == 0 {
		return fmt.Errorf("not enough arguments to run")
	}

	if logFilePath != nil && *logFilePath != "" {
		originalLogFile, err := os.OpenFile(*logFilePath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, os.FileMode(uint32(*permission)))
		if err != nil {
			return fmt.Errorf("failed to create log file %v: %w", *logFilePath, err)
		}

		logFile, err := tryWrapFileOutput(originalLogFile)
		if err != nil {
			return fmt.Errorf("failed to wrap log file %v: %w", *logFilePath, err)
		}
		if *alsoToStdOut {
			outputStream = io.MultiWriter(os.Stdout, logFile)
		} else {
			outputStream = logFile
		}
	}

	if *redirectStderr {
		errStream = outputStream
	}

	exe := args[0]
	var exeArgs []string
	if len(args) > 1 {
		exeArgs = args[1:]
	}
	if *enableDaemonSubprocess {
		return startCmd(exe, exeArgs, outputStream, errStream)
	}
	cmd := exec.Command(exe, exeArgs...)
	cmd.Stdout = outputStream
	cmd.Stderr = errStream

	log.Printf("Running command:\n%v", cmdInfo(cmd))
	err := cmd.Start()
	if err != nil {
		return fmt.Errorf("starting command: %w", err)
	}

	// Handle signals and shutdown process gracefully.
	go setupSigHandler(cmd.Process)
	if err := cmd.Wait(); err != nil {
		return fmt.Errorf("running command: %w", err)
	}
	return nil
}

// cmdInfo generates a useful look at what the command is for printing/debug.
func cmdInfo(cmd *exec.Cmd) string {
	return fmt.Sprintf(
		`Command env: (log-file=%v, also-stdout=%v, redirect-stderr=%v)
Run from directory: %v
Executable path: %v
Args (comma-delimited): %v`, *logFilePath, *alsoToStdOut, *redirectStderr,
		cmd.Dir, cmd.Path, strings.Join(cmd.Args, ","),
	)
}

// setupSigHandler will forward any termination signals to the process
func setupSigHandler(process *os.Process) {
	// terminationSignals are signals that cause the program to exit in the
	// supported platforms (linux, darwin, windows).
	terminationSignals := []os.Signal{syscall.SIGHUP, syscall.SIGINT, syscall.SIGTERM, syscall.SIGQUIT}

	c := make(chan os.Signal, 1)
	signal.Notify(c, terminationSignals...)

	// Block until a signal is received.
	log.Println("Now listening for interrupts")
	s := <-c
	log.Printf("Got signal: %v. Sending down to process (PID: %v)", s, process.Pid)
	if err := process.Signal(s); err != nil {
		log.Fatalf("Failed to signal process: %v", err)
	}
	log.Printf("Signalled process %v successfully.", process.Pid)
}
