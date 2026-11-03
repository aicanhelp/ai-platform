// Copyright(c) Huawei Technologies Co., Ltd. 2025-2025. All right reserved.

// Package pprof
package pprof

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime/pprof"
	"strings"
	"syscall"
	"time"

	"github.com/coredns/coredns/plugin/pkg/log"
)

const (
	createIfAbsent     = os.O_WRONLY | os.O_CREATE | os.O_TRUNC
	mode               = 0600
	timeFormat         = "20060102-150405"
	hideDebuggingInfo  = 0
	cpuProfileDuration = 15
)

type profilingHandler struct {
	modules       string
	profilingPath string
}

// FY listening signal
func (p *profilingHandler) run(ch chan os.Signal) {
	for {
		select {
		case s := <-ch:
			switch s {
			case syscall.SIGUSR2:
				log.Infof("[%v] receive signal: %v, try collection pprof", time.Now(), s)
				p.handle()
			default:
				log.Infof("received signal: %v", time.Now())
			}
		}
	}
}

// FY collection pprof
func (p *profilingHandler) handle() {
	p.cleanOldProfileFile()
	go p.cpuProfile()
	go p.heapProfile()
	go p.blockProfile()
	go p.mutexProfile()
	go p.goroutineProfile()
}

// FY clean old profile
func (p *profilingHandler) cleanOldProfileFile() {
	err := filepath.Walk(p.profilingPath, func(absPath string, fileInfo os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if !fileInfo.IsDir() && strings.HasPrefix(fileInfo.Name(), p.modules) && strings.HasSuffix(fileInfo.Name(), ".profile") {
			log.Infof("cleaning old profile %q", fileInfo.Name())
			if err := os.Remove(absPath); err != nil && !os.IsNotExist(err) {
				return err
			}
		}
		return nil
	})
	if err != nil {
		log.Errorf("clean old profile file error, reason: %v", err)
	}
}

// FY generate pprof file name
func (p *profilingHandler) getProfileFile(fileExt string) (*os.File, error) {
	timestamp := time.Now().Format(timeFormat)
	f := fmt.Sprintf("%s-%s-%s", p.modules, timestamp, fileExt)
	filePath := filepath.Clean(filepath.Join(p.profilingPath, f))
	return os.OpenFile(filePath, createIfAbsent, mode)
}

// FY generate cpu profiling, default 15 seconds
func (p *profilingHandler) cpuProfile() {
	file, err := p.getProfileFile("cpu.profile")
	if err != nil {
		log.Errorf("open cup profiling file error, reason: %v", err)
		return
	}
	defer func(file *os.File) {
		err := file.Close()
		if err != nil {
			log.Errorf("close cpu profiling file error, reason: %v", err)
		}
	}(file)
	log.Infof("trying to generate cpu profile file, default duration is 15s")
	if err := pprof.StartCPUProfile(file); err != nil {
		log.Errorf("fail generate cpu profiling file, reason: %v", err)
		return
	}
	defer pprof.StopCPUProfile()
	t := time.NewTimer(time.Duration(cpuProfileDuration) * time.Second)
	<-t.C
	t.Stop()
}

// FY generate heap profiling
func (p *profilingHandler) heapProfile() {
	file, err := p.getProfileFile("heap.profile")
	if err != nil {
		log.Errorf("open heap profile file error, reason: %v", err)
		return
	}
	defer func(file *os.File) {
		err := file.Close()
		if err != nil {
			log.Errorf("close heap profile file error, reason: %v", err)
		}
	}(file)
	log.Infof("trying to generate heap profile file")
	if err := pprof.WriteHeapProfile(file); err != nil {
		log.Errorf("fail generate heap profile file, reason: %v", err)
		return
	}
}

// FY generate block profiling
func (p *profilingHandler) blockProfile() {
	file, err := p.getProfileFile("block.profile")
	if err != nil {
		log.Errorf("open block profile file error, reason: %v", err)
		return
	}
	defer func(file *os.File) {
		err := file.Close()
		if err != nil {
			log.Errorf("close block profile file error, reason: %v", err)
		}
	}(file)
	contentionProfile(file, "block")
}

// FY generate mutex profiling
func (p *profilingHandler) mutexProfile() {
	file, err := p.getProfileFile("mutex.profile")
	if err != nil {
		log.Errorf("open mutex profile file error, reason: %v", err)
		return
	}
	defer func(file *os.File) {
		err := file.Close()
		if err != nil {
			log.Errorf("close mutex profile file error, reason: %v", err)
		}
	}(file)
	contentionProfile(file, "mutex")
}

func (p *profilingHandler) goroutineProfile() {
	file, err := p.getProfileFile("goroutine.profile")
	if err != nil {
		log.Errorf("open goroutine profile file error, reason: %v", err)
		return
	}
	defer func(file *os.File) {
		err := file.Close()
		if err != nil {
			log.Errorf("close mutex profile file error, reason: %v", err)
		}
	}(file)
	contentionProfile(file, "goroutine")
}

// FY generate block mutex goroutine profiling
func contentionProfile(file io.Writer, name string) {
	log.Infof("trying to generate %q profiling file", name)
	profile := pprof.Lookup(name)
	if profile == nil {
		log.Infof("generate %q profile file error.", name)
		return
	}
	if err := profile.WriteTo(file, hideDebuggingInfo); err != nil {
		log.Errorf("fail generate %q profile file. %v", name, err)
		return
	}
}
