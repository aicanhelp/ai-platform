// Copyright(c) Huawei Technologies Co., Ltd. 2025-2025. All right reserved.

// Package pprof
package pprof

import (
	"os"
	"os/signal"
	"syscall"

	"github.com/coredns/coredns/fypkg/crypto/caascore/util/env"
	"github.com/coredns/coredns/plugin/pkg/log"
)

const (
	profilingHandlerChanSize = 1
	defaultProfilingPath     = "/opt/log/dump"
	profilingPathPrem        = 750
	profilingPath            = "FY_PROFILING_PATH"
)

// FYRegisterPProfSignal collection entry, default collection cpu heap block mutex goroutine
func FYRegisterPProfSignal() {
	profilingPath := env.GetEnvAsStringOrDefault(profilingPath, defaultProfilingPath)
	// create profilingPath
	if _, err := os.Stat(profilingPath); os.IsNotExist(err) {
		if err := os.Mkdir(profilingPath, profilingPathPrem); err != nil {
			log.Errorf("create  profilingPath error, register pprof signal failed, ersaon: %v", err)
			return
		}
		log.Infof("create path profilingPath success.")
	}

	log.Infof("register coredns pprof collection")
	ch := make(chan os.Signal, profilingHandlerChanSize)

	// listening signal
	sig := []os.Signal{syscall.SIGUSR2}
	signal.Notify(ch, sig...)

	ph := &profilingHandler{
		modules:       "coredns",
		profilingPath: profilingPath,
	}

	// start listening
	go ph.run(ch)
}
