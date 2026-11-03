/******************************************************************
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * cluster-api-provider-bke is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package remote

import (
	"fmt"
	"strings"
	"sync"
)

type CombineOut struct {
	Host    string
	Command string
	Out     string
}

type StdErrs StdCombine
type StdOuts StdCombine

type CombineOuts []CombineOut

func (s CombineOuts) String() string {
	var out []string
	for _, v := range s {
		out = append(out, v.String())
	}
	return strings.Join(out, "\n")
}

func NewCombineOut(host string, command string, out string) CombineOut {
	return CombineOut{
		Host:    host,
		Command: command,
		Out:     out,
	}
}

func (s CombineOut) String() string {
	if s.Command != "" {
		return fmt.Sprintf("host: %q, cmd: %q, combined out: %q", s.Host, s.Command, s.Out)
	}
	return fmt.Sprintf("host: %q, error: %q", s.Host, s.Out)
}

type StdCombine struct {
	records map[string]CombineOuts
	mux     *sync.RWMutex
}

func NewStdCombine() StdCombine {
	return StdCombine{
		records: make(map[string]CombineOuts),
		mux:     &sync.RWMutex{},
	}
}

func (e *StdCombine) Add(err CombineOut) {
	e.mux.Lock()
	defer e.mux.Unlock()
	e.records[err.Host] = append(e.records[err.Host], err)
}

func (e *StdCombine) Get(host string) CombineOuts {
	e.mux.RLock()
	defer e.mux.RUnlock()
	return e.records[host]
}

func (e *StdCombine) Len() int {
	return len(e.records)
}

func (e *StdCombine) Out() map[string]CombineOuts {
	e.mux.RLock()
	defer e.mux.RUnlock()

	return e.records
}
