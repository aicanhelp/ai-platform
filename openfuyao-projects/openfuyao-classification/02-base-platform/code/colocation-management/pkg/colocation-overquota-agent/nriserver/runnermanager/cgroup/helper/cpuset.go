/*
Copyright 2017 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Original file: https://github.com/kubernetes/kubernetes/pkg/kubelet/cm/cpuset/cpuset.go
Copyright (c) 2024 China Unicom Digital Technology Co., Ltd.

*/

// Package helper
package helper

import (
	"fmt"
	"reflect"
	"strconv"
	"strings"
)

const (
	expectedRangeParts = 2
	splitLimit         = 2
)

// CPUSet is a thread-safe, immutable set-like data structure for CPU IDs.
type CPUSet struct {
	elems map[int]struct{}
}

// New returns a new CPUSet containing the supplied elements.
func New(cpus ...int) CPUSet {
	s := CPUSet{
		elems: map[int]struct{}{},
	}
	for _, c := range cpus {
		s.add(c)
	}
	return s
}

// add adds the supplied elements to the CPUSet.
// It is intended for internal use only, since it mutates the CPUSet.
func (s CPUSet) add(elems ...int) {
	for _, elem := range elems {
		s.elems[elem] = struct{}{}
	}
}

// Equals returns true if the supplied set contains exactly the same elements
// as this set (s IsSubsetOf s2 and s2 IsSubsetOf s).
func (s CPUSet) Equals(s2 CPUSet) bool {
	return reflect.DeepEqual(s.elems, s2.elems)
}

// Parse CPUSet constructs a new CPU set from a Linux CPU list formatted string.
//
// See: http://man7.org/linux/man-pages/man7/cpuset.7.html#FORMATS
func Parse(s string) (CPUSet, error) {
	// Handle empty string.
	if s == "" {
		return New(), nil
	}

	result := New()

	// Split CPU list string:
	// "0-5,34,46-48" => ["0-5", "34", "46-48"]
	ranges := strings.Split(s, ",")

	for _, r := range ranges {
		boundaries := strings.SplitN(r, "-", splitLimit)
		if len(boundaries) == 1 {
			// Handle ranges that consist of only one element like "34".
			elem, err := strconv.Atoi(boundaries[0])
			if err != nil {
				return New(), err
			}
			result.add(elem)
		} else if len(boundaries) == expectedRangeParts {
			// Handle multi-element ranges like "0-5".
			start, err := strconv.Atoi(boundaries[0])
			if err != nil {
				return New(), err
			}
			end, err := strconv.Atoi(boundaries[1])
			if err != nil {
				return New(), err
			}
			if start > end {
				return New(), fmt.Errorf("invalid range %q (%d > %d)", r, start, end)
			}
			// start == end is acceptable (1-1 -> 1)

			// Add all elements to the result.
			// e.g. "0-5", "46-48" => [0, 1, 2, 3, 4, 5, 46, 47, 48].
			for e := start; e <= end; e++ {
				result.add(e)
			}
		}
	}
	return result, nil
}

// IsCpuSetEqual checks if two CPU sets are equal.
func IsCpuSetEqual(a, b string) bool {
	parseA, err := Parse(a)
	if err != nil {
		return false
	}
	parseB, err := Parse(b)
	if err != nil {
		return false
	}
	return parseA.Equals(parseB)
}
