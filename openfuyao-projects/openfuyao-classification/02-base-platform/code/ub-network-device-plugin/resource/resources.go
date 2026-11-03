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

package resource

import (
	"fmt"
	"regexp"
	"strings"

	"k8s.io/klog/v2"

	"gitcode.com/openFuyao/ub-network-device-plugin/utils"
)

type Pattern string

type Name string

type Resource struct {
	Pattern Pattern `json:"pattern" yaml:"pattern"`
	Name    Name    `json:"name"    yaml:"name"`
}

// Resources lists full SHARED devices separately.
type Resources struct {
	SHAREDs []Resource `json:"shared,omitempty"  yaml:"shared,omitempty"`
}

func InitDefaultResources() (*Resources, error) {
	res := &Resources{}
	err := res.AddSHAREResource("*", utils.URMAResource)
	if err != nil {
		return res, err
	}
	return res, nil
}

func NewResourceName(n string) (Name, error) {
	if !strings.HasPrefix(n, utils.ResourceNamePrefix+"/") {
		n = utils.ResourceNamePrefix + "/" + n
	}

	if len(n) > utils.MaxResourceNameLength {
		return "", fmt.Errorf("fully-qualified name name must be %v characters or less: %v", utils.MaxResourceNameLength, n)
	}

	return Name(n), nil
}

func NewResource(pattern, name string) (*Resource, error) {
	resourceName, err := NewResourceName(name)
	if err != nil {
		return nil, fmt.Errorf("invalid name name: %v", err)
	}
	r := &Resource{
		Pattern: Pattern(pattern),
		Name:    resourceName,
	}
	return r, nil
}

func (r Name) Split() (string, string) {
	split := strings.SplitN(string(r), "/", 2)
	if len(split) != 2 {
		return "", string(r)
	}
	return split[0], split[1]
}

// AddSHAREResource adds a SHARE name to the list of SHARE resources.
func (r *Resources) AddSHAREResource(pattern, name string) error {
	resource, err := NewResource(pattern, name)
	if err != nil {
		return err
	}
	r.SHAREDs = append(r.SHAREDs, *resource)
	return nil
}

// Matches checks if the provided string matches the ResourcePattern or not.
func (p Pattern) Matches(s string) bool {
	result, err := regexp.MatchString(wildCardToRegexp(string(p)), s)
	if err != nil {
		klog.Errorf("failed to match pattern: %v", err)
		return false
	}
	return result
}

// wildCardToRegexp converts a wildcard pattern to a regular expression pattern.
func wildCardToRegexp(pattern string) string {
	var result strings.Builder
	for i, literal := range strings.Split(pattern, "*") {
		// Replace * with .*
		if i > 0 {
			result.WriteString(".*")
		}
		// Quote any regular expression meta characters in the literal text.
		result.WriteString(regexp.QuoteMeta(literal))
	}
	return result.String()
}
