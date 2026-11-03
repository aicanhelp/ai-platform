/*
 * Copyright OAuth2_proxy Authors.
 * Licensed under the MIT License (the "License"); you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://opensource.org/licenses/MIT
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
 * an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
 * specific language governing permissions and limitations under the License.
 */

package main

import (
	"flag"
	"strings"
)

var _ flag.Getter = StringArray{}

type StringArray struct {
	stringArray *[]string
}

func NewStringArray() *StringArray {
	return &StringArray{
		stringArray: &[]string{},
	}
}

func (a StringArray) Get() interface{} {
	if a.stringArray == nil {
		return nil
	}
	return *a.stringArray
}

func (a StringArray) Set(s string) error {
	*a.stringArray = append(*a.stringArray, s)
	return nil
}

func (a StringArray) String() string {
	if a.stringArray == nil {
		return ""
	}
	return strings.Join(*a.stringArray, ",")
}
