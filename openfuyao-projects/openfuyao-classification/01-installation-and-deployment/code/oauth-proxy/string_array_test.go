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
	"testing"

	"github.com/bmizerany/assert"
)

func TestStringArray(t *testing.T) {
	sa := NewStringArray()
	assert.Equal(t, "", sa.String())
	err := sa.Set("foo")
	if err != nil {
		t.Errorf("unexpected error %v", err)
	}
	assert.Equal(t, "foo", sa.String())
	err = sa.Set("bar")
	if err != nil {
		t.Errorf("unexpected error %v", err)
	}
	assert.Equal(t, "foo,bar", sa.String())
}

func TestStringArrayInFlagSet(t *testing.T) {
	sa := NewStringArray()

	// flagset tries to grab an empty interface of each flag and invokes
	// their .String() method to check whether they are empty
	flagSet := flag.NewFlagSet("cool flagset", flag.ExitOnError)
	flagSet.Var(sa, "favourite colours", "colourful storm")
	flagSet.PrintDefaults()
}
