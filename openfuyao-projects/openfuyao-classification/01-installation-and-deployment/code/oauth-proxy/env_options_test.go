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
	"os"
	"testing"

	"github.com/bmizerany/assert"
)

type envTest struct {
	testField string `cfg:"target_field" env:"TEST_ENV_FIELD"`
}

func TestLoadEnvForStruct(t *testing.T) {

	cfg := make(EnvOptions)
	cfg.LoadEnvForStruct(&envTest{})

	_, ok := cfg["target_field"]
	assert.Equal(t, ok, false)

	os.Setenv("TEST_ENV_FIELD", "1234abcd")
	cfg.LoadEnvForStruct(&envTest{})
	v := cfg["target_field"]
	assert.Equal(t, v, "1234abcd")
}
