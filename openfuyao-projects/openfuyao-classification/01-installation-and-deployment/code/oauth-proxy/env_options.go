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
	"reflect"
	"strings"
)

type EnvOptions map[string]interface{}

func (cfg EnvOptions) LoadEnvForStruct(options interface{}) {
	val := reflect.ValueOf(options).Elem()
	typ := val.Type()
	for i := 0; i < typ.NumField(); i++ {
		// pull out the struct tags:
		//    flag - the name of the command line flag
		//    deprecated - (optional) the name of the deprecated command line flag
		//    cfg - (optional, defaults to underscored flag) the name of the config file option
		field := typ.Field(i)
		flagName := field.Tag.Get("flag")
		envName := field.Tag.Get("env")
		cfgName := field.Tag.Get("cfg")
		if cfgName == "" && flagName != "" {
			cfgName = strings.Replace(flagName, "-", "_", -1)
		}
		if envName == "" || cfgName == "" {
			// resolvable fields must have the `env` and `cfg` struct tag
			continue
		}
		v := os.Getenv(envName)
		if v != "" {
			cfg[cfgName] = v
		}
	}
}
