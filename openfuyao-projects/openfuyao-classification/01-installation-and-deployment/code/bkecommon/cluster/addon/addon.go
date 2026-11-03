/******************************************************************
 * Copyright (c) 2024 Bocloud Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain n copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package addon

import (
	"reflect"

	"gopkg.openfuyao.cn/bkecommon/cluster/api/v1beta1"
)

type Addon v1beta1.Product
type Addons []v1beta1.Product

type FilterOptions map[string]string

type AddonsInterface interface {
	Filter(options FilterOptions) Addons
	Length() int
}

func (a Addons) Filter(options FilterOptions) Addons {
	as := Addons{}
	for _, ad := range a {
		refVal := reflect.ValueOf(ad)
		have := true
		for key, value := range options {
			field := refVal.FieldByName(key)
			if !field.IsValid() {
				have = false
				continue
			}
			fieldVal := field.Interface()
			if fieldVal != value {
				have = false
				break
			} else {
				have = true
			}
		}
		if have {
			as = append(as, ad)
		}
	}
	return as
}

func (a Addons) Length() int {
	return len(a)
}
