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

import "gopkg.openfuyao.cn/bkecommon/cluster/api/v1beta1"

type AddonOperate string

const (
	CreateAddon  AddonOperate = "create"
	UpdateAddon  AddonOperate = "update"
	RemoveAddon  AddonOperate = "delete"
	UpgradeAddon AddonOperate = "upgrade"
)

type AddonTransfer struct {
	Addon   *v1beta1.Product
	Operate AddonOperate
}

// CompareBKEConfigAddon compare addons in []v1beta1.Product
func CompareBKEConfigAddon(old []v1beta1.Product, dst []v1beta1.Product) ([]*AddonTransfer, bool) {
	var transferAddon []*AddonTransfer
	for _, p := range dst {
		t := p.DeepCopy()
		if pp, ok := IsExistInProductList(p, old); !ok {
			transferAddon = append(transferAddon, &AddonTransfer{Addon: t, Operate: CreateAddon})
		} else {
			if IsProductUpgrade(&p, pp) {
				transferAddon = append(transferAddon, &AddonTransfer{Addon: t, Operate: UpgradeAddon})
				continue
			}
			if !IsProductEqual(&p, pp) {
				transferAddon = append(transferAddon, &AddonTransfer{Addon: t, Operate: UpdateAddon})
			}
		}
	}
	for _, p := range old {
		t := p.DeepCopy()
		if _, ok := IsExistInProductList(p, dst); !ok {
			transferAddon = append(transferAddon, &AddonTransfer{Addon: t, Operate: RemoveAddon})
		}
	}
	return transferAddon, len(transferAddon) > 0
}

func IsExistInProductList(p v1beta1.Product, dst []v1beta1.Product) (*v1beta1.Product, bool) {
	for i, d := range dst {
		if p.Name == d.Name {
			return &dst[i], true
		}
	}
	return nil, false
}

func IsProductEqual(p, d *v1beta1.Product) bool {
	if len(p.Param) != len(d.Param) {
		return false
	}
	if p.Version != d.Version {
		return false
	}
	if p.Name != d.Name {
		return false
	}
	for k, v := range p.Param {
		if v != d.Param[k] {
			return false
		}
	}
	return true
}

func IsProductUpgrade(p, d *v1beta1.Product) bool {
	if p.Version != d.Version {
		return true
	}
	return false
}
