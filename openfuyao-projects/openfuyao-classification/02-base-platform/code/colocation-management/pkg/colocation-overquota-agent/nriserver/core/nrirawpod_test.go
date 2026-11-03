/*
* Copyright (c) 2024 China Unicom Digital Technology Co., Ltd.
* openFuyao is licensed under Mulan PSL v2.
* You can use this software according to the terms and conditions of the Mulan PSL v2.
* You may obtain a copy of Mulan PSL v2 at:
*          http://license.coscl.org.cn/MulanPSL2
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
* EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
* MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
* See the Mulan PSL v2 for more details.
* Author: YuXiang Guo
* Date: 2024-11-25
 */

package core

import (
	"reflect"
	"testing"

	"github.com/containerd/nri/pkg/api"

	"openfuyao.com/colocation-management/pkg/common"
)

func TestFixContainerCgroupParent(t *testing.T) {
	cgroupDriver = common.CgroupDriverSystemd
	currentContainerEngines = 4
	container := api.Container{
		Id: "66ff3a44a254533880e6b50e8fb52e0311c9158eb66426ae244066a4f11b26e5",
		Linux: &api.LinuxContainer{
			CgroupsPath: "kubepods-burstable-podbb29f378_0c50_4da4_b070_be919e350db2.slice:" +
				"crio:66ff3a44a254533880e6b50e8fb52e0311c9158eb66426ae244066a4f11b26e5",
		},
	}
	parent, err := FixContainerCgroupParent(&container)
	if err != nil {
		t.Log(err)
	}

	expectParent := "kubepods.slice/" +
		"kubepods-burstable.slice/kubepods-burstable-podbb29f378_0c50_4da4_b070_be919e350db2.slice/" +
		"crio-66ff3a44a254533880e6b50e8fb52e0311c9158eb66426ae244066a4f11b26e5.scope"

	if !reflect.DeepEqual(expectParent, parent) {
		t.Errorf("FixContainerCgroupParent expect %s, but got %s", expectParent, parent)
	}
}
func TestFixContainerEngine(t *testing.T) {
	FixCurrentContainerEngine("cri-containerd")
	t.Log(currentContainerEngines)
	FixCurrentContainerEngine("isulad")
	t.Log(currentContainerEngines)

	GetContainerEngine(map[string]string{
		common.ContainerEngineAnnotation: "isulad",
	})
	t.Log(currentContainerEngines)
	GetContainerEngine(map[string]string{
		common.ContainerEngineAnnotation: "docker",
	})
	t.Log(currentContainerEngines)
	GetContainerEngine(map[string]string{})
	t.Log(currentContainerEngines)

}
