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

package common

import (
	econd "gopkg.openfuyao.cn/bkeadm/pkg/executor/containerd"
	"gopkg.openfuyao.cn/bkeadm/pkg/global"
	"gopkg.openfuyao.cn/bkeadm/pkg/infrastructure"
	"gopkg.openfuyao.cn/bkeadm/utils"
)

// LoadLocalRepositoryFromFile loads images from the specified file into the local container runtime
func LoadLocalRepositoryFromFile(imageFilePath string) error {
	if utils.Exists(imageFilePath) {
		if infrastructure.IsDocker() {
			_, err := global.Docker.Load(imageFilePath)
			if err != nil {
				return err
			}
		}
		if infrastructure.IsContainerd() {
			err := econd.Load(imageFilePath)
			if err != nil {
				return err
			}
		}
	}
	return nil
}
