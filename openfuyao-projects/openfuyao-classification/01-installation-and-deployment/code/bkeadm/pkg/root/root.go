/******************************************************************
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain n copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package root

import (
	"fmt"

	"gopkg.openfuyao.cn/bkeadm/pkg/executor/k8s"
	"gopkg.openfuyao.cn/bkeadm/pkg/global"
)

type Options struct {
	KubeConfig string   `json:"kubeConfig"`
	Args       []string `json:"arg"`
}

// ClusterPre initializes the global Kubernetes client if not already initialized
func (op *Options) ClusterPre() error {
	// k8s client
	if global.K8s == nil {
		var err error
		global.K8s, err = k8s.NewKubernetesClient(op.KubeConfig)
		if err != nil {
			return err
		}
	}
	return nil
}

func (op *Options) Print() {
	content := `
 .----------------.  .----------------.  .----------------.
| .--------------. || .--------------. || .--------------. |
| |   ______     | || |  ___  ____   | || |  _________   | |
| |  |_   _ \    | || | |_  ||_  _|  | || | |_   ___  |  | |
| |    | |_) |   | || |   | |_/ /    | || |   | |_  \_|  | |
| |    |  __'.   | || |   |  __'.    | || |   |  _|  _   | |
| |   _| |__) |  | || |  _| |  \ \_  | || |  _| |___/ |  | |
| |  |_______/   | || | |____||____| | || | |_________|  | |
| |              | || |              | || |              | |
| '--------------' || '--------------' || '--------------' |
 '----------------'  '----------------'  '----------------'

The default working directory is /bke.
Other working directories can be specified using the environment variable BKE_WORKSPACE.
`
	fmt.Print(content)
}
