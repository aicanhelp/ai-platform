/******************************************************************
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package kubeadm

import (
	"testing"

	"github.com/stretchr/testify/assert"
	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/bkeagent/mfutil"
)

func TestApplyCommandEtcdVersion(t *testing.T) {
	k := &KubeadmPlugin{
		boot: &mfutil.BootScope{
			BkeConfig: &confv1beta1.BKEConfig{
				Cluster: confv1beta1.Cluster{EtcdVersion: "v3.5.12-of.1"},
			},
			Extra: map[string]interface{}{},
		},
	}
	k.applyCommandEtcdVersion(map[string]string{"etcdVersion": "v3.5.21-of.1"})
	assert.Equal(t, "v3.5.21-of.1", k.boot.BkeConfig.Cluster.EtcdVersion)
	assert.Equal(t, "v3.5.21-of.1", k.boot.Extra["etcdVersion"])
}

func TestApplyCommandEtcdVersionEmpty(t *testing.T) {
	k := &KubeadmPlugin{
		boot: &mfutil.BootScope{
			BkeConfig: &confv1beta1.BKEConfig{
				Cluster: confv1beta1.Cluster{EtcdVersion: "v3.5.12-of.1"},
			},
		},
	}
	k.applyCommandEtcdVersion(map[string]string{})
	assert.Equal(t, "v3.5.12-of.1", k.boot.BkeConfig.Cluster.EtcdVersion)
}
