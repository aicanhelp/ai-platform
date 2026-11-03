/******************************************************************
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package upgradepath

import (
	"strings"

	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/common"
)

const (
	bkeConfigKeyDomain        = "domain"
	bkeConfigKeyHost          = "host"
	bkeConfigKeyImageRepoPort = "imageRepoPort"
	bkeConfigKeyOtherRepo     = "otherRepo"
	bkeConfigKeyOtherRepoIP   = "otherRepoIp"
	bkeConfigKeyOnlineImage   = "onlineImage"

	bkeConfigHostPortFields = 2
)

// RepoFromBKEConfigData derives cluster imageRepo from cluster-system/bke-config .Data,
// matching bkeadm prepareImageRepoConfig() without requiring new ConfigMap keys.
func RepoFromBKEConfigData(data map[string]string) (confv1beta1.Repo, bool) {
	if data == nil {
		return confv1beta1.Repo{}, false
	}

	otherRepo := strings.TrimSpace(data[bkeConfigKeyOtherRepo])
	onlineImage := strings.TrimSpace(data[bkeConfigKeyOnlineImage])
	domain := strings.TrimSpace(data[bkeConfigKeyDomain])
	host := strings.TrimSpace(data[bkeConfigKeyHost])
	portCLI := strings.TrimSpace(data[bkeConfigKeyImageRepoPort])
	repoIP := strings.TrimSpace(data[bkeConfigKeyOtherRepoIP])

	if otherRepo != "" {
		return repoFromOtherRepo(otherRepo, repoIP), true
	}
	if onlineImage != "" {
		return confv1beta1.Repo{
			Domain: "default",
			Ip:     host,
			Port:   portCLI,
			Prefix: "",
		}, true
	}
	if domain == "" && host == "" {
		return confv1beta1.Repo{}, false
	}
	return confv1beta1.Repo{
		Domain: domain,
		Ip:     host,
		Port:   portCLI,
		Prefix: common.ImageRegistryKubernetes,
	}, true
}

func repoFromOtherRepo(otherRepo, repoIP string) confv1beta1.Repo {
	trimmed := strings.TrimSuffix(strings.TrimSpace(otherRepo), "/")
	parts := strings.SplitN(trimmed, "/", 2)
	hostPort := parts[0]
	prefix := ""
	if len(parts) > 1 {
		prefix = strings.Trim(parts[1], "/")
	}

	hp := strings.Split(hostPort, ":")
	port := "443"
	domain := hostPort
	if len(hp) == bkeConfigHostPortFields {
		domain = hp[0]
		port = hp[1]
	}

	return confv1beta1.Repo{
		Domain: domain,
		Ip:     repoIP,
		Port:   port,
		Prefix: prefix,
	}
}
