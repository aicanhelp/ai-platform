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

package bkeagent

import (
	_ "embed"
	"fmt"
	"os"

	"gopkg.openfuyao.cn/bkeadm/pkg/executor/k8s"
	"gopkg.openfuyao.cn/bkeadm/pkg/global"
	"gopkg.openfuyao.cn/bkeadm/utils"
	"gopkg.openfuyao.cn/bkeadm/utils/log"
)

var (
	//go:embed bkeagent.yaml
	bkeAgent []byte
)

// InstallBKEAgentCRD installs the BKE Agent Custom Resource Definition
func InstallBKEAgentCRD() error {
	log.Info("Installing BKE Agent CRD...")
	if global.K8s == nil {
		var err error
		global.K8s, err = k8s.NewKubernetesClient("")
		if err != nil {
			log.Errorf("Failed to initialize k8s client for BKE Agent: %v", err)
			return err
		}
	}
	bkeAgentFile := fmt.Sprintf("%s/tmpl/bkeagent.yaml", global.Workspace)
	err := os.WriteFile(bkeAgentFile, bkeAgent, utils.DefaultFilePermission)
	if err != nil {
		log.Errorf("Failed to write BKE Agent YAML file: %v", err)
		return err
	}

	err = global.K8s.InstallYaml(bkeAgentFile, map[string]string{}, "")
	if err != nil {
		log.Errorf("Failed to install BKE Agent CRD: %v", err)
		return err
	}
	log.Info("BKE Agent CRD installed successfully")
	return nil
}
