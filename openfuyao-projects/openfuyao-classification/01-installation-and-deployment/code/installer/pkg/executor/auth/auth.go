/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *           http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package auth

import (
	"fmt"
	"installer/pkg/zlog"
	"path/filepath"

	"installer/pkg/common"
	"installer/pkg/constants"
	"installer/pkg/fuyao/v1beta1/cluster"
	"installer/pkg/utils"
)

type function func(nodes []*cluster.Node) error

// OauthWebhook is oauth webhook cfg
type OauthWebhook struct {
	webhookLocalDir  string
	webhookRemoteDir string
}

// NewOauthWebHook is new OauthWebhook
func NewOauthWebHook() *OauthWebhook {
	return &OauthWebhook{
		webhookRemoteDir: filepath.Join(constants.EtcPath, constants.Kubernetes, constants.WebHookPath),
		webhookLocalDir:  filepath.Join(constants.OptPath, constants.OpenFuyao, constants.Tmp, constants.WebHookPath),
	}
}

func (o *OauthWebhook) downloadFile(node *cluster.Node) error {
	if err := utils.MkdirAll(o.webhookLocalDir, constants.RwxRxRx); err != nil {
		return err
	}

	filePathMap := map[string]string{
		filepath.Join(o.webhookRemoteDir, caPem):          filepath.Join(o.webhookLocalDir, caPem),
		filepath.Join(o.webhookRemoteDir, serverCrt):      filepath.Join(o.webhookLocalDir, serverCrt),
		filepath.Join(o.webhookRemoteDir, serverKey):      filepath.Join(o.webhookLocalDir, serverKey),
		filepath.Join(o.webhookRemoteDir, webhookCfgFile): filepath.Join(o.webhookLocalDir, webhookCfgFile),
	}

	client := node.Client()
	for remote, local := range filePathMap {
		_, err := client.SudoExecCmd(fmt.Sprintf("chmod %o %s", constants.RwxRxRx, remote))
		if err != nil {
			return err
		}

		err = client.DownloadFile(local, remote)
		if err != nil {
			return err
		}
		_, err = client.SudoExecCmd(fmt.Sprintf("chmod %o %s", constants.Rw, remote))
		if err != nil {
			return err
		}
	}
	return nil
}

func (o *OauthWebhook) copyFileToRemote(nodes []*cluster.Node) error {
	for _, node := range nodes {
		client := node.Client()
		if err := client.CopyFilesToRemote(o.webhookLocalDir, o.webhookRemoteDir); err != nil {
			return err
		}
		_, err := client.SudoExecCmd(fmt.Sprintf("chmod %o %s && chmod %o %s",
			constants.Rw, filepath.Join(o.webhookRemoteDir, constants.Asterisk),
			constants.RwxRxRx, o.webhookRemoteDir))
		if err != nil {
			return err
		}
	}
	return nil
}

func (o *OauthWebhook) setK8sManifests(nodes []*cluster.Node) error {
	for _, node := range nodes {
		pkgPath, err := common.GetOptPkgPathByNode(node)
		if err != nil {
			return err
		}

		client := node.Client()
		utilsPath := filepath.Join(pkgPath, constants.Utils)
		zlog.Debugf("restart kubernetes api-server on %s", node.IP)
		_, err = client.ExecCmd(fmt.Sprintf("cd %s && bash %s", utilsPath, constants.SetK8sManifests))
		if err != nil {
			return err
		}
	}
	return nil
}

func (o *OauthWebhook) modifyFilePrem(node *cluster.Node) error {
	client := node.Client()
	_, err := client.SudoExecCmd(fmt.Sprintf("chmod %o %s && chmod %o %s",
		constants.Rw, filepath.Join(o.webhookRemoteDir, constants.Asterisk),
		constants.RwxRxRx, o.webhookRemoteDir))
	if err != nil {
		zlog.Errorf("modifyFilePrem failed, err: %s", err)
	}
	return err
}

// OauthOperate is oauth webhook operate
func OauthOperate(nodes []*cluster.Node) error {
	oauth := NewOauthWebHook()
	if err := oauth.downloadFile(nodes[0]); err != nil {
		return err
	}

	operateSlice := []function{
		oauth.copyFileToRemote,
		oauth.setK8sManifests,
	}

	for _, def := range operateSlice {
		if err := def(nodes[1:]); err != nil {
			return err
		}
	}
	return oauth.modifyFilePrem(nodes[0])
}
