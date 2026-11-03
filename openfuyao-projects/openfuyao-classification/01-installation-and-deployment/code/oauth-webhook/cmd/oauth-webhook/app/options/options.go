/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

// Package options store the configfile and will possibly extend other datastructures (genericapiserver) in the future
package options

import (
	"context"
	"errors"

	"github.com/spf13/viper"
	"k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"

	"openfuyao/oauth-webhook/cmd/oauth-webhook/app/config"
	"openfuyao/oauth-webhook/pkg/fuyaoerrors"
	"openfuyao/oauth-webhook/pkg/k8sconfig"
	"openfuyao/oauth-webhook/pkg/zlog"
)

const (
	secretNamespace = "openfuyao-system"
	secretName      = "oauth-jwt-secret"
)

// OAuthWebhookOption stores the overall configfile and its loading method for the whole oauth-webhook service
type OAuthWebhookOption struct {
	ConfigFile string `json:"ConfigFile"`
}

// NewOAuthWebhookOption inits the option
func NewOAuthWebhookOption() *OAuthWebhookOption {
	return &OAuthWebhookOption{}
}

// Validate validates the options is legal to use
func (o *OAuthWebhookOption) Validate() error {
	if len(o.ConfigFile) == 0 {
		return fuyaoerrors.ErrOAuthWebhookConfigFileMissing
	}

	return nil
}

// ReadConfig loads the configfile from disk
func (o *OAuthWebhookOption) ReadConfig() (*config.OAuthWebhookConfig, error) {
	v := viper.New()
	v.SetConfigFile(o.ConfigFile)
	if err := v.ReadInConfig(); err != nil {
		return nil, err
	}

	var oAuthWebhookConfig config.OAuthWebhookConfig
	if err := v.Unmarshal(&oAuthWebhookConfig); err != nil {
		return nil, err
	}

	// K8sConfig is allowed to be nil since we will read from incluster config / default path
	k8sClient := k8sconfig.GetKubernetesClient(nil)

	// manually add secret keys
	jwtPrivateKeyDecoded, err := readDataFromK8sSecret(k8sClient, "oauth-jwt.key")
	if err != nil {
		return nil, err
	}
	oAuthWebhookConfig.JWTPrivateKey = jwtPrivateKeyDecoded

	return &oAuthWebhookConfig, nil
}

func readDataFromK8sSecret(k8sClient kubernetes.Interface, key string) ([]byte, error) {
	secret, err := k8sClient.CoreV1().Secrets(secretNamespace).Get(context.TODO(), secretName, v1.GetOptions{})
	if err != nil {
		return nil, errors.New("fail to load data from jwt secret")
	}

	rawData := secret.Data[key]
	if rawData == nil {
		zlog.LogErrorf("cannot load data from jwt secret")
		return nil, errors.New("fail to load data from jwt secret")
	}

	return rawData, nil
}
