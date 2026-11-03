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
	"encoding/json"

	"github.com/spf13/viper"
	"k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"

	"openfuyao/oauth-server/cmd/oauth-server/app/config"
	k8sconfig "openfuyao/oauth-server/pkg/config"
	"openfuyao/oauth-server/pkg/fuyaoerrors"
	"openfuyao/oauth-server/pkg/zlog"
)

const (
	secretNamespace = "openfuyao-system"
	jwtSecretName   = "oauth-jwt-cookie-secret"
	oauthSecretName = "oauth-client-secrets"
)

// OAuthServerOption stores the overall configfile and its loading method for the whole oauthserver service
type OAuthServerOption struct {
	ConfigFile string `json:"ConfigFile"`
}

// NewOAuthServerOption inits the option
func NewOAuthServerOption() *OAuthServerOption {
	return &OAuthServerOption{}
}

// Validate validates the options is legal to use
func (o *OAuthServerOption) Validate() error {
	if len(o.ConfigFile) == 0 {
		return fuyaoerrors.ErrOAuthServerConfigFileMissing
	}

	return nil
}

// ReadConfig loads the configfile from disk
func (o *OAuthServerOption) ReadConfig() (*config.OAuthServerAPIServerConfig, error) {
	v := viper.New()
	v.SetConfigFile(o.ConfigFile)
	if err := v.ReadInConfig(); err != nil {
		return nil, err
	}

	var oAuthServerConfig config.OAuthServerAPIServerConfig
	if err := v.Unmarshal(&oAuthServerConfig); err != nil {
		return nil, err
	}

	// oAuthServerConfig.K8sConfig is allowed to be nil since we will read from incluster config / default path
	k8sClient := k8sconfig.GetKubernetesClient(oAuthServerConfig.K8sConfig)

	// manually add secret keys
	jwtPrivateKeyDecoded, err := readDataFromK8sSecret(k8sClient, jwtSecretName, "oauth-jwt.key")
	if err != nil {
		return nil, err
	}
	oAuthServerConfig.OAuthServerConfig.JWTPrivateKey = jwtPrivateKeyDecoded

	signKeyDecoded, err := readDataFromK8sSecret(k8sClient, jwtSecretName, "oauth-cookie-sign.key")
	if err != nil {
		return nil, err
	}
	oAuthServerConfig.IDPLoginStoreConfig.SigningKey = signKeyDecoded

	encryptKeyDecoded, err := readDataFromK8sSecret(k8sClient, jwtSecretName, "oauth-cookie-encrypt.key")
	if err != nil {
		return nil, err
	}
	oAuthServerConfig.IDPLoginStoreConfig.EncryptionKey = encryptKeyDecoded

	oAuthIDSecretsDecoded, err := readDataFromK8sSecret(k8sClient, oauthSecretName, "client-mapper")
	if err != nil {
		return nil, err
	}
	var clientMapper map[string]string
	err = json.Unmarshal(oAuthIDSecretsDecoded, &clientMapper)
	if err != nil {
		zlog.LogErrorf("cannot load data from client-id-secrets-mapper, err: %v", err)
		return nil, err
	}

	oAuthServerConfig.OAuthServerConfig.ClientMapper = clientMapper
	return &oAuthServerConfig, nil

}

func readDataFromK8sSecret(k8sClient kubernetes.Interface, name, key string) ([]byte, error) {
	secret, err := k8sClient.CoreV1().Secrets(secretNamespace).Get(context.TODO(), name, v1.GetOptions{})
	if err != nil {
		return nil, fuyaoerrors.ErrFailToGetSecret
	}

	rawData := secret.Data[key]
	if rawData == nil {
		zlog.LogErrorf("cannot load data from %s", name)
		return nil, fuyaoerrors.ErrFailToGetSecret
	}

	return rawData, nil
}
