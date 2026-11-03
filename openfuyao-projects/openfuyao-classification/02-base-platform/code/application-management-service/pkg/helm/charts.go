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

// Package helm
package helm

import (
	"crypto/tls"
	goErrors "errors"
	"fmt"
	"io"
	"net/http"
	"time"

	"helm.sh/helm/v3/pkg/chart"

	"application-management-service/pkg/constant"
	"application-management-service/pkg/models/helm"
	"application-management-service/pkg/utils/httputil"
	"application-management-service/pkg/zlog"
)

const (
	token           = "token"
	authorization   = "Authorization"
	wwwAuthenticate = "Www-Authenticate"
	realmParam      = "realm"
	serviceParam    = "service"
	scopeParam      = "scope"
)

// DeployRequest is the parameter needed during the installation
type DeployRequest struct {
	Chart       *chart.Chart      `json:"chart"`
	DeployParam *helm.DeployParam `json:"deployParam"`
}

func getMarketplaceURL(host, repoName, chartName, chartVersion string) string {
	return fmt.Sprintf("%s/helm-repos/%s/charts/%s/versions/%s/files?fileType=raw", host, repoName,
		chartName, chartVersion)
}

func getHarborHttpClient() *http.Client {
	return &http.Client{
		Timeout: time.Second * constant.DefaultHttpRequestSeconds,
		Transport: &http.Transport{
			TLSClientConfig: &tls.Config{
				MinVersion:         tls.VersionTLS12,
				InsecureSkipVerify: true,
			},
		},
	}
}

func getHttpRequestResponse(req *http.Request, skipServerCertVerify bool) (*http.Response, []byte, error) {
	var client *http.Client
	var err error
	if skipServerCertVerify {
		client = getHarborHttpClient()
	} else {
		client, err = httputil.GetDefaultHttpClient()
		if err != nil {
			zlog.Errorf("error create http client %v", err)
			return nil, nil, err
		}
	}
	if client == nil {
		return nil, nil, goErrors.New("error create http client")
	}

	resp, err := client.Do(req)
	if err != nil {
		zlog.Errorf("error sending HTTP request: %v", err)
		return nil, nil, err
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		zlog.Errorf("error reading response body: %v", err)
		return nil, nil, err
	}
	return resp, body, nil
}
