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

package helm

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"helm.sh/helm/v3/pkg/chart"
	"helm.sh/helm/v3/pkg/chart/loader"
	"helm.sh/helm/v3/pkg/registry"
	"helm.sh/helm/v3/pkg/repo"
	"sigs.k8s.io/yaml"

	"marketplace-service/pkg/constant"
	"marketplace-service/pkg/utils/httputil"
	"marketplace-service/pkg/zlog"
)

var (
	timeLimit = 120
)

// LoadChart load helm chart tar file from repo url
func LoadChart(chartUrl string, repoEntry *repo.Entry) (*chart.Chart, error) {
	chartBytes, err := LoadChartBytes(chartUrl, repoEntry)
	if err != nil {
		fmt.Println("Error loading chartStruct archive:", err)
		return nil, err
	}
	chartStruct, err := loader.LoadArchive(chartBytes)
	if err != nil {
		fmt.Println("Error loading chartStruct archive:", err)
		return nil, err
	}
	return chartStruct, nil
}

// LoadChartBytes load helm chart tar file from repo url
func LoadChartBytes(chartUrl string, repoEntry *repo.Entry) (*bytes.Buffer, error) {
	if registry.IsOCI(chartUrl) {
		return LoadOCI(chartUrl, repoEntry)
	}
	if !(strings.HasPrefix(chartUrl, "https://") || strings.HasPrefix(chartUrl, "http://")) {
		u := repoEntry.URL
		if !strings.HasSuffix(u, "/") {
			chartUrl = fmt.Sprintf("%s/%s", u, chartUrl)
		} else {
			chartUrl = fmt.Sprintf("%s%s", u, chartUrl)
		}
	}
	resp, err := LoadData(chartUrl, repoEntry)
	if err != nil {
		return nil, err
	}
	return resp, nil
}

// LoadRepoIndex load repository index.yaml from url
func LoadRepoIndex(repoEntry *repo.Entry) (*repo.IndexFile, error) {
	u := repoEntry.URL
	if !strings.HasSuffix(u, "/") {
		u = fmt.Sprintf("%s/%s", u, "index.yaml")
	} else {
		u = fmt.Sprintf("%s%s", u, "index.yaml")
	}

	resp, err := LoadData(u, repoEntry)
	if err != nil {
		return nil, err
	}

	indexFile, err := loadIndex(resp.Bytes())
	if err != nil {
		return nil, err
	}

	return indexFile, nil
}

func loadIndex(data []byte) (*repo.IndexFile, error) {
	i := &repo.IndexFile{}
	if err := yaml.Unmarshal(data, i); err != nil {
		return i, err
	}
	i.SortEntries()
	if i.APIVersion == "" {
		return i, repo.ErrNoAPIVersion
	}
	return i, nil
}

type limitedBuffer struct {
	buffer  *bytes.Buffer
	maxSize int64
	written int64
}

func (l *limitedBuffer) Write(p []byte) (int, error) {
	if l.written+int64(len(p)) > l.maxSize {
		return 0, errors.New("response body exceeds the size limit")
	}
	n, err := l.buffer.Write(p)
	l.written += int64(n)
	return n, err
}

func newLimitedBuffer(maxSize int64) *limitedBuffer {
	return &limitedBuffer{
		buffer:  new(bytes.Buffer),
		maxSize: maxSize,
	}
}

// LoadData load index.yaml from target url
func LoadData(u string, repoEntry *repo.Entry) (*bytes.Buffer, error) {
	req, err := http.NewRequest("GET", u, nil)
	if err != nil {
		zlog.Errorf("error creating index.yaml request: %v", err)
		return nil, err
	}
	if repoEntry.Username != "" && repoEntry.Password != "" {
		req.SetBasicAuth(repoEntry.Username, repoEntry.Password)
	}

	if err != nil {
		zlog.Errorf("error get http config %v", err)
	}
	tlsClient, err := httputil.GetHarborHTTPConfig(repoEntry)
	if err != nil {
		zlog.Errorf("error get http config %v", err)
	}
	client := &http.Client{Transport: &http.Transport{
		TLSHandshakeTimeout: time.Duration(timeLimit) * time.Second,
		TLSClientConfig:     tlsClient,
	}}
	resp, err := client.Do(req)
	if err != nil {
		zlog.Errorf("error making index.yaml request: %v", err)
		return nil, err
	}
	defer resp.Body.Close()

	maxSize := int64(constant.ResponseBodyLimit << constant.ToMegabytes)
	limitedBuf := newLimitedBuffer(maxSize)

	_, err = io.Copy(limitedBuf, resp.Body)
	if err != nil {
		zlog.Errorf("failed to read response body: %v", err)
		return nil, errors.New(fmt.Sprintf(""))
	}

	if resp.StatusCode != http.StatusOK {
		return nil, errors.New(fmt.Sprintf("Error: received non-200 status code: %d", resp.StatusCode))
	}
	return limitedBuf.buffer, nil
}

func LoadOCI(ref string, repoEntry *repo.Entry) (*bytes.Buffer, error) {
	var opts []registry.ClientOption
	if !repoEntry.InsecureSkipTLSverify {
		tlsClient, err := httputil.GetHarborHTTPConfig(repoEntry)
		if err != nil {
			zlog.Errorf("error get http config %v", err)
		}
		opts = append(opts, registry.ClientOptHTTPClient(&http.Client{Transport: &http.Transport{
			TLSHandshakeTimeout: time.Duration(timeLimit) * time.Second,
			TLSClientConfig:     tlsClient,
		}}))
	}
	if repoEntry.Username != "" && repoEntry.Password != "" {
		opts = append(opts, registry.ClientOptBasicAuth(repoEntry.Username, repoEntry.Password))
	}
	client, err := registry.NewClient(opts...)
	if err != nil {
		return nil, fmt.Errorf("error creating OCI client: %w", err)
	}
	result, err := client.Pull(ref, registry.PullOptWithChart(true))
	if err != nil {
		return nil, fmt.Errorf("error pulling OCI chart: %w", err)
	}
	return bytes.NewBuffer(result.Chart.Data), nil
}
