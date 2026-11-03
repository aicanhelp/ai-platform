/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *           http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package download

import (
	"io"
	"net/http"
	"os"
	"path"
	"path/filepath"
	goruntime "runtime"
	"strconv"
	"strings"

	"github.com/pkg/errors"

	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/log"
)

const (
	parseUintBase = 8
	bitSize       = 32
)

// ExecDownload downloads a file from url into saveto, substituting {.arch} with the local GOARCH.
func ExecDownload(url, saveto, rename, chmod string) error {
	return ExecDownloadForArch(url, saveto, rename, chmod, goruntime.GOARCH)
}

// ExecDownloadForArch downloads a file from url into saveto, substituting {.arch} with arch.
func ExecDownloadForArch(url, saveto, rename, chmod, arch string) error {
	if arch == "" {
		arch = goruntime.GOARCH
	}
	if !utils.Exists(saveto) {
		if err := os.MkdirAll(saveto, utils.RwxRxRx); err != nil {
			return errors.Wrapf(err, "create directory %q failed", saveto)
		}
	}

	// string to FileMode
	perm, err := strconv.ParseUint(chmod, parseUintBase, bitSize)
	if err != nil {
		log.Warnf("parse perm %q failed, use default 0644", chmod)
		perm = 0644
	}

	url = strings.ReplaceAll(url, "{.arch}", arch)

	resp, err := http.Get(url)
	if err != nil {
		return errors.Wrapf(err, "download %s failed", url)
	}
	if resp.StatusCode != http.StatusOK {
		return errors.Errorf("download %s failed, status code %d", url, resp.StatusCode)
	}
	defer resp.Body.Close()
	var savePath string
	if rename != "" {
		savePath = path.Join(saveto, rename)
	} else {
		savePath = path.Join(saveto, path.Base(url))
	}

	newFile, err := os.OpenFile(savePath, os.O_CREATE|os.O_WRONLY, os.FileMode(perm))
	if err != nil {
		return err
	}
	defer newFile.Close()
	size, err := io.Copy(newFile, resp.Body)
	if err != nil {
		return err
	}
	size = size / 1024 / 1024

	log.Infof("download %q to %q, size %dM, chmod %s %q", url, savePath, size, chmod,
		os.FileMode(perm).String())
	defer func() {
		if err := recover(); err != nil {
			log.Errorf("download file %q failed, err: %v", url, err)
			err = os.Remove(savePath)
			if err != nil {
				log.Errorf("remove file %q failed, err: %v", savePath, err)
			}
		}
	}()
	return nil
}

// DownloadBytes downloads url and returns the response body.
func DownloadBytes(url string) ([]byte, error) {
	resp, err := http.Get(url)
	if err != nil {
		return nil, errors.Wrapf(err, "download %s failed", url)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, errors.Errorf("download %s failed, status code %d", url, resp.StatusCode)
	}
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	log.Infof("download %q, size %d bytes", url, len(data))
	return data, nil
}

// DownloadToFile downloads url and writes the response body to destPath.
func DownloadToFile(url, destPath, chmod string) error {
	perm, err := strconv.ParseUint(chmod, parseUintBase, bitSize)
	if err != nil {
		log.Warnf("parse perm %q failed, use default 0644", chmod)
		perm = 0644
	}

	resp, err := http.Get(url)
	if err != nil {
		return errors.Wrapf(err, "download %s failed", url)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return errors.Errorf("download %s failed, status code %d", url, resp.StatusCode)
	}

	if err := os.MkdirAll(filepath.Dir(destPath), utils.RwxRxRx); err != nil {
		return errors.Wrapf(err, "create directory for %q failed", destPath)
	}

	newFile, err := os.OpenFile(destPath, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, os.FileMode(perm))
	if err != nil {
		return err
	}
	defer newFile.Close()

	if _, err := io.Copy(newFile, resp.Body); err != nil {
		return err
	}
	log.Infof("download %q to %q", url, destPath)
	return nil
}
