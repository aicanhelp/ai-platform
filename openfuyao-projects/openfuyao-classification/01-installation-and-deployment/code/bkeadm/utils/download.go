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

package utils

import (
	"bufio"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path"
	"path/filepath"
	"regexp"
	"strings"

	"gopkg.openfuyao.cn/bkeadm/utils/log"
)

func DownloadAllFiles(url, targetDirectory string) error {
	htmlData, err := httpGet(url)
	if err != nil {
		return err
	}
	if len(htmlData) == 0 {
		return errors.New(fmt.Sprintf("url: %s, Failed to get download list", url))
	}
	re := regexp.MustCompile(`<a href="(.*?)">(.*?)</a>`)
	result := re.FindAllStringSubmatch(htmlData, -1)

	for _, res := range result {
		if len(res) < HttpUrlFields {
			continue
		}
		if !strings.HasSuffix(res[1], ".rpm") {
			continue
		}
		fmt.Println(res[1])
		err = DownloadFile(url+res[1], path.Join(targetDirectory, res[1]))
		if err != nil {
			return err
		}
	}
	return nil
}

func DownloadSignalFile(url, targetPath string) error {
	req, err := http.NewRequest(http.MethodGet, url, nil)
	if err != nil {
		return fmt.Errorf("failed to create HTTP request for URL %s: %w", url, err)
	}

	req.Header.Set("User-Agent", "Mozilla/5.0 (compatible; Go-Client/1.0)")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return fmt.Errorf("failed to fetch remote content from %s: %w", url, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		errorBody, err := io.ReadAll(resp.Body)
		return fmt.Errorf("HTTP request failed: status %d (%s) for URL %s, response: %s, err: %v",
			resp.StatusCode, http.StatusText(resp.StatusCode), url, string(errorBody), err)
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("failed to read response body from %s: %w", url, err) // 修复：添加 return
	}

	if err = os.MkdirAll(filepath.Dir(targetPath), DefaultDirPermission); err != nil {
		return fmt.Errorf("failed to create directory: %w", err)
	}

	if err = os.WriteFile(targetPath, body, DefaultFilePermission); err != nil {
		return fmt.Errorf("failed to write file: %w", err)
	}

	return nil
}

func httpGet(url string) (string, error) {
	resp, err := http.Get(url)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode != HTTPStatusOK {
		return "", errors.New(fmt.Sprintf(" get url %s, status code %d", url, resp.StatusCode))
	}
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", err
	}
	return string(body), nil
}

func DownloadFile(url, destinationFile string) error {
	resp, err := http.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != HTTPStatusOK {
		return errors.New(fmt.Sprintf("File cannot be found %d", resp.StatusCode))
	}
	reader := bufio.NewReaderSize(resp.Body, 32*1024)
	file, err := os.Create(destinationFile)
	if err != nil {
		return err
	}
	defer file.Close()
	if err := file.Chmod(DefaultFilePermission); err != nil {
		log.Warnf("failed to set file permission for %s: %v", destinationFile, err)
	}
	writer := bufio.NewWriter(file)
	_, err = io.Copy(writer, reader)
	if err != nil {
		return err
	}
	if err := writer.Flush(); err != nil {
		return fmt.Errorf("failed to flush writer for %s: %w", destinationFile, err)
	}
	return nil
}
