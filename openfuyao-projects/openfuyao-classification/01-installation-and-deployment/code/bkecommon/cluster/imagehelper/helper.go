/******************************************************************
 * Copyright (c) 2024 Bocloud Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain n copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package imagehelper

import (
	"fmt"
	"strings"
)

func imageMapJoinRepo(repo string, images map[string]string) []string {
	var res []string
	for _, v := range images {
		res = append(res, RepoJoinImageName(repo, v))
	}
	return res
}

// GetFullImageName returns the full image name with repo, name and tag.
func GetFullImageName(repo, name, tag string) string {
	return fmt.Sprintf("%s/%s:%s", CleanImageRepo(repo), CleanImageName(name), CleanImageTag(tag))
}

// GetImageNameWithTag returns the image name with tag.
func GetImageNameWithTag(name, tag string) string {
	return fmt.Sprintf("%s:%s", CleanImageName(name), CleanImageTag(tag))
}

// GetImageNameWithTagAndOf returns the image name with tag and of.
func GetImageNameWithTagAndOf(name, tag string) string {
	return fmt.Sprintf("%s:of-%s", CleanImageName(name), CleanImageTag(tag))
}

func RepoJoinImageName(repo string, imageName string) string {
	return fmt.Sprintf("%s/%s", CleanImageRepo(repo), CleanImageName(imageName))
}

// CleanImageName removes the leading and trailing slashes and colons from the image name.
func CleanImageName(name string) string {
	name = strings.TrimPrefix(name, "/")
	name = strings.TrimSuffix(name, ":")
	return name
}

// CleanImageTag removes the leading colon from the image tag.
func CleanImageTag(tag string) string {
	tag = strings.TrimPrefix(tag, ":")
	return tag
}

func CleanImageRepo(repo string) string {
	repo = strings.TrimSuffix(repo, "/")
	return repo
}
