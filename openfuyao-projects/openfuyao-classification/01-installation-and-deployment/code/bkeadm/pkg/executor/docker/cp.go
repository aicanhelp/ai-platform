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

package docker

import (
	"github.com/docker/docker/pkg/archive"
)

func (c *Client) CopyFromContainer(containerId, srcPath, dstPath string) error {
	content, stat, err := c.Client.CopyFromContainer(c.ctx, containerId, srcPath)
	if err != nil {
		return err
	}
	defer content.Close()

	info := archive.CopyInfo{
		RebaseName: "",
		IsDir:      stat.Mode.IsDir(),
		Exists:     true,
		Path:       srcPath,
	}
	return archive.CopyTo(content, info, dstPath)
}
