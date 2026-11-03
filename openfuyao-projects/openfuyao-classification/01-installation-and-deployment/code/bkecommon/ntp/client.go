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

package ntp

import (
	"fmt"
	"os/exec"

	"github.com/beevik/ntp"
)

// Date 同步本地时间到指定 NTP 服务器的时间
func Date(ntpServer string) error {
	t, err := ntp.Time(ntpServer)
	if err != nil {
		return fmt.Errorf("failed to query ntp server %s: %w", ntpServer, err)
	}

	ts := t.Format("2006-01-02 15:04:05")

	cmd := exec.Command("date", "-s", ts)
	out, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("failed to set system time: %s, err: %w", string(out), err)
	}

	fmt.Println(string(out))
	return nil
}
