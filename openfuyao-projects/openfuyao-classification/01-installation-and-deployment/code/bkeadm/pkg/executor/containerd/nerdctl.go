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

package containerd

import (
	"encoding/json"
	"errors"
	"fmt"

	"gopkg.openfuyao.cn/bkeadm/pkg/executor/exec"
	"gopkg.openfuyao.cn/bkeadm/utils"
	"gopkg.openfuyao.cn/bkeadm/utils/log"
)

type NerdContainerInfo struct {
	Id    string `json:"Id"`
	State struct {
		Status     string `json:"Status"`
		Running    bool   `json:"Running"`
		Paused     bool   `json:"Paused"`
		Restarting bool   `json:"Restarting"`
		Pid        uint   `json:"Pid"`
		ExitCode   uint   `json:"ExitCode"`
		FinishedAt string `json:"FinishedAt"`
	} `json:"State"`
	Image           string `json:"Image"`
	Name            string `json:"Name"`
	RestartCount    uint   `json:"RestartCount"`
	Platform        string `json:"Platform"`
	NetworkSettings struct {
		IPAddress  string `json:"IPAddress"`
		MacAddress string `json:"MacAddress"`
	} `json:"NetworkSettings"`
}

type NerdImageInfo struct {
	Id           string   `json:"Id"`
	RepoTags     []string `json:"RepoTags"`
	Architecture string   `json:"Architecture"`
	OS           string   `json:"Os"`
}

var cmd = exec.CommandExecutor{}

func EnsureImageExists(image string) error {
	info, err := ImageInspect(image)
	if err != nil || len(info.Id) == 0 {
		res, pullErr := cmd.ExecuteCommandWithOutput(utils.NerdCtl, "pull", image)
		if pullErr != nil {
			return fmt.Errorf("nerdctl pull %s: %w: %s", image, pullErr, res)
		}
	}
	return nil
}

func ImageInspect(image string) (NerdImageInfo, error) {
	var info []NerdImageInfo
	result, err := cmd.ExecuteCommandWithOutput(utils.NerdCtl, "inspect", image)
	if err != nil {
		return NerdImageInfo{}, err
	}

	log.Debug(result)
	err = json.Unmarshal([]byte(result), &info)
	if err != nil {
		return NerdImageInfo{}, err
	}
	if len(info) == 1 {
		return info[0], nil
	}
	return NerdImageInfo{}, errors.New("not found")
}

func EnsureContainerRun(containerId string) (bool, error) {
	info, exist := ContainerExists(containerId)
	if exist {
		if info.State.Running {
			return true, nil
		}
		if err := ContainerRemove(containerId); err != nil {
			return false, err
		}
	}
	return false, nil
}

func ContainerExists(containerId string) (NerdContainerInfo, bool) {
	var info []NerdContainerInfo
	result, err := cmd.ExecuteCommandWithOutput(utils.NerdCtl, "inspect", containerId)
	if err != nil {
		return NerdContainerInfo{}, false
	}
	log.Debug(result)
	err = json.Unmarshal([]byte(result), &info)
	if err != nil {
		return NerdContainerInfo{}, false
	}
	if len(info) == 1 {
		return info[0], true
	}
	return NerdContainerInfo{}, false
}

func ContainerInspect(containerId string) (NerdContainerInfo, error) {
	var info []NerdContainerInfo
	result, err := cmd.ExecuteCommandWithOutput(utils.NerdCtl, "inspect", containerId)
	if err != nil {
		return NerdContainerInfo{}, err
	}
	log.Debug(result)
	err = json.Unmarshal([]byte(result), &info)
	if err != nil {
		return NerdContainerInfo{}, err
	}
	if len(info) == 1 {
		return info[0], nil
	}
	return NerdContainerInfo{}, errors.New("not found")
}

func ContainerRemove(containerId string) error {
	err := cmd.ExecuteCommand(utils.NerdCtl, "rm", "-f", containerId)
	if err != nil {
		return err
	}
	return nil
}

func Load(imageFile string) error {
	return cmd.ExecuteCommand(utils.NerdCtl, "load", "--input", imageFile)
}

func Run(script []string) error {
	return cmd.ExecuteCommand(utils.NerdCtl, script...)
}

func CP(src, dst string) error {
	return cmd.ExecuteCommand(utils.NerdCtl, "cp", src, dst)
}
