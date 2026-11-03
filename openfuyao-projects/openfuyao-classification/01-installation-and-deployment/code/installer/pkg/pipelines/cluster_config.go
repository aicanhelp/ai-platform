/******************************************************************
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package pipelines

import (
	"fmt"
	"os"

	"installer/pkg/common"
	"installer/pkg/config"
	"installer/pkg/constants"
)

// CreateClusterConfigFile 创建集群配置文件并保存到指定路径
func CreateClusterConfigFile(args common.Argument) error {
	if args.FileSaveFolder != "" {
		if _, err := os.Stat(args.FileSaveFolder); os.IsNotExist(err) {
			// 如果文件夹不存在，则创建嵌套文件夹
			err := os.MkdirAll(args.FileSaveFolder, constants.RwxRxRx)
			if err != nil {
				return fmt.Errorf("failed to create dir: %v", err)
			}
		}
	}

	runtime := config.ClusterConfigRuntime{
		AdvancedParam: args.EnableAdvancedParams,
		Ha:            args.EnableHaParams,
		AllParam:      args.EnableAllParams,
		SavePath:      args.FileSaveFolder,
	}

	var err error
	if runtime.AllParam {
		err = runtime.CreateAllParamCfgFile()
	} else if runtime.Ha && runtime.AdvancedParam {
		err = runtime.CreateHaCfgFileWithAdvanced()
	} else if runtime.Ha && !runtime.AdvancedParam {
		err = runtime.CreateHaCfgFile()
	} else if runtime.AdvancedParam {
		err = runtime.CreateSimplestCfgFileWithAdvanced()
	} else {
		err = runtime.CreateSimplestCfgFile()
	}
	if err != nil {
		return err
	}
	fmt.Println("The cluster configuration file is successfully generated.")
	return nil
}
