/******************************************************************
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * cluster-api-provider-bke is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package version

import (
	"fmt"

	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/log"
)

var (
	GitCommitID  = "dev"
	Version      = "v1.0.0"
	Architecture = "unknown"
	BuildTime    = "unknown"
)

func PrintVersion() {
	fmt.Printf("🤯 Version: %s\n", Version)
	fmt.Printf("🤔 GitCommitId: %s\n", GitCommitID)
	fmt.Printf("👉 Architecture: %s\n", Architecture)
	fmt.Printf("⏲ BuildTime: %s\n", BuildTime)
}

func LogPrintVersion() {
	log.Info("--------------Starting the BKEAgent---------------")
	log.Info(fmt.Sprintf("🤯 Version: %s", Version))
	log.Info(fmt.Sprintf("🤔 GitCommitId: %s ", GitCommitID))
	log.Info(fmt.Sprintf("👉 Architecture: %s", Architecture))
	log.Info(fmt.Sprintf("⏲ BuildTime: %s", BuildTime))
	log.Info("--------------------------------------------------")
}

func String() string {
	return fmt.Sprintf("%s %s %s %s", Version, GitCommitID, Architecture, BuildTime)
}
