package version

import (
	"fmt"

	"gopkg.openfuyao.cn/bkeagent/utils/log"
)

var (
	GitCommitID  = "dev"
	Version      = "v1.0.0"
	Architecture = "unknown"
	BuildTime    = "unknown"
)

func PrintVersion() {
	fmt.Println(GitCommitID)
	fmt.Println(fmt.Sprintf("🤯 Version: %s", Version))
	fmt.Println(fmt.Sprintf("🤔 GitCommitId: %s ", GitCommitID))
	fmt.Println(fmt.Sprintf("👉 Architecture: %s", Architecture))
	fmt.Println(fmt.Sprintf("⏲ BuildTime: %s", BuildTime))
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
