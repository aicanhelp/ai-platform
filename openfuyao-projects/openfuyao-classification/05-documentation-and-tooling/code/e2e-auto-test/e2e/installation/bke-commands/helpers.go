package bke_commands

import (
	"time"

	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework"
)

const InstallationItTimeout = 45 * time.Minute

var BKECommandDescribe = framework.SIGDescribe("installation")
