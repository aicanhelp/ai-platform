package utils

import (
	"fmt"
	"io"
	"os"
	"strings"
	"sync"
	"time"

	. "github.com/onsi/ginkgo/v2"
)

var (
	liveStepLogMu sync.Mutex
	liveStepLogTo io.Writer
)

// EnableLiveStepLog configures whether LogStep mirrors to stderr immediately.
// Default is on (direct os.Stderr write, not GinkgoWriter.TeeTo — go test may still buffer tee).
// Set INFERNEX_BRIDGE_LIVE_STDERR=0 to disable; =1 to force on.
func EnableLiveStepLog() {
	switch strings.TrimSpace(os.Getenv("INFERNEX_BRIDGE_LIVE_STDERR")) {
	case "0", "false", "no":
		liveStepLogMu.Lock()
		liveStepLogTo = nil
		liveStepLogMu.Unlock()
	default:
		liveStepLogMu.Lock()
		liveStepLogTo = os.Stderr
		liveStepLogMu.Unlock()
	}
}

func liveStepLogWriter() io.Writer {
	liveStepLogMu.Lock()
	defer liveStepLogMu.Unlock()
	return liveStepLogTo
}

func writeLogLine(line string) {
	if w := liveStepLogWriter(); w != nil {
		_, _ = io.WriteString(w, line)
	}
	_, _ = GinkgoWriter.Write([]byte(line))
}

func LogStep(fitID, format string, args ...any) {
	msg := fmt.Sprintf(format, args...)
	line := fmt.Sprintf("[%s] %s | %s\n", time.Now().Format("15:04:05"), fitID, msg)
	writeLogLine(line)
}

func LogFailContext(fitID, detail string) {
	writeLogLine(fmt.Sprintf("[%s] FAIL-CONTEXT | %s\n", fitID, detail))
}
