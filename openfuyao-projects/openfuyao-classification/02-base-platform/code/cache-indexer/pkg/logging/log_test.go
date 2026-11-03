package logging

import "testing"

const (
	testLevelDebug   = "debug"
	testLevelInfo    = "info"
	testLevelError   = "error"
	testLevelInvalid = "invalid-level"
)

func TestNew_AcceptsKnownLevels(t *testing.T) {
	levels := []string{testLevelDebug, testLevelInfo, testLevelError}
	for _, level := range levels {
		level := level
		t.Run(level, func(t *testing.T) {
			logger, err := New(level)
			if err != nil {
				t.Fatalf("New(%q): %v", level, err)
			}
			logger.Info("log")
		})
	}
}

func TestNew_FallsBackForInvalidLevel(t *testing.T) {
	logger, err := New(testLevelInvalid)
	if err != nil {
		t.Fatalf("New(%q): %v", testLevelInvalid, err)
	}
	logger.Info("fallback")
}
