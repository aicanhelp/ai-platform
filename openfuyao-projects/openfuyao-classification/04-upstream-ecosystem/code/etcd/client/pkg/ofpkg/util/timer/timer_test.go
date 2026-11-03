package timer

import (
	"testing"
	"time"
)

func TestUntil(t *testing.T) {
	stopCh := make(chan struct{})
	count := 0
	go Until(func() {
		count++
	}, time.Second, stopCh)

	// wait for seconds and stop
	time.Sleep(2 * time.Second)
	close(stopCh)

	// wait and check
	time.Sleep(2 * time.Second)
	if count < 2 || count > 3 {
		t.Errorf("expect count to be between 2 and 3, but got %d", count)
	}
}
