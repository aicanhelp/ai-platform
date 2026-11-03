package timer

import "time"

// Until 周期性地执行函数f，立即执行一次，然后每隔period时间重复执行，直到stopCh通道关闭
func Until(f func(), period time.Duration, stopCh <-chan struct{}) {
	// call it immediately
	f()

	// then call it intermittently
	ticker := time.NewTicker(period)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			f()
		case <-stopCh:
			return
		}
	}
}
