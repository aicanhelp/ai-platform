package runtime

import (
	"fmt"
	"testing"
)

func TestDetectRuntime(t *testing.T) {
	fmt.Println(DetectRuntime())
}
