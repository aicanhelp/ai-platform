package framework

import (
	"github.com/onsi/ginkgo/v2"
)

// SIGDescribe 用于添加sig组标签
func SIGDescribe(name string) func(string, ...any) bool {
	return func(text string, args ...any) bool {
		argsWithSIGLabel := append([]interface{}{ginkgo.Label("sig-" + name + "")}, args...)
		return ginkgo.Describe(text, argsWithSIGLabel...)
	}
}
