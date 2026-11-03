package utils

import (
	"fmt"
	"path/filepath"
	"runtime"
)

func packageDir() (string, error) {
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		return "", fmt.Errorf("runtime.Caller failed")
	}
	// .../infernex-bridge/utils -> infernex-bridge
	return filepath.Dir(filepath.Dir(file)), nil
}

func FixturesDir() (string, error) {
	dir, err := packageDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(dir, "fixtures"), nil
}

func FixturePath(parts ...string) (string, error) {
	base, err := FixturesDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(append([]string{base}, parts...)...), nil
}
