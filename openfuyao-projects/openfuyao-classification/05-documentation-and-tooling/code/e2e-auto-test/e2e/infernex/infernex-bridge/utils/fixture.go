package utils

import (
	"os"
)

func LoadFixture(parts ...string) (string, error) {
	path, err := FixturePath(parts...)
	if err != nil {
		return "", err
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}
	return RenderNamespace(string(data), E2ENS()), nil
}
