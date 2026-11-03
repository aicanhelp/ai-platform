package utils

import (
	"os"
)

// RenderOptions substitutes placeholders in fixture YAML.
type RenderOptions struct {
	Namespace     string
	Name          string
	ExternalGW    string
	ExternalRoute string
	ExternalPool  string
	KServeNS      string
	PrefillDeploy string
	DecodeDeploy  string
}

func (o RenderOptions) ns() string {
	if o.Namespace != "" {
		return o.Namespace
	}
	return E2ENS()
}

// RenderFixtureBytes loads a fixture and applies placeholders.
// Registered fixtures are built from one of four canonical templates; others fall back to legacy files.
func RenderFixtureBytes(parts []string, opts RenderOptions) ([]byte, error) {
	return RenderFixtureBytesFromTemplate(parts, opts)
}

func WriteRenderedFixtureOpts(parts []string, opts RenderOptions) (path string, err error) {
	data, err := RenderFixtureBytes(parts, opts)
	if err != nil {
		return "", err
	}
	f, err := os.CreateTemp("", "infernex-bridge-fixture-*.yaml")
	if err != nil {
		return "", err
	}
	if _, err := f.Write(data); err != nil {
		f.Close()
		os.Remove(f.Name())
		return "", err
	}
	f.Close()
	return f.Name(), nil
}
