/*
Copyright 2017 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Original file: https://github.com/kubernetes/kubernetes/blob/release-1.15/pkg/util/initsystem/initsystem_unix.go
*/

package initsystem

import (
	"fmt"
	"os/exec"
	"strings"
)

// OpenRCInitSystem defines openrc
type OpenRCInitSystem struct{}

// ServiceStart tries to start a specific service
func (openrc OpenRCInitSystem) ServiceStart(service string) error {
	args := []string{service, "start"}
	return exec.Command("rc-service", args...).Run()
}

// ServiceStop tries to stop a specific service
func (openrc OpenRCInitSystem) ServiceStop(service string) error {
	args := []string{service, "stop"}
	return exec.Command("rc-service", args...).Run()
}

// ServiceRestart tries to reload the environment and restart the specific service
func (openrc OpenRCInitSystem) ServiceRestart(service string) error {
	args := []string{service, "restart"}
	return exec.Command("rc-service", args...).Run()
}

// ServiceExists ensures the service is defined for this init system.
// openrc writes to stderr if a service is not found or not enabled
// this is in contrast to systemd which only writes to stdout.
// Hence, we use the Combinedoutput, and ignore the error.
func (openrc OpenRCInitSystem) ServiceExists(service string) bool {
	args := []string{service, "status"}
	outBytes, err := exec.Command("rc-service", args...).CombinedOutput()
	if err != nil {
		return false
	}
	if strings.Contains(string(outBytes), "does not exist") {
		return false
	}
	return true
}

// ServiceIsEnabled ensures the service is enabled to start on each boot.
func (openrc OpenRCInitSystem) ServiceIsEnabled(service string) bool {
	args := []string{"show", "default"}
	outBytes, err := exec.Command("rc-update", args...).Output()
	if err != nil {
		return false
	}
	if strings.Contains(string(outBytes), service) {
		return true
	}
	return false
}

// ServiceIsActive ensures the service is running, or attempting to run. (crash looping in the case of kubelet)
func (openrc OpenRCInitSystem) ServiceIsActive(service string) bool {
	args := []string{service, "status"}
	outBytes, err := exec.Command("rc-service", args...).CombinedOutput()
	if err != nil {
		return false
	}
	outStr := string(outBytes)
	return !strings.Contains(outStr, "stopped") && !strings.Contains(outStr, "does not exist")
}

// EnableCommand return a string describing how to enable a service
func (openrc OpenRCInitSystem) EnableCommand(service string) string {
	return fmt.Sprintf("rc-update add %s default", service)
}

func (openrc OpenRCInitSystem) ServiceDisable(service string) error {
	args := []string{service, "stop"}
	return exec.Command("rc-update", args...).Run()
}

func (openrc OpenRCInitSystem) ServiceEnable(service string) error {
	args := []string{service, "start"}
	return exec.Command("rc-update", args...).Run()
}

// SystemdInitSystem defines systemd
type SystemdInitSystem struct{}

// EnableCommand return a string describing how to enable a service
func (sysd SystemdInitSystem) EnableCommand(service string) string {
	return fmt.Sprintf("systemctl enable %s.service", service)
}

// reloadSystemd reloads the systemd daemon
func (sysd SystemdInitSystem) reloadSystemd() error {
	if err := exec.Command("systemctl", "daemon-reload").Run(); err != nil {
		return fmt.Errorf("failed to reload systemd: %w", err)
	}
	return nil
}

// ServiceStart tries to start a specific service
func (sysd SystemdInitSystem) ServiceStart(service string) error {
	// Before we try to start any service, make sure that systemd is ready
	if err := sysd.reloadSystemd(); err != nil {
		return err
	}
	args := []string{"start", service}
	return exec.Command("systemctl", args...).Run()
}

// ServiceRestart tries to reload the environment and restart the specific service
func (sysd SystemdInitSystem) ServiceRestart(service string) error {
	// Before we try to restart any service, make sure that systemd is ready
	if err := sysd.reloadSystemd(); err != nil {
		return err
	}
	args := []string{"restart", service}
	return exec.Command("systemctl", args...).Run()
}

// ServiceStop tries to stop a specific service
func (sysd SystemdInitSystem) ServiceStop(service string) error {
	args := []string{"stop", service}
	return exec.Command("systemctl", args...).Run()
}

// ServiceExists ensures the service is defined for this init system.
func (sysd SystemdInitSystem) ServiceExists(service string) bool {
	args := []string{"status", service}
	outBytes, err := exec.Command("systemctl", args...).Output()
	if err != nil {
		return false
	}
	output := string(outBytes)
	return !strings.Contains(output, "could not be found") && output != ""
}

// ServiceIsEnabled ensures the service is enabled to start on each boot.
func (sysd SystemdInitSystem) ServiceIsEnabled(service string) bool {
	args := []string{"is-enabled", service}
	err := exec.Command("systemctl", args...).Run()
	return err == nil
}

// ServiceIsActive will check is the service is "active". In the case of
// crash looping services (kubelet in our case) status will return as
// "activating", so we will consider this active as well.
func (sysd SystemdInitSystem) ServiceIsActive(service string) bool {
	args := []string{"is-active", service}
	// Ignoring error here, command returns non-0 if in "activating" status:
	outBytes, err := exec.Command("systemctl", args...).Output()
	if err != nil {
		return false
	}
	output := strings.TrimSpace(string(outBytes))
	if output == "active" || output == "activating" {
		return true
	}
	return false
}

func (sysd SystemdInitSystem) ServiceDisable(service string) error {
	args := []string{"disable", service}
	return exec.Command("systemctl", args...).Run()
}

func (sysd SystemdInitSystem) ServiceEnable(service string) error {
	args := []string{"enable", service}
	return exec.Command("systemctl", args...).Run()
}

// GetInitSystem returns an InitSystem for the current system, or nil
// if we cannot detect a supported init system.
// This indicates we will skip init system checks, not an error.
func GetInitSystem() (InitSystem, error) {
	// Assume existence of systemctl in path implies this is a systemd system:
	_, err := exec.LookPath("systemctl")
	if err == nil {
		return &SystemdInitSystem{}, nil
	}
	_, err = exec.LookPath("openrc")
	if err == nil {
		return &OpenRCInitSystem{}, nil
	}

	return nil, fmt.Errorf("no supported init system detected, skipping checking for services")
}
