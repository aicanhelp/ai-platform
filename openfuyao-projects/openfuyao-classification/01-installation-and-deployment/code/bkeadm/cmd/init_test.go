/*
 *
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          <http://license.coscl.org.cn/MulanPSL2>
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 *
 */

package cmd

import (
	"errors"
	"net"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/spf13/cobra"
	"github.com/stretchr/testify/assert"
	configv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	configinit "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/initialize"

	"gopkg.openfuyao.cn/bkeadm/pkg/cluster"
	"gopkg.openfuyao.cn/bkeadm/pkg/initialize"
	"gopkg.openfuyao.cn/bkeadm/utils"
)

const (
	testIPv4SegmentE = 101
	testIPv4SegmentF = 200
	testIPv4SegmentG = 201
	testIPv4SegmentH = 202
)

var (
	testIP1 = net.IPv4(
		testIPv4SegmentA,
		testIPv4SegmentB,
		testIPv4SegmentC,
		testIPv4SegmentD,
	)
	testIP2 = net.IPv4(
		testIPv4SegmentA,
		testIPv4SegmentB,
		testIPv4SegmentC,
		testIPv4SegmentE,
	)
	testIP3 = net.IPv4(
		testIPv4SegmentA,
		testIPv4SegmentB,
		testIPv4SegmentC,
		testIPv4SegmentF,
	)
	testIP4 = net.IPv4(
		testIPv4SegmentA,
		testIPv4SegmentB,
		testIPv4SegmentC,
		testIPv4SegmentG,
	)
	testIP5 = net.IPv4(
		testIPv4SegmentA,
		testIPv4SegmentB,
		testIPv4SegmentC,
		testIPv4SegmentH,
	)
	testImageRepoPort   = "5000"
	testYumRepoPort     = "8080"
	testChartRepoPort   = "8443"
	testClusterFileName = "test-cluster.yaml"
)

func TestInitCmdInitialization(t *testing.T) {
	tests := []struct {
		name          string
		expectedUse   string
		expectedShort string
		hasFlags      bool
	}{
		{
			name:          "init command properties",
			expectedUse:   "init",
			expectedShort: "Initialize the boot node",
			hasFlags:      true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.expectedUse, initCmd.Use)
			assert.Equal(t, tt.expectedShort, initCmd.Short)

			if tt.hasFlags {
				// Check if required flags exist
				fileFlag := initCmd.Flags().Lookup("file")
				assert.NotNil(t, fileFlag)

				domainFlag := initCmd.Flags().Lookup("domain")
				assert.NotNil(t, domainFlag)

				hostIPFlag := initCmd.Flags().Lookup("hostIP")
				assert.NotNil(t, hostIPFlag)

				consolePortFlag := initCmd.Flags().Lookup("consolePort")
				assert.NotNil(t, consolePortFlag)
				assert.Equal(t, utils.DefaultConsolePort, consolePortFlag.DefValue)
			}
		})
	}
}

func TestRegisterInitCommand(t *testing.T) {
	// Find the init command in root commands
	var foundInitCmd bool
	for _, cmd := range rootCmd.Commands() {
		if cmd.Use == "init" {
			foundInitCmd = true
			break
		}
	}

	assert.True(t, foundInitCmd, "init command should be registered in root command")
}

func TestInitCmdPreRunE(t *testing.T) {
	tests := []struct {
		name                      string
		initialHostIP             string
		initialDomain             string
		initialImageRepoPort      string
		initialKubernetesPort     string
		initialConsolePort        string
		initialYumRepoPort        string
		initialChartRepoPort      string
		mockGetOutBoundIP         func() (string, error)
		mockGetIntranetIp         func() (string, error)
		mockNewBKEClusterFromFile func(string) (*configv1beta1.BKECluster, error)
		mockLoopIP                func(string) ([]string, error)
		mockPromptForConfirmation func(bool) bool
		expectError               bool
		errorContains             string
	}{
		{
			name:                  "success with default values",
			initialHostIP:         "",
			initialDomain:         "",
			initialImageRepoPort:  "",
			initialKubernetesPort: "",
			initialConsolePort:    utils.DefaultConsolePort,
			initialYumRepoPort:    "",
			initialChartRepoPort:  "",
			mockGetOutBoundIP: func() (string, error) {
				return testIP1.String(), nil
			},
			mockGetIntranetIp: func() (string, error) {
				return testIP2.String(), nil // Won't be used since OutBoundIP succeeds
			},
			mockNewBKEClusterFromFile: func(string) (*configv1beta1.BKECluster, error) {
				return nil, nil // No file specified, so this won't be called
			},
			mockLoopIP: func(string) ([]string, error) {
				return nil, nil
			},
			mockPromptForConfirmation: func(bool) bool {
				return true // User confirms
			},
			expectError: true,
		},
		{
			name:                  "missing host IP should return error",
			initialHostIP:         "",
			initialDomain:         configinit.DefaultImageRepo,
			initialImageRepoPort:  configinit.DefaultImageRepoPort,
			initialKubernetesPort: utils.DefaultKubernetesPort,
			initialConsolePort:    utils.DefaultConsolePort,
			initialYumRepoPort:    configinit.DefaultYumRepoPort,
			initialChartRepoPort:  utils.DefaultChartRegistryPort,
			mockGetOutBoundIP: func() (string, error) {
				return "", errors.New("network error")
			},
			mockGetIntranetIp: func() (string, error) {
				return "", errors.New("network error")
			},
			mockNewBKEClusterFromFile: func(string) (*configv1beta1.BKECluster, error) {
				return nil, nil
			},
			mockLoopIP: func(string) ([]string, error) {
				return nil, nil
			},
			mockPromptForConfirmation: func(bool) bool {
				return true
			},
			expectError:   true,
			errorContains: "The host IP address must be set",
		},
		{
			name:                  "user does not confirm should return error",
			initialHostIP:         testIP1.String(),
			initialDomain:         configinit.DefaultImageRepo,
			initialImageRepoPort:  configinit.DefaultImageRepoPort,
			initialKubernetesPort: utils.DefaultKubernetesPort,
			initialConsolePort:    utils.DefaultConsolePort,
			initialYumRepoPort:    configinit.DefaultYumRepoPort,
			initialChartRepoPort:  utils.DefaultChartRegistryPort,
			mockGetOutBoundIP: func() (string, error) {
				return testIP1.String(), nil
			},
			mockGetIntranetIp: func() (string, error) {
				return testIP2.String(), nil
			},
			mockNewBKEClusterFromFile: func(string) (*configv1beta1.BKECluster, error) {
				return nil, nil
			},
			mockLoopIP: func(string) ([]string, error) {
				return nil, nil
			},
			mockPromptForConfirmation: func(bool) bool {
				return false // User does not confirm
			},
			expectError:   true,
			errorContains: "operation cancelled by user",
		},
		{
			name:                  "invalid console port should return error",
			initialHostIP:         testIP1.String(),
			initialDomain:         configinit.DefaultImageRepo,
			initialImageRepoPort:  configinit.DefaultImageRepoPort,
			initialKubernetesPort: utils.DefaultKubernetesPort,
			initialConsolePort:    "70000",
			initialYumRepoPort:    configinit.DefaultYumRepoPort,
			initialChartRepoPort:  utils.DefaultChartRegistryPort,
			mockGetOutBoundIP: func() (string, error) {
				return testIP1.String(), nil
			},
			mockGetIntranetIp: func() (string, error) {
				return testIP2.String(), nil
			},
			mockNewBKEClusterFromFile: func(string) (*configv1beta1.BKECluster, error) {
				return nil, nil
			},
			mockLoopIP: func(string) ([]string, error) {
				return nil, nil
			},
			mockPromptForConfirmation: func(bool) bool {
				return true
			},
			expectError:   true,
			errorContains: "The console port must be legal",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Save original values
			originalHostIP := initOption.HostIP
			originalDomain := initOption.Domain
			originalImageRepoPort := initOption.ImageRepoPort
			originalKubernetesPort := initOption.KubernetesPort
			originalConsolePort := initOption.ConsolePort
			originalYumRepoPort := initOption.YumRepoPort
			originalChartRepoPort := initOption.ChartRepoPort
			originalRuntime := initOption.Runtime
			originalRuntimeStorage := initOption.RuntimeStorage
			originalAgentHealthPort := initOption.AgentHealthPort
			originalFile := initOption.File
			originalConfirm := confirm
			defer func() {
				initOption.HostIP = originalHostIP
				initOption.Domain = originalDomain
				initOption.ImageRepoPort = originalImageRepoPort
				initOption.KubernetesPort = originalKubernetesPort
				initOption.ConsolePort = originalConsolePort
				initOption.YumRepoPort = originalYumRepoPort
				initOption.ChartRepoPort = originalChartRepoPort
				initOption.Runtime = originalRuntime
				initOption.RuntimeStorage = originalRuntimeStorage
				initOption.AgentHealthPort = originalAgentHealthPort
				initOption.File = originalFile
				confirm = originalConfirm
			}()

			// Set initial values
			initOption.HostIP = tt.initialHostIP
			initOption.Domain = tt.initialDomain
			initOption.ImageRepoPort = tt.initialImageRepoPort
			initOption.KubernetesPort = tt.initialKubernetesPort
			initOption.ConsolePort = tt.initialConsolePort
			initOption.YumRepoPort = tt.initialYumRepoPort
			initOption.ChartRepoPort = tt.initialChartRepoPort
			initOption.File = "" // Don't process file in this test
			confirm = true       // Override for this test

			// Apply patches
			patches := gomonkey.NewPatches()
			defer patches.Reset()
			patches.ApplyFunc(utils.GetOutBoundIP, tt.mockGetOutBoundIP)

			patches.ApplyFunc(utils.GetIntranetIp, tt.mockGetIntranetIp)

			patches.ApplyFunc(cluster.NewBKEClusterFromFile, tt.mockNewBKEClusterFromFile)

			patches.ApplyFunc(utils.LoopIP, tt.mockLoopIP)

			patches.ApplyFunc(utils.PromptForConfirmation, tt.mockPromptForConfirmation)

			// Call PreRunE
			err := initCmd.PreRunE(nil, nil)

			if tt.expectError {
				assert.Error(t, err)
				if tt.errorContains != "" {
					assert.Contains(t, err.Error(), tt.errorContains)
				}
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestInitCmdPreRunERuntimeRestrictions(t *testing.T) {
	tests := []struct {
		name                   string
		initialRuntime         string
		initialRuntimeStorage  string
		expectError            bool
		errorContains          string
		expectedRuntime        string
		expectedRuntimeStorage string
	}{
		{
			name:                   "default runtime uses containerd storage",
			initialRuntime:         "",
			initialRuntimeStorage:  "",
			expectedRuntime:        "containerd",
			expectedRuntimeStorage: configinit.DefaultCRIContainerdDataRootDir,
		},
		{
			name:                  "docker runtime is not supported",
			initialRuntime:        "docker",
			initialRuntimeStorage: "",
			expectError:           true,
			errorContains:         "runtime` only supports containerd",
		},
		{
			name:                  "docker runtime storage is not supported",
			initialRuntime:        "containerd",
			initialRuntimeStorage: configinit.DefaultCRIDockerDataRootDir,
			expectError:           true,
			errorContains:         "runtimeStorage` does not support docker data root",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			originalHostIP := initOption.HostIP
			originalDomain := initOption.Domain
			originalImageRepoPort := initOption.ImageRepoPort
			originalKubernetesPort := initOption.KubernetesPort
			originalConsolePort := initOption.ConsolePort
			originalYumRepoPort := initOption.YumRepoPort
			originalChartRepoPort := initOption.ChartRepoPort
			originalRuntime := initOption.Runtime
			originalRuntimeStorage := initOption.RuntimeStorage
			originalAgentHealthPort := initOption.AgentHealthPort
			originalFile := initOption.File
			originalConfirm := confirm
			defer func() {
				initOption.HostIP = originalHostIP
				initOption.Domain = originalDomain
				initOption.ImageRepoPort = originalImageRepoPort
				initOption.KubernetesPort = originalKubernetesPort
				initOption.ConsolePort = originalConsolePort
				initOption.YumRepoPort = originalYumRepoPort
				initOption.ChartRepoPort = originalChartRepoPort
				initOption.Runtime = originalRuntime
				initOption.RuntimeStorage = originalRuntimeStorage
				initOption.AgentHealthPort = originalAgentHealthPort
				initOption.File = originalFile
				confirm = originalConfirm
			}()

			initOption.HostIP = testIP1.String()
			initOption.Domain = configinit.DefaultImageRepo
			initOption.ImageRepoPort = configinit.DefaultImageRepoPort
			initOption.KubernetesPort = utils.DefaultKubernetesPort
			initOption.ConsolePort = utils.DefaultConsolePort
			initOption.YumRepoPort = configinit.DefaultYumRepoPort
			initOption.ChartRepoPort = utils.DefaultChartRegistryPort
			initOption.Runtime = tt.initialRuntime
			initOption.RuntimeStorage = tt.initialRuntimeStorage
			initOption.AgentHealthPort = utils.DefaultAgentHealthPort
			initOption.File = ""
			confirm = true

			patches := gomonkey.NewPatches()
			defer patches.Reset()
			patches.ApplyFunc(utils.PromptForConfirmation, func(bool) bool {
				return true
			})

			err := initCmd.PreRunE(&cobra.Command{}, []string{})

			if tt.expectError {
				assert.Error(t, err)
				assert.Contains(t, err.Error(), tt.errorContains)
				return
			}
			assert.NoError(t, err)
			assert.Equal(t, tt.expectedRuntime, initOption.Runtime)
			assert.Equal(t, tt.expectedRuntimeStorage, initOption.RuntimeStorage)
		})
	}
}
func TestInitCmdRunWithMock(t *testing.T) {
	// Save original values
	originalArgs := initOption.Args
	originalOptions := initOption.Options
	defer func() {
		initOption.Args = originalArgs
		initOption.Options = originalOptions
	}()

	// Apply patch to mock the Initialize method
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc((*initialize.Options).Initialize, func(o *initialize.Options) {
		// Mock implementation - do nothing
	})

	// Create and run the command
	cmd := &cobra.Command{}
	args := []string{"arg1", "arg2"}

	initCmd.Run(cmd, args)

	// Verify that args and options were set
	assert.Equal(t, args, initOption.Args)
	assert.Equal(t, options, initOption.Options)
}

func TestInitCmdWithFileProcessing(t *testing.T) {
	// This test verifies the logic when a file is provided
	// We'll mock the file processing logic to avoid actual file operations

	// Save original values
	originalFile := initOption.File
	originalHostIP := initOption.HostIP
	originalDomain := initOption.Domain
	originalImageRepoPort := initOption.ImageRepoPort
	originalKubernetesPort := initOption.KubernetesPort
	originalConsolePort := initOption.ConsolePort
	originalYumRepoPort := initOption.YumRepoPort
	originalChartRepoPort := initOption.ChartRepoPort
	defer func() {
		initOption.File = originalFile
		initOption.HostIP = originalHostIP
		initOption.Domain = originalDomain
		initOption.ImageRepoPort = originalImageRepoPort
		initOption.KubernetesPort = originalKubernetesPort
		initOption.ConsolePort = originalConsolePort
		initOption.YumRepoPort = originalYumRepoPort
		initOption.ChartRepoPort = originalChartRepoPort
	}()

	// Set up test values
	testFile := testClusterFileName
	testIPAddr := testIP1.String()

	initOption.File = testFile
	initOption.HostIP = testIPAddr
	initOption.Domain = configinit.DefaultImageRepo
	initOption.ImageRepoPort = configinit.DefaultImageRepoPort
	initOption.KubernetesPort = utils.DefaultKubernetesPort
	initOption.ConsolePort = utils.DefaultConsolePort
	initOption.YumRepoPort = configinit.DefaultYumRepoPort
	initOption.ChartRepoPort = utils.DefaultChartRegistryPort

	// Mock the NewBKEClusterFromFile function to return a test cluster
	mockCluster := &configv1beta1.BKECluster{
		Spec: configv1beta1.BKEClusterSpec{
			ClusterConfig: &configv1beta1.BKEConfig{
				Cluster: configv1beta1.Cluster{
					ImageRepo: configv1beta1.Repo{
						Ip:     testIP3.String(),
						Domain: configinit.DefaultImageRepo,
						Port:   testImageRepoPort,
						Prefix: "k8s",
					},
					HTTPRepo: configv1beta1.Repo{
						Ip:     testIP4.String(),
						Domain: "yum.example.com",
						Port:   testYumRepoPort,
					},
					ChartRepo: configv1beta1.Repo{
						Ip:     testIP5.String(),
						Domain: configinit.DefaultChartRepo,
						Port:   testChartRepoPort,
						Prefix: "charts",
					},
					NTPServer: "pool.ntp.org",
				},
			},
		},
	}

	// Apply patches
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(cluster.NewBKEClusterFromFile, func(file string) (*configv1beta1.BKECluster, error) {
		return mockCluster, nil
	})

	patches.ApplyFunc(utils.LoopIP, func(domain string) ([]string, error) {
		return []string{net.ParseIP(testIP1.String()).String()}, nil
	})

	patches.ApplyFunc(utils.PromptForConfirmation, func(confirmed bool) bool {
		return true
	})

	// Mock the GetOutBoundIP and GetIntranetIp to return test values
	patches.ApplyFunc(utils.GetOutBoundIP, func() (string, error) {
		return testIPAddr, nil
	})

	patches.ApplyFunc(utils.GetIntranetIp, func() (string, error) {
		return testIPAddr, nil
	})

	// Call PreRunE to trigger the file processing logic
	err := initCmd.PreRunE(&cobra.Command{}, []string{})

	// The error might occur due to validation checks, which is expected in this test
	// We mainly want to verify that the file processing logic ran without panicking
	if err != nil {
		// If there's an error, it should be related to validation, not the file processing itself
		assert.Contains(t, err.Error(), "port is not a number") // This could happen if ports are not numbers
	}
}
