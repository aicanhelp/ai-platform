/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"github.com/spf13/cobra"

	"env-check/pkg/config"
)

// captureStdout captures stdout produced by f and returns it as a string.
// It restores stdout before returning.
func captureStdout(f func()) string {
	old := os.Stdout
	r, w, err := os.Pipe()
	if err != nil {
		return ""
	}
	os.Stdout = w

	outC := make(chan string)
	go func() {
		var buf bytes.Buffer
		_, _ = io.Copy(&buf, r)
		outC <- buf.String()
	}()

	// run the function while stdout is captured
	f()

	// close writer and restore stdout
	_ = w.Close()
	os.Stdout = old

	out := <-outC
	return out
}

// Group tests per target function to make intent clear and allow
// adding focused BeforeEach/AfterEach in future if needed.

var _ = Describe("cmd functions", func() {
	var dir string

	BeforeEach(func() {
		// create a single temp dir and switch cwd to it so all outputs go there
		dir = GinkgoT().TempDir()

		orig, err := os.Getwd()
		if err != nil {
			Fail("unable to get cwd: " + err.Error())
		}

		if err := os.Chdir(dir); err != nil {
			Fail("unable to chdir to temp dir: " + err.Error())
		}

		DeferCleanup(func() { _ = os.Chdir(orig) })
	})

	Describe("runProgram", func() {
		It("completes and reports missing program", func() {
			cfgFilePath := filepath.Join(dir, "cfg.json")
			cfgContent := `{"program_list":[{"name":"no-such-program","should_exist":true}]}`
			Expect(os.WriteFile(cfgFilePath, []byte(cfgContent), config.FileMode)).To(Succeed())

			origCfg := cfgFile
			cfgFile = cfgFilePath
			DeferCleanup(func() { cfgFile = origCfg })

			Expect(runProgram(nil, nil)).To(Succeed())

			_, statErr := os.Stat(filepath.Join(dir, "check.html"))
			Expect(statErr).ToNot(HaveOccurred())
		})

		It("fails when config lacks program_list", func() {
			cfgFilePath := filepath.Join(dir, "cfg_invalid.json")
			// empty config -> program_list missing
			Expect(os.WriteFile(cfgFilePath, []byte(`{}`), config.FileMode)).To(Succeed())

			origCfg := cfgFile
			cfgFile = cfgFilePath
			DeferCleanup(func() { cfgFile = origCfg })

			// Should return error due to validation
			err := runProgram(nil, nil)
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("program list is required"))
		})

		It("generates JSON when output_format=json", func() {
			cfgFilePath := filepath.Join(dir, "cfg_json.json")
			cfgContent := `{"program_list":[{"name":"no-such-program","should_exist":false}], "output_format":"json"}`
			Expect(os.WriteFile(cfgFilePath, []byte(cfgContent), config.FileMode)).To(Succeed())

			origCfg := cfgFile
			cfgFile = cfgFilePath
			DeferCleanup(func() { cfgFile = origCfg })

			Expect(runProgram(nil, nil)).To(Succeed())

			// verify json file exists and is valid JSON
			jsPath := filepath.Join(dir, "program-check-result.json")
			data, err := os.ReadFile(jsPath)
			Expect(err).ToNot(HaveOccurred())
			var j interface{}
			Expect(json.Unmarshal(data, &j)).To(Succeed())
		})

		It("detects installed 'go' and completes successfully", func() {
			// skip if go is not available in PATH on this machine
			if _, err := exec.LookPath("go"); err != nil {
				Skip("go not available in PATH; skipping installed-go test")
			}

			cfgFilePath := filepath.Join(dir, "cfg_go.json")
			cfgContent := `{"program_list":[{"name":"go","should_exist":true}]}`
			Expect(os.WriteFile(cfgFilePath, []byte(cfgContent), config.FileMode)).To(Succeed())

			origCfg := cfgFile
			cfgFile = cfgFilePath
			DeferCleanup(func() { cfgFile = origCfg })

			Expect(runProgram(nil, nil)).To(Succeed())
		})

		It("fails when config file is not valid JSON", func() {
			cfgFilePath := filepath.Join(dir, "cfg_not_json.json")
			// write invalid JSON content
			Expect(os.WriteFile(cfgFilePath, []byte("not a json file"), config.FileMode)).To(Succeed())

			origCfg := cfgFile
			cfgFile = cfgFilePath
			DeferCleanup(func() { cfgFile = origCfg })

			err := runProgram(nil, nil)
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("failed to load configuration"))
		})

		It("fails when config file is YAML (yaml suffix, YAML content)", func() {
			cfgFilePath := filepath.Join(dir, "cfg_yaml.yaml")
			// write valid YAML content which json.Unmarshal will not accept
			yamlContent := `program_list:\n  - name: no-such-program\n    should_exist: false\n` + "\n"
			Expect(os.WriteFile(cfgFilePath, []byte(yamlContent), config.FileMode)).To(Succeed())

			origCfg := cfgFile
			cfgFile = cfgFilePath
			DeferCleanup(func() { cfgFile = origCfg })

			err := runProgram(nil, nil)
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("failed to load configuration"))
		})
	})

	Describe("runQuery", func() {
		It("finds existing file and generates html", func() {
			filePath := filepath.Join(dir, "f.txt")
			Expect(os.WriteFile(filePath, []byte("data"), config.SecretFileMode)).To(Succeed())

			cfgFilePath := filepath.Join(dir, "cfg_query.json")
			cfgContent := fmt.Sprintf(`{"paths":[%q]}`, filePath)
			Expect(os.WriteFile(cfgFilePath, []byte(cfgContent), config.FileMode)).To(Succeed())

			origCfg := cfgFile
			cfgFile = cfgFilePath
			DeferCleanup(func() { cfgFile = origCfg })

			Expect(runQuery(nil, nil)).To(Succeed())

			_, statErr := os.Stat(filepath.Join(dir, "query.html"))
			Expect(statErr).ToNot(HaveOccurred())
		})

		It("fails when config lacks paths", func() {
			cfgFilePath := filepath.Join(dir, "cfg_invalid_query.json")
			Expect(os.WriteFile(cfgFilePath, []byte(`{}`), config.FileMode)).To(Succeed())

			origCfg := cfgFile
			cfgFile = cfgFilePath
			DeferCleanup(func() { cfgFile = origCfg })

			err := runQuery(nil, nil)
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("at least one path is required"))
		})

		It("generates JSON when output_format=json", func() {
			// create file and config
			filePath := filepath.Join(dir, "f_json.txt")
			Expect(os.WriteFile(filePath, []byte("data"), config.SecretFileMode)).To(Succeed())

			cfgFilePath := filepath.Join(dir, "cfg_query_json.json")
			cfgContent := fmt.Sprintf(`{"paths":[%q], "output_format":"json"}`, filePath)
			Expect(os.WriteFile(cfgFilePath, []byte(cfgContent), config.FileMode)).To(Succeed())

			origCfg := cfgFile
			cfgFile = cfgFilePath
			DeferCleanup(func() { cfgFile = origCfg })

			Expect(runQuery(nil, nil)).To(Succeed())

			jsPath := filepath.Join(dir, "query-result.json")
			data, err := os.ReadFile(jsPath)
			Expect(err).ToNot(HaveOccurred())
			var j interface{}
			Expect(json.Unmarshal(data, &j)).To(Succeed())
		})

		It("fails when config file is not valid JSON", func() {
			cfgFilePath := filepath.Join(dir, "cfg_query_not_json.json")
			Expect(os.WriteFile(cfgFilePath, []byte("this is not json"), config.FileMode)).To(Succeed())

			origCfg := cfgFile
			cfgFile = cfgFilePath
			DeferCleanup(func() { cfgFile = origCfg })

			err := runQuery(nil, nil)
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("failed to load configuration"))
		})

		It("fails when config file is YAML (yaml suffix, YAML content)", func() {
			cfgFilePath := filepath.Join(dir, "cfg_query_yaml.yaml")
			// valid YAML which is not JSON
			yamlContent := "paths:\n  - /tmp/somefile\n"
			Expect(os.WriteFile(cfgFilePath, []byte(yamlContent), config.FileMode)).To(Succeed())

			origCfg := cfgFile
			cfgFile = cfgFilePath
			DeferCleanup(func() { cfgFile = origCfg })

			err := runQuery(nil, nil)
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("failed to load configuration"))
		})
	})

	Describe("runClean", func() {
		It("deletes target file when forced and generates html", func() {
			filePath := filepath.Join(dir, "to-delete.txt")
			Expect(os.WriteFile(filePath, []byte("delete me"), config.SecretFileMode)).To(Succeed())

			cfgFilePath := filepath.Join(dir, "cfg_clean.json")
			cfgContent := fmt.Sprintf(`{"paths":[%q]}`, filePath)
			Expect(os.WriteFile(cfgFilePath, []byte(cfgContent), config.FileMode)).To(Succeed())

			origCfg := cfgFile
			cfgFile = cfgFilePath
			DeferCleanup(func() { cfgFile = origCfg })

			cmd := &cobra.Command{}
			cmd.Flags().Bool("force", false, "force clean")
			Expect(cmd.Flags().Set("force", "true")).To(Succeed())

			Expect(runClean(cmd, nil)).To(Succeed())

			_, statErr := os.Stat(filepath.Join(dir, "clean.html"))
			Expect(statErr).ToNot(HaveOccurred())

			_, err := os.Stat(filePath)
			Expect(os.IsNotExist(err)).To(BeTrue())
		})

		It("fails when config lacks paths", func() {
			cfgFilePath := filepath.Join(dir, "cfg_invalid_clean.json")
			Expect(os.WriteFile(cfgFilePath, []byte(`{}`), config.FileMode)).To(Succeed())

			origCfg := cfgFile
			cfgFile = cfgFilePath
			DeferCleanup(func() { cfgFile = origCfg })

			cmd := &cobra.Command{}
			cmd.Flags().Bool("force", false, "force clean")
			Expect(cmd.Flags().Set("force", "true")).To(Succeed())

			err := runClean(cmd, nil)
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("at least one path is required"))
		})

		It("generates JSON when output_format=json", func() {
			filePath := filepath.Join(dir, "to-delete-json.txt")
			Expect(os.WriteFile(filePath, []byte("delete me"), config.SecretFileMode)).To(Succeed())

			cfgFilePath := filepath.Join(dir, "cfg_clean_json.json")
			cfgContent := fmt.Sprintf(`{"paths":[%q], "output_format":"json"}`, filePath)
			Expect(os.WriteFile(cfgFilePath, []byte(cfgContent), config.FileMode)).To(Succeed())

			origCfg := cfgFile
			cfgFile = cfgFilePath
			DeferCleanup(func() { cfgFile = origCfg })

			cmd := &cobra.Command{}
			cmd.Flags().Bool("force", false, "force clean")
			Expect(cmd.Flags().Set("force", "true")).To(Succeed())

			Expect(runClean(cmd, nil)).To(Succeed())

			jsPath := filepath.Join(dir, "clean-result.json")
			data, err := os.ReadFile(jsPath)
			Expect(err).ToNot(HaveOccurred())
			var j interface{}
			Expect(json.Unmarshal(data, &j)).To(Succeed())
		})

		It("fails when config file is not valid JSON", func() {
			cfgFilePath := filepath.Join(dir, "cfg_clean_not_json.json")
			Expect(os.WriteFile(cfgFilePath, []byte("not-json"), config.FileMode)).To(Succeed())

			origCfg := cfgFile
			cfgFile = cfgFilePath
			DeferCleanup(func() { cfgFile = origCfg })

			cmd := &cobra.Command{}
			cmd.Flags().Bool("force", false, "force clean")
			Expect(cmd.Flags().Set("force", "true")).To(Succeed())

			err := runClean(cmd, nil)
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("failed to load configuration"))
		})

		It("fails when config file is YAML (yaml suffix, YAML content)", func() {
			cfgFilePath := filepath.Join(dir, "cfg_clean_yaml.yaml")
			yamlContent := "paths:\n  - /tmp/to-delete.txt\n"
			Expect(os.WriteFile(cfgFilePath, []byte(yamlContent), config.FileMode)).To(Succeed())

			origCfg := cfgFile
			cfgFile = cfgFilePath
			DeferCleanup(func() { cfgFile = origCfg })

			cmd := &cobra.Command{}
			cmd.Flags().Bool("force", false, "force clean")
			Expect(cmd.Flags().Set("force", "true")).To(Succeed())

			err := runClean(cmd, nil)
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("failed to load configuration"))
		})
	})

	Describe("runClock", func() {
		It("returns error for unreachable hosts", func() {
			cfgFilePath := filepath.Join(dir, "cfg_clock.json")
			cfgContent := `{"hosts":[{"ip":"192.168.0.1","role":["bootstrap"]},
{"ip":"192.168.0.2","role":["node"]}],"clock_threshold":10}`
			Expect(os.WriteFile(cfgFilePath, []byte(cfgContent), config.FileMode)).To(Succeed())

			origCfg := cfgFile
			cfgFile = cfgFilePath
			DeferCleanup(func() { cfgFile = origCfg })

			err := runClock(nil, nil)
			Expect(err).To(HaveOccurred())
		})

		It("fails when config lacks hosts", func() {
			cfgFilePath := filepath.Join(dir, "cfg_invalid_clock.json")
			Expect(os.WriteFile(cfgFilePath, []byte(`{}`), config.FileMode)).To(Succeed())

			origCfg := cfgFile
			cfgFile = cfgFilePath
			DeferCleanup(func() { cfgFile = origCfg })

			err := runClock(nil, nil)
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("hosts field is required"))
		})

		It("fails when config contains only bootstrap hosts", func() {
			cfgFilePath := filepath.Join(dir, "cfg_clock_only_bootstrap.json")
			cfgContent := `{"hosts":[{"ip":"127.0.0.1","role":["bootstrap"]}], "clock_threshold":10}`
			Expect(os.WriteFile(cfgFilePath, []byte(cfgContent), config.FileMode)).To(Succeed())

			origCfg := cfgFile
			cfgFile = cfgFilePath
			DeferCleanup(func() { cfgFile = origCfg })

			err := runClock(nil, nil)
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("no non-bootstrap host found"))
		})

		It("fails when config contains no bootstrap hosts", func() {
			cfgFilePath := filepath.Join(dir, "cfg_clock_no_bootstrap.json")
			cfgContent := `{"hosts":[{"ip":"192.168.0.2","role":["node"]}], "clock_threshold":10}`
			Expect(os.WriteFile(cfgFilePath, []byte(cfgContent), config.FileMode)).To(Succeed())

			origCfg := cfgFile
			cfgFile = cfgFilePath
			DeferCleanup(func() { cfgFile = origCfg })

			err := runClock(nil, nil)
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("no bootstrap host found"))
		})

		It("passes when bootstrap and master are localhost (both 127.0.0.1)", func() {
			cfgFilePath := filepath.Join(dir, "cfg_clock_local_dupe.json")
			cfgContent := `{"hosts":[{"ip":"127.0.0.1","role":["bootstrap"]},{"ip":"127.0.0.1","role":["master"]}], "clock_threshold":10}`
			Expect(os.WriteFile(cfgFilePath, []byte(cfgContent), config.FileMode)).To(Succeed())

			origCfg := cfgFile
			cfgFile = cfgFilePath
			DeferCleanup(func() { cfgFile = origCfg })

			Expect(runClock(nil, nil)).To(Succeed())

			_, statErr := os.Stat(filepath.Join(dir, "clock.html"))
			Expect(statErr).ToNot(HaveOccurred())
		})

		It("fails when config file is not valid JSON", func() {
			cfgFilePath := filepath.Join(dir, "cfg_clock_not_json.json")
			Expect(os.WriteFile(cfgFilePath, []byte("{invalid json"), config.FileMode)).To(Succeed())

			origCfg := cfgFile
			cfgFile = cfgFilePath
			DeferCleanup(func() { cfgFile = origCfg })

			err := runClock(nil, nil)
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("failed to load configuration"))
		})

		It("fails when config file is YAML (yaml suffix, YAML content)", func() {
			cfgFilePath := filepath.Join(dir, "cfg_clock_yaml.yaml")
			yamlContent := `hosts:\n  - ip: 192.168.0.1\n    role:\n      - bootstrap\n  - ip: 192.168.0.2\n    role:\n      - node\nclock_threshold: 10\n`
			Expect(os.WriteFile(cfgFilePath, []byte(yamlContent), config.FileMode)).To(Succeed())

			origCfg := cfgFile
			cfgFile = cfgFilePath
			DeferCleanup(func() { cfgFile = origCfg })

			err := runClock(nil, nil)
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("failed to load configuration"))
		})
	})

	Describe("runRoute", func() {
		It("completes route check and generates html", func() {
			cfgFilePath := filepath.Join(dir, "cfg_route.json")
			cfgContent := `{}`
			Expect(os.WriteFile(cfgFilePath, []byte(cfgContent), config.FileMode)).To(Succeed())

			origCfg := cfgFile
			cfgFile = cfgFilePath
			DeferCleanup(func() { cfgFile = origCfg })

			Expect(runRoute(nil, nil)).To(Succeed())

			_, statErr := os.Stat(filepath.Join(dir, "route.html"))
			Expect(statErr).ToNot(HaveOccurred())
		})

		It("generates JSON when output_format=json", func() {
			cfgFilePath := filepath.Join(dir, "cfg_route_json.json")
			cfgContent := `{"output_format":"json"}`
			Expect(os.WriteFile(cfgFilePath, []byte(cfgContent), config.FileMode)).To(Succeed())

			origCfg := cfgFile
			cfgFile = cfgFilePath
			DeferCleanup(func() { cfgFile = origCfg })

			Expect(runRoute(nil, nil)).To(Succeed())

			jsPath := filepath.Join(dir, "route-check-result.json")
			data, err := os.ReadFile(jsPath)
			Expect(err).ToNot(HaveOccurred())
			var j interface{}
			Expect(json.Unmarshal(data, &j)).To(Succeed())
		})

		It("fails when config file is not valid JSON", func() {
			cfgFilePath := filepath.Join(dir, "cfg_route_not_json.json")
			Expect(os.WriteFile(cfgFilePath, []byte("not a json file"), config.FileMode)).To(Succeed())

			origCfg := cfgFile
			cfgFile = cfgFilePath
			DeferCleanup(func() { cfgFile = origCfg })

			err := runRoute(nil, nil)
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("failed to load configuration"))
		})
	})

	Describe("runIptables", func() {
		It("completes iptables check and generates html", func() {
			cfgFilePath := filepath.Join(dir, "cfg_iptables.json")
			cfgContent := `{}`
			Expect(os.WriteFile(cfgFilePath, []byte(cfgContent), config.FileMode)).To(Succeed())

			origCfg := cfgFile
			cfgFile = cfgFilePath
			DeferCleanup(func() { cfgFile = origCfg })

			Expect(runIptables(nil, nil)).To(Succeed())

			_, statErr := os.Stat(filepath.Join(dir, "iptables.html"))
			Expect(statErr).ToNot(HaveOccurred())
		})

		It("generates JSON when output_format=json", func() {
			cfgFilePath := filepath.Join(dir, "cfg_iptables_json.json")
			cfgContent := `{"output_format":"json"}`
			Expect(os.WriteFile(cfgFilePath, []byte(cfgContent), config.FileMode)).To(Succeed())

			origCfg := cfgFile
			cfgFile = cfgFilePath
			DeferCleanup(func() { cfgFile = origCfg })

			Expect(runIptables(nil, nil)).To(Succeed())

			jsPath := filepath.Join(dir, "iptables-check-result.json")
			data, err := os.ReadFile(jsPath)
			Expect(err).ToNot(HaveOccurred())
			var j interface{}
			Expect(json.Unmarshal(data, &j)).To(Succeed())
		})

		It("fails when config file is not valid JSON", func() {
			cfgFilePath := filepath.Join(dir, "cfg_iptables_not_json.json")
			Expect(os.WriteFile(cfgFilePath, []byte("not a json file"), config.FileMode)).To(Succeed())

			origCfg := cfgFile
			cfgFile = cfgFilePath
			DeferCleanup(func() { cfgFile = origCfg })

			err := runIptables(nil, nil)
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("failed to load configuration"))
		})
	})

})
