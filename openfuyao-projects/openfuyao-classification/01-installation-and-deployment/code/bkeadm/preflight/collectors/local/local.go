/******************************************************************
 * Copyright (c) 2026 Bocloud Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

// Package local collects read-only facts about the host running preflight.
//
// It reads OS family, distribution, kernel, architecture, CPU, memory, disk,
// path, port and tool facts. All access is read-only: it never creates files,
// writes sysctls, or starts services. Platform-specific acquisition is behind
// the HostProbe interface so tests can inject fake facts.
package local

import (
	"context"
	"crypto/x509"
	"encoding/pem"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"syscall"

	"gopkg.openfuyao.cn/preflight/api"
	"gopkg.openfuyao.cn/preflight/normalize"
)

const (
	writePermissionMask = 0o222
	ownerWriteBit       = 0o200
	portKeyParts        = 2
	minimumMemoryFields = 2
	decimalBase         = 10
	bytesPerKilobyte    = 1024
)

// HostProbe is the narrow read-only surface the local collector uses to
// acquire host facts. The concrete implementation reads /etc/os-release,
// uname, gopsutil and the filesystem; tests inject a fake.
type HostProbe interface {
	// OSFamily returns linux/windows/darwin/other.
	OSFamily() string
	// Distribution returns (id, version) from /etc/os-release.
	Distribution() (id, version string)
	// Kernel returns the uname release string.
	Kernel() string
	// Architecture returns the host arch (amd64/arm64/other).
	Architecture() string
	// CPUCount returns logical CPU count.
	CPUCount() int
	// MemoryTotalBytes returns MemTotal.
	MemoryTotalBytes() uint64
	// SystemDiskTotalBytes returns the total capacity of the system filesystem
	// mounted at /, in bytes. 0 means unknown.
	SystemDiskTotalBytes() uint64
	// FileExists reports whether a path exists (stat, no follow).
	FileExists(path string) bool
	// FileReadable reports whether the executing identity can read the path.
	FileReadable(path string) bool
	// PortFree reports whether a TCP or UDP port is free on the given address.
	PortFree(protocol string, port int) bool
	// ToolExists reports whether a binary is on PATH and executable.
	ToolExists(name string) bool
}

// Collector is the local-host fact collector.
type Collector struct {
	probe             HostProbe
	verifiedPlatforms []api.VerifiedPlatform
}

// New returns a Collector backed by the given probe.
func New(probe HostProbe) *Collector {
	return &Collector{probe: probe}
}

// NewWithVerifiedPlatforms returns a local collector with release-verified platform metadata.
func NewWithVerifiedPlatforms(probe HostProbe, platforms []api.VerifiedPlatform) *Collector {
	return &Collector{probe: probe, verifiedPlatforms: append([]api.VerifiedPlatform(nil), platforms...)}
}

// ID returns the collector identifier.
func (c *Collector) ID() string { return "local-host" }

// Collect returns facts for the requested keys, scoped to subjects. The local
// collector uses the first subject's ID (the bootstrap host) as the fact
// Subject so engine.Run's findFact (which keys on subject+key) can match.
func (c *Collector) Collect(ctx context.Context, req api.FactRequest) ([]api.Fact, error) {
	if c.probe == nil {
		return c.errorFacts(req.Keys, "UNSUPPORTED", "no host probe configured"), nil
	}
	// Local facts are about the host running preflight; use the first subject
	// (or empty) as the Subject field so engine.Run can index them.
	subject := ""
	if len(req.Subjects) > 0 {
		subject = req.Subjects[0].ID
	}
	out := make([]api.Fact, 0, len(req.Keys))
	for _, k := range req.Keys {
		select {
		case <-ctx.Done():
			ctxErr := ctx.Err()
			if ctxErr == nil {
				ctxErr = context.Canceled
			}
			out = append(out, api.Fact{Key: k, Subject: subject, Available: false,
				Error: &api.FactError{Code: api.FactErrTimeout, Message: ctxErr.Error()}})
			continue
		default:
		}
		var target api.Subject
		if len(req.Subjects) > 0 {
			target = req.Subjects[0]
		}
		f := c.collectKey(target, k)
		f.Subject = subject
		out = append(out, f)
	}
	return out, nil
}

// collectKey maps a fact key to its host fact.
func (c *Collector) collectKey(subject api.Subject, key string) api.Fact {
	if exact, ok := c.collectExactKey(key); ok {
		return exact
	}
	switch {
	case strings.HasPrefix(key, "sysctl."):
		return c.sysctlFact(key)
	case strings.HasPrefix(key, "path."):
		return c.semanticPathFact(subject, key)
	case key == "runtime.dataRoot.conflict":
		return fact(key, nonEmptyDirectory(subject.Metadata["runtimeStorage"]))
	case strings.HasPrefix(key, "artifact.runc."):
		return c.executableFact(key)
	case strings.HasPrefix(key, "certificate."):
		return c.certificateFact(subject, key)
	case strings.HasPrefix(key, "privateKey."):
		return c.privateKeyFact(subject, key)
	case strings.HasPrefix(key, "port."):
		return c.portFact(key)
	case strings.HasPrefix(key, "tool."):
		name := strings.TrimPrefix(key, "tool.")
		if name == "sudo" && os.Geteuid() == 0 {
			return fact(key, true)
		}
		exists := c.probe.ToolExists(name)
		return fact(key, exists)
	default:
		return api.Fact{Key: key, Available: false,
			Error: &api.FactError{Code: api.FactErrUnsupported, Message: "unknown fact key: " + key}}
	}
}

func (c *Collector) collectExactKey(key string) (api.Fact, bool) {
	switch key {
	case "exec.identity.euid":
		return fact(key, os.Geteuid()), true
	case "platform.os":
		return fact(key, normalize.OSFamily(c.probe.OSFamily())), true
	case "platform.distribution":
		id, version := c.probe.Distribution()
		return fact(key, []string{normalize.Distribution(id), version}), true
	case "platform.kernel":
		return fact(key, c.probe.Kernel()), true
	case "platform.kernel.matched":
		return c.verifiedPlatformFact(key), true
	case "platform.architecture":
		return fact(key, normalize.Architecture(c.probe.Architecture())), true
	case "resources.cpu.logical":
		return fact(key, c.probe.CPUCount()), true
	case "resources.memory.totalBytes":
		return fact(key, c.probe.MemoryTotalBytes()), true
	case "resources.systemDisk.totalBytes":
		return fact(key, c.probe.SystemDiskTotalBytes()), true
	case "filesystem.root.availableBytes":
		return fact(key, availableBytes("/")), true
	case "runtime.socket.conflict":
		return fact(key, anyExists("/run/containerd/containerd.sock", "/var/run/docker.sock")), true
	default:
		return api.Fact{}, false
	}
}

func (c *Collector) verifiedPlatformFact(key string) api.Fact {
	id, version := c.probe.Distribution()
	distribution := normalize.Distribution(id)
	architecture := normalize.Architecture(c.probe.Architecture())
	kernel := c.probe.Kernel()
	return api.Fact{
		Key: key, Available: true,
		Value: matchesVerifiedPlatform(c.verifiedPlatforms, distribution, version, architecture, kernel),
		Evidence: []api.Evidence{{
			Kind: "command", Source: c.ID(), Locator: "uname -r; /etc/os-release; uname -m",
			Observed: "distribution=" + distribution + " version=" + version +
				" arch=" + architecture + " kernel=" + kernel,
		}},
	}
}

func matchesVerifiedPlatform(
	platforms []api.VerifiedPlatform,
	distribution, version, architecture, kernel string,
) bool {
	for _, platform := range platforms {
		if platform.Distribution != distribution || platform.Version != version ||
			!containsString(platform.Architectures, architecture) {
			continue
		}
		for _, prefix := range platform.KernelPrefixes {
			if strings.HasPrefix(kernel, prefix) {
				return true
			}
		}
	}
	return false
}

func containsString(values []string, expected string) bool {
	for _, value := range values {
		if value == expected {
			return true
		}
	}
	return false
}

func (c *Collector) semanticPathFact(subject api.Subject, key string) api.Fact {
	if strings.HasPrefix(key, "path./") {
		return c.pathFact(key, strings.TrimPrefix(key, "path."))
	}
	path := resolveSemanticPath(subject, key)
	switch {
	case key == "path.local.accessible":
		return fact(key, c.probe.FileReadable(existingAncestor(path)))
	case key == "path.local.output.noUnknown":
		return fact(key, expectedOutputsAbsent(path, subject.Metadata["expectedOutputs"]))
	case key == "path.local.output.type":
		return fact(key, expectedOutputsRegular(path, subject.Metadata["expectedOutputs"]))
	case key == "path.local.output.noSymlink":
		return fact(key, expectedOutputsNoSymlink(path, subject.Metadata["expectedOutputs"]))
	case strings.HasSuffix(key, ".exists"):
		return fact(key, c.probe.FileExists(path))
	case strings.HasSuffix(key, ".readable"):
		return fact(key, c.probe.FileReadable(path))
	case strings.HasSuffix(key, ".creatable"), strings.HasSuffix(key, ".writable"):
		return fact(key, pathCreatable(path))
	case strings.HasSuffix(key, ".mountReadOnly"):
		return fact(key, mountReadOnly(existingAncestor(path)))
	case strings.HasSuffix(key, ".availableBytes"):
		return fact(key, availableBytes(existingAncestor(path)))
	case strings.HasSuffix(key, ".availableInodes"):
		return fact(key, availableInodes(existingAncestor(path)))
	case strings.HasSuffix(key, ".writeProtected"):
		return writeProtectedFact(key, path)
	default:
		return api.Fact{
			Key:       key,
			Available: false,
			Error:     &api.FactError{Code: api.FactErrUnsupported, Message: "unknown path fact"},
		}
	}
}

func resolveSemanticPath(subject api.Subject, key string) string {
	path := subject.Metadata["workspace"]
	switch {
	case strings.HasPrefix(key, "path.local."):
		path = subject.Metadata["outputDir"]
		if path == "" {
			path = filepath.Join(subject.Metadata["workspace"], "cluster")
		}
	case strings.HasPrefix(key, "path.runc-dest."):
		path = "/usr/local/bin"
	case strings.HasPrefix(key, "path.customRepoCA."), strings.HasPrefix(key, "path.caCert."),
		strings.HasPrefix(key, "path.caKey."):
		path = subject.Metadata["imageRepoCAFile"]
	default:
		return path
	}
	return path
}

func writeProtectedFact(key, path string) api.Fact {
	info, err := os.Stat(path)
	if err != nil {
		return fact(key, false)
	}
	return fact(key, info.Mode().Perm()&writePermissionMask == 0)
}

func expectedOutputPaths(dir, csv string) []string {
	var paths []string
	for _, name := range strings.Split(csv, ",") {
		name = strings.TrimSpace(name)
		if name != "" && filepath.Base(name) == name {
			paths = append(paths, filepath.Join(dir, name))
		}
	}
	return paths
}

func expectedOutputsAbsent(dir, csv string) bool {
	for _, path := range expectedOutputPaths(dir, csv) {
		if _, err := os.Lstat(path); err == nil || !os.IsNotExist(err) {
			return false
		}
	}
	return true
}

func expectedOutputsRegular(dir, csv string) bool {
	for _, path := range expectedOutputPaths(dir, csv) {
		info, err := os.Lstat(path)
		if os.IsNotExist(err) {
			continue
		}
		if err != nil || !info.Mode().IsRegular() {
			return false
		}
	}
	return true
}

func expectedOutputsNoSymlink(dir, csv string) bool {
	for _, path := range expectedOutputPaths(dir, csv) {
		info, err := os.Lstat(path)
		if os.IsNotExist(err) {
			continue
		}
		if err != nil || info.Mode()&os.ModeSymlink != 0 {
			return false
		}
	}
	return true
}

func (c *Collector) sysctlFact(key string) api.Fact {
	path := "/proc/sys/" + strings.ReplaceAll(strings.TrimPrefix(key, "sysctl."), ".", "/")
	b, err := os.ReadFile(path)
	if err != nil {
		return api.Fact{
			Key:       key,
			Available: false,
			Error:     &api.FactError{Code: api.FactErrNotFound, Message: err.Error()},
		}
	}
	n, err := strconv.Atoi(strings.TrimSpace(string(b)))
	if err != nil {
		return api.Fact{
			Key:       key,
			Available: false,
			Error:     &api.FactError{Code: api.FactErrParseError, Message: err.Error()},
		}
	}
	return fact(key, n)
}

func (c *Collector) executableFact(key string) api.Fact {
	exe, err := os.Executable()
	if err != nil {
		return api.Fact{
			Key:       key,
			Available: false,
			Error:     &api.FactError{Code: api.FactErrNotFound, Message: err.Error()},
		}
	}
	switch {
	case strings.HasSuffix(key, ".exists"):
		return fact(key, c.probe.FileExists(exe))
	case strings.HasSuffix(key, ".readable"):
		return fact(key, c.probe.FileReadable(exe))
	case strings.HasSuffix(key, ".arch"):
		return fact(key, normalize.Architecture(runtime.GOARCH))
	default:
		return api.Fact{Key: key, Available: false, Error: &api.FactError{Code: api.FactErrUnsupported}}
	}
}

func (c *Collector) certificateFact(subject api.Subject, key string) api.Fact {
	path := subject.Metadata["imageRepoCAFile"]
	block, failure := readPEMBlock(key, path)
	if failure != nil {
		return *failure
	}
	cert, err := x509.ParseCertificate(block.Bytes)
	if err != nil {
		return fact(key, false)
	}
	if strings.HasSuffix(key, ".parseable") {
		return fact(key, true)
	}
	if strings.HasSuffix(key, ".isCA") {
		return fact(key, cert.IsCA)
	}
	return fact(key, true)
}

func (c *Collector) privateKeyFact(subject api.Subject, key string) api.Fact {
	path := subject.Metadata["imageRepoCAFile"]
	block, failure := readPEMBlock(key, path)
	if failure != nil {
		return *failure
	}
	if strings.HasSuffix(key, ".format") {
		return fact(key, block.Type)
	}
	return api.Fact{
		Key:       key,
		Available: false,
		Error:     &api.FactError{Code: api.FactErrUnsupported, Message: "key matching requires separate key input"},
	}
}

func readPEMBlock(key, path string) (*pem.Block, *api.Fact) {
	b, err := os.ReadFile(path)
	if err != nil {
		failure := api.Fact{
			Key:       key,
			Available: false,
			Error:     &api.FactError{Code: api.FactErrNotFound, Message: err.Error()},
		}
		return nil, &failure
	}
	block, _ := pem.Decode(b)
	if block == nil {
		failure := fact(key, false)
		return nil, &failure
	}
	return block, nil
}

func existingAncestor(path string) string {
	if path == "" {
		return "/"
	}
	for {
		if _, err := os.Stat(path); err == nil {
			return path
		}
		parent := filepath.Dir(path)
		if parent == path {
			return "/"
		}
		path = parent
	}
}

func pathCreatable(path string) bool {
	info, err := os.Stat(existingAncestor(path))
	if err != nil {
		return false
	}
	if os.Geteuid() == 0 {
		return info.IsDir()
	}
	return info.IsDir() && info.Mode().Perm()&ownerWriteBit != 0
}

func availableBytes(path string) int64 {
	var s syscall.Statfs_t
	if syscall.Statfs(path, &s) != nil {
		return 0
	}
	return int64(s.Bavail) * int64(s.Bsize)
}
func availableInodes(path string) int64 {
	var s syscall.Statfs_t
	if syscall.Statfs(path, &s) != nil {
		return 0
	}
	return int64(s.Ffree)
}
func mountReadOnly(path string) bool {
	var s syscall.Statfs_t
	if syscall.Statfs(path, &s) != nil {
		return true
	}
	// POSIX statfs read-only flag is bit 0 on supported Unix targets.
	return s.Flags&1 != 0
}
func anyExists(paths ...string) bool {
	for _, p := range paths {
		if _, err := os.Stat(p); err == nil {
			return true
		}
	}
	return false
}
func nonEmptyDirectory(path string) bool {
	entries, err := os.ReadDir(path)
	return err == nil && len(entries) > 0
}

// pathFact collects existence/readability of a path.
func (c *Collector) pathFact(key, path string) api.Fact {
	exists := c.probe.FileExists(path)
	if !exists {
		return api.Fact{Key: key, Available: true, Value: map[string]any{"exists": false}}
	}
	readable := c.probe.FileReadable(path)
	return api.Fact{Key: key, Available: true, Value: map[string]any{
		"exists":   true,
		"readable": readable,
	}}
}

// portFact collects whether a port is free. Key format: port.<port>.<proto>.free
// or port.<port>.free (defaults to tcp).
func (c *Collector) portFact(key string) api.Fact {
	rest := strings.TrimPrefix(key, "port.")
	parts := strings.SplitN(rest, ".", portKeyParts)
	port, err := normalize.ParsePort(parts[0])
	if err != nil {
		return api.Fact{Key: key, Available: false,
			Error: &api.FactError{Code: api.FactErrParseError, Message: err.Error()}}
	}
	proto := "tcp"
	if len(parts) > 1 {
		tag := strings.TrimSuffix(parts[1], ".free")
		if tag == "udp" || tag == "tcp" {
			proto = tag
		}
	}
	free := c.probe.PortFree(proto, port)
	return api.Fact{Key: key, Available: true, Value: free}
}

// fact is a convenience constructor for an available fact with a value.
func fact(key string, value any) api.Fact {
	return api.Fact{Key: key, Value: value, Available: true,
		Evidence: []api.Evidence{{Kind: "local", Source: "local-host"}}}
}

// errorFacts builds a slice of unavailable facts for every key.
func (c *Collector) errorFacts(keys []string, code, msg string) []api.Fact {
	out := make([]api.Fact, len(keys))
	for i, k := range keys {
		out[i] = api.Fact{Key: k, Available: false, Error: &api.FactError{Code: code, Message: msg}}
	}
	return out
}

// --- default probe using the stdlib + runtime ---

// DefaultProbe returns a HostProbe that reads real host facts using only the
// Go standard library. On Linux it reads /proc/meminfo, /proc/sys/kernel/osrelease
// and uses syscall.Statfs for disk; on non-Linux the OS-family check returns the
// GOOS value and resource facts may be 0 (the platform check then fails as
// expected). No third-party deps (gopsutil) — keeps the SDK self-contained.
func DefaultProbe() HostProbe { return &stdlibProbe{} }

type stdlibProbe struct{}

func (p *stdlibProbe) OSFamily() string { return runtime.GOOS }

func (p *stdlibProbe) Distribution() (string, string) {
	b, err := os.ReadFile("/etc/os-release")
	if err != nil {
		return "", ""
	}
	return parseOSRelease(string(b))
}

func (p *stdlibProbe) Kernel() string {
	// On Linux, read /proc/sys/kernel/osrelease. On other platforms, return
	// empty (the kernel check will be UNKNOWN → BLOCK, which is correct).
	b, err := os.ReadFile("/proc/sys/kernel/osrelease")
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(b))
}

func (p *stdlibProbe) Architecture() string { return runtime.GOARCH }

func (p *stdlibProbe) CPUCount() int { return runtime.NumCPU() }

func (p *stdlibProbe) MemoryTotalBytes() uint64 {
	// Parse /proc/meminfo for MemTotal (kB). Linux-only; returns 0 elsewhere.
	b, err := os.ReadFile("/proc/meminfo")
	if err != nil {
		return 0
	}
	return parseMemTotal(string(b))
}

func (p *stdlibProbe) SystemDiskTotalBytes() uint64 {
	// Use statfs on "/" to get the total block-device capacity. Linux-only.
	return diskTotalBytes("/")
}

func (p *stdlibProbe) FileExists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}

func (p *stdlibProbe) FileReadable(path string) bool {
	f, err := os.Open(path)
	if err != nil {
		return false
	}
	defer func() {
		if err := f.Close(); err != nil {
			return
		}
	}()
	return true
}

func (p *stdlibProbe) PortFree(protocol string, port int) bool {
	return canBind(protocol, port)
}

func (p *stdlibProbe) ToolExists(name string) bool {
	_, err := lookPath(name)
	return err == nil
}

// parseMemTotal extracts MemTotal (in bytes) from /proc/meminfo content.
// The line looks like: "MemTotal:       16384000 kB"
func parseMemTotal(content string) uint64 {
	for _, line := range strings.Split(content, "\n") {
		if !strings.HasPrefix(line, "MemTotal:") {
			continue
		}
		// "MemTotal:       16384000 kB" → extract the number.
		fields := strings.Fields(line)
		if len(fields) < minimumMemoryFields {
			return 0
		}
		var kb uint64
		for _, c := range fields[1] {
			if c >= '0' && c <= '9' {
				kb = kb*decimalBase + uint64(c-'0')
			} else {
				break
			}
		}
		return kb * bytesPerKilobyte // kB → bytes
	}
	return 0
}

// diskTotalBytes returns the total capacity (in bytes) of the filesystem
// containing the given path, using syscall.Statfs. Linux-only.
func diskTotalBytes(path string) uint64 {
	var stat syscall.Statfs_t
	if err := syscall.Statfs(path, &stat); err != nil {
		return 0
	}
	return stat.Blocks * uint64(stat.Bsize)
}

// parseOSRelease extracts ID and VERSION_ID from /etc/os-release content.
func parseOSRelease(content string) (id, version string) {
	for _, line := range strings.Split(content, "\n") {
		if strings.HasPrefix(line, "ID=") {
			id = strings.Trim(strings.TrimPrefix(line, "ID="), `"'`)
		}
		if strings.HasPrefix(line, "VERSION_ID=") {
			version = strings.Trim(strings.TrimPrefix(line, "VERSION_ID="), `"'`)
		}
	}
	return id, version
}
