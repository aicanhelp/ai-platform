package utils

import (
	"fmt"
	"os"
	"regexp"
	"strconv"
	"strings"
	"time"

	. "github.com/onsi/gomega"
)

// MooncakeMasterMetrics is parsed from mooncake-master "Master Metrics:" log lines.
type MooncakeMasterMetrics struct {
	MemUsed          string
	MemCapacity      string
	MemUsedBytes     float64
	MemCapacityBytes float64
	Keys             int
	GetOK            int
	GetTotal         int
	GetBatchOK       int
	GetBatchReq      int
	PutEndOK         int
	PutEndTotal      int
	PutStartOK       int
	Line             string
}

var (
	reMooncakeMasterMetrics = regexp.MustCompile(`Master Metrics:\s*Mem Storage:\s*([^|]+?)\s*/\s*([^|(]+)`)
	reMooncakeKeys          = regexp.MustCompile(`Keys:\s*(\d+)`)
	reMooncakeGet           = regexp.MustCompile(`(?:^|[\s|])Get=(\d+)/(\d+)`)
	reMooncakePutEnd        = regexp.MustCompile(`PutEnd=(\d+)/(\d+)`)
	reMooncakePutStart      = regexp.MustCompile(`PutStart=(\d+)/(\d+)`)
	reMooncakeGetBatch      = regexp.MustCompile(`Get:\(Req=(\d+)/\d+/(\d+),\s*Item=(\d+)/(\d+)\)`)
)

// MooncakeE2EMemCapacity optionally pins an exact capacity substring (e.g. "80.00 GB").
// When unset, baseline only requires Mem capacity (denominator) > 0.
func MooncakeE2EMemCapacity() string {
	return strings.TrimSpace(os.Getenv("MOONCAKE_E2E_MEM_CAPACITY"))
}

// parseMooncakeStorageBytes parses sizes like "0 B", "1.23 GB", "80.00 GB".
func parseMooncakeStorageBytes(s string) (float64, bool) {
	s = strings.TrimSpace(s)
	if idx := strings.Index(s, "("); idx >= 0 {
		s = strings.TrimSpace(s[:idx])
	}
	fields := strings.Fields(s)
	if len(fields) < 2 {
		return 0, false
	}
	val, err := strconv.ParseFloat(fields[0], 64)
	if err != nil || val < 0 {
		return 0, false
	}
	switch strings.ToUpper(fields[1]) {
	case "B", "BYTE", "BYTES":
		return val, true
	case "KB", "KIB":
		return val * 1024, true
	case "MB", "MIB":
		return val * 1024 * 1024, true
	case "GB", "GIB":
		return val * 1024 * 1024 * 1024, true
	case "TB", "TIB":
		return val * 1024 * 1024 * 1024 * 1024, true
	default:
		return 0, false
	}
}

// MooncakeE2ELongPrompt builds a long shared prefix for KV warm-up (FIT 48–51).
func MooncakeE2ELongPrompt(tag string) string {
	const chunk = "Mooncake KV cache e2e shared prefix. "
	var b strings.Builder
	for b.Len() < 3600 {
		b.WriteString(chunk)
		b.WriteString(tag)
		b.WriteByte(' ')
	}
	return b.String()
}

// ParseMooncakeMasterMetricsLine extracts counters from a mooncake-master metrics log line.
func ParseMooncakeMasterMetricsLine(line string) (MooncakeMasterMetrics, bool) {
	if !strings.Contains(line, "Master Metrics") {
		return MooncakeMasterMetrics{}, false
	}
	m := reMooncakeMasterMetrics.FindStringSubmatch(line)
	if len(m) < 3 {
		return MooncakeMasterMetrics{}, false
	}
	out := MooncakeMasterMetrics{
		MemUsed:     strings.TrimSpace(m[1]),
		MemCapacity: strings.TrimSpace(m[2]),
		Line:        strings.TrimSpace(line),
	}
	if km := reMooncakeKeys.FindStringSubmatch(line); len(km) == 2 {
		out.Keys, _ = strconv.Atoi(km[1])
	}
	if gm := reMooncakeGet.FindStringSubmatch(line); len(gm) == 3 {
		out.GetOK, _ = strconv.Atoi(gm[1])
		out.GetTotal, _ = strconv.Atoi(gm[2])
	}
	if pm := reMooncakePutEnd.FindStringSubmatch(line); len(pm) == 3 {
		out.PutEndOK, _ = strconv.Atoi(pm[1])
		out.PutEndTotal, _ = strconv.Atoi(pm[2])
	}
	if sm := reMooncakePutStart.FindStringSubmatch(line); len(sm) == 3 {
		out.PutStartOK, _ = strconv.Atoi(sm[1])
	}
	if bm := reMooncakeGetBatch.FindStringSubmatch(line); len(bm) == 5 {
		out.GetBatchOK, _ = strconv.Atoi(bm[1])
		out.GetBatchReq, _ = strconv.Atoi(bm[2])
		if itemOK, _ := strconv.Atoi(bm[3]); itemOK > out.GetBatchOK {
			out.GetBatchOK = itemOK
		}
	}
	if used, ok := parseMooncakeStorageBytes(out.MemUsed); ok {
		out.MemUsedBytes = used
	}
	if cap, ok := parseMooncakeStorageBytes(out.MemCapacity); ok {
		out.MemCapacityBytes = cap
	}
	return out, true
}

func scrapeMooncakeMasterMetrics(fitID, ns, deploy string, tail int, logEachScrape bool) (MooncakeMasterMetrics, error) {
	kc := localKubeconfig()
	cmd := fmt.Sprintf("KUBECONFIG=%q kubectl logs -n %q deploy/%q --tail=%d 2>/dev/null", kc, ns, deploy, tail)
	out, stderr, err := ExecLocal(cmd)
	if err != nil {
		return MooncakeMasterMetrics{}, fmt.Errorf("kubectl logs deploy/%s: %w stderr=%s", deploy, err, stderr)
	}
	var latest MooncakeMasterMetrics
	found := false
	for _, line := range strings.Split(out, "\n") {
		if m, ok := ParseMooncakeMasterMetricsLine(line); ok {
			latest = m
			found = true
		}
	}
	if !found {
		return MooncakeMasterMetrics{}, fmt.Errorf("no Master Metrics line in last %d log lines for deploy/%s", tail, deploy)
	}
	if logEachScrape {
		logMooncakeMasterMetrics(fitID, deploy, latest)
	}
	return latest, nil
}

func logMooncakeMasterMetrics(fitID, deploy string, latest MooncakeMasterMetrics) {
	LogStep(fitID, "mooncake-master %s metrics: Mem=%s/%s Keys=%d Get=%d/%d batchGetOK=%d PutEnd=%d/%d PutStart=%d",
		deploy, latest.MemUsed, latest.MemCapacity, latest.Keys, latest.GetOK, latest.GetTotal,
		latest.GetBatchOK, latest.PutEndOK, latest.PutEndTotal, latest.PutStartOK)
	if strings.TrimSpace(latest.Line) != "" {
		LogStep(fitID, "mooncake-master metrics line: %s", latest.Line)
	}
}

func mooncakeMemCapacityPositive(m MooncakeMasterMetrics) bool {
	if m.MemCapacityBytes <= 0 {
		return false
	}
	if want := MooncakeE2EMemCapacity(); want != "" {
		return strings.Contains(strings.TrimSpace(m.MemCapacity), want)
	}
	return true
}

func mooncakeMemUsedPositive(m MooncakeMasterMetrics) bool {
	return m.MemUsedBytes > 0
}

// AssertMooncakeMasterMetricsBaseline ensures master reports Mem Storage with capacity (denominator) > 0.
func AssertMooncakeMasterMetricsBaseline(fitID, ns, owner string) MooncakeMasterMetrics {
	deploy := BridgeWorkloadDeployName(owner, "mooncake-master")
	phase := fmt.Sprintf("mooncake-master baseline metrics owner=%s", owner)
	LogPhaseBegin(fitID, phase)
	defer LogPhaseEnd(fitID, phase)
	var snap MooncakeMasterMetrics
	Eventually(func(g Gomega) {
		m, err := scrapeMooncakeMasterMetrics(fitID, ns, deploy, 200, false)
		g.Expect(err).NotTo(HaveOccurred())
		g.Expect(m.Line).To(ContainSubstring("Master Metrics"))
		g.Expect(m.Line).To(ContainSubstring("Mem Storage"))
		if !mooncakeMemCapacityPositive(m) {
			want := MooncakeE2EMemCapacity()
			if want == "" {
				want = "capacity > 0"
			}
			LogPollThrottled(fitID, "mooncake-cap-"+owner, defaultPollLogInterval,
				"Mem capacity %q (bytes=%.0f) not ready (want %s)", m.MemCapacity, m.MemCapacityBytes, want)
		}
		g.Expect(mooncakeMemCapacityPositive(m)).To(BeTrue(),
			"Mem capacity %q must be > 0 (set MOONCAKE_E2E_MEM_CAPACITY to pin exact value)", m.MemCapacity)
		snap = m
	}).WithTimeout(30 * time.Second).WithPolling(5 * time.Second).Should(Succeed())
	logMooncakeMasterMetrics(fitID, deploy, snap)
	LogStep(fitID, "mooncake-master baseline OK Mem=%s/%s", snap.MemUsed, snap.MemCapacity)
	return snap
}

// AssertMooncakeKVCacheHitAfterWarmup polls mooncake-master until Mem used (numerator) > 0 after two inference passes.
func AssertMooncakeKVCacheHitAfterWarmup(fitID, ns, owner string, baseline MooncakeMasterMetrics) {
	deploy := BridgeWorkloadDeployName(owner, "mooncake-master")
	phase := fmt.Sprintf("mooncake-master KV cache hit owner=%s", owner)
	LogPhaseBegin(fitID, phase)
	defer LogPhaseEnd(fitID, phase)

	Eventually(func(g Gomega) {
		cur, err := scrapeMooncakeMasterMetrics(fitID, ns, deploy, 400, false)
		g.Expect(err).NotTo(HaveOccurred())
		g.Expect(mooncakeMemCapacityPositive(cur)).To(BeTrue())

		memUsedHit := mooncakeMemUsedPositive(cur)
		getHit := cur.GetOK > 0 || cur.GetBatchOK > 0
		getIncreased := cur.GetOK > baseline.GetOK || cur.GetBatchOK > baseline.GetBatchOK

		LogPollThrottled(fitID, "mooncake-kv-hit-"+owner, defaultPollLogInterval,
			"mooncake KV: Mem used=%s (%.0f B) cap=%s Get=%d/%d (base %d) batchGetOK=%d Keys=%d memUsedHit=%v getHit=%v",
			cur.MemUsed, cur.MemUsedBytes, cur.MemCapacity,
			cur.GetOK, cur.GetTotal, baseline.GetOK, cur.GetBatchOK, cur.Keys,
			memUsedHit, getHit || getIncreased)

		g.Expect(memUsedHit).To(BeTrue(),
			"KV cache should use Mem Storage (numerator > 0); baseline used=%.0f B now %s (%.0f B); line=%s",
			baseline.MemUsedBytes, cur.MemUsed, cur.MemUsedBytes, cur.Line)
	}).WithTimeout(5 * time.Minute).WithPolling(10 * time.Second).Should(Succeed())

	final, err := scrapeMooncakeMasterMetrics(fitID, ns, deploy, 400, true)
	Expect(err).NotTo(HaveOccurred())
	LogStep(fitID, "mooncake KV cache hit verified: Mem used %s→%s (%.0f B) Get %d→%d Keys %d→%d",
		baseline.MemUsed, final.MemUsed, final.MemUsedBytes,
		baseline.GetOK, final.GetOK, baseline.Keys, final.Keys)
}

// AssertInferenceEngineKVCacheLogHint optionally checks engine pod logs for prefix/cache hints (FIT CSV).
func AssertInferenceEngineKVCacheLogHint(fitID, ns, owner, engineDeploy string) {
	if engineDeploy == "" {
		return
	}
	phase := fmt.Sprintf("engine KV log hint deploy/%s", engineDeploy)
	LogPhaseBegin(fitID, phase)
	defer LogPhaseEnd(fitID, phase)
	kc := localKubeconfig()
	cmd := fmt.Sprintf(
		`KUBECONFIG=%q kubectl logs -n %q deploy/%q --tail=600 2>/dev/null | grep -iE 'prefix|cache|mooncake|kv.*hit' | tail -5`,
		kc, ns, engineDeploy)
	out, _, _ := ExecLocal(cmd)
	if strings.TrimSpace(out) == "" {
		LogStep(fitID, "engine %s: no prefix/cache hit log lines in tail (non-fatal; mooncake-master Get counters are primary)", engineDeploy)
		return
	}
	LogStep(fitID, "engine %s KV/prefix log hints:\n%s", engineDeploy, out)
}
