package compliance_operator

import (
	"fmt"
	"sort"
	"strings"

	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/executor"
	"gitcode.com/openFuyao/e2e-auto-test/e2e/installation/compliance-operator/utils"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

// DFX-04 complements the product-side FIT-17 Bats unit test by auditing the
// artifacts produced by the real ordered hardening and rollback lifecycle.
// Passwords are sent to remote audit processes through stdin and are never
// embedded in commands or printed in Ginkgo output.
func defineSensitiveAuditSpecs() {
	Describe("[DFX][DFX-04] Real artifact sensitive information audit",
		Ordered, Label("DFX", "DFX-04", "sensitive_info_audit", "P1", "26.09", "online"), func() {
			It("[DFX][DFX-04] should keep all real hardening and rollback artifacts free of sensitive content", func() {
				passwords := auditPasswordInput()

				By("审查 entry node 的编排日志、节点结果日志和 Markdown/JSON 集群报告")
				entryResult := runSensitiveAudit(entry, entryAuditCommand(), passwords)
				assertSensitiveAuditPassed("entry node", entryResult)

				By("逐节点审查真实备份 changes.json，确保所有节点均有审查素材且无敏感内容")
				for _, node := range cfg.Nodes {
					nodeResult := runSensitiveAudit(nodeExecs[node.IP], nodeAuditCommand(), passwords)
					assertSensitiveAuditPassed(fmt.Sprintf("node %s", node.IP), nodeResult)
				}
			})
		})
}

func auditPasswordInput() string {
	unique := map[string]struct{}{}
	if cfg.Entry.Password != "" {
		unique[cfg.Entry.Password] = struct{}{}
	}
	for _, node := range cfg.Nodes {
		if node.Password != "" {
			unique[node.Password] = struct{}{}
		}
	}
	passwords := make([]string, 0, len(unique))
	for password := range unique {
		passwords = append(passwords, password)
	}
	sort.Strings(passwords)
	return strings.Join(passwords, "\n") + "\n"
}

func runSensitiveAudit(e *executor.SSHExecutor, command, passwords string) *executor.ExecResult {
	result, err := e.ExecWithInput(command, passwords)
	if result == nil {
		Expect(err).NotTo(HaveOccurred(), "敏感信息审查 SSH 通道应可用")
		Fail("敏感信息审查未返回执行结果")
	}
	return result
}

func assertSensitiveAuditPassed(scope string, result *executor.ExecResult) {
	GinkgoWriter.Printf("[%s 敏感信息审查]\n%s\n", scope, strings.TrimSpace(result.Stdout))
	Expect(result.ExitCode).To(BeZero(),
		"%s 敏感信息审查应通过（仅输出状态与命中文件名，不输出敏感值）\nstdout:\n%s\nstderr:\n%s",
		scope, result.Stdout, result.Stderr)
	Expect(result.Stdout).To(ContainSubstring("AUDIT_STATUS=PASS"), "%s 应输出通过标记", scope)
}

func entryAuditCommand() string {
	rules := make([]string, 0, len(utils.SecretRuleIDs))
	for _, ruleID := range utils.SecretRuleIDs {
		rules = append(rules, auditShellLiteral(ruleID))
	}
	return fmt.Sprintf(`set -u
umask 077
patterns="$(mktemp)"
files="$(mktemp)"
trap 'rm -f "$patterns" "$files"' EXIT
cat > "$patterns"
{
  yq -r '.cluster.ssh.default_password // empty' %s 2>/dev/null || true
  yq -r '.cluster.nodes[]? | .password // empty' %s 2>/dev/null || true
} >> "$patterns"
sed -i '/^$/d' "$patterns"
sort -u "$patterns" -o "$patterns"

find /tmp -mindepth 2 -maxdepth 2 -type f \( \
  -path '/tmp/cluster-harden-results-*/*' -o \
  -path '/tmp/cluster-rollback-results-*/*' \
\) -print 2>/dev/null >> "$files" || true
find %s -type f \( -name 'cluster-harden-report-*.md' -o -name 'cluster-harden-report-*.json' \) \
  -print 2>/dev/null >> "$files" || true
find %s -maxdepth 1 -type f \( -name 'cluster-harden-*.log' -o -name 'cluster-rollback-*.log' \) \
  -print 2>/dev/null >> "$files" || true
sort -u "$files" -o "$files"

rc=0
require_files() {
  label="$1"
  expression="$2"
  count="$(grep -Ec "$expression" "$files" 2>/dev/null || true)"
  echo "ARTIFACT_${label}=${count}"
  if [ "$count" -eq 0 ]; then rc=1; fi
}
require_files HARDEN_NODE_LOG '/cluster-harden-results-[^/]+/harden-[^/]+[.]log$'
require_files ROLLBACK_NODE_LOG '/cluster-rollback-results-[^/]+/rollback-[^/]+[.]log$'
require_files HARDEN_ORCHESTRATION_LOG '/cluster-harden-[^/]+[.]log$'
require_files ROLLBACK_ORCHESTRATION_LOG '/cluster-rollback-[^/]+[.]log$'
require_files MARKDOWN_REPORT '/cluster-harden-report-[^/]+[.]md$'
require_files JSON_REPORT '/cluster-harden-report-[^/]+[.]json$'

credential_re='BEGIN[[:space:]]+CERTIFICATE|BEGIN[[:space:]]+([A-Z0-9]+[[:space:]]+)*PRIVATE[[:space:]]+KEY|client-certificate-data|client-key-data|MIIC|MIIE|MIIF'
token_re='(^|[^[:alnum:]_])(token|password|passwd|secret)[[:space:]]*[:=][[:space:]]*["]?[[:alnum:]_.~+/=-]{8,}'
while IFS= read -r file; do
  [ -f "$file" ] || continue
  if grep -IqE "$credential_re" "$file"; then
    echo "SENSITIVE_MARKER_FILE=$file"
    rc=1
  fi
  if grep -IqE "$token_re" "$file"; then
    echo "TOKEN_LIKE_VALUE_FILE=$file"
    rc=1
  fi
  if [ -s "$patterns" ] && grep -IqF -f "$patterns" "$file"; then
    echo "PASSWORD_VALUE_FILE=$file"
    rc=1
  fi
done < "$files"

for rule_id in %s; do
  found=0
  while IFS= read -r file; do
    if grep -IqF "$rule_id" "$file"; then found=1; break; fi
  done < "$files"
  echo "RULE_${rule_id}=${found}"
  if [ "$found" -ne 1 ]; then rc=1; fi
done

if [ "$rc" -eq 0 ]; then echo 'AUDIT_STATUS=PASS'; else echo 'AUDIT_STATUS=FAIL'; fi
exit "$rc"`,
		auditShellLiteral(cfg.ClusterYAML), auditShellLiteral(cfg.ClusterYAML),
		auditShellLiteral(cfg.ReportDir), auditShellLiteral(cfg.LogDir), strings.Join(rules, " "))
}

func nodeAuditCommand() string {
	return fmt.Sprintf(`set -u
umask 077
patterns="$(mktemp)"
files="$(mktemp)"
trap 'rm -f "$patterns" "$files"' EXIT
cat > "$patterns"
sed -i '/^$/d' "$patterns"
sort -u "$patterns" -o "$patterns"

find %s -type f -name changes.json -print 2>/dev/null > "$files" || true
count="$(grep -c . "$files" 2>/dev/null || true)"
echo "ARTIFACT_CHANGES_JSON=${count}"
rc=0
if [ "$count" -eq 0 ]; then rc=1; fi

credential_re='BEGIN[[:space:]]+CERTIFICATE|BEGIN[[:space:]]+([A-Z0-9]+[[:space:]]+)*PRIVATE[[:space:]]+KEY|client-certificate-data|client-key-data|MIIC|MIIE|MIIF'
token_re='(^|[^[:alnum:]_])(token|password|passwd|secret)[[:space:]]*[:=][[:space:]]*["]?[[:alnum:]_.~+/=-]{8,}'
while IFS= read -r file; do
  [ -f "$file" ] || continue
  if grep -IqE "$credential_re" "$file"; then
    echo "SENSITIVE_MARKER_FILE=$file"
    rc=1
  fi
  if grep -IqE "$token_re" "$file"; then
    echo "TOKEN_LIKE_VALUE_FILE=$file"
    rc=1
  fi
  if [ -s "$patterns" ] && grep -IqF -f "$patterns" "$file"; then
    echo "PASSWORD_VALUE_FILE=$file"
    rc=1
  fi
done < "$files"

if [ "$rc" -eq 0 ]; then echo 'AUDIT_STATUS=PASS'; else echo 'AUDIT_STATUS=FAIL'; fi
exit "$rc"`, auditShellLiteral(cfg.BackupDir))
}

func auditShellLiteral(value string) string {
	return "'" + strings.ReplaceAll(value, "'", "'\"'\"'") + "'"
}
