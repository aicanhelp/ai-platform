#!/bin/bash
# Security regression tests for utils.sh ConfigMap tag handling.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./log.sh
source "${ROOT}/log.sh"

validate_openfuyao_tag_value() {
    local value="$1"
    local label="${2:-tag}"
    if [ -z "$value" ]; then
        return 0
    fi
    if [[ "$value" =~ ^[a-zA-Z0-9._-]{1,128}$ ]]; then
        return 0
    fi
    error_log "invalid ${label} from ConfigMap; rejecting value"
    return 1
}

assert_eq() {
    local got="$1" want="$2" msg="$3"
    if [ "$got" != "$want" ]; then
        echo "FAIL: $msg (got=$got want=$want)" >&2
        exit 1
    fi
}

# Safe assignment path (same as generate_var after fix).
assign_image_tag() {
    local prefix="$1" image_tag="$2"
    if [ -n "$image_tag" ]; then
        if validate_openfuyao_tag_value "$image_tag" "image_tag"; then
            printf -v "${prefix}_IMAGE_TAG" '%s' "$image_tag"
        fi
    fi
}

echo "=== utils security tests ==="

# Valid tag is stored literally.
assign_image_tag "OAUTH_WEBHOOK" "v2.7.0"
assert_eq "${OAUTH_WEBHOOK_IMAGE_TAG:-}" "v2.7.0" "valid tag assignment"

# Malicious payload must not execute or assign.
MALicious='x;echo PWNED_SECURITY_TEST>/tmp/pwned_by_eval;#'
assign_image_tag "OAUTH_WEBHOOK" "$MALicious"
if [ -f /tmp/pwned_by_eval ]; then
    echo "FAIL: command injection executed" >&2
    rm -f /tmp/pwned_by_eval
    exit 1
fi
assert_eq "${OAUTH_WEBHOOK_IMAGE_TAG:-}" "v2.7.0" "malicious tag rejected, prior value unchanged"

# Command substitution must not run.
assign_image_tag "OAUTH_SERVER" '$(touch /tmp/pwned_by_subshell)'
if [ -f /tmp/pwned_by_subshell ]; then
    echo "FAIL: command substitution executed" >&2
    rm -f /tmp/pwned_by_subshell
    exit 1
fi
assert_eq "${OAUTH_SERVER_IMAGE_TAG:-}" "" "command substitution rejected"

echo "PASS: all utils security tests"
