#!/bin/bash
###############################################################
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# installer is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
###############################################################

source ./log.sh
source ./consts.sh

# [MOD-1] Reject ConfigMap-sourced tag/chart strings before any shell assignment (defense in depth).
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

function generate_var() {
    CONFIGMAP_NAME="patch-config"
    NAMESPACE="openfuyao-system-controller"
    CM_KEY="patch-data"
    TEMP_YAML="/tmp/openfuyao-cm-data.yaml"

    # === 1. get cm ===
    local ESCAPED_KEY
    ESCAPED_KEY=$(echo "$CM_KEY" | sed 's/\./\\./g')
    if ! kubectl get configmap "$CONFIGMAP_NAME" -n "$NAMESPACE" -o jsonpath="{.data.$ESCAPED_KEY}" > "$TEMP_YAML"; then
        error_log "Failed to fetch ConfigMap $CONFIGMAP_NAME in namespace $NAMESPACE, use default in consts.sh"
        rm -f "$TEMP_YAML"
        return 1
    fi

    # === 2. component list ===
    local components=(
        "harbor,HARBOR"
        "oauth-webhook,OAUTH_WEBHOOK"
        "oauth-server,OAUTH_SERVER"
        "console-website,CONSOLE_WEBSITE"
        "console-service,CONSOLE_SERVICE"
        "monitoring-service,MONITORING_SERVICE"
        "marketplace-service,MARKETPLACE_SERVICE"
        "application-management-service,APPLICATION_MANAGEMENT_SERVICE"
        "plugin-management-service,PLUGIN_MANAGEMENT_SERVICE"
        "user-management-operator,USER_MANAGEMENT_OPERATOR"
        "web-terminal-service,WEB_TERMINAL_SERVICE"
        "installer-service,INSTALLER_SERVICE"
        "installer-website,INSTALLER_WEBSITE"
    )

    local chart_name prefix image_tag chart_ver
    for comp in "${components[@]}"; do
        IFS=',' read -r chart_name prefix <<< "$comp"
        image_tag=$(yq eval ".openFuyaoCharts[] | select(.name == \"$chart_name\") | .tagVersion // \"\"" "$TEMP_YAML" 2>/dev/null)
        chart_ver=$(yq eval ".openFuyaoCharts[] | select(.name == \"$chart_name\") | .chartVersion // \"\"" "$TEMP_YAML" 2>/dev/null)

        # [MOD-2] Use printf -v / indirect expansion instead of eval (ConfigMap → shell injection fix).
        if [ -n "$image_tag" ]; then
            if validate_openfuyao_tag_value "$image_tag" "image_tag for ${chart_name}"; then
                printf -v "${prefix}_IMAGE_TAG" '%s' "$image_tag"
            else
                image_tag=""
            fi
        fi
        if [ -n "$chart_ver" ]; then
            if validate_openfuyao_tag_value "$chart_ver" "chartVersion for ${chart_name}"; then
                printf -v "${prefix}_CHART_VERSION" '%s' "$chart_ver"
            else
                chart_ver=""
            fi
        fi
    done

    # === 3. image list ===
    local images=(
        "busy-box,BUSY_BOX"
        "kubectl-openfuyao,KUBECTL_OPENFUYAO"
        "oauth-proxy,OAUTH_PROXY"
        "openfuyao-system-controller,OPENFUYAO"
    )

    local img_name var_prefix tag_val
    for img in "${images[@]}"; do
        IFS=',' read -r img_name var_prefix <<< "$img"
        tag_val=$(yq eval ".openFuyaoImagesTag[] | select(.name == \"$img_name\") | .version // \"\"" "$TEMP_YAML" 2>/dev/null)
        if [ -n "$tag_val" ]; then
            if validate_openfuyao_tag_value "$tag_val" "image version for ${img_name}"; then
                printf -v "${var_prefix}_IMAGE_TAG" '%s' "$tag_val"
            fi
        fi
    done

    rm -f "$TEMP_YAML"

    # === 4. print ===
    info_log "=== Extracted Tags Summary ==="
    {
        local prefixes=(
            HARBOR
            OAUTH_WEBHOOK
            OAUTH_SERVER
            CONSOLE_WEBSITE
            CONSOLE_SERVICE
            MONITORING_SERVICE
            MARKETPLACE_SERVICE
            APPLICATION_MANAGEMENT_SERVICE
            PLUGIN_MANAGEMENT_SERVICE
            USER_MANAGEMENT_OPERATOR
            WEB_TERMINAL_SERVICE
            INSTALLER_SERVICE
            INSTALLER_WEBSITE
        )
        for p in "${prefixes[@]}"; do
            local img_var="${p}_IMAGE_TAG" chart_var="${p}_CHART_VERSION"
            local img="${!img_var:-}" chart="${!chart_var:-}"
            [ -n "$img"  ] && echo "${p}_IMAGE_TAG=$img"
            [ -n "$chart" ] && echo "${p}_CHART_VERSION=$chart"
        done

        local img_vars=(
            BUSY_BOX_IMAGE_TAG
            KUBECTL_OPENFUYAO_IMAGE_TAG
            OAUTH_PROXY_IMAGE_TAG
            OPENFUYAO_IMAGE_TAG
        )
        for v in "${img_vars[@]}"; do
            local val="${!v:-}"
            [ -n "$val" ] && echo "$v=$val"
        done
    } | while IFS= read -r line; do
        info_log "$line"
    done
    info_log "=== End of Tags Summary ==="
}

function get_cm_image_tag() {
    local cm_name="$1"
    local ns="$2"
    local img_name="$3"
    local temp_yaml="/tmp/cm-$$"

    if ! kubectl get configmap "$cm_name" -n "$ns" -o go-template='{{index .data "patch-data"}}' > "$temp_yaml"; then
        info_log "Failed to fetch ConfigMap '$cm_name' in namespace '$ns'"
        rm -f "$temp_yaml"
        return 1
    fi

    if [ ! -s "$temp_yaml" ]; then
        info_log "ConfigMap key 'patch-data' is empty"
        rm -f "$temp_yaml"
        return 1
    fi

    local tag
    tag=$(yq eval ".repos[].subImages[].images[] | select(.name == \"$img_name\") | .tag[0] // \"\"" "$temp_yaml" 2>/dev/null)
    rm -f "$temp_yaml"

    if [ -z "$tag" ]; then
        info_log "No tag found for image '$img_name'"
        return 1
    fi

    # [MOD-3] Validate before returning tag used by update_image_tag_from_cm / yq.
    if ! validate_openfuyao_tag_value "$tag" "image tag for ${img_name}"; then
        return 1
    fi

    echo "$tag"
}

function update_image_tag_from_cm() {
    local IMAGE_NAME_IN_CM="$1"
    local YAML_FILE="$2"
    local CM_NAME="${3:-patch-config}"
    local NAMESPACE="${4:-openfuyao-system-controller}"

    if [ -z "$IMAGE_NAME_IN_CM" ] || [ -z "$YAML_FILE" ]; then
        error_log "Usage: update_image_tag_from_cm <image_name_in_cm> <yaml_file>"
        return
    fi

    # 1. get image tag from config map
    local NEW_TAG
    if ! NEW_TAG=$(get_cm_image_tag "$CM_NAME" "$NAMESPACE" "$IMAGE_NAME_IN_CM"); then
        info_log "No update needed for '$IMAGE_NAME_IN_CM' tag as it was not found in ConfigMap."
        return
    fi

    info_log "Found tag for '$IMAGE_NAME_IN_CM': $NEW_TAG"

    # 2. update image tag in yaml
    if ! yq eval "(.. | select(has(\"image\") and (.image | contains(\"$IMAGE_NAME_IN_CM\"))).image) |= sub(\":[^:]*$\", \":$NEW_TAG\")" "$YAML_FILE" > "${YAML_FILE}.tmp"; then
        error_log "Failed to update image tag in $YAML_FILE"
        rm -f "${YAML_FILE}.tmp"
        return
    fi

    mv -f "${YAML_FILE}.tmp" "$YAML_FILE"
    info_log "Successfully updated '$IMAGE_NAME_IN_CM' image tag to $NEW_TAG in $YAML_FILE"
}