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

function generate_var() {
    local CONFIGMAP_NAME="cm.${OPENFUYAO_VERSION}"
    local NAMESPACE="openfuyao-patch"
    local CM_KEY="$OPENFUYAO_VERSION"
    local TEMP_YAML="/tmp/openfuyao-cm-data.yaml"

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
        "bke-console-website,BKE_CONSOLE_WEBSITE"
        "bke-console-service,BKE_CONSOLE_SERVICE"
        "plugin-management-service,PLUGIN_MANAGEMENT_SERVICE"
        "installer-service,INSTALLER_SERVICE"
        "installer-website,INSTALLER_WEBSITE"
        "oauth-server,OAUTH_SERVER"
        "oauth-webhook,OAUTH_WEBHOOK"
        "user-management-operator,USER_MANAGEMENT_OPERATOR"
    )

    local chart_name prefix image_tag chart_ver
    for comp in "${components[@]}"; do
        IFS=',' read -r chart_name prefix <<< "$comp"
        image_tag=$(yq eval ".openFuyaoCharts[] | select(.name == \"$chart_name\") | .tagVersion // \"\"" "$TEMP_YAML" 2>/dev/null)
        chart_ver=$(yq eval ".openFuyaoCharts[] | select(.name == \"$chart_name\") | .chartVersion // \"\"" "$TEMP_YAML" 2>/dev/null)

        [ -n "$image_tag" ] && eval "${prefix}_IMAGE_TAG=\$image_tag"
        [ -n "$chart_ver" ] && eval "${prefix}_CHART_VERSION=\$chart_ver"
    done

    # === 3. image list ===
    local images=(
        "busy-box,BUSY_BOX"
        "kubectl-openfuyao,KUBECTL_OPENFUYAO"
        "oauth-proxy,OAUTH_PROXY"
    )

    local img_name var_prefix tag_val
    for img in "${images[@]}"; do
        IFS=',' read -r img_name var_prefix <<< "$img"
        tag_val=$(yq eval ".openFuyaoImagesTag[] | select(.name == \"$img_name\") | .version // \"\"" "$TEMP_YAML" 2>/dev/null)
        [ -n "$tag_val" ] && eval "${var_prefix}_IMAGE_TAG=\$tag_val"
    done

    rm -f "$TEMP_YAML"

    # === 4. print ===
    info_log "=== Extracted Tags Summary ==="
    {
        local prefixes=(
            BKE_CONSOLE_WEBSITE
            BKE_CONSOLE_SERVICE
            PLUGIN_MANAGEMENT_SERVICE
            INSTALLER_SERVICE
            INSTALLER_WEBSITE
            OAUTH_SERVER
            OAUTH_WEBHOOK
            USER_MANAGEMENT_OPERATOR
        )
        for p in "${prefixes[@]}"; do
            eval "local img=\${${p}_IMAGE_TAG:-} chart=\${${p}_CHART_VERSION:-}"
            [ -n "$img"  ] && echo "${p}_IMAGE_TAG=$img"
            [ -n "$chart" ] && echo "${p}_CHART_VERSION=$chart"
        done

        local img_vars=(
            BUSY_BOX_IMAGE_TAG
            KUBECTL_OPENFUYAO_IMAGE_TAG
            OAUTH_PROXY_IMAGE_TAG
        )
        for v in "${img_vars[@]}"; do
            eval "local val=\${$v:-}"
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
    local cm_key="$4"
    local temp_yaml="/tmp/cm-$$"

    if ! kubectl get configmap "$cm_name" -n "$ns" --template="{{index .data \"$cm_key\"}}" > "$temp_yaml"; then
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

    echo "$tag"
}

function update_image_tag_from_cm() {
    local IMAGE_NAME_IN_CM="$1"
    local YAML_FILE="$2"
    local CM_NAME="$3"
    local NAMESPACE="$4"
    local CM_KEY="$5"

    if [ -z "$IMAGE_NAME_IN_CM" ] || [ -z "$YAML_FILE" ]; then
        error_log "Usage: update_image_tag_from_cm <image_name_in_cm> <yaml_file>"
        return 1
    fi

    # 1. get image tag from config map
    local NEW_TAG
    if ! NEW_TAG=$(get_cm_image_tag "${CM_NAME}" "${NAMESPACE}" "${IMAGE_NAME_IN_CM}" "${CM_KEY}"); then
        info_log "No update needed for '$IMAGE_NAME_IN_CM' tag as it was not found in ConfigMap."
        return 0
    fi

    info_log "Found tag for '$IMAGE_NAME_IN_CM': $NEW_TAG"

    # 2. update image tag in yaml
    if ! yq eval "(.. | select(has(\"image\") and (.image | contains(\"$IMAGE_NAME_IN_CM\"))).image) |= sub(\":[^:]*$\", \":$NEW_TAG\")" \
               "$YAML_FILE" > "${YAML_FILE}.tmp"; then
        error_log "Failed to update image tag in $YAML_FILE"
        rm -f "${YAML_FILE}.tmp"
        return 1
    fi

    mv -f "${YAML_FILE}.tmp" "$YAML_FILE"
    info_log "Successfully updated '$IMAGE_NAME_IN_CM' image tag to $NEW_TAG in $YAML_FILE"
}