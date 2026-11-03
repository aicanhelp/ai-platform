#!/bin/bash
###############################################################
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# installer is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
###############################################################
source ../../utils/log.sh "harbor"
source ../../utils/consts.sh

sleep 10

IMAGES=(
    docker.io/goharbor/harbor-exporter:v${HARBOR_VERSION}
    docker.io/goharbor/redis-photon:v${HARBOR_VERSION}
    docker.io/goharbor/trivy-adapter-photon:v${HARBOR_VERSION}
    docker.io/goharbor/harbor-registryctl:v${HARBOR_VERSION}
    docker.io/goharbor/registry-photon:v${HARBOR_VERSION}
    docker.io/goharbor/nginx-photon:v${HARBOR_VERSION}
    docker.io/goharbor/harbor-log:v${HARBOR_VERSION}
    docker.io/goharbor/harbor-jobservice:v${HARBOR_VERSION}
    docker.io/goharbor/harbor-core:v${HARBOR_VERSION}
    docker.io/goharbor/harbor-portal:v${HARBOR_VERSION}
    docker.io/goharbor/harbor-db:v${HARBOR_VERSION}
    docker.io/goharbor/prepare:v${HARBOR_VERSION}
)

if sudo which nerdctl; then
    if [ $(nerdctl ps | wc -l) -le 1 ]; then
        fatal_log "failed to install harbor"
    fi

    for index in $(seq 1 $(nerdctl ps | wc -l)); do
        if [ ${index} -eq 1 ]; then
            continue
        fi

        RAW_DATA=$(sudo nerdctl ps | awk "NR==${index}")
        FLAG="0"
        for image_name in ${IMAGES[@]} ; do
            if [[ "${RAW_DATA}" =~ "${image_name}" ]] && [[ "${RAW_DATA}" =~ "Up" ]]; then
                FLAG="1"
                continue
            fi
        done

        if [ "$FLAG" == "0" ]; then
            error_log "The harbor status is abnormal."
            fatal_log "${RAW_DATA}"
        fi
    done
    info_log "Successfully installed harbor"
    exit 0
fi
