#!/bin/sh
set -e

echo "========================================="
echo "Pre-Script: init-os.sh"
echo "NODE_IP: ${NODE_IP}"
echo "HTTP_REPO: ${HTTP_REPO}"
echo "========================================="

MARKER_DIR="/tmp/prescript-markers"
mkdir -p ${MARKER_DIR}

echo "Configuring OS settings..."
echo "Using repo: ${HTTP_REPO}"

echo "NODE_IP=${NODE_IP}" > ${MARKER_DIR}/init-os.env
echo "HTTP_REPO=${HTTP_REPO}" >> ${MARKER_DIR}/init-os.env
echo "EXECUTED_AT=$(date +%s)" >> ${MARKER_DIR}/init-os.env

touch ${MARKER_DIR}/init-os.done
echo "init-os.sh completed successfully"
