#!/bin/sh
set -e

echo "========================================="
echo "Pre-Script: install-agent.sh"
echo "NODE_IP: ${NODE_IP}"
echo "HTTP_REPO: ${HTTP_REPO}"
echo "========================================="

MARKER_DIR="/tmp/prescript-markers"
mkdir -p ${MARKER_DIR}

echo "Installing agent on node ${NODE_IP}..."
echo "Downloading from: ${HTTP_REPO}/agent"

echo "NODE_IP=${NODE_IP}" > ${MARKER_DIR}/install-agent.env
echo "HTTP_REPO=${HTTP_REPO}" >> ${MARKER_DIR}/install-agent.env
echo "EXECUTED_AT=$(date +%s)" >> ${MARKER_DIR}/install-agent.env

touch ${MARKER_DIR}/install-agent.done
echo "install-agent.sh completed successfully"
