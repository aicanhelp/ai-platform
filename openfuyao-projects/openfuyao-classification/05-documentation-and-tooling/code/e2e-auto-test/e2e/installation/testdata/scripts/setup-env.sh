#!/bin/sh
set -e

echo "========================================="
echo "Pre-Script: setup-env.sh"
echo "NODE_IP: ${NODE_IP}"
echo "HTTP_REPO: ${HTTP_REPO}"
echo "========================================="

MARKER_DIR="/tmp/prescript-markers"
mkdir -p ${MARKER_DIR}

echo "Setting up environment..."
echo "Using repo: ${HTTP_REPO}"

echo "NODE_IP=${NODE_IP}" > ${MARKER_DIR}/setup-env.env
echo "HTTP_REPO=${HTTP_REPO}" >> ${MARKER_DIR}/setup-env.env
echo "EXECUTED_AT=$(date +%s)" >> ${MARKER_DIR}/setup-env.env

touch ${MARKER_DIR}/setup-env.done
echo "setup-env.sh completed successfully"
