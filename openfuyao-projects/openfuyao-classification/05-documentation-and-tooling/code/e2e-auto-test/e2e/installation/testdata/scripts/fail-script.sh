#!/bin/sh

echo "========================================="
echo "Pre-Script: fail-script.sh (intentionally failing)"
echo "NODE_IP: ${NODE_IP}"
echo "PKG_VERSION: ${PKG_VERSION}"
echo "========================================="

echo "Attempting to download non-existent package version ${PKG_VERSION}..."
echo "ERROR: Package version ${PKG_VERSION} not found in repository"

exit 1
