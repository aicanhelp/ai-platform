#!/bin/bash
# Install tracing module into vLLM-Ascend container's site-packages.
# Runs inside the main container after InitContainer copies files to /tracing-install.

set -e

INSTALL_DIR="/tracing-install"
SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")

# Copy tracing module
cp -r "$INSTALL_DIR/vllm_ascend_tracing" "$SITE_PACKAGES/"

# Handle sitecustomize.py (append if exists to avoid overwriting other hooks)
if [ -f "$SITE_PACKAGES/sitecustomize.py" ]; then
    echo "" >> "$SITE_PACKAGES/sitecustomize.py"
    cat "$INSTALL_DIR/sitecustomize.py" >> "$SITE_PACKAGES/sitecustomize.py"
else
    cp "$INSTALL_DIR/sitecustomize.py" "$SITE_PACKAGES/"
fi

# Copy OTel dependencies (skip if already installed in the base image)
if [ -d "$INSTALL_DIR/deps" ]; then
    cp -rn "$INSTALL_DIR/deps/"* "$SITE_PACKAGES/" 2>/dev/null || true
fi

echo "[tracing] Installed to $SITE_PACKAGES"
