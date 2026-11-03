#!/bin/bash

set -e

# the ascend deployer is already there
cd /root/ascend-deployer/ascend_deployer || exit

cp -af /root/large_scale_inventory.ini /root/ascend-deployer/ascend_deployer/large_scale_inventory.ini

rm -f /root/large_scale_inventory.ini

architecture=$(uname -m)

if [ "$architecture" = "aarch64" ]; then
    bash /root/ascend-deployer/ascend_deployer/start_download.sh --os-list=OpenEuler_22.03LTS_aarch64
fi

if [ "$architecture" = "x86_64" ]; then
    bash /root/ascend-deployer/ascend_deployer/start_download.sh --os-list=OpenEuler_22.03LTS_x86_64
fi

python3 large_scale_inventory.py --install=sys_pak,python --skip_check << EOF
y
EOF


