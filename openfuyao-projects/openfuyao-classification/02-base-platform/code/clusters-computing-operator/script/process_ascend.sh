#!/usr/bin/bash

# Quit when error
set -e

cd /root || exit

scp /root/install_ascend.sh root@"$TARGET_IP":/root/install_ascend.sh || exit
echo "$INI_FILE" > large_scale_inventory.ini
scp /root/large_scale_inventory.ini root@"$TARGET_IP":/root/large_scale_inventory.ini || exit

ssh root@"$TARGET_IP" "bash /root/install_ascend.sh" || exit


