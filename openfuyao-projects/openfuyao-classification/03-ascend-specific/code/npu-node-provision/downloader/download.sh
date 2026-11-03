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

VERSION=$1
json_dir=$(dirname "$0")
json_file="$json_dir/software/$VERSION/config.json"
echo "file: $json_file"
component_keys=$(jq -r 'keys[]' "$json_file")
arch=$(uname -m)
http_referer="https://www.hiascend.com/"

for component in $component_keys; do
  enable=$(jq -r ".${component}.enable" "$json_file")
  url_x86=$(jq -r ".${component}.url_x86" "$json_file")
  url_arm=$(jq -r ".${component}.url_arm" "$json_file")


  if [ -z "$enable" ] || [ -z "$url_x86" ] || [ -z "$url_arm" ]; then
    echo "Error: Missing 'enable' or 'url_x86' or 'url_arm' for $component in the configuration file."
    exit 1
  fi

  if [ "$enable" == "true" ]; then

    if [ "$arch" == "x86_64" ]; then
      echo "System architecture is x86_64. Downloading from $url_x86 for $component"
      download_url=$url_x86
    elif [ "$arch" == "aarch64" ]; then
      echo "System architecture is aarch64. Downloading from $url_arm for $component"
      download_url=$url_arm
    else
      echo "Unsupported architecture: $arch"
      exit 1
    fi

    install_path="/tmp/${component}"
    mkdir -p "$install_path"


    if ! wget --referer=$http_referer -O "$install_path/$(basename "$download_url")" "$download_url"; then
      echo "Error: Failed to download file from $download_url for $component."
      exit 1
    fi

    echo "Download completed successfully for $component."

    echo "Extracting $component..."
    unzip "$install_path/$(basename "$download_url")" -d "$install_path"
    rm -f "$install_path/$(basename "$download_url")"

    echo "$component extracted to $install_path"
  else
    echo "Download not enabled for $component."
  fi
done



