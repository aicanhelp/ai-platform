#!/bin/bash
# ******************************************************************
# Copyright (c) 2025 Bocloud Technologies Co., Ltd.
# cluster-api-provider-bke is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# ******************************************************************

image=$1
tag=$2

manifests=$(docker buildx imagetools inspect $image:$tag --raw)

imagePath="/tmp/$image"
mkdir -p "$imagePath"

imageName=${image##*/}
echo "package save at $imagePath"
echo "start package image $image:$tag"

for i in $(seq 0 $(echo $manifests | jq '.manifests|length'-1)); do
  arch=$(jq -r .manifests[$i].platform.architecture <<<$manifests)
  digest=$(jq -r .manifests[$i].digest <<<$manifests)

  rm -rf "$imagePath/$imageName.tar.gz-$arch-$tag"

  echo "package $imagePath/$imageName.tar.gz-$arch-$tag ..."

  docker rmi "$image:$arch-$tag"
  docker pull --platform="$arch" "$image@$digest"
  docker tag "$image@$digest" "$image:$arch-$tag"
  docker save "$image:$arch-$tag" -o "$imagePath/$imageName.tar.gz-$arch-$tag"
  docker rmi $image:$arch-$tag

done
