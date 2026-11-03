#!/bin/bash

image=$1
tag=$2

docker rmi -f  `docker images | grep '<none>' | awk '{print $3}'`

manifests=$(docker buildx imagetools inspect $image:$tag --raw)

for i in `seq 0 $(echo $manifests | jq '.manifests|length'-1)`; do

   arch=$(jq -r .manifests[$i].platform.architecture <<< $manifests)
   digest=$(jq -r .manifests[$i].digest <<< $manifests)

   rm -rf /tmp/addons.tar.gz-$tag-$arch
   echo "package /tmp/addons.tar.gz-$tag-$arch ..."

   docker rmi $image:$tag-$arch
   docker pull --platform=$arch $image@$digest
   docker tag $image@$digest $image:$tag-$arch
   docker save $image:$tag-$arch -o /tmp/addons.tar.gz-$tag-$arch

   docker rmi $image:$tag-$arch

done