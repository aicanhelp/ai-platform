#!/bin/bash
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# openFuyao is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

COUNTRY=CN
PROVINCE=shanghai
CITY=shanghai
ORGANIZATION=openfuyao
GROUP=OEM
CN=openfuyao.com
SUBJ="/C=$COUNTRY/ST=$PROVINCE/L=$CITY/O=$ORGANIZATION/OU=$GROUP/CN=$CN"

# 1.生成根证书RSA私钥
openssl genrsa -out deploy.bocloud.k8s.key 4096

# 2.用根证书RSA私钥生成自签名的根证书
openssl req -newkey rsa:4096 -nodes -sha256 -keyout deploy.bocloud.k8s.key --addext "subjectAltName=DNS:*.bocloud.k8s\
,IP:0.0.0.0,IP:127.0.0.1" -x509 -days 36500 -out deploy.bocloud.k8s.crt -subj $SUBJ

# 设置私钥权限为只读
chmod -f 0400 deploy.bocloud.k8s.key
