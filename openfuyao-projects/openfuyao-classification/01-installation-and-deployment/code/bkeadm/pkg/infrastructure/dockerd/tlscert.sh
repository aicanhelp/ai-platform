#!/bin/bash
###############################################################
# Copyright (c) 2025 Bocloud Technologies Co., Ltd.
# installer is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
###############################################################

if [ $# != 1 ] ; then
echo "USAGE: $0 [HOST_IP]"
exit 1;
fi

#============================================#
#   下面为证书密钥及相关信息配置，注意修改   #
#============================================#
COUNTRY=CN
PROVINCE=jiangsu
CITY=suzhou
ORGANIZATION=Bocloud
GROUP=OEM
NAME=Bocloud
HOST=$1
SUBJ="/C=$COUNTRY/ST=$PROVINCE/L=$CITY/O=$ORGANIZATION/OU=$GROUP/CN=$HOST"

echo "your host is: $1"

# 1.生成根证书RSA私钥，PASSWORD作为私钥文件的密码
openssl genrsa -out ca-key.pem 4096

# 2.用根证书RSA私钥生成自签名的根证书
openssl req -new -x509 -days 36500 -key ca-key.pem -sha256 -out ca.pem -subj $SUBJ

#============================================#
#          用根证书签发server端证书          #
#============================================#

# 3.生成服务端私钥
openssl genrsa -out server-key.pem 4096

# 4.生成服务端证书请求文件
openssl req -new -sha256 -key server-key.pem -out server.csr -subj "/CN=$HOST"

# 5.使tls连接能通过ip地址方式，绑定IP
echo subjectAltName = IP:127.0.0.1,IP:$HOST > extfile.cnf

# 6.使用根证书签发服务端证书
openssl x509 -req -days 36500 -sha256 -in server.csr -CA ca.pem -CAkey ca-key.pem -CAcreateserial -out server-cert.pem -extfile extfile.cnf

#============================================#
#          用根证书签发client端证书          #
#============================================#

# 7.生成客户端私钥
openssl genrsa -out key.pem 4096

# 8.生成客户端证书请求文件
openssl req -subj '/CN=client' -new -key key.pem -out client.csr

# 9.客户端证书配置文件
echo extendedKeyUsage = clientAuth > extfile.cnf

# 10.使用根证书签发客户端证书
openssl x509 -req -days 36500 -sha256 -in client.csr -CA ca.pem -CAkey ca-key.pem -CAcreateserial -out cert.pem -extfile extfile.cnf

#============================================#
#                    清理                    #
#============================================#
# 删除中间文件
rm -f client.csr server.csr ca.srl extfile.cnf

# 转移目录
mkdir client server
cp {ca,cert,key}.pem client
cp {ca,server-cert,server-key}.pem server
rm {cert,key}.pem

# 设置私钥权限为只读
chmod -f 0400 ca-key.pem server/server-key.pem client/key.pem
