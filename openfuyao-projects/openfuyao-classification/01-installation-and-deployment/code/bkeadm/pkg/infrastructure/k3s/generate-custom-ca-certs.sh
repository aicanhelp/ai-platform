#!/usr/bin/env bash

# ==============================================================================
# This file is based on the K3s project's CA generation script
# Original file: https://github.com/k3s-io/k3s/blob/master/contrib/util/generate-custom-ca-certs.sh
#
# Original copyright belongs to the K3s project and is licensed under Apache License 2.0
# Copyright (c) 2019-2023 Rancher Labs, Inc.
#
# Modifications we made to the original script (if any):
# [Describe any modifications here, or remove this section if none]
#
# Original License text:
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


set -e
umask 027

TIMESTAMP=$(date +%s)
PRODUCT="${PRODUCT:-k3s}"
DATA_DIR="${DATA_DIR:-/var/lib/rancher/${PRODUCT}}"

if type -t openssl-3 &>/dev/null; then
  OPENSSL=openssl-3
else
  OPENSSL=openssl
fi

echo "Using $(type -p ${OPENSSL}): $(${OPENSSL} version)"

if ! ${OPENSSL} ecparam -name prime256v1 -genkey -noout -out /dev/null &>/dev/null; then
  echo "openssl not found or missing Elliptic Curve (ecparam) support."
  exit 1
fi

${OPENSSL} version | grep -qF 'OpenSSL 3' && OPENSSL_GENRSA_FLAGS=-traditional

mkdir -p "${DATA_DIR}/server/tls/etcd"
cd "${DATA_DIR}/server/tls"

# Set up temporary openssl configuration
mkdir -p ".ca/certs"
trap "rm -rf .ca" EXIT
touch .ca/index
openssl rand -hex 8 > .ca/serial
cat >.ca/config <<'EOF'
[ca]
default_ca = ca_default
[ca_default]
dir = ./.ca
database = $dir/index
serial = $dir/serial
new_certs_dir = $dir/certs
default_md = sha256
policy = policy_anything
[policy_anything]
commonName = supplied
[req]
distinguished_name = req_distinguished_name
[req_distinguished_name]
[v3_ca]
subjectKeyIdentifier = hash
authorityKeyIdentifier = keyid:always
basicConstraints = critical, CA:true
keyUsage = critical, digitalSignature, keyEncipherment, keyCertSign
EOF

# Don't overwrite the service account issuer key; we pass the key into both the controller-manager
# and the apiserver instead of passing a cert list into the apiserver, so there's no facility for
# rotation and things will get very angry if all the SA keys are invalidated.
if [[ -e service.key ]]; then
  echo "Generating additional Kubernetes service account issuer RSA key"
  OLD_SERVICE_KEY="$(cat service.key)"
else
  echo "Generating Kubernetes service account issuer RSA key"
fi
${OPENSSL} genrsa ${OPENSSL_GENRSA_FLAGS:-} -out service.key 2048
echo "${OLD_SERVICE_KEY}" >> service.key

# Use existing root CA if present
if [[ -e root-ca.pem ]]; then
  echo "Using existing root certificate"
else
  echo "Generating root certificate authority RSA key and certificate"
  ${OPENSSL} genrsa ${OPENSSL_GENRSA_FLAGS:-} -out root-ca.key 4096
  ${OPENSSL} req -x509 -new -nodes -sha256 -days 7300 \
                 -subj "/CN=${PRODUCT}-root-ca@${TIMESTAMP}" \
                 -key root-ca.key \
                 -out root-ca.pem \
                 -config .ca/config \
                 -extensions v3_ca
fi
cat root-ca.pem > root-ca.crt

# Use existing intermediate CA if present
if [[ -e intermediate-ca.pem ]]; then
  echo "Using existing intermediate certificate"
else
  if [[ ! -e root-ca.key ]]; then
    echo "Cannot generate intermediate certificate without root certificate private key"
    exit 1
  fi

  echo "Generating intermediate certificate authority RSA key and certificate"
  ${OPENSSL} genrsa ${OPENSSL_GENRSA_FLAGS:-} -out intermediate-ca.key 4096
  ${OPENSSL} req -new -nodes \
                 -subj "/CN=${PRODUCT}-intermediate-ca@${TIMESTAMP}" \
                 -key intermediate-ca.key |
  ${OPENSSL} ca  -batch -notext -days 3700 \
                 -in /dev/stdin \
                 -out intermediate-ca.pem \
                 -keyfile root-ca.key \
                 -cert root-ca.pem \
                 -config .ca/config \
                 -extensions v3_ca
fi
cat intermediate-ca.pem root-ca.pem > intermediate-ca.crt

if [[ ! -e intermediate-ca.key ]]; then
  echo "Cannot generate leaf certificates without intermediate certificate private key"
  exit 1
fi

# Generate new leaf CAs for all the control-plane and etcd components
for TYPE in client server request-header etcd/peer etcd/server; do
  CERT_NAME="${PRODUCT}-$(echo ${TYPE} | tr / -)-ca"
  echo "Generating ${CERT_NAME} leaf certificate authority EC key and certificate"
  ${OPENSSL} ecparam -name prime256v1 -genkey -noout -out ${TYPE}-ca.key
  ${OPENSSL} req -new -nodes \
                 -subj "/CN=${CERT_NAME}@${TIMESTAMP}" \
                 -key ${TYPE}-ca.key |
  ${OPENSSL} ca  -batch -notext -days 3700 \
                 -in /dev/stdin \
                 -out ${TYPE}-ca.pem \
                 -keyfile intermediate-ca.key \
                 -cert intermediate-ca.pem \
                 -config .ca/config \
                 -extensions v3_ca
  cat ${TYPE}-ca.pem \
      intermediate-ca.pem \
      root-ca.pem > ${TYPE}-ca.crt
done

echo
echo "CA certificate generation complete. Required files are now present in: ${DATA_DIR}/server/tls"
echo "For security purposes, you should make a secure copy of the following files and remove them from cluster members:"
ls ${DATA_DIR}/server/tls/root-ca.* ${DATA_DIR}/server/tls/intermediate-ca.* | xargs -n1 echo -e "\t"

if [ "${DATA_DIR}" != "/var/lib/rancher/${PRODUCT}" ]; then
  echo
  echo "To update certificates on an existing cluster, you may now run:"
  echo "    k3s certificate rotate-ca --path=${DATA_DIR}/server"
fi
