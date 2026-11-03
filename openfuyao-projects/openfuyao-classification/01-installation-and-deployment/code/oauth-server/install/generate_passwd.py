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

import base64
import argparse
import hashlib
import os

def encrypt_password(raw_password, salt_length=16, iterations=100000, key_length=64, encrypt_method='sha256'):
    # 生成随机的盐值
    salt = os.urandom(salt_length)
    # 使用 PBKDF2 算法生成密文
    encrypted_password = hashlib.pbkdf2_hmac(encrypt_method, raw_password.encode('utf-8'), salt, iterations, dklen=key_length)
    # 将盐值和密文合并并编码为 Base64 字符串
    encrypted_data = salt + encrypted_password
    encrypted_password_base64 = base64.b64encode(base64.b64encode(encrypted_data))
    # 返回加密后的密码
    return encrypted_password_base64

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Encrypt a password using PBKDF2 and Base64 encoding.")
    parser.add_argument("--password", required=True, help="The password to be encrypted.")

    args = parser.parse_args()
    raw_password = args.password

    encrypted_password = encrypt_password(raw_password)
    print(encrypted_password.decode('utf-8'))

