/**
 *  Copyright (c) 2024 Huawei Technologies Co., Ltd.
 *  openFuyao is licensed under Mulan PSL v2.
 *  You can use this software according to the terms and conditions of the Mulan PSL v2.
 *  You may obtain a copy of Mulan PSL v2 at:
  
 *       http://license.coscl.org.cn/MulanPSL2
  
 *   THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 *   EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 *   MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 *   See the Mulan PSL v2 for more details.
 */

import { GET, POST, PATCH } from '@/api/request';
import qs from 'query-string';

export function logout() {
  return POST('/rest/auth/logout', {});
}

export function updatePassword(username, originalPassword, newPassword) {
  return POST('/password/modify', {
    username,
    'original_password': originalPassword,
    'new_password': newPassword,
  });
}

export function logoutSSO() {
  return POST('/oauth2/auth/logout/fuyaoPasswordProvider?redirect_uri=%2Frest%2Fauth%2Flogin', {});
}

export function getUserName() {
  const url = '/rest/auth/user';
  return GET(url);
}

export function getUserInfoByname(name) {
  const url = `/rest/user/v1/users/${name}/user-descriptions`;
  return GET(url);
}

export function editUserInfo(name, data) {
  const url = qs.stringifyUrl({
    url: `/rest/user/v1/users/${name}/user-descriptions`,
  });
  return PATCH(url, data, { 'Content-Type': 'application/json' });
}

export function getUserAuth(name) {
  const url = `/rest/user/v1/users/${name}/check-permission`;
  return GET(url);
}

