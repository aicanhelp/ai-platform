/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import { GET, POST, DELETE } from '@/api/request';
import qs from 'query-string';
import { buildQueryString } from '@/tools/utils';
// 统一公共前缀
function commonApiPrefix() {
  let clusterName = sessionStorage.getItem('cluster');
  let paramObj = {
    alertUrl: `/clusters/${clusterName}/rest/alert/api/v2/`,
    lokiUrl: `/clusters/${clusterName}/rest/application-management/v1beta1/helm-releases/releasestatus/`,
  };
  return paramObj;
}

/**
 * 获取告警选择options
 */
export function getAlertOptions(string) {
  const url = qs.stringifyUrl({ url: `${commonApiPrefix().alertUrl}alerts` + `${string ? `?${string}` : ''}` });
  return GET(url);
}

/**
 * 创建静默告警
 */
export function creatSilentAlert(data) {
  const url = qs.stringifyUrl({ url: `${commonApiPrefix().alertUrl}silences` });
  return POST(url, data, { 'Content-Type': 'application/json' });
}

/**
 * 获取静默告警列表
 */
export function getSilentList(string) {
  const url = qs.stringifyUrl({ url: `${commonApiPrefix().alertUrl}silences` + `${string ? `?${string}` : ''}` });
  if (url.indexOf('?') + 1 === '&') {
    url.replace('&', '');
  }
  return GET(url);
}

/**
 * 获取静默告警详情
 */
export function getSilentListDetail(id) {
  const url = qs.stringifyUrl({ url: `${commonApiPrefix().alertUrl}silence/${id}` });
  if (url.indexOf('?') + 1 === '&') {
    url.replace('&', '');
  }
  return GET(url);
}

/**
 * 删除静默告警
 */
export function deleteSilentAlert(id) {
  const url = qs.stringifyUrl({ url: `${commonApiPrefix().alertUrl}silence/${id}` });
  return DELETE(url);
}

/**
 * getlock
 */
export function getLoki() {
  const url = qs.stringifyUrl({ url: `${commonApiPrefix().lokiUrl}logging-package` });
  return GET(url);
}
