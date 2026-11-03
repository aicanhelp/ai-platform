/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import { GET, POST, PUT, DELETE, PATCH } from '@/api/request';
import { solveParams } from '@/tools/utils';
import qs from 'query-string';

// 统一公共前缀
function commonApiPrefix() {
  let paramObj = {
    logUrl: `/rest/logging/v1/`,
  };
  return paramObj;
}


/**
 * 获取日志采集源
 */
export function getLogCollectionData() {
  const url = qs.stringifyUrl(
    { url: `${commonApiPrefix().logUrl}sources` }
  );
  return GET(url);
}
/**
 * 获取日志采集任务
 */
export function getLogJobData() {
  const url = qs.stringifyUrl(
    { url: `${commonApiPrefix().logUrl}configmap` }
  );
  return GET(url);
}
/**
 * 更新采集任务
 */
export function addLogJob(data) {
  const url = qs.stringifyUrl(
    { url: `${commonApiPrefix().logUrl}update-config` }
  );
  return PATCH(url, data);
}
/**
 * 删除采集任务
 */
export function deleteLogJob(data) {
  const url = qs.stringifyUrl(
    { url: `${commonApiPrefix().logUrl}delete-job/${data}` }
  );
  return DELETE(url);
}
/**
 * 获取日志采集任务保存时长
 */
export function getLogJobTime() {
  const url = qs.stringifyUrl(
    { url: `${commonApiPrefix().logUrl}secret` }
  );
  return GET(url);
}
/**
 * 获取日志设置告警列表
 */
export function getLogAlertData() {
  const url = qs.stringifyUrl(
    { url: `${commonApiPrefix().logUrl}rules` }
  );
  return GET(url);
}
/**
 * 获取日志查询options 命名空间 pod 容器
 */
export function getLogOptions(name, type) {
  const url = qs.stringifyUrl(
    { url: `${commonApiPrefix().logUrl}resource-list` + `${type && name ? `?${type}=${name}` : ''}` }
  );
  return GET(url);
}
/**
 * 获取日志所有的条件
 */
export function getLogOptionsAll(namespace, podName, container) {
  const url = qs.stringifyUrl(
    {
      url: `${commonApiPrefix().logUrl}resource-list`, query: solveParams({
        namespace,
        podName,
        container,
      }),
    }, { skipNull: true },
  );
  return GET(url);
}

/**
 * 获取日志查询options filename
 */
export function getLogFilenameOptions() {
  const url = qs.stringifyUrl(
    { url: `${commonApiPrefix().logUrl}filelist` }
  );
  return GET(url);
}
/**
 * 获取日志查询列表 按资源类型查询
 */
export function getLogSearchData(param) {
  let url = qs.stringifyUrl(
    { url: `${commonApiPrefix().logUrl}` + `${param.namespace ? `namespaces/${param.namespace}/` : ''}` + `${param.pod ? `pods/${param.pod}/` : ''}` + `${param.container ? `containers/${param.container}/` : ''}` + `logs?` + `${param.start ? `&startTime=${param.start}` : ''}` + `${param.end ? `&endTime=${param.end}` : ''}` + `${param.level ? `&level=${param.level}` : ''}` + `${param.keyword ? `&keyword=${param.keyword}` : ''}` + `${param.size ? `&size=${param.size}` : ''}` }
  );
  if (url.indexOf('?') + 1 === '&') {
    url.replace('&', '');
  }
  return GET(url);
}
/**
 * 获取日志查询列表 按采集源查询
 */
export function getLogSearchDataBycollecting(param) {
  let url = qs.stringifyUrl(
    { url: `${commonApiPrefix().logUrl}filenames/` + `logs?` + `${param.filename ? `&fileName=${param.filename}` : ''}` + `${param.start ? `&startTime=${param.start}` : ''}` + `${param.end ? `&endTime=${param.end}` : ''}` + `${param.level ? `&level=${param.level}` : ''}` + `${param.keyword ? `&keyword=${param.keyword}` : ''}` + `${param.size ? `&size=${param.size}` : ''}` }
  );
  if (url.indexOf('?') + 1 === '&') {
    url.replace('&', '');
  }
  return GET(url);
}
/**
 * 获取日志上下文查询列表
 */
export function getLogDetail(param) {
  let url = qs.stringifyUrl(
    { url: `${commonApiPrefix().logUrl}detail?` + `${param.filename ? `fileName=${param.filename}` : ''}` + `${param.level ? `&level=${param.level}` : ''}` + `${param.keyword ? `&keyword=${param.keyword}` : ''}` + `${param.timestamp ? `&timestamp=${param.timestamp}` : ''}` + `${param.timePeriod ? `&timePeriod=${param.timePeriod}` : ''}` + `${param.size ? `&size=${param.size}` : ''}` }
  );
  return GET(url);
}

/**
 * 查询上下文
 */
export function getLogSearch({ fileName, timestamp, size, timePeriod, keyword, level }) {
  const url = qs.stringifyUrl(
    {
      url: `${commonApiPrefix().logUrl}detail?`, query:
        solveParams({ fileName, timestamp, size, timePeriod, keyword, level }),
    }
    , {
      skipNull: true,
    }
  );
  return GET(url);
}

