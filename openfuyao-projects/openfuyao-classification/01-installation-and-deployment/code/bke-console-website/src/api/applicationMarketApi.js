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
import { buildQueryString } from '@/tools/utils';
import qs from 'query-string';

// 统一公共前缀
function getClusterName() {
  let clusterName = sessionStorage.getItem('cluster');
  return clusterName;
}

// 获取应用市场的列表信息
export function getHelmChartList(params, currentPage = 2, limit = 10) {
  let appTypeStr = '';
  if (params.sceneList.length) {
    if (params.sceneList.includes('compute-power-engine-plugin')) {
      appTypeStr = '&appType=compute-power-engine-plugin';
      params.sceneList.splice(params.sceneList.indexOf('compute-power-engine-plugin'), 1);
    }
  }
  const sceneQueryString = (params.sceneList[0] === 'undefined' || params.sceneList[0] === 'null' || params.sceneList.length === 0) ? '' : buildQueryString('scene', params.sceneList);
  const sourceQueryString = buildQueryString('repo', params.sourceList);
  const url = `/clusters/${getClusterName()}/rest/marketplace/v1beta1/helm-charts?page=${currentPage}&limit=${limit}${params.chart ? `&chart=${params.chart}` : ''}${params.sortType ? `&sortType=${params.sortType}` : ''}${sceneQueryString}${sourceQueryString}${params.isSelectExtensionComponent ? `&appType=${params.isSelectExtensionComponent}${appTypeStr}` : `${appTypeStr}`}`;
  return GET(url, { 'Content-Type': 'application/json; charset=UTF-8' });
}
// 获取远端仓库中最新版本的Chart信息
export function getHelmChartDetails(chart, currentPage, limit) {
  const url = `/clusters/${getClusterName()}/rest/marketplace/v1beta1/helm-charts?repo=local${chart ? `&chart=${chart}` : ''}&page=${currentPage}&limit=${limit}`;
  return GET(url);
}

// 上传Helm Chart到内置仓库
export function upLoadHelmChart(chart) {
  const url = `/clusters/${getClusterName()}/rest/marketplace/v1beta1/helm-charts`;
  return POST(url, chart, { 'Content-Type': 'multipart/form-data' });
}

// 删除内置仓库中的Helm Chart
export function deleteHelmChart(chart) {
  const url = `/clusters/${getClusterName()}/rest/marketplace/v1beta1/helm-charts/${chart}`;
  return DELETE(url);
}

// 删除内置仓库中指定的Helm Chart版本
export function deleteHelmChartVersion(chart, version) {
  const url = `/clusters/${getClusterName()}/rest/marketplace/v1beta1/helm-charts/${chart}/versions/${version}`;
  return DELETE(url);
}

// 安装helm release到指定的命名空间
export function installReleaseToNameSpace(namespace, release, data) {
  const url = `/clusters/${getClusterName()}/rest/application-management/v1beta1/helm-releases/namespace/${namespace}/release/${release}`;
  return POST(url, data, { 'Content-Type': 'application/json' });
}

// 获取指定仓库，指定 Chart，指定版本的文件
export function getRepoChartVersionFile(repo, chart, version, type) {
  const url = `/clusters/${getClusterName()}/rest/marketplace/v1beta1/helm-repos/${repo}/charts/${chart}/versions/${version}/files?fileType=${type}`;
  return GET(url);
}

// 获取指定仓库中指定 Chart 的所有版本的信息
export function getAllVersionInfo(repo, chart, sortType, sortOrder) {
  const url = `/clusters/${getClusterName()}/rest/marketplace/v1beta1/helm-repos/${repo}/charts/${chart}${sortType ? `?sortType=${sortType}` : ''}${sortOrder ? `&order=${sortOrder}` : ''}`;
  return GET(url);
}

// 获取仓库中指定chart最新版本的信息
export function getAppointChartInfo(chart, sortType, sortOrder) {
  const url = `/clusters/${getClusterName()}/rest/marketplace/v1beta1/helm-charts?chart=${encodeURI(chart)}&repo=local${sortType ? `&sortType=${sortType}` : ''}${sortOrder ? `&order=${sortOrder}` : ''}`;
  return GET(url);
}

// 获取helm仓库信息
export function getHelmRepoList(repo, sortType, currentPage, limit, sortOrder) {
  const url = `/clusters/${getClusterName()}/rest/marketplace/v1beta1/helm-repos?page=${currentPage}&limit=${limit}${repo ? `&repo=${repo}` : ''}${sortType ? `&sortType=${sortType}` : ''}${sortOrder ? `&order=${sortOrder}` : ''}`;
  return GET(url);
}

// 获取远端仓库中最新版本的Chart信息
export function getNewHelmChartList(repo, chart, params, currentPage = 1, limit = 5) {
  let filterStr = '';
  if (params.filterType) {
    params.filterType.forEach(type => {
      filterStr += `&appType=${encodeURIComponent(type)}`;
    });
  }
  const url = `/clusters/${getClusterName()}/rest/marketplace/v1beta1/helm-charts?page=${currentPage}&limit=${limit}${repo ? `&repo=${repo}` : ''}${chart ? `&chart=${chart}` : ''}${params.sortType ? `&sortType=${params.sortType}` : ''}${params.sortOrder ? `&order=${params.sortOrder}` : ''}${filterStr}`;
  return GET(url);
}

// 删除helm仓库
export function deleteHelmRepo(repo) {
  const url = `/clusters/${getClusterName()}/rest/marketplace/v1beta1/helm-repos/${repo}`;
  return DELETE(url);
}

// 更新helm仓库
export function updateHelmRepo(data) {
  const url = `/clusters/${getClusterName()}/rest/marketplace/v1beta1/helm-repos/${data.name}`;
  return PUT(url, data);
}

// 获取远端仓库中最新版本的Chart信息
export function getHelmRepoChartsList(chart, currentPage = 1, limit = 5) {
  const url = `/clusters/${getClusterName()}/rest/marketplace/v1beta1/helm-charts?page=${currentPage}&limit=${limit}${chart ? `&chart=${chart}` : ''}`;
  return GET(url);
}

// 获取指定仓库信息
export function getHelmRepoDetail(repo) {
  const url = `/clusters/${getClusterName()}/rest/marketplace/v1beta1/helm-repos/${repo}`;
  return GET(url);
}

// 同步指定远端仓库和本地缓存的Chart信息（新helm chart，新版本等）
export function synchronizationHelmRepo(repo) {
  const url = `/clusters/${getClusterName()}/rest/marketplace/v1beta1/helm-repos/${repo}/sync`;
  return POST(url);
}

// 添加helm仓库
export function createHelmRepo(data) {
  const url = `/clusters/${getClusterName()}/rest/marketplace/v1beta1/helm-repos`;
  return POST(url, data);
}

// 获取指定版本指定chart的详细信息
export function getAppointVersionChart(repo, charts, version) {
  const url = `/clusters/${getClusterName()}/rest/marketplace/v1beta1/helm-repos/${repo}/charts/${charts}/versions/${version}`;
  return GET(url);
}

// 获取同步仓库状态
export function getsynchronizationStatus(repo) {
  const url = `/clusters/${getClusterName()}/rest/marketplace/v1beta1/helm-repos/${repo}/sync`;
  return GET(url);
}

// 查询仓库中的推荐列表
export function getRecommendHelmChartList(accelerator) {
  let url = '';
  if (accelerator) {
    url = qs.stringifyUrl({ url: `/clusters/${getClusterName()}/rest/marketplace/v1beta1/helm-charts/official-tags`, query: { tag: accelerator } });
  } else {
    url = qs.stringifyUrl({ url: `/clusters/${getClusterName()}/rest/marketplace/v1beta1/helm-charts/official-tags`, query: { tag: 'openfuyao-premium' } });
  }
  return GET(url);
};

// 查询仓库中的扩展组件列表
export function getExtensionHelmChartList(appType) {
  const url = `/clusters/${getClusterName()}/rest/marketplace/v1beta1/helm-charts${appType ? `?appType=${appType}` : ''}`;
  return GET(url);
};

// 包管理查询上传数量
export function getLimitPackageNumber() {
  const url = `/clusters/${getClusterName()}/rest/marketplace/v1beta1/helm-charts/count`;
  return GET(url);
}