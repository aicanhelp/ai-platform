/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */

import { DEFAULT_LOCALE } from './index';

export const getCurrentLanguage = () => {
  return localStorage.getItem('locale') || DEFAULT_LOCALE;
};

/**
 * Hardcoded overrides for plugin display names when displayNameI18n is not set.
 * TODO: Remove once all ConsolePlugin CRs have displayNameI18n populated.
 * @type {Array<{ key: string, 'zh-CN'?: string, 'en-US'?: string }>}
 */
export const CONSOLE_PLUGIN_DISPLAY_NAME_OVERRIDES = [
  { key: 'colocation', 'zh-CN': '在离线混部', 'en-US': 'Colocation' },
  { key: 'colocation_overview', 'zh-CN': '概览', 'en-US': 'Overview' },
  { key: 'colocation_configuration', 'zh-CN': '混部策略配置', 'en-US': 'Configuration' },
  { key: 'colocation_monitoring', 'zh-CN': '混部监控', 'en-US': 'Colocation Monitoring' },
  { key: 'cluster', 'zh-CN': '集群生命周期管理', 'en-US': 'Cluster Lifecycle Management' },
  { key: 'log', 'zh-CN': '日志', 'en-US': 'Logs' },
  { key: 'log_query', 'zh-CN': '日志查询', 'en-US': 'Query' },
  { key: 'log_config', 'zh-CN': '日志配置', 'en-US': 'Configuration' },
  { key: 'monitoringDashboard', 'zh-CN': '自定义监控看板', 'en-US': 'Custom Monitoring' },
  { key: 'numa', 'zh-CN': 'NUMA亲和调度', 'en-US': 'NUMA Affinity' },
  { key: 'numa_overview', 'zh-CN': '概览', 'en-US': 'Overview' },
  { key: 'numa_configuration', 'zh-CN': '亲和策略配置', 'en-US': 'Affinity Policy Configuration' },
  { key: 'numa_monitoring', 'zh-CN': '集群NUMA监控', 'en-US': 'Cluster NUMA Monitoring' },
  { key: 'ray', 'zh-CN': 'openFuyao Ray', 'en-US': 'openFuyao Ray' },
  { key: 'ray_overview', 'zh-CN': '概览', 'en-US': 'Overview' },
  { key: 'ray_cluster', 'zh-CN': 'RayCluster', 'en-US': 'RayCluster' },
  { key: 'ray_service', 'zh-CN': 'RayService', 'en-US': 'RayService' },
  { key: 'ray_job', 'zh-CN': 'RayJob', 'en-US': 'RayJob' },
];

const getHardcodedConsolePluginDisplayName = (item, language) => {
  if (language === 'zh-CN') {
    return '';
  }
  const lookupKey = item.displayName;
  if (!lookupKey) {
    return '';
  }
  const entry = CONSOLE_PLUGIN_DISPLAY_NAME_OVERRIDES.find(({ 'zh-CN': zhCN }) => zhCN === lookupKey);
  return entry?.['en-US'] || '';
};

/**
 * Resolve console plugin display name for the current locale.
 * Prefers displayNameI18n[locale], then hardcoded overrides, then displayName / pluginName / pageName.
 * @param {{ displayName?: string, displayNameI18n?: Record<string, string>, pluginName?: string, pageName?: string }} item
 * @param {string} [language]
 * @returns {string}
 */
export const getConsolePluginDisplayName = (item, language = getCurrentLanguage()) => {
  if (!item) {
    return '';
  }
  const localized = item.displayNameI18n?.[language];
  if (localized) {
    return localized;
  }
  const hardcoded = getHardcodedConsolePluginDisplayName(item, language);
  if (hardcoded) {
    return hardcoded;
  }
  return item.displayName || item.pluginName || item.pageName || '';
};
