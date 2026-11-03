/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */

/**
 * 与 i18next 语言对齐的 dayjs 全局 locale（单例 dayjs 的 locale 状态）。
 * 在 i18n 初始化与 languageChanged 时调用，避免各业务文件分散 Dayjs.locale。
 */
import dayjs from 'dayjs';
import 'dayjs/locale/zh-cn';
import 'dayjs/locale/en';

/**
 * @param {string} [lng] i18next 当前语言，如 zh、en
 * @returns {'zh-cn' | 'en'} dayjs 已注册的语言 key
 */
export function getDayjsLocaleForI18nLng(lng) {
  if (!lng || typeof lng !== 'string') { return 'zh-cn'; }
  const lower = lng.toLowerCase();
  if (lower === 'en' || lower.startsWith('en-')) { return 'en'; }
  return 'zh-cn';
}

/**
 * 根据 i18n 语言设置全局 dayjs locale（与 Ant Design DatePicker 等共用同一 dayjs 实例行为）
 * @param {string} [lng]
 */
export function syncDayjsLocale(lng) {
  dayjs.locale(getDayjsLocaleForI18nLng(lng));
}
