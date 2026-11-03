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

import { useMemo, useStore } from 'openinula';
import { Space } from 'antd';
import ApplicationListLayout from '@/components/layouts/ApplicationListLayout';
import { useTranslation } from 'react-i18next';

export default function ExtendIndex() {
  const consolePluginStore = useStore('consolePlugins');
  const { t } = useTranslation('application');

  // Precompute enabled/disabled plugin counts per release (avoid 2x filter per row)
  const pluginCounts = useMemo(() => {
    const map = new Map();
    consolePluginStore.$s.consolePlugins.forEach(p => {
      if (!map.has(p.release)) {
        map.set(p.release, { enabled: 0, disabled: 0 });
      }
      if (p.enabled) {
        map.get(p.release).enabled++;
      } else {
        map.get(p.release).disabled++;
      }
    });
    return map;
  }, [consolePluginStore.$s.consolePlugins]);

  // Extra column for extension: plugin toggle
  const extraColumns = [
    {
      title: t('extend.startStop'),
      key: 'enablement',
      render: (_, record) => {
        const counts = pluginCounts.get(record.name);
        return (
          <Space size="middle">
            <span>
              {counts ? counts.enabled : 0}
              /
              {counts ? counts.disabled : 0}
            </span>
          </Space>
        );
      },
    },
  ];

  return (
    <ApplicationListLayout
      type="extension"
      extraColumns={extraColumns}
      pluginStore={consolePluginStore}
    />
  );
}