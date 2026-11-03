/**
 *  Copyright (c) 2024 Huawei Technologies Co., Ltd.
 *  openFuyao is licensed under Mulan PSL v2.
 *  You can use this software according to the terms and conditions of the Mulan PSL v2.
 *  You may obtain a copy of Mulan PSL v2 at:
 *
 *       http://license.coscl.org.cn/MulanPSL2
 *
 *   THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 *   EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 *   MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 *   See the Mulan PSL v2 for more details.
 */
import { useMemo } from 'openinula';
import BreadCrumbCom from '@/components/BreadCrumbCom';
import { Link } from 'inula-router';
import { listLimitRangeLists, deleteLimitRange } from '@/api/containerApi';
import Dayjs from 'dayjs';
import ResourceList from '@/components/layouts/ResourceList';
import { sorterFirstAlphabet } from '@/tools/utils';
import '@/styles/pages/workload.less';
import { containerRouterPrefix } from '@/constant.js';
import { useTranslation } from 'react-i18next';

export default function LimitRange() {
  const { t, i18n } = useTranslation('namespace');

  const limitRangeColumns = useMemo(() => [
    {
      title: t('limitRange.nameLabel'),
      key: 'limitRange_name',
      sorter: (a, b) => sorterFirstAlphabet(a.metadata.name, b.metadata.name),
      render: (_, record) => (
        <Link to={`/${containerRouterPrefix}/namespace/limitRange/${record.metadata.namespace}/${record.metadata.name}`}>
          {record.metadata.name}
        </Link>
      ),
    },
    {
      title: t('limitRange.namespace'),
      key: 'limitRange_namespace',
      render: (_, record) => record.metadata.namespace,
    },
    {
      title: t('limitRange.createdAt'),
      key: 'limitRange_created_time',
      sorter: (a, b) => Dayjs(a.metadata.creationTimestamp) - Dayjs(b.metadata.creationTimestamp),
      render: (_, record) => Dayjs(record.metadata.creationTimestamp).format('YYYY-MM-DD HH:mm'),
    },
  ], [t, i18n.language]);

  return (
    <div className="child_content withBread_content">
      <BreadCrumbCom
        className="create_bread"
        items={[
          { title: t('module.title'), path: `/${containerRouterPrefix}/namespace/limitRange`, disabled: true },
          { title: 'LimitRange', path: '/' },
        ]}
      />
      <ResourceList
        resourceType="LimitRange"
        columns={limitRangeColumns}
        getResourceFn={listLimitRangeLists}
        deleteResourceFn={(record) => deleteLimitRange(record.metadata.namespace, record.metadata.name)}
        yamlEditPath={(record) => `/${containerRouterPrefix}/namespace/limitRange/${record.metadata.namespace}/${record.metadata.name}/yaml`}
      />
    </div>
  );
}
