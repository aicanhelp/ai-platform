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
import { useMemo } from 'openinula';
import { useTranslation } from 'react-i18next';
import { getScData, deleteSc } from '@/api/containerApi';
import { Link } from 'inula-router';
import Dayjs from 'dayjs';
import { sorterFirstAlphabet } from '@/tools/utils';
import BreadCrumbCom from '@/components/BreadCrumbCom';
import '@/styles/pages/workload.less';
import ResourceList from '@/components/layouts/ResourceList';
import { sortResourceByTime } from '@/utils/common';
import { containerRouterPrefix } from '@/constant.js';

export default function Sc() {
  const { t, i18n } = useTranslation('storage');

  const scColumns = useMemo(() => [
    {
      title: t('sc.name'),
      key: 'sc_name',
      sorter: (a, b) => sorterFirstAlphabet(a.metadata.name, b.metadata.name),
      render: (_, record) => <Link to={`/${containerRouterPrefix}/storage/sc/${record.metadata.name}`}>{record.metadata.name}</Link>,
    },
    {
      title: t('sc.provisioner'),
      key: 'sc_provider',
      render: (_, record) => record.provisioner ? record.provisioner : '--',
    },
    {
      title: t('sc.reclaimPolicy'),
      key: 'sc_strategy',
      render: (_, record) => record.reclaimPolicy ? record.reclaimPolicy : '--',
    },
    {
      title: t('sc.created'),
      key: 'create_time',
      sorter: (a, b) => sortResourceByTime(a.metadata, b.metadata),
      render: (_, record) =>
        Dayjs(record.metadata.creationTimestamp ? record.metadata.creationTimestamp : '--').format('YYYY-MM-DD HH:mm'),
    },
  ], [t, i18n.language]);

  return <div className="child_content withBread_content">
    <BreadCrumbCom className="create_bread" items={[
      { title: t('module.title'), disabled: true },
      { title: t('sc.title') },
    ]} />
    <ResourceList
      resourceType={t('sc.displayName')}
      columns={scColumns}
      getResourceFn={getScData}
      deleteResourceFn={(record) => deleteSc(record.metadata.name)}
      yamlEditPath={(record) => `/${containerRouterPrefix}/storage/sc/${record.metadata.name}/yaml`}
    />
  </div>;
}
