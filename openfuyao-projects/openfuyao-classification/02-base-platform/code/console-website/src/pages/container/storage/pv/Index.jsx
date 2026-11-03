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
import { useMemo, useCallback } from 'openinula';
import { useTranslation } from 'react-i18next';
import { getPvData, deletePv } from '@/api/containerApi';
import { Link } from 'inula-router';
import Dayjs from 'dayjs';
import { sorterFirstAlphabet, sorterStorage } from '@/tools/utils';
import BreadCrumbCom from '@/components/BreadCrumbCom';
import '@/styles/pages/workload.less';
import ResourceList from '@/components/layouts/ResourceList';
import { sortResourceByTime } from '@/utils/common';
import { containerRouterPrefix } from '@/constant.js';

export default function Pv() {
  const { t, i18n } = useTranslation('storage');

  const validateDelete = useCallback((record) => (
    record.status?.phase === 'Bound' ? t('pv.deleteWarning', { name: record.metadata.name }) : null
  ), [t, i18n.language]);

  const pvColumns = useMemo(() => [
    {
      title: t('pv.name'),
      key: 'pv_name',
      sorter: (a, b) => sorterFirstAlphabet(a.metadata.name, b.metadata.name),
      render: (_, record) => <Link to={`/${containerRouterPrefix}/storage/pv/${record.metadata.name}`}>{record.metadata.name}</Link>,
    },
    {
      title: t('pv.capacity'),
      key: 'pv_capacity',
      sorter: (a, b) => sorterStorage(a.spec?.capacity?.storage ?? '', b.spec?.capacity?.storage ?? ''),
      render: (_, record) => record.spec?.capacity?.storage ? record.spec.capacity.storage : '--',
    },
    {
      title: t('pv.storageClass'),
      key: 'pv_sc',
      render: (_, record) => record.spec?.storageClassName ? record.spec.storageClassName : '--',
    },
    {
      title: t('pv.boundPvc'),
      key: 'pv_pvc',
      render: (_, record) => record.spec?.claimRef?.name ? record.spec.claimRef.name : '--',
    },
    {
      title: t('pv.created'),
      key: 'create_time',
      sorter: (a, b) => sortResourceByTime(a.metadata, b.metadata),
      render: (_, record) =>
        record.metadata?.creationTimestamp ? Dayjs(record.metadata.creationTimestamp).format('YYYY-MM-DD HH:mm') : '--',
    },
  ], [t, i18n.language]);

  return <div className="child_content withBread_content">
    <BreadCrumbCom className="create_bread" items={[
      { title: t('module.title'), disabled: true },
      { title: t('pv.title') },
    ]} />
    <ResourceList
      resourceType={t('pv.displayName')}
      columns={pvColumns}
      getResourceFn={getPvData}
      deleteResourceFn={(record) => deletePv(record.metadata.name)}
      yamlEditPath={(record) => `/${containerRouterPrefix}/storage/pv/${record.metadata.name}/yaml`}
      validateDelete={validateDelete}
    />
  </div>;
}
