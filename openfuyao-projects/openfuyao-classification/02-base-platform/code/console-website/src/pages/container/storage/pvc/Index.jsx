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
import { getPvcData, deletePvc } from '@/api/containerApi';
import { Link } from 'inula-router';
import Dayjs from 'dayjs';
import { sorterFirstAlphabet, sorterStorage } from '@/tools/utils';
import BreadCrumbCom from '@/components/BreadCrumbCom';
import '@/styles/pages/workload.less';
import ResourceList from '@/components/layouts/ResourceList';
import { sortResourceByTime } from '@/utils/common';
import { containerRouterPrefix } from '@/constant.js';

export default function Pvc() {
  const { t, i18n } = useTranslation('storage');

  const pvcColumns = useMemo(() => [
    {
      title: t('pvc.name'),
      key: 'pvc_name',
      sorter: (a, b) => sorterFirstAlphabet(a.metadata.name, b.metadata.name),
      render: (_, record) => <Link to={`/${containerRouterPrefix}/storage/pvc/${record.metadata.namespace}/${record.metadata.name}`}>{record.metadata.name}</Link>,
    },
    {
      title: t('pvc.capacity'),
      key: 'pvc_capacity',
      sorter: (a, b) => sorterStorage(a.spec?.resources?.requests?.storage ?? '', b.spec?.resources?.requests?.storage ?? ''),
      render: (_, record) => record.spec?.resources?.requests?.storage ? record.spec.resources.requests.storage : '--',
    },
    {
      title: t('pvc.namespace'),
      key: 'pvc_namespace',
      sorter: (a, b) => sorterFirstAlphabet(a.metadata.namespace, b.metadata.namespace),
      render: (_, record) => record.metadata?.namespace ? record.metadata.namespace : '--',
    },
    {
      title: t('pvc.storageClass'),
      key: 'pvc_sc',
      render: (_, record) => record.spec?.storageClassName ? record.spec.storageClassName : '--',
    },
    {
      title: t('pvc.created'),
      key: 'create_time',
      sorter: (a, b) => sortResourceByTime(a.metadata, b.metadata),
      render: (_, record) =>
        Dayjs(record.metadata.creationTimestamp ? record.metadata.creationTimestamp : '--').format('YYYY-MM-DD HH:mm'),
    },
  ], [t, i18n.language]);

  return <div className="child_content withBread_content">
    <BreadCrumbCom className="create_bread" items={[
      { title: t('module.title'), disabled: true },
      { title: t('pvc.title') },
    ]} />
    <ResourceList
      resourceType={t('pvc.displayName')}
      columns={pvcColumns}
      getResourceFn={getPvcData}
      deleteResourceFn={(record) => deletePvc(record.metadata.namespace, record.metadata.name)}
      yamlEditPath={(record) => `/${containerRouterPrefix}/storage/pvc/${record.metadata.namespace}/${record.metadata.name}/yaml`}
    />
  </div>;
}
