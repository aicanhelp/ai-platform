/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import { useMemo } from 'openinula';
import { useTranslation } from 'react-i18next';
import { workloadStatus } from '@/common/constants';
import { listDaemonSets, deleteDaemonSet } from '@/api/containerApi';
import { Link } from 'inula-router';
import Dayjs from 'dayjs';
import { firstAlphabetUp, sorterFirstAlphabet, getDaemonSetStatus } from '@/tools/utils';
import BreadCrumbCom from '@/components/BreadCrumbCom';
import '@/styles/pages/workload.less';
import ResourceList from '@/components/layouts/ResourceList';
import { containerRouterPrefix } from '@/constant.js';

export default function DaemonSet() {
  const { t, i18n } = useTranslation('workload');

  const filterStatus = useMemo(() => {
    return workloadStatus.map(item => ({ text: firstAlphabetUp(item), value: item }));
  }, []);
  // 列表项
  const daemonSetColumns = useMemo(() => [
    {
      title: t('daemonSet.name'),
      key: 'daemonSet_name',
      sorter: (a, b) => sorterFirstAlphabet(a.metadata.name, b.metadata.name),
      render: (_, record) => <Link to={`/${containerRouterPrefix}/workload/daemonSet/${record.metadata.namespace}/${record.metadata.name}`}>{record.metadata.name}</Link>,
    },
    {
      title: t('daemonSet.status'),
      key: 'status',
      width: 220,
      sorter: (a, b) => sorterFirstAlphabet(getDaemonSetStatus(a.status), getDaemonSetStatus(b.status)),
      render: (_, record) => <p className={`resource_status ${getDaemonSetStatus(record.status).toLowerCase()}_circle`}>
        {getDaemonSetStatus(record.status)}
      </p>,
      filters: filterStatus,
      onFilter: (value, record) => getDaemonSetStatus(record.status).toLowerCase() === String(value).toLowerCase(),
      filterMultiple: false,
    },
    {
      title: t('daemonSet.namespace'),
      key: 'daemonSet_namespace',
      sorter: (a, b) => sorterFirstAlphabet(a.metadata.namespace, b.metadata.namespace),
      render: (_, record) => record.metadata.namespace,
    },
    {
      title: t('deployment.instances'),
      key: 'daemonSet_example',
      render: (_, record) => `${record.status.currentNumberScheduled}/${record.status.desiredNumberScheduled}`,
    },
    {
      title: t('pod.created'),
      key: 'daemonSet_create_time',
      sorter: (a, b) => Dayjs(a.metadata.creationTimestamp) - Dayjs(b.metadata.creationTimestamp),
      render: (_, record) => Dayjs(record.metadata.creationTimestamp).format('YYYY-MM-DD HH:mm'),
    },
  ], [filterStatus, t, i18n.language]);

  return <div className="child_content withBread_content">
    <BreadCrumbCom
      className="create_bread"
      items={[
        { title: t('module.title'), disabled: true },
        { title: t('daemonSet.title') },
      ]}
    />
    <ResourceList
      resourceType={'DaemonSet'}
      columns={daemonSetColumns}
      getResourceFn={listDaemonSets}
      deleteResourceFn={(record) => deleteDaemonSet(record.metadata.namespace, record.metadata.name)}
      yamlEditPath={(record) => `/${containerRouterPrefix}/workload/daemonSet/${record.metadata.namespace}/${record.metadata.name}/yaml`}
    />
  </div>;
}