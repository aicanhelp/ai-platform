/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import { Link } from 'inula-router';
import { containerRouterPrefix } from '@/constant.js';
import Dayjs from 'dayjs';
import BreadCrumbCom from '@/components/BreadCrumbCom';
import '@/styles/pages/workload.less';
import ResourceList from '@/components/layouts/ResourceList';
import { getIngressData, deleteIngress } from '@/api/containerApi';
import { sorterFirstAlphabet } from '@/tools/utils';
import { Space } from 'antd';
import { useMemo } from 'openinula';
import { useTranslation } from 'react-i18next';

function filterService(arr) {
  if (!arr?.length) {
    return '--';
  }
  const names = [];
  arr.forEach(item => {
    if (item.http?.paths) {
      item.http.paths.forEach(i => {
        const name = i.backend?.service?.name;
        if (name) { names.push(name); }
      });
    }
  });
  return names.length ? names.join(', ') : '--';
}

export default function Ingress() {
  const { t, i18n } = useTranslation('network');

  const ingressColumns = useMemo(() => [
    {
      title: t('ingress.name'),
      key: 'ingress_name',
      sorter: (a, b) => sorterFirstAlphabet(a.metadata.name, b.metadata.name),
      render: (_, record) => (
        <Link to={`/${containerRouterPrefix}/network/ingress/${record.metadata.namespace}/${record.metadata.name}`}>
          {record.metadata.name}
        </Link>
      ),
    },
    {
      title: t('ingress.service'),
      key: 'ingress_service',
      sorter: (a, b) => sorterFirstAlphabet(filterService(a.spec.rules), filterService(b.spec.rules)),
      render: (_, record) => <Space>{filterService(record.spec?.rules)}</Space>,
    },
    {
      title: t('ingress.namespace'),
      key: 'ingress_namespace',
      sorter: (a, b) => sorterFirstAlphabet(a.metadata.namespace, b.metadata.namespace),
      render: (_, record) => <Space>{record.metadata.namespace}</Space>,
    },
    {
      title: t('ingress.created'),
      sorter: (a, b) => Dayjs(a.metadata.creationTimestamp) - Dayjs(b.metadata.creationTimestamp),
      key: 'create_time',
      render: (_, record) => (
        <Space>
          {record.metadata.creationTimestamp
            ? Dayjs(record.metadata.creationTimestamp).format('YYYY-MM-DD HH:mm')
            : '--'}
        </Space>
      ),
    },
  ], [t, i18n.language]);

  return (
    <div className="child_content withBread_content">
      <BreadCrumbCom
        className="create_bread"
        items={[
          { title: t('module.title'), path: `/${containerRouterPrefix}/network`, disabled: true },
          { title: t('ingress.title'), path: '/' },
        ]}
      />
      <ResourceList
        resourceType={t('ingress.title')}
        columns={ingressColumns}
        getResourceFn={getIngressData}
        deleteResourceFn={(record) => deleteIngress(record.metadata.namespace, record.metadata.name)}
        yamlEditPath={(record) => `/${containerRouterPrefix}/network/ingress/${record.metadata.namespace}/${record.metadata.name}/yaml`}
      />
    </div>
  );
}
