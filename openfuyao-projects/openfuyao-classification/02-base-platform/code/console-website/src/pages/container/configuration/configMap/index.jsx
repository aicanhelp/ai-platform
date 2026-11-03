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
import Dayjs from 'dayjs';
import BreadCrumbCom from '@/components/BreadCrumbCom';
import '@/styles/pages/workload.less';
import ResourceList from '@/components/layouts/ResourceList';
import { getConfigMapsList, deleteConfigMaps } from '@/api/containerApi';
import { sorterFirstAlphabet } from '@/tools/utils';
import { Tooltip } from 'antd';
import { useMemo } from 'openinula';
import { useTranslation } from 'react-i18next';
import { containerRouterPrefix } from '@/constant.js';

export default function ConfigMap() {
  const { t, i18n } = useTranslation('configuration');

  const configMapColumns = useMemo(() => [
    {
      title: t('configMap.name'),
      key: 'configmap_name',
      sorter: (a, b) => sorterFirstAlphabet(a.metadata.name, b.metadata.name),
      render: (_, record) => (
        <Link to={`/${containerRouterPrefix}/configuration/configMap/${record.metadata.namespace}/${record.metadata.name}`}>
          <Tooltip title={record.metadata.name}>
            <div className='word_break'>{record.metadata.name}</div>
          </Tooltip>
        </Link>
      ),
    },
    {
      title: t('configMap.namespace'),
      key: 'configmap_namespace',
      sorter: (a, b) => sorterFirstAlphabet(a.metadata.namespace, b.metadata.namespace),
      render: (_, record) => record.metadata.namespace,
    },
    {
      title: t('configMap.created'),
      key: 'creationTimestamp',
      sorter: (a, b) => Dayjs(a.metadata.creationTimestamp) - Dayjs(b.metadata.creationTimestamp),
      render: (_, record) =>
        Dayjs(record.metadata.creationTimestamp).format('YYYY-MM-DD HH:mm:ss'),
    },
  ], [t, i18n.language]);

  return (
    <div className='configmap child_content withBread_content'>
      <BreadCrumbCom
        className='create_bread'
        items={[
          { title: t('module.title'), disabled: true },
          { title: t('configMap.title') },
        ]}
      />
      <ResourceList
        resourceType='ConfigMap'
        columns={configMapColumns}
        getResourceFn={getConfigMapsList}
        deleteResourceFn={(record) => deleteConfigMaps(record.metadata.namespace, record.metadata.name)}
        yamlEditPath={(record) => `/${containerRouterPrefix}/configuration/configMap/${record.metadata.namespace}/${record.metadata.name}/yaml`}
      />
    </div>
  );
}
