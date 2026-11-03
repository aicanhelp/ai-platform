/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import { useMemo, useCallback, useState } from 'openinula';
import { Link } from 'inula-router';
import Dayjs from 'dayjs';
import BreadCrumbCom from '@/components/BreadCrumbCom';
import '@/styles/pages/workload.less';
import ResourceList from '@/components/layouts/ResourceList';
import { getSecretList, deleteSecret } from '@/api/containerApi';
import { sorterFirstAlphabet } from '@/tools/utils';
import { Tooltip } from 'antd';
import { ResponseCode } from '@/common/constants';
import { useTranslation } from 'react-i18next';
import { containerRouterPrefix } from '@/constant.js';

export default function Secret() {
  const { t, i18n } = useTranslation('configuration');

  const [secretTypeFilterOptions, setSecretTypeFilterOptions] = useState([]);

  const getSecretListWrapped = useCallback(async (ns) => {
    let res;
    try {
      res = await getSecretList(ns);
    } catch {
      setSecretTypeFilterOptions([]);
      return null;
    }
    if (res.status === ResponseCode.OK && res.data?.items?.length) {
      const types = [...new Set(res.data.items.map(i => i.type).filter(Boolean))].map(t => ({
        text: t,
        value: t,
      }));
      setSecretTypeFilterOptions(types);
    } else {
      setSecretTypeFilterOptions([]);
    }
    return res;
  }, []);

  const secretColumns = useMemo(() => {
    const typeCol = {
      title: t('secret.type'),
      key: 'secret_type',
      sorter: (a, b) => sorterFirstAlphabet(a.type || '', b.type || ''),
      render: (_, record) => record.type || '--',
    };
    if (secretTypeFilterOptions.length) {
      typeCol.filters = secretTypeFilterOptions;
      typeCol.onFilter = (value, record) => (record.type || '').toLowerCase() === String(value).toLowerCase();
      typeCol.filterMultiple = false;
    }
    return [
      {
        title: t('secret.name'),
        key: 'secret_name',
        sorter: (a, b) => sorterFirstAlphabet(a.metadata.name, b.metadata.name),
        render: (_, record) => (
          <Link to={`/${containerRouterPrefix}/configuration/secret/${record.metadata.namespace}/${record.metadata.name}`}>
            <Tooltip title={record.metadata.name}>
              <div className='word_break'>{record.metadata.name}</div>
            </Tooltip>
          </Link>
        ),
      },
      {
        title: t('secret.namespace'),
        key: 'secret_namespace',
        sorter: (a, b) => sorterFirstAlphabet(a.metadata.namespace, b.metadata.namespace),
        render: (_, record) => record.metadata.namespace,
      },
      typeCol,
      {
        title: t('secret.created'),
        key: 'creationTimestamp',
        sorter: (a, b) => Dayjs(a.metadata.creationTimestamp) - Dayjs(b.metadata.creationTimestamp),
        render: (_, record) =>
          Dayjs(record.metadata.creationTimestamp).format('YYYY-MM-DD HH:mm:ss'),
      },
    ];
  }, [secretTypeFilterOptions, t, i18n.language]);

  return (
    <div className='child_content withBread_content secret'>
      <BreadCrumbCom
        className='create_bread'
        items={[
          { title: t('module.title'), disabled: true },
          { title: t('secret.title') },
        ]}
      />
      <ResourceList
        resourceType='Secret'
        columns={secretColumns}
        getResourceFn={getSecretListWrapped}
        deleteResourceFn={(record) => deleteSecret(record.metadata.namespace, record.metadata.name)}
        yamlEditPath={(record) => `/${containerRouterPrefix}/configuration/secret/${record.metadata.namespace}/${record.metadata.name}/yaml`}
      />
    </div>
  );
}
