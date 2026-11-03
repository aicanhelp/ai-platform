/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import BreadCrumbCom from '@/components/BreadCrumbCom';
import ResourceList from '@/components/layouts/ResourceList';
import { useTranslation } from 'react-i18next';
import { useMemo } from 'openinula';
import { Link } from 'inula-router';
import { getCustomResourceDefinitionList, deleteCustomResourceDefinition } from '@/api/containerApi';
import { containerRouterPrefix } from '@/constant';
import { solveEncodePath, sorterFirstAlphabet, solveCustomResourceStatus } from '@/tools/utils';

export default function CrdIndex() {
  const { t, i18n } = useTranslation('crd');

  const filterStatusOptions = useMemo(() => ([
    { text: t('status.normal'), value: 'normal' },
    { text: t('status.abnormal'), value: 'abnormal' },
  ]), [t, i18n.language]);

  const yamlEditPath = (record) => {
    return {
      pathname: `/${containerRouterPrefix}/customResourceDefinition/${solveEncodePath(record.metadata.name)}/yaml`,
      state: { crdName: record.metadata.name },
    };
  };

  const columns = useMemo(() => [
    {
      title: t('table.name'),
      key: 'name',
      sorter: (a, b) => sorterFirstAlphabet(a.metadata.name, b.metadata.name),
      render: (_, record) => (
        <Link to={`/${containerRouterPrefix}/customResourceDefinition/${solveEncodePath(record.metadata.name)}`}>
          {record.metadata.name}
        </Link>
      ),
    },
    {
      title: t('table.latestVersion'),
      key: 'version',
      render: (_, record) => record.spec.versions?.[0]?.name || '--',
    },
    {
      title: t('table.status'),
      key: 'status',
      filters: filterStatusOptions,
      onFilter: (value, record) => (solveCustomResourceStatus(record.status?.conditions) ? 'normal' : 'abnormal').toLowerCase() === String(value).toLowerCase(),
      filterMultiple: false,
      render: (_, record) => (
        <div className="status_group">
          <span className={solveCustomResourceStatus(record.status?.conditions) ? 'running_circle' : 'failed_circle'}></span>
          <span>{t(`status.${solveCustomResourceStatus(record.status?.conditions) ? 'normal' : 'abnormal'}`)}</span>
        </div>
      ),
    },
  ], [t, i18n.language, filterStatusOptions]);

  return (
    <div className="child_content withBread_content">
      <BreadCrumbCom className="create_bread" items={[{ title: t('module.title'), disabled: true }]} />
      <ResourceList
        resourceType={t('module.title')}
        columns={columns}
        getResourceFn={() => getCustomResourceDefinitionList()}
        deleteResourceFn={(record) => deleteCustomResourceDefinition(record.metadata.name)}
        deleteSuccessMessage={t('messages.deleteSuccess')}
        yamlEditPath={yamlEditPath}
        createLabel={t('actions.create')}
        createPath={`/${containerRouterPrefix}/customResourceDefinition/createCustomResourceDefinition`}
      />
    </div>
  );
}