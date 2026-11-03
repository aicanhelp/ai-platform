/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import { DownOutlined } from '@ant-design/icons';
import { useCallback, useMemo, useState } from 'openinula';
import { Link, useHistory } from 'inula-router';
import { useTranslation } from 'react-i18next';
import { containerRouterPrefix } from '@/constant.js';
import { DEFAULT_CURRENT_PAGE, ResponseCode } from '@/common/constants';
import {
  getRoleBindsData,
  deleteRoleBind,
  getClusterRoleBindingsData,
  deleteClusterRoleBinding,
} from '@/api/containerApi';
import BreadCrumbCom from '@/components/BreadCrumbCom';
import ResourceList from '@/components/layouts/ResourceList';
import { sorterFirstAlphabet } from '@/tools/utils';

export default function RoleBinding() {
  const { t, i18n } = useTranslation('rbac');
  const history = useHistory();
  const [roleFilterOptions, setRoleFilterOptions] = useState([]);
  const [themeNameFilterOptions, setThemeNameFilterOptions] = useState([]);

  const getAllRoleBinds = useCallback(async (namespace) => {
    const roleBindingItems = [];
    const clusterRoleBindingItems = [];

    try {
      const res = await getRoleBindsData(namespace, '', DEFAULT_CURRENT_PAGE, 10000);
      if (res.status === ResponseCode.OK) {
        (res.data?.items || []).forEach((item) => {
          roleBindingItems.push({ ...item, type: 'roleBinding' });
        });
      }
    } catch (e) {
      if (e?.response?.data?.code !== ResponseCode.NotFound) {
        throw e;
      }
    }

    try {
      const res = await getClusterRoleBindingsData('', '', DEFAULT_CURRENT_PAGE, 10000);
      if (res.status === ResponseCode.OK) {
        (res.data?.items || []).forEach((item) => {
          clusterRoleBindingItems.push({
            ...item,
            type: 'clusterRoleBinding',
            metadata: { ...item.metadata, namespace: 'all' },
          });
        });
      }
    } catch (e) {
      if (e?.response?.data?.code !== ResponseCode.NotFound) {
        throw e;
      }
    }

    const merged = [...clusterRoleBindingItems, ...roleBindingItems];
    merged.sort((a, b) => sorterFirstAlphabet(a.metadata.name, b.metadata.name));

    const roleNames = [...new Set(merged.map(item => item.roleRef?.name).filter(Boolean))];
    const themeNames = [...new Set(merged.map(item => item.subjects?.[0]?.name).filter(Boolean))];
    setRoleFilterOptions(roleNames.map(item => ({ text: item, value: item })));
    setThemeNameFilterOptions(themeNames.map(item => ({ text: item, value: item })));

    return {
      status: ResponseCode.OK,
      data: { items: merged },
    };
  }, []);

  const yamlEditPath = (record) => {
    if (record.type === 'roleBinding') {
      return `/${containerRouterPrefix}/userManage/roleBinding/${record.metadata.namespace}/${record.metadata.name}/yaml`;
    }
    return `/${containerRouterPrefix}/userManage/clusterRoleBinding/${record.metadata.name}/yaml`;
  };

  const roleBindColumns = useMemo(() => ([
    {
      title: t('roleBinding.bindName'),
      key: 'roleBind_name',
      sorter: (a, b) => sorterFirstAlphabet(a.metadata.name, b.metadata.name),
      render: (_, record) => <Link to={
        record.type === 'roleBinding'
          ? `/${containerRouterPrefix}/userManage/roleBinding/${record.metadata.namespace}/${record.metadata.name}`
          : `/${containerRouterPrefix}/userManage/clusterRoleBinding/${record.metadata.name}`
      }>{record.metadata.name}</Link>,
    },
    {
      title: t('roleBinding.bindType'),
      key: 'roleBind_type',
      sorter: (a, b) => sorterFirstAlphabet(
        a.type === 'roleBinding' ? t('roleBinding.roleBindingType') : t('roleBinding.clusterRoleBindingType'),
        b.type === 'roleBinding' ? t('roleBinding.roleBindingType') : t('roleBinding.clusterRoleBindingType'),
      ),
      filters: [
        { text: t('roleBinding.roleBindingType'), value: t('roleBinding.roleBindingType') },
        { text: t('roleBinding.clusterRoleBindingType'), value: t('roleBinding.clusterRoleBindingType') },
      ],
      onFilter: (value, record) => (record.type === 'roleBinding' ? t('roleBinding.roleBindingType') : t('roleBinding.clusterRoleBindingType')).toLowerCase() === String(value).toLowerCase(),
      filterMultiple: false,
      render: (_, record) => (record.type === 'roleBinding' ? t('roleBinding.roleBindingType') : t('roleBinding.clusterRoleBindingType')),
    },
    {
      title: t('roleBinding.role'),
      key: 'role_name',
      sorter: (a, b) => sorterFirstAlphabet(a.roleRef?.name || '--', b.roleRef?.name || '--'),
      filters: roleFilterOptions,
      onFilter: (value, record) => (record.roleRef?.name || '').toLowerCase() === String(value).toLowerCase(),
      filterMultiple: false,
      render: (_, record) => record.roleRef?.name || '--',
    },
    {
      title: t('roleBinding.roleType'),
      key: 'role_type',
      sorter: (a, b) => sorterFirstAlphabet(
        a.roleRef?.kind === 'Role' ? t('role.roleTitle') : t('role.clusterTitle'),
        b.roleRef?.kind === 'Role' ? t('role.roleTitle') : t('role.clusterTitle'),
      ),
      filters: [
        { text: t('role.roleTitle'), value: t('role.roleTitle') },
        { text: t('role.clusterTitle'), value: t('role.clusterTitle') },
      ],
      onFilter: (value, record) => (record.roleRef?.kind === 'Role' ? t('role.roleTitle') : t('role.clusterTitle')).toLowerCase() === String(value).toLowerCase(),
      filterMultiple: false,
      render: (_, record) => (record.roleRef?.kind === 'Role' ? t('role.roleTitle') : t('role.clusterTitle')),
    },
    {
      title: t('roleBinding.subjectKind'),
      key: 'roleBind_theme_kind',
      sorter: (a, b) => sorterFirstAlphabet(a.subjects?.[0]?.kind || '--', b.subjects?.[0]?.kind || '--'),
      render: (_, record) => record.subjects?.[0]?.kind || '--',
    },
    {
      title: t('roleBinding.subjectName'),
      key: 'roleBind_theme_title',
      sorter: (a, b) => sorterFirstAlphabet(a.subjects?.[0]?.name || '--', b.subjects?.[0]?.name || '--'),
      filters: themeNameFilterOptions,
      onFilter: (value, record) => (record.subjects?.[0]?.name || '').toLowerCase() === String(value).toLowerCase(),
      filterMultiple: false,
      render: (_, record) => record.subjects?.[0]?.name || '--',
    },
    {
      title: t('roleBinding.namespace'),
      key: 'roleBind_namespace',
      sorter: (a, b) => sorterFirstAlphabet(a.metadata.namespace, b.metadata.namespace),
      render: (_, record) => record.metadata.namespace || '--',
    },
  ]), [roleFilterOptions, themeNameFilterOptions, t, i18n.language]);

  const createPopMenuItems = useMemo(() => ([
    {
      key: 'roleBinding',
      label: t('roleBinding.roleBindingTitle'),
      onClick: () => history.push(`/${containerRouterPrefix}/userManage/roleBinding/create`),
    },
    {
      key: 'clusterRoleBinding',
      label: t('roleBinding.clusterTitle'),
      onClick: () => history.push(`/${containerRouterPrefix}/userManage/clusterRoleBinding/create`),
    },
  ]), [history, t, i18n.language]);

  return (
    <div className="child_content withBread_content">
      <BreadCrumbCom
        className="create_bread"
        items={[
          { title: t('module.title'), disabled: true },
          { title: t('roleBinding.roleBindingTitle') },
        ]}
      />
      <ResourceList
        resourceType={t('roleBinding.roleBindingTitle')}
        columns={roleBindColumns}
        getResourceFn={getAllRoleBinds}
        deleteResourceFn={(record) => (record.type === 'roleBinding'
          ? deleteRoleBind(record.metadata.namespace, record.metadata.name)
          : deleteClusterRoleBinding(record.metadata.name))}
        yamlEditPath={yamlEditPath}
        createLabel={<>{t('actions.create')} <DownOutlined className='small_margin_adjust' /></>}
        createMenuItems={createPopMenuItems}
      />
    </div>
  );
}