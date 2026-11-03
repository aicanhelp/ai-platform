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
import { useCallback, useMemo } from 'openinula';
import { useTranslation } from 'react-i18next';
import { ResponseCode } from '@/common/constants';
import { getRolesData, deleteRole, getClusterRolesData, deleteClusterRole } from '@/api/containerApi';
import { Link, useHistory } from 'inula-router';
import { containerRouterPrefix } from '@/constant.js';
import { sorterFirstAlphabet } from '@/tools/utils';
import BreadCrumbCom from '@/components/BreadCrumbCom';
import '@/styles/pages/workload.less';
import ResourceList from '@/components/layouts/ResourceList';

export default function Role() {
  const { t, i18n } = useTranslation('rbac');
  const history = useHistory();

  const filterRoleTypeOptions = useMemo(() => ([
    { text: t('role.roleType'), value: 'role' },
    { text: t('role.clusterRoleType'), value: 'clusterRole' },
  ]), [t, i18n.language]);

  const yamlEditPath = (record) => {
    if (record.type === 'role') {
      return {
        pathname: `/${containerRouterPrefix}/userManage/role/${record.metadata.namespace}/${record.metadata.name}/yaml`,
        state: {
          roleType: 'role',
          roleNamespace: record.metadata.namespace,
          roleName: record.metadata.name,
        },
      };
    } else {
      return {
        pathname: `/${containerRouterPrefix}/userManage/clusterRole/${record.metadata.name}/yaml`,
        state: {
          roleName: record.metadata.name,
        },
      };
    }
  };

  const getAllRoles = useCallback(async (namespace) => {
    const roleItems = [];
    const clusterRoleItems = [];

    // role
    try {
      const res = await getRolesData(namespace, '', 1, 10000);
      if (res.status === ResponseCode.OK) {
        (res.data?.items || []).forEach((item) => {
          roleItems.push({ ...item, type: 'role' });
        });
      }
    } catch (e) {
      if (e?.response?.data?.code !== ResponseCode.NotFound) {
        throw e;
      }
    }

    // clusterRole
    try {
      const res = await getClusterRolesData('', '', 1, 10000);
      if (res.status === ResponseCode.OK) {
        (res.data?.items || []).forEach((item) => {
          clusterRoleItems.push({
            ...item,
            type: 'clusterRole',
            metadata: { ...item.metadata, namespace: 'all' },
          });
        });
      }
    } catch (e) {
      if (e?.response?.data?.code !== ResponseCode.NotFound) {
        throw e;
      }
    }

    const merged = [...clusterRoleItems, ...roleItems];
    merged.sort((a, b) => sorterFirstAlphabet(a.metadata.name, b.metadata.name));
    return {
      status: ResponseCode.OK,
      data: { items: merged },
    };
  }, []);

  const roleColumns = useMemo(() => ([
    {
      title: t('role.name'),
      key: 'role_name',
      sorter: (a, b) => sorterFirstAlphabet(a.metadata.name, b.metadata.name),
      render: (_, record) => {
        if (record.type === 'role') {
          return <Link
            to={{
              pathname: `/${containerRouterPrefix}/userManage/role/${record.metadata.namespace}/${record.metadata.name}`,
            }}
          >
            {record.metadata.name}
          </Link>;
        } else {
          return <Link
            to={{
              pathname: `/${containerRouterPrefix}/userManage/clusterRole/${record.metadata.name}`,
            }}
          >
            {record.metadata.name}
          </Link>;
        }
      },
    },
    {
      title: t('role.type'),
      key: 'role_type',
      sorter: (a, b) => sorterFirstAlphabet(
        a.type === 'role' ? t('role.roleType') : t('role.clusterRoleType'),
        b.type === 'role' ? t('role.roleType') : t('role.clusterRoleType'),
      ),
      filters: filterRoleTypeOptions,
      onFilter: (value, record) => (record.type).toLowerCase() === String(value).toLowerCase(),
      filterMultiple: false,
      render: (_, record) => (record.type === 'role' ? t('role.roleType') : t('role.clusterRoleType')),
    },
    {
      title: t('role.namespace'),
      key: 'role_namespace',
      sorter: (a, b) => sorterFirstAlphabet(a.metadata.namespace, b.metadata.namespace),
      render: (_, record) => (record.metadata.namespace ? record.metadata.namespace : '--'),
    },
  ]), [filterRoleTypeOptions, t, i18n.language]);

  const createPopMenuItems = useMemo(() => ([
    {
      key: 'role',
      label: t('role.roleTitle'),
      onClick: () => history.push(`/${containerRouterPrefix}/userManage/role/create`),
    },
    {
      key: 'clusterRole',
      label: t('role.clusterTitle'),
      onClick: () => history.push(`/${containerRouterPrefix}/userManage/clusterRole/create`),
    },
  ]), [history, t, i18n.language]);

  return (
    <div className="child_content withBread_content">
      <BreadCrumbCom
        className="create_bread"
        items={[
          { title: t('module.title'), disabled: true },
          { title: t('role.roleTitle') },
        ]}
      />
      <ResourceList
        resourceType={t('role.roleTitle')}
        columns={roleColumns}
        getResourceFn={getAllRoles}
        deleteResourceFn={(record) => (record.type === 'role'
          ? deleteRole(record.metadata.namespace, record.metadata.name)
          : deleteClusterRole(record.metadata.name))}
        yamlEditPath={yamlEditPath}
        createLabel={<>{t('actions.create')} <DownOutlined className='small_margin_adjust' /></>}
        createMenuItems={createPopMenuItems}
      />
    </div>
  );
}