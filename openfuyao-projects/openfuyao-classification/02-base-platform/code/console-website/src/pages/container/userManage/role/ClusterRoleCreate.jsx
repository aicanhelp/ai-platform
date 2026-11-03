/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import { containerRouterPrefix } from '@/constant.js';
import BreadCrumbCom from '@/components/BreadCrumbCom';
import '@/styles/pages/podDetail.less';
import ResourceCreate from '@/components/layouts/ResourceCreate';
import { addClusterRoleYamlData } from '@/api/containerApi';
import { clusterRoleYamlExample } from '@/common/exampleYaml';
import { useTranslation } from 'react-i18next';

export default function ClusterRoleCreate() {
  const { t } = useTranslation('rbac');
  return (
    <div className="child_content">
      <BreadCrumbCom
        className="create_bread"
        items={[
          { title: t('module.title'), disabled: true },
          { title: t('role.roleTitle'), path: `/${containerRouterPrefix}/userManage/role` },
          { title: t('role.create') },
        ]}
      />
      <ResourceCreate
        type="ClusterRole"
        yamlExample={clusterRoleYamlExample}
        submitApi={addClusterRoleYamlData}
        jumpPath={`/${containerRouterPrefix}/userManage/role`}
        clusterScoped
      />
    </div>
  );
}