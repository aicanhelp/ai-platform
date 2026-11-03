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
import ResourceCreate from '@/components/layouts/ResourceCreate';
import { addNamespaceYamlData } from '@/api/containerApi';
import { namespaceYamlExample } from '@/common/exampleYaml';
import { containerRouterPrefix } from '@/constant.js';
import { useTranslation } from 'react-i18next';

export default function NamespaceCreate() {
  const { t } = useTranslation('namespace');
  return (
    <div className="child_content">
      <BreadCrumbCom className="create_bread" items={[
        { title: t('module.title'), link: `/${containerRouterPrefix}/namespace/namespaceManage` },
        { title: t('namespace.title'), link: `/${containerRouterPrefix}/namespace/namespaceManage` },
        { title: t('namespace.create') },
      ]} />
      <ResourceCreate
        type="Namespace"
        yamlExample={namespaceYamlExample}
        submitApi={addNamespaceYamlData}
        jumpPath={`/${containerRouterPrefix}/namespace/namespaceManage`}
        clusterScoped
      />
    </div>
  );
}
