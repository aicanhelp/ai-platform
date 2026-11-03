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
import { createDeployment } from '@/api/containerApi';
import { deploymentYamlExample } from '@/common/exampleYaml';
import { containerRouterPrefix } from '@/constant.js';
import { useTranslation } from 'react-i18next';

export default function DeploymentCreate() {
  const { t } = useTranslation('workload');
  return (
    <div className="child_content">
      <BreadCrumbCom className="create_bread" items={[
        { title: t('module.title'), link: `/${containerRouterPrefix}/workload` },
        { title: t('deployment.title'), link: `/${containerRouterPrefix}/workload/deployment` },
        { title: t('deployment.create') },
      ]} />
      <ResourceCreate
        type="Deployment"
        yamlExample={deploymentYamlExample}
        submitApi={createDeployment}
        jumpPath={`/${containerRouterPrefix}/workload/deployment`}
      />
    </div>
  );
}