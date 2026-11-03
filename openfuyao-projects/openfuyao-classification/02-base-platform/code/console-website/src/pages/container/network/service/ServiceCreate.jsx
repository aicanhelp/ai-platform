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
import { addServiceYamlData } from '@/api/containerApi';
import { serviceYamlExample } from '@/common/exampleYaml';
import ResourceCreate from '@/components/layouts/ResourceCreate';
import { useTranslation } from 'react-i18next';

export default function ServiceCreate() {
  const { t } = useTranslation('network');
  return <div className="child_content">
    <BreadCrumbCom
      className='create_bread'
      items={[
        { title: t('module.title'), link: `/${containerRouterPrefix}/network` },
        { title: t('service.title'), link: `/${containerRouterPrefix}/network/service` },
        { title: t('service.create') },
      ]} />
    <ResourceCreate
      type="Service"
      yamlExample={serviceYamlExample}
      submitApi={addServiceYamlData}
      jumpPath={`/${containerRouterPrefix}/network/service`}
    />
  </div>;
}