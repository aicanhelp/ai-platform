/**
 *  Copyright (c) 2024 Huawei Technologies Co., Ltd.
 *  openFuyao is licensed under Mulan PSL v2.
 *  You can use this software according to the terms and conditions of the Mulan PSL v2.
 *  You may obtain a copy of Mulan PSL v2 at:
 *       http://license.coscl.org.cn/MulanPSL2
 *   THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 *   EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 *   MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 *   See the Mulan PSL v2 for more details.
 */

import { useParams } from 'inula-router';
import ApplicationDetailLayout from '@/components/layouts/ApplicationDetailLayout';
import HelmDetailInfo from '@/pages/container/applicationManage/helm/detail/HelmDetailInfo';
import HelmDetailYaml from '@/pages/container/applicationManage/helm/detail/HelmDetailYaml';
import HelmDetailResource from '@/pages/container/applicationManage/helm/detail/HelmDetailResource';
import HelmDetailLog from '@/pages/container/applicationManage/helm/detail/HelmDetailLog';
import HelmDetailEvent from '@/pages/container/applicationManage/helm/detail/HelmDetailEvent';
import HelmDetailMonitor from '@/pages/container/applicationManage/helm/detail/HelmDetailMonitor';

export default function HelmDetail() {
  const param = useParams();
  const helmName = param.helm_name;
  const helmNamespace = param.helm_namespace;

  // Info component with props
  const infoComponent = (
    <HelmDetailInfo
      helmName={helmName}
    />
  );

  // YAML component with props
  const yamlComponent = (
    <HelmDetailYaml
      helmName={helmName}
      helmNamespace={helmNamespace}
    />
  );

  // Resource component with props
  const resourceComponent = (
    <HelmDetailResource />
  );

  // Log component with props
  const logComponent = (
    <HelmDetailLog
      helmName={helmName}
      helmNamespace={helmNamespace}
    />
  );

  // Event component with props
  const eventComponent = (
    <HelmDetailEvent
      helmName={helmName}
      helmNamespace={helmNamespace}
    />
  );

  // Monitor component with props
  const monitorComponent = (
    <HelmDetailMonitor />
  );

  return (
    <ApplicationDetailLayout
      type="application"
      infoComponent={infoComponent}
      yamlComponent={yamlComponent}
      resourceComponent={resourceComponent}
      logComponent={logComponent}
      eventComponent={eventComponent}
      monitorComponent={monitorComponent}
    />
  );
}