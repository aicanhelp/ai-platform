/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import { useParams } from 'inula-router';
import ResourceYaml from '@/components/layouts/ResourceYaml';
import { updatePvYamlData } from '@/api/containerApi';

export default function PvYaml({ pvYamlProps, isPvReadyOnly, handleEditFn, refreshFn }) {
  const { name: pvResourceName } = useParams();

  // Keep PV update behavior consistent: always update current route resource.
  const updateFn = (_, yamlJson) => updatePvYamlData(pvResourceName, yamlJson);

  return <ResourceYaml
    yamlProps={pvYamlProps}
    readOnly={isPvReadyOnly}
    handleEditFn={handleEditFn}
    updateFn={updateFn}
    refreshFn={refreshFn}
    clusterScoped
  />;
}
