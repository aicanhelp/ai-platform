/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import ResourceYaml from '@/components/layouts/ResourceYaml';
import { updateRoleYamlData } from '@/api/containerApi';

export default function RoleYaml({ roleYamlProps, readOnly, handleEditFn, refreshFn }) {
  return <ResourceYaml
    yamlProps={roleYamlProps}
    readOnly={readOnly}
    handleEditFn={handleEditFn}
    updateFn={(namespace, name, yamlJson) => updateRoleYamlData(namespace, name, yamlJson)}
    refreshFn={refreshFn}
  />;
}