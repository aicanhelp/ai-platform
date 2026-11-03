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
import { updateResourceInstanceYamlData } from '@/api/containerApi';

export default function CrYaml({
  yamlData,
  isReadyOnly,
  handleReadyOnly,
  getDetailInfo,
  prefixObj,
}) {
  const { namespace: routeNamespace } = useParams();
  const namespace = prefixObj?.namespace || routeNamespace || '';
  const clusterScoped = prefixObj?.scopeNamespace
    ? prefixObj.scopeNamespace !== 'Namespaced'
    : !namespace;

  // ResourceYaml：集群级 updateFn(name, yaml)；命名空间级 updateFn(namespace, name, yaml)
  const updateFn = (namespaceOrName, nameOrYaml, yamlJson) => {
    if (clusterScoped) {
      return updateResourceInstanceYamlData(prefixObj, namespaceOrName, nameOrYaml);
    }
    return updateResourceInstanceYamlData(
      { ...prefixObj, namespace: namespaceOrName },
      nameOrYaml,
      yamlJson,
    );
  };

  return (
    <ResourceYaml
      yamlProps={yamlData}
      readOnly={isReadyOnly}
      handleEditFn={handleReadyOnly}
      updateFn={updateFn}
      refreshFn={getDetailInfo}
      clusterScoped={clusterScoped}
    />
  );
}
