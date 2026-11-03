/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import { containerRouterPrefix } from '@/constant';
import BreadCrumbCom from '@/components/BreadCrumbCom';
import '@/styles/pages/podDetail.less';
import { useState, useEffect, useRef, useStore } from 'openinula';
import { ExportOutlined, CopyOutlined } from '@ant-design/icons';
import { useTranslation } from 'react-i18next';
import copy from 'copy-to-clipboard';
import { ResponseCode } from '@/common/constants';
import { message, Button } from 'antd';
import { addResourceInstanceYamlData, getCustomResourceDefinitionDetailDescription } from '@/api/containerApi';
import CodeMirrorEditor from '@/components/CodeMirrorEditor';
import { useHistory, useParams } from 'inula-router';
import { yamlTojson, solveEncodePath, exportYamlOutPut, jsonToYaml, solveDecodePath, forbiddenMsg, showSuccessThen } from '@/tools/utils';
import {
  namespacedCrYamlExample,
  clusterCrYamlExample,
  podGroupExample,
  queueSchedulingExampe,
  jobBatchExample,
  virtualMachineYamlExample,
} from '@/common/exampleYaml';
import useButtonLoading from '@/hooks/useButtonLoading';

export default function CrCreate() {
  const { t } = useTranslation(['crd', 'common']);
  const { customResourceName } = useParams();
  const history = useHistory();
  const [messageApi, contextHolder] = message.useMessage();
  const themeStore = useStore('theme');
  const childCodeMirrorRef = useRef(null);
  const crdMetaRef = useRef(null);

  const [yamlData, setYamlData] = useState('');
  const [saveLoading, setSaveLoading] = useButtonLoading();
  const [initialYaml, setInitialYaml] = useState('');
  const [loaded, setLoaded] = useState(false);

  const handleCopyYaml = () => {
    copy(yamlData);
    messageApi.success(t('messages.copySuccess', { ns: 'common' }));
  };

  const handleChangeYaml = (str) => {
    setYamlData(str);
  };

  const handleSaveYaml = async () => {
    setSaveLoading(true);
    try {
      await handleSaveYamlInner();
    } finally {
      setSaveLoading(false);
    }
  };
  const handleSaveYamlInner = async () => {
    if (!crdMetaRef.current?.group || !crdMetaRef.current?.version || !crdMetaRef.current?.plural) {
      messageApi.error(t('messages.crdMetaMissing'));
      return;
    }
    let errorCount = 0;
    let yamlJson = '';
    try {
      yamlJson = yamlTojson(yamlData);
    } catch (e) {
      messageApi.error(t('resourceYaml.invalidYaml', {
        ns: 'common',
        message: e.message,
        interpolation: { escapeValue: false },
      }));
      errorCount++;
    }

    if (!errorCount) {
      const { group, version, plural, scope } = crdMetaRef.current;
      const clusterScoped = scope !== 'Namespaced';
      const writtenNamespace = yamlJson?.metadata?.namespace || '';
      try {
        const res = await addResourceInstanceYamlData(
          { group, version, plural, namespace: clusterScoped ? '' : (writtenNamespace || 'default') },
          yamlJson,
        );
        if (res.status === ResponseCode.Created) {
          showSuccessThen(messageApi, t('messages.createSuccess'), () => {
            history.push(`/${containerRouterPrefix}/customResourceDefinition/${solveEncodePath(customResourceName)}/example`);
          });
        }
      } catch (error) {
        if (error.response?.status === ResponseCode.Forbidden) {
          forbiddenMsg(messageApi, error, t);
        } else {
          messageApi.error(t('resourceYaml.createFailed', {
            ns: 'common',
            message: error.response?.data?.message ?? error.message ?? '',
            interpolation: { escapeValue: false },
          }));
        }
      }
    }
  };

  const handleResetCode = () => {
    setYamlData(initialYaml);
    childCodeMirrorRef.current?.resetCodeEditor(initialYaml);
  };

  const exportYaml = () => {
    exportYamlOutPut('Cr', yamlData);
    messageApi.success(t('messages.exportSuccess', { ns: 'common' }));
  };

  useEffect(() => {
    const init = async () => {
      setLoaded(false);
      const decodedName = solveDecodePath(customResourceName);
      let group;
      let version;
      let plural;
      let kind;
      let scope;

      try {
        const res = await getCustomResourceDefinitionDetailDescription(decodedName);
        if (res.status === ResponseCode.OK) {
          const data = res.data;
          group = data.spec.group;
          version = data.spec.versions[0].name;
          plural = data.spec.names.plural;
          kind = data.spec.names.kind;
          scope = data.spec.scope;
        }
      } catch (e) {
        messageApi.error(t('messages.fetchFailed', { ns: 'common', message: e?.response?.data?.message }));
      }

      crdMetaRef.current = { group, version, plural, scope };

      const baseTemplate = scope === 'Namespaced' ? namespacedCrYamlExample : clusterCrYamlExample;
      let templateObj = yamlTojson(baseTemplate);

      if (group && version) {
        templateObj.apiVersion = `${group}/${version}`;
      }
      if (kind) {
        templateObj.kind = kind;
      }

      if (decodedName === 'podgroups.scheduling.volcano.sh') {
        templateObj = yamlTojson(podGroupExample);
      } else if (decodedName === 'queues.scheduling.volcano.sh') {
        templateObj = yamlTojson(queueSchedulingExampe);
      } else if (decodedName === 'jobs.batch.volcano.sh') {
        templateObj = yamlTojson(jobBatchExample);
      } else if (decodedName === 'virtualmachines.kubevirt.io') {
        templateObj = yamlTojson(virtualMachineYamlExample);
      }

      const resultYaml = jsonToYaml(JSON.stringify(templateObj));
      setYamlData(resultYaml);
      setInitialYaml(resultYaml);
      setLoaded(true);
    };
    init();
  }, [customResourceName]);

  return (
    <div className="child_content">
      <div style={{ background: themeStore.$s.theme === 'dark' ? '#2a2d34ff' : '#fff', color: themeStore.$s.theme === 'dark' ? '#fff' : '#333' }}>
        {contextHolder}
      </div>
      <BreadCrumbCom items={[
        { title: t('module.title'), path: `/${containerRouterPrefix}/customResourceDefinition` },
        { title: t('module.crdDetail'), path: `${solveEncodePath(customResourceName)}/example` },
        { title: t('actions.createInstance') },
      ]} />
      <div className="tab_container container_margin_box normal_container_height">
        <div className="yaml_card">
          <div className="yaml_flex_box">
            <h3>{t('detail.tabYaml')}</h3>
            <div className="yaml_tools">
              <div className="tool_word_group" onClick={exportYaml}>
                <ExportOutlined className="common_antd_icon primary_color" />
                <span>{t('resourceYaml.export', { ns: 'common' })}</span>
              </div>
              <div className="tool_word_group" onClick={handleCopyYaml}>
                <CopyOutlined className="common_antd_icon primary_color" />
                <span>{t('resourceYaml.copy', { ns: 'common' })}</span>
              </div>
            </div>
          </div>
        </div>
        <div className="yaml_space_box">
          {loaded && <CodeMirrorEditor yamlData={yamlData} changeYaml={handleChangeYaml} ref={childCodeMirrorRef} />}
          <div className="btn_footer">
            <Button className="cancel_btn borderRadiusRight6" onClick={() => history.go(-1)}>{t('actions.cancel', { ns: 'common' })}</Button>
            <Button className="cancel_btn borderRadiusRight6" onClick={handleResetCode}>{t('actions.reset', { ns: 'common' })}</Button>
            <Button className="primary_btn borderRadiusRight6" onClick={handleSaveYaml} loading={saveLoading}>{t('actions.confirm', { ns: 'common' })}</Button>
          </div>
        </div>
      </div>
    </div>
  );
}
