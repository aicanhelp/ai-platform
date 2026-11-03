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
import { useState, useRef, useStore, useEffect } from 'openinula';
import { ExportOutlined, CopyOutlined } from '@ant-design/icons';
import { useTranslation } from 'react-i18next';
import copy from 'copy-to-clipboard';
import { ResponseCode } from '@/common/constants';
import { message, Button } from 'antd';
import { addCustomResourceDefinitionYamlData, getCustomResourceDefinitionDetailDescription, updateCustomResourceDefinitionYamlData } from '@/api/containerApi';
import CodeMirrorEditor from '@/components/CodeMirrorEditor';
import SkeletonArea from '@/components/SkeletonArea';
import { useHistory, useParams } from 'inula-router';
import { yamlTojson, jsonToYaml, exportYamlOutPut, forbiddenMsg, solveDecodePath, showSuccessThen } from '@/tools/utils';
import { customResourceDefinitionYamlExample } from '@/common/exampleYaml';
import useButtonLoading from '@/hooks/useButtonLoading';

export default function CrdCreate() {
  const { t } = useTranslation('crd');
  const { customResourceName } = useParams();
  const history = useHistory();
  const [messageApi, contextHolder] = message.useMessage();
  const themeStore = useStore('theme');
  const childCodeMirrorRef = useRef(null);

  const isEditMode = !!customResourceName && customResourceName !== 'createCustomResourceDefinition';
  const [yamlData, setYamlData] = useState(isEditMode ? '' : customResourceDefinitionYamlExample);
  const [saveLoading, setSaveLoading] = useButtonLoading();
  const [originalYaml, setOriginalYaml] = useState('');
  const [yamlLoaded, setYamlLoaded] = useState(!isEditMode); // 编辑模式等待 YAML 加载

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
    let errorCount = 0;
    let yamlJson = '';
    try {
      yamlJson = yamlTojson(yamlData);
    } catch (e) {
      messageApi.error(`${t('messages.yamlInvalid')}${e.message}`);
      errorCount++;
    }

    if (!errorCount) {
      if (isEditMode) {
        if (!originalYaml) {
          messageApi.error(t('messages.yamlNotLoaded'));
          return;
        }
        try {
          const name = yamlTojson(originalYaml).metadata.name;
          const res = await updateCustomResourceDefinitionYamlData(name, yamlJson);
          if (res.status === ResponseCode.OK) {
            showSuccessThen(messageApi, t('messages.yamlModifySuccess'), () => {
              history.push(`/${containerRouterPrefix}/customResourceDefinition/${solveDecodePath(name)}`);
            });
          }
        } catch (error) {
          if (error.response?.status === ResponseCode.Forbidden) {
            forbiddenMsg(messageApi, error);
          } else {
            messageApi.error(`${t('messages.yamlModifyFailed')}${error.response?.data?.message}`);
          }
        }
      } else {
        try {
          const res = await addCustomResourceDefinitionYamlData(yamlJson);
          if (res.status === ResponseCode.Created) {
            showSuccessThen(messageApi, t('messages.createSuccess'), () => {
              history.push(`/${containerRouterPrefix}/customResourceDefinition`);
            });
          }
        } catch (error) {
          if (error.response?.status === ResponseCode.Forbidden) {
            forbiddenMsg(messageApi, error);
          } else {
            messageApi.error(`${t('messages.createFailed')}${error.response?.data?.message}`);
          }
        }
      }
    }
  };

  const handleResetCode = () => {
    const resetData = isEditMode ? originalYaml : customResourceDefinitionYamlExample;
    setYamlData(resetData);
    childCodeMirrorRef.current.resetCodeEditor(resetData);
  };

  const exportYaml = () => {
    const fileName = isEditMode ? solveDecodePath(customResourceName) : 'CustomResourceDefinition';
    exportYamlOutPut(fileName, yamlData);
    messageApi.success(t('messages.exportSuccess', { ns: 'common' }));
  };

  useEffect(() => {
    if (!isEditMode) { return undefined; }
    let cancelled = false;
    const fetchYaml = async () => {
      setYamlLoaded(false);
      try {
        const name = solveDecodePath(customResourceName);
        const res = await getCustomResourceDefinitionDetailDescription(name);
        if (cancelled) { return; }
        if (res.status === ResponseCode.OK) {
          const yaml = jsonToYaml(JSON.stringify(res.data));
          setYamlData(yaml);
          setOriginalYaml(yaml);
        }
      } catch (e) {
        if (!cancelled) {
          messageApi.error(t('messages.fetchFailed', { ns: 'common', message: e?.response?.data?.message }));
        }
      } finally {
        if (!cancelled) { setYamlLoaded(true); }
      }
    };
    fetchYaml();
    return () => {
      cancelled = true;
    };
  }, [customResourceName]);

  return (
    <div className="child_content">
      <div style={{ background: themeStore.$s.theme === 'dark' ? '#2a2d34ff' : '#fff', color: themeStore.$s.theme === 'dark' ? '#fff' : '#333' }}>
        {contextHolder}
      </div>
      <BreadCrumbCom className="create_bread" items={[
        { title: t('module.title'), path: `/${containerRouterPrefix}/customResourceDefinition` },
        { title: isEditMode ? t('actions.modify') : t('actions.create'), path: isEditMode ? '/yaml' : '/createCustomResourceDefinition' },
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
          {isEditMode && !yamlLoaded ? <SkeletonArea /> : <CodeMirrorEditor yamlData={yamlData} changeYaml={handleChangeYaml} ref={childCodeMirrorRef} />}
          <div className="btn_footer">
            <Button className="cancel_btn borderRadiusRight6" onClick={() => history.go(-1)}>{t('actions.cancel', { ns: 'common' })}</Button>
            <Button className="cancel_btn borderRadiusRight6" onClick={handleResetCode}>{t('actions.reset', { ns: 'common' })}</Button>
            <Button className="primary_btn borderRadiusRight6" onClick={handleSaveYaml} loading={saveLoading} disabled={isEditMode && !yamlLoaded}>{t('actions.confirm', { ns: 'common' })}</Button>
          </div>
        </div>
      </div>
    </div>
  );
}