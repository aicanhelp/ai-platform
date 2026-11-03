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

import { useCallback, useEffect, useState, useRef, useStore, useMemo } from 'openinula';
import { useTranslation } from 'react-i18next';
import { useParams, useHistory } from 'inula-router';
import BreadCrumbCom from '@/components/BreadCrumbCom';
import { containerRouterPrefix } from '@/constant.js';
import { getHelmUpgradeYamlData, getHelmDetailDescriptionData, getHelmTemplateDetailVersion, updateHelmLevelYaml } from '@/api/containerApi';
import { Form, Select, Tabs, Button, message } from 'antd';
import { ResponseCode } from '@/common/constants';
import copy from 'copy-to-clipboard';
import CodeMirrorEditor from '@/components/CodeMirrorEditor';
import DiffComponent from '@/components/YamlDiff';
import { jsonToYaml, exportYamlOutPut, showSuccessThen } from '@/tools/utils';
import { ExportOutlined, CopyOutlined } from '@ant-design/icons';
import defaultIcon from '@/assets/images/helmIcon.png';
import Dayjs from 'dayjs';
import { filterRepeat } from '@/utils/common';
import ToastMsg from '@/components/ToastMsg';
import DetailBaseInfoList from '@/components/DetailBaseInfoList';

/**
 * Application/Extension Upgrade Layout Component
 *
 * @param {'application'|'extension'} type - Type: 'application' or 'extension'
 */
export default function ApplicationUpgradeLayout({ type = 'application' }) {
  const isExtension = type === 'extension';
  const { t } = useTranslation('application');
  const routePrefix = isExtension ? 'extendManage' : 'applicationManageHelm';
  const paramName = isExtension ? 'extend_name' : 'helm_name';
  const paramNamespace = isExtension ? 'extend_namespace' : 'helm_namespace';

  const param = useParams();
  const history = useHistory();
  const [messageApi, contextHolder] = message.useMessage();
  const [form] = Form.useForm();
  const themeStore = useStore('theme');

  // State
  const [levelOptions, setLevelOptions] = useState([{ label: '', value: '' }]);
  const [loading, setLoading] = useState(true);
  const [upgradeLoading, setUpgradeLoading] = useState(false);
  const [tabKey, setTabKey] = useState('1');
  const childCodeMirrorRef = useRef(null);
  const resetYamlTimerRef = useRef(null);
  const [yamlData, setYamlData] = useState();
  const [beforeYamlData, setBeforeYamlData] = useState();
  const [upgradeInfoData, setUpgradeInfoData] = useState({});
  const [chartName, setChartName] = useState('');
  const [repoName, setRepoName] = useState('');
  const [currentVersion, setCurrentVersion] = useState('');
  const [currentYamlValue, setCurrentYamlValue] = useState('');

  const name = param[paramName];
  const namespace = param[paramNamespace];

  // Text configuration
  const textConfig = {
    breadcrumbTitle: isExtension ? t('extension.title') : t('module.title'),
    detailTitle: isExtension ? t('upgrade.extensionDetail') : t('upgrade.helmDetail'),
    upgradeTitle: isExtension ? t('upgrade.extensionUpgrade') : t('upgrade.helmUpgrade'),
  };

  // Handlers
  const handleCopy = () => {
    copy(yamlData);
    messageApi.success(t('messages.copySuccess', { ns: 'common' }));
  };

  const handleExportYaml = () => {
    exportYamlOutPut('Values', beforeYamlData);
    messageApi.success(t('messages.exportSuccess', { ns: 'common' }));
  };

  const handleChangeYaml = (yaml) => setYamlData(yaml);

  const handleSetTabKey = (key) => setTabKey(key);

  const handleResetCode = () => {
    if (isExtension) {
      setYamlData('');
      getYamlData(repoName, chartName);
    } else {
      setYamlData(beforeYamlData);
      childCodeMirrorRef.current.resetCodeEditor(beforeYamlData);
    }
  };

  useEffect(() => () => clearTimeout(resetYamlTimerRef.current), []);

  const handleYamlOnchange = (value) => {
    if (currentVersion === value) {
      setLoading(true);
      resetYamlTimerRef.current = setTimeout(() => {
        setBeforeYamlData(currentYamlValue);
        setYamlData(currentYamlValue);
        setLoading(false);
      }, 2000);
    } else {
      getYamlData(repoName, chartName);
    }
  };

  // Get level options
  const getLevelOptions = useCallback(async (repo, chart) => {
    try {
      const res = await getHelmTemplateDetailVersion(repo, chart);
      if (res.status === ResponseCode.OK) {
        let options = [];
        res.data.data.forEach(item => {
          options.push({ value: item.metadata.version, label: item.metadata.version });
        });
        options = filterRepeat(options);
        setLevelOptions([...options]);
      }
    } catch {
      messageApi.error(t('messages.fetchVersionFailed'));
    }
  }, []);

  // Get YAML data
  const getYamlData = useCallback(async (repo, chart) => {
    setLoading(true);
    try {
      const resYaml = await getHelmUpgradeYamlData(repo, chart, form.getFieldsValue().version);
      if (resYaml.status === ResponseCode.OK) {
        let replaceData = resYaml.data.data;
        if (!replaceData['values.yaml']) {
          replaceData['values.yaml'] = '';
        }
        replaceData['values.yaml'] = replaceData['values.yaml'].replace(/\r/g, '');
        setBeforeYamlData(replaceData['values.yaml']);
        setYamlData(replaceData['values.yaml']);
        setLoading(false);
      }
    } catch {
      messageApi.error(t('messages.fetchYamlFailed'));
    }
  }, []);

  // Get detail data
  const getDetail = useCallback(async () => {
    try {
      const res = await getHelmDetailDescriptionData(namespace, name);
      if (res.status === ResponseCode.OK) {
        setUpgradeInfoData(res.data.data);
        form.setFieldsValue({ version: res.data.data.chart.metadata.version });
        setCurrentVersion(res.data.data.chart.metadata.version);
        setCurrentYamlValue(jsonToYaml(JSON.stringify(res.data.data.values)));
        setRepoName(res.data.data.labels['openfuyao.io.repo']);
        setChartName(res.data.data.chart.metadata.name);

        if (!res.data.data.labels['openfuyao.io.repo']) {
          messageApi.error(t('messages.notPlatformSource'));
          return;
        }

        if (res.data.data.chart.metadata.version === form.getFieldValue('version')) {
          setBeforeYamlData(jsonToYaml(JSON.stringify(res.data.data.values)));
          setYamlData(jsonToYaml(JSON.stringify(res.data.data.values)));
          setLoading(false);
        } else {
          getYamlData(res.data.data.labels['openfuyao.io.repo'], res.data.data.chart.metadata.name);
        }
        getLevelOptions(res.data.data.labels['openfuyao.io.repo'], res.data.data.chart.metadata.name);
      }
    } catch (e) {
      messageApi.error(t('messages.fetchDataError', { ns: 'application', message: e.response?.data?.msg }));
    }
  }, [namespace, name]);

  // Update YAML
  const updateYaml = async () => {
    setUpgradeLoading(true);
    let data = {
      chartName: upgradeInfoData.chart.metadata.name,
      repoName: upgradeInfoData.labels['openfuyao.io.repo'],
      version: form.getFieldsValue().version,
      values: yamlData,
    };
    try {
      const res = await updateHelmLevelYaml(upgradeInfoData.namespace, upgradeInfoData.name, data);
      if (res.status === ResponseCode.Accepted) {
        await showSuccessThen(messageApi, t('messages.upgradeSuccess'), () => {
          history.push(`/${containerRouterPrefix}/${routePrefix}/${namespace}/${name}`);
        });
      }
    } catch (error) {
      if (error.response.status === ResponseCode.Forbidden) {
        messageApi.error(t('messages.noPermission'));
      } else {
        messageApi.error(t('messages.upgradeFailed', { message: error.response?.data?.msg }));
      }
    }
    setUpgradeLoading(false);
  };

  // Tab items
  const items = [
    {
      label: t('upgrade.valuesYaml'),
      key: '1',
      children: (
        <div>
          {!loading ? (
            <div className={isExtension ? 'extend_level_flex_box' : 'level_flex_box'}>
              <h3 className={`level_title ${isExtension ? '' : 'defaultClass'}`}>{t('upgrade.yamlReadWrite')}</h3>
              <div className="level_tools">
                <div className="level_tool_word_group" onClick={handleExportYaml}>
                  <ExportOutlined className="common_antd_icon primary_color" />
                  <span>{t('resourceYaml.export', { ns: 'common' })}</span>
                </div>
                <div className="level_tool_word_group" onClick={handleCopy}>
                  <CopyOutlined className="common_antd_icon primary_color" />
                  <span>{t('resourceYaml.copy', { ns: 'common' })}</span>
                </div>
              </div>
            </div>
          ) : <div></div>}
          <div className={`level_yaml_space_box ${isExtension ? '' : 'defaultClass'}`}>
            {!loading && (
              <CodeMirrorEditor
                yamlData={yamlData}
                changeYaml={handleChangeYaml}
                ref={childCodeMirrorRef}
              />
            )}
          </div>
        </div>
      ),
    },
    {
      key: '2',
      label: t('upgrade.diff'),
      children: (
        <div className={isExtension ? 'extend_diff_box' : 'helm_diff_box'}>
          <DiffComponent
            diffDataList={[{
              oldData: beforeYamlData,
              newData: yamlData,
              isYamls: true,
              oldHeader: t('upgrade.oldConfig'),
              newHeader: t('upgrade.newConfig'),
            }]}
            outputFormat={'side-by-side'}
          />
        </div>
      ),
    },
  ];

  useEffect(() => {
    getDetail();
  }, [getDetail]);

  // CSS class names based on type
  const tabTopClass = isExtension ? 'extend-tab-top' : 'helm-tab-top';
  const upgradeTitleClass = isExtension ? 'extend_upgrade_title' : 'helm_upgrade_title';
  const basicBoxClass = isExtension ? 'extend_upgrade_basic_box' : 'upgrade_basic_box';
  const upgradeBoxClass = isExtension ? 'extend_upgrade_box' : 'upgrade_box';
  const levelCardClass = isExtension ? 'extend_level_card' : 'level_card';
  const formClass = isExtension ? 'extend_level_form' : 'level_form';
  const versionSelectClass = isExtension ? 'extend_version_select' : 'version_select';

  const baseInfoItems = useMemo(() => [
    { key: t('detail.appName'), value: upgradeInfoData.name },
    { key: t('detail.chartVersion'), value: upgradeInfoData.chart?.metadata?.version },
    { key: t('detail.namespace'), value: upgradeInfoData.namespace },
    { key: t('detail.createTime'), value: Dayjs(upgradeInfoData.info?.firstDeployed).format('YYYY-MM-DD HH:mm') },
    { key: t('detail.chartName'), value: upgradeInfoData.chart?.metadata?.name },
    { key: t('detail.updateTime'), value: Dayjs(upgradeInfoData.info?.lastDeployed).format('YYYY-MM-DD HH:mm') },
  ], [upgradeInfoData, t]);

  return (
    <div className={`child_content withBread_content ${isExtension ? 'extend_all' : ''}`}>
      <ToastMsg contextHolder={contextHolder} />
      <div className={tabTopClass}>
        <BreadCrumbCom items={[
          { title: textConfig.breadcrumbTitle, path: `/${containerRouterPrefix}/${routePrefix}` },
          { title: textConfig.detailTitle, path: `/${namespace}/${name}` },
          { title: textConfig.upgradeTitle },
        ]} />
      </div>
      <div className={upgradeTitleClass}>
        <div style={{ display: 'flex' }}>
          <div>
            <img
              src={defaultIcon}
              alt=""
              style={{ height: '30px', width: '30px', marginRight: '8px' }}
              className='title_image'
            />
          </div>
          <div className={`upgrade_descript_group ${isExtension ? '' : 'defaultClass'}`}>
            <div style={{ marginRight: '64px' }}>
              <h3 className={`upgrade_descript_group_name ${isExtension ? '' : 'defaultClass'}`}>
                {upgradeInfoData.name}
              </h3>
            </div>
            <div style={{ marginRight: '64px' }}>
              <p className={`upgrade_descript_group_description ${isExtension ? '' : 'defaultClass'}`}>
                {upgradeInfoData.chart?.metadata?.description}
              </p>
            </div>
          </div>
        </div>
      </div>
      <div className={basicBoxClass}>
        <div
          className={`detail_card basic_level_card ${isExtension ? '' : 'defaultClass'}`}
          style={{
            padding: '32px 32px 0px 32px',
            background: themeStore.$s.theme === 'light' ? '#fff' : '#2a2d34',
          }}
        >
          <h3 className="box_title_h3">{t('upgrade.currentVersionInfo')}</h3>
          <div className="detail_info_box">
            <DetailBaseInfoList items={baseInfoItems} columns={2} />
          </div>
          <div className={levelCardClass}>
            <h3 className="box_title_h3">{t('upgrade.upgradeInfo')}</h3>
            <Form form={form} layout="vertical" className={formClass}>
              <Form.Item
                label={t('upgrade.versionSelect')}
                name="version"
                rules={[{ required: true, message: t('upgrade.versionRequired') }]}
              >
                <Select
                  options={levelOptions}
                  className={versionSelectClass}
                  onChange={handleYamlOnchange}
                />
              </Form.Item>
            </Form>
          </div>
        </div>
      </div>

      <div className={upgradeBoxClass}>
        <div className="level_card">
          <h3 className="box_title_h3">{t('upgrade.paramConfig')}</h3>
          <Tabs items={items} onChange={handleSetTabKey} activeKey={tabKey} />
        </div>
      </div>
      <div className="level_btn_footer">
        <div className='level_btn_footer_button'>
          <Button className='cancel_btn' style={isExtension ? {} : { marginRight: '16px' }} onClick={() => history.go(-1)}>
            {t('actions.cancel')}
          </Button>
          <Button
            className={loading ? 'disable_btn' : 'cancel_btn'}
            style={{ marginRight: '16px' }}
            disabled={loading}
            onClick={handleResetCode}
          >
            {t('actions.reset')}
          </Button>
          <Button
            className={loading ? 'disable_btn' : 'primary_btn'}
            disabled={loading}
            loading={upgradeLoading}
            onClick={updateYaml}
          >
            {t('actions.confirm')}
          </Button>
        </div>
      </div>
    </div>
  );
}