/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import { Form, Input, Select, Tabs, Button, message } from 'antd';
import DetailsHeader from '@/pages/applicationMarket/component/DetailsHeader';
import '@/styles/applicationMarket/index.less';
import { useEffect, useState, useRef, useStore } from 'openinula';
import { containerRouterPrefix } from '@/constant.js';
import { Link, useHistory, useParams } from 'inula-router';
import {
  installReleaseToNameSpace,
  getAllVersionInfo,
  getRepoChartVersionFile,
  getAppointVersionChart,
} from '@/api/applicationMarketApi';
import { ResponseCode } from '@/common/constants';
import { listNamespaces } from '@/api/containerApi';
import DiffComponent from '@/components/YamlDiff';
import { exportYamlOutPut, showSuccessThen } from '@/tools/utils';
import CodeMirrorEditor from '@/components/CodeMirrorEditor';
import {
  ExportOutlined,
  CopyOutlined,
} from '@ant-design/icons';
import copy from 'copy-to-clipboard';
import '@/styles/applicationMarket/dark.less';
import BreadCrumbCom from '@/components/BreadCrumbCom';
import { useTranslation } from 'react-i18next';
import useButtonLoading from '@/hooks/useButtonLoading';

export default function Deploy() {
  const history = useHistory();
  const { chart, repo, versionSelect, defaultNameSpace } = useParams();
  const { t } = useTranslation(['appMarket', 'common']);
  const [form] = Form.useForm();
  const [loaded, setLoaded] = useState(false);
  const [isLoading, setIsLoading] = useButtonLoading();
  const [nameSpaceList, setNameSpaceList] = useState([]);
  const [versionList, setVersionList] = useState([]);
  const [version, setVersion] = useState('');
  const [nameSpace, setNameSpace] = useState('');
  const [applicationName, setApplicationName] = useState('');
  const [basicInfo, setBasicInfo] = useState(null);
  const [beforeYamlData, setBeforeYamlData] = useState('');
  const [paramsYamlData, setParamsYamlData] = useState('');
  const childCodeMirrorRef = useRef(null);
  const resetYamlTimerRef = useRef(null);
  useEffect(() => () => clearTimeout(resetYamlTimerRef.current), []);
  const themeStore = useStore('theme');
  const [messageApi, contextHolder] = message.useMessage();
  // 初始表单数值
  const initialValues = {
    version: versionSelect,
    nameSpace: defaultNameSpace,
  };
  const handleChangeYaml = (yaml) => {
    setParamsYamlData(yaml);
  };

  const exportYaml = () => {
    exportYamlOutPut('Values', paramsYamlData);
    messageApi.success(t('messages.exportSuccess', { ns: 'common' }));
  };
  const handleCopyYaml = () => {
    copy(paramsYamlData);
    messageApi.success(t('messages.copySuccess', { ns: 'common' }));
  };

  const items = [
    {
      key: '1',
      label: 'Values.yaml',
      children: (
        <div>
          <div className='yaml_card'>
            <div className='yaml_flex_box'>
              <h3 style={{ color: themeStore.$s.theme === 'light' ? 'black' : '#fff' }}>{t('deploy.valueReadWrite')}</h3>
              <div className='yaml_tools'>
                <div className='tool_word_group deployExport' onClick={exportYaml}>
                  <ExportOutlined className='common_antd_icon primary_color' />
                  <span>{t('deploy.export')}</span>
                </div>
                <div className='tool_word_group deployCopy' onClick={handleCopyYaml}>
                  <CopyOutlined className='common_antd_icon primary_color' />
                  <span>{t('deploy.copy')}</span>
                </div>
              </div>
            </div>
          </div>
          <CodeMirrorEditor
            yamlData={paramsYamlData}
            changeYaml={handleChangeYaml}
            ref={childCodeMirrorRef}
          />
        </div>
      ),
    },
    {
      key: '2',
      label: t('deploy.changes'),
      children: (
        <DiffComponent
          diffDataList={[
            {
              oldData: beforeYamlData,
              newData: paramsYamlData,
              isYamls: true,
              oldHeader: t('deploy.oldConfig'),
              newHeader: t('deploy.newConfig'),
            },
          ]}
          outputFormat={'side-by-side'}
        />
      ),
    },
  ];

  const getNameSpacesListInfo = async () => {
    try {
      const res = await listNamespaces();
      const arr = (res.data?.items || []).map((el) => ({
        label: el.metadata.name,
        value: el.metadata.name,
      }));
      setNameSpace(defaultNameSpace || arr[0]?.value || '');
      setNameSpaceList(arr);
    } catch (e) {
      messageApi.error(t('messages.fetchDataError', { ns: 'application', message: e.response?.data?.msg }));
    }
  };

  const getAllVersionInfoList = async () => {
    const res = await getAllVersionInfo(repo, chart);
    const versionArr = res.data.data.map((el) => ({
      label: el.metadata.version,
      value: el.metadata.version,
    }));
    setVersionList(versionArr);
  };

  const isGoToOperationComponent = (arr) => {
    return arr.includes('openfuyao-extension');
  };

  const handleDeploy = async () => {
    try {
      await form.validateFields();
    } catch {
      return;
    }
    setIsLoading(true);
    try {
      await installReleaseToNameSpace(nameSpace, applicationName, {
        chartName: chart,
        repoName: repo,
        version,
        values: paramsYamlData,
      });
      if (basicInfo?.keywords && isGoToOperationComponent(basicInfo?.keywords)) {
        await showSuccessThen(messageApi, t('deploy.deployExtensionSuccess'), () => {
          setIsLoading(false);
          window.location = `/${containerRouterPrefix}/extendManage`;
        });
      } else {
        await showSuccessThen(messageApi, t('deploy.deploySuccess'), () => {
          setIsLoading(false);
          history.push(`/${containerRouterPrefix}/applicationManageHelm`);
        });
      }
    } catch (error) {
      setIsLoading(false);
      if (error?.response?.status === ResponseCode.Forbidden) {
        messageApi.error(t('messages.noPermission'));
      } else {
        messageApi.error(t('messages.fetchDataError', { ns: 'application', message: error.response?.data?.msg }));
      }
    }
  };

  const detailRequestSeqRef = useRef(0);
  const getDetailInfo = async (targetVersion) => {
    const seq = ++detailRequestSeqRef.current;
    try {
      const res = await getRepoChartVersionFile(
        repo,
        chart,
        targetVersion || versionSelect,
        'detail',
      );
      if (seq !== detailRequestSeqRef.current) {
        return;
      }
      const replaceYamlData = res.data.data;
      replaceYamlData.values = (replaceYamlData.values || '').replace(/\r/g, '');
      setParamsYamlData(replaceYamlData.values);
      setBeforeYamlData(replaceYamlData.values);
      clearTimeout(resetYamlTimerRef.current);
      resetYamlTimerRef.current = setTimeout(() => {
        childCodeMirrorRef?.current?.resetCodeEditor(replaceYamlData.values);
      }, 0);
    } catch (e) {
      if (seq !== detailRequestSeqRef.current) {
        return;
      }
      messageApi.error(t('messages.fetchDataError', { ns: 'application', message: e.response?.data?.msg }));
    } finally {
      setLoaded(true);
    }
  };

  const handleSelectVersion = (e) => {
    setVersion(e);
  };

  const handleSelectNameSpace = (e) => {
    setNameSpace(e);
  };

  const handleInputName = (e) => {
    setApplicationName(e.target.value);
  };

  const getAppointVersionChartInfo = async () => {
    try {
      const res = await getAppointVersionChart(repo, chart, version);
      setBasicInfo(res.data?.data?.[0]?.metadata);
    } catch (e) {
      messageApi.error(t('messages.fetchDataError', { ns: 'application', message: e.response?.data?.msg }));
    }
  };

  const onFinish = () => { };
  useEffect(() => {
    setVersion(versionSelect);
    getAllVersionInfoList();
    getNameSpacesListInfo();
  }, []);

  useEffect(() => {
    if (!version) {
      return;
    }
    setLoaded(false);
    getDetailInfo(version);
  }, [version]);

  useEffect(() => {
    if (version) {
      getAppointVersionChartInfo();
    }
  }, [version]);

  return (
    <div style={{ backgroundColor: themeStore.$s.theme === 'light' ? '#f7f7f7' : '#171a1f' }} className={themeStore.$s.theme === 'light' ? '' : 'dark_box'}>
      <div style={{ background: themeStore.$s.theme === 'dark' ? '#2a2d34ff' : '#fff', color: themeStore.$s.theme === 'dark' ? '#fff' : '#333' }}>
        {contextHolder}
      </div>
      <div className='application_breadcrumb'>
        <BreadCrumbCom
          items={[
            {
              title: t('details.appMarket'),
              path: `/${containerRouterPrefix}/appMarket`,
            },
            {
              title: t('details.appList'),
              path: '/marketCategory',
            },

            {
              title: t('deploy.appDetails'),
              path: `/ApplicationDetails/${chart}/${repo}/${versionSelect}`,
            },
            {
              title: t('deploy.title'),
            },
          ]}
        />
      </div>
      <div>
        <div className='detail_container' style={{ marginBottom: '20px' }}>
          <DetailsHeader
            chart={chart}
            version={version}
            icon={basicInfo?.icon}
            content={basicInfo?.description}
          ></DetailsHeader>
        </div>
        <Form
          layout='vertical'
          form={form}
          initialValues={initialValues}
          name='form_in_modal'
          onFinish={onFinish}
          style={{ paddingBottom: '100px' }}
        >
          <div
            className='detail_container installation_information'
            style={{ marginBottom: '20px', backgroundColor: themeStore.$s.theme === 'light' ? '#fff' : '#2a2d34ff' }}
          >
            <div className='installation_information_title'>{t('deploy.installInfo')}</div>
            <div style={{ display: 'flex', flexDirection: 'column', marginBottom: '8px' }}>
              <Form.Item label={t('deploy.appName')} name='applicationName'
                rules={[
                  {
                    required: true,
                    message: t('deploy.appNameRequired'),
                  },
                  {
                    pattern: /^[a-z0-9](?:[a-z0-9.-]{0,50}[a-z0-9])?$/,
                    message: t('deploy.appNamePattern'),
                  },
                ]}
              >
                <Input style={{ width: '600px' }} placeholder={t('deploy.appNamePlaceholder')} onChange={handleInputName} />
              </Form.Item>
            </div>
            <Form.Item label={t('deploy.version')} name='version'
              rules={[
                {
                  required: true,
                  message: t('deploy.versionRequired'),
                },
              ]}
              style={{ display: 'flex', flexDirection: 'column', marginBottom: '0px' }}
            >
              <Select
                options={versionList}
                onChange={handleSelectVersion}
                style={{ width: '600px', marginBottom: '32px' }}
              ></Select>
            </Form.Item>
            <Form.Item label={t('deploy.namespace')} name='nameSpace'
              rules={[
                {
                  required: true,
                  message: t('deploy.namespaceRequired'),
                },
              ]}
              style={{ display: 'flex', flexDirection: 'column', marginBottom: '0px' }}
            >
              <Select
                options={nameSpaceList}
                onChange={handleSelectNameSpace}
                style={{ width: '600px' }}
              ></Select>
            </Form.Item>
          </div>
          <div>
            <div className='detail_container' style={{ backgroundColor: themeStore.$s.theme === 'light' ? '#fff' : '#2a2d34ff', padding: '32px' }}>
              <div className='installation_information_title' style={{ color: themeStore.$s.theme === 'light' ? 'black' : '#fff' }}>{t('deploy.paramsConfig')}</div>
              {loaded && <Tabs items={items} tabBarStyle={{ backgroundColor: themeStore.$s.theme === 'light' ? '#fff' : '#2a2d34ff', color: '#000' }} />}
            </div>
          </div>
          <div className='deploy_footer'>
            <div
              className='detail_container'
              style={{ display: 'flex', gap: '16px' }}
            >
              <Link
                to={`/${containerRouterPrefix}/appMarket/marketCategory/ApplicationDetails/${chart}/${repo}/${versionSelect}`}
              >
                <Button className='cancel_btn'>{t('deploy.cancel')}</Button>
              </Link>
              <Button
                className='primary_btn'
                loading={isLoading}
                htmlType='submit'
                onClick={handleDeploy}
                disabled={!loaded}
              >
                {isLoading ? t('deploy.deploying') : t('actions.confirm', { ns: 'common' })}
              </Button>
            </div>
          </div>
        </Form>
      </div>
    </div>
  );
}
