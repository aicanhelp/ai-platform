/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import { Checkbox, Tooltip, message } from 'antd';
import { ClearOutlined } from '@ant-design/icons';
import { useState, useEffect, useMemo } from 'openinula';
import { getHelmRepoList } from '@/api/applicationMarketApi';
import '@/styles/applicationMarket/index.less';
import { QuestionCircleOutlined, UpOutlined, DownOutlined } from '@ant-design/icons';
import { useTranslation } from 'react-i18next';
import ToastMsg from '@/components/ToastMsg';

export function ApplicationType({
  sceneCheckedList = [],
  sourceCheckedList = [],
  extensionChecked = false,
  computingChecked = false,
  onSelectType,
  onSelectSource,
  onExtensionComponent,
  onComputingEngine,
  onClearFilter,
}) {
  const { t, i18n } = useTranslation('appMarket');
  const [messageApi, contextHolder] = message.useMessage();

  const [sourceOptions, setSourceOptions] = useState([]);
  const [isSceneExtend, setIsSceneExtend] = useState(true);
  const [isSourceExtend, setIsSourceExtend] = useState(true);
  const [moreSourceOption, setMoreSourceOption] = useState([]);
  const CheckboxGroup = Checkbox.Group;
  const onChange = (list) => {
    onSelectType(list);
  };

  const plainOptions = useMemo(() => [
    { label: t('category.artificialIntelligence'), value: 'artificial-intelligence' },
    { label: t('category.computing'), value: 'computing' },
    { label: t('category.database'), value: 'database' },
    { label: t('category.developerTool'), value: 'developer-tool' },
    { label: 'CI/CD', value: 'CI/CD' },
    { label: t('category.monitor'), value: 'monitor' },
    { label: t('category.log'), value: 'log' },
    { label: t('category.network'), value: 'network' },
    { label: t('category.observability'), value: 'observability' },
    { label: t('category.security'), value: 'security' },
    { label: t('category.storage'), value: 'storage' },
    { label: t('category.bigData'), value: 'big-data' },
  ], [t, i18n.language]);

  const handleExtensionComponent = (e) => {
    if (e.target.checked) {
      onExtensionComponent('openfuyao-extension');
    } else {
      onExtensionComponent('');
    }
  };

  const handleComputingEngine = (e) => {
    onComputingEngine(e.target.checked);
  };
  const handleSelectSource = (list) => {
    onSelectSource(list);
  };

  const handleClear = () => {
    if (onClearFilter) {
      onClearFilter();
      return;
    }
    onSelectSource([]);
    onSelectType([]);
    onExtensionComponent('');
    onComputingEngine(false);
  };

  const getHelmRepoListInfo = async () => {
    try {
      const res = await getHelmRepoList('', '', 1, -1);
      const data = (res.data?.data?.items ?? []).map((el) => el.name);
      if (data.length > 10) {
        setMoreSourceOption(data.slice(0, 10));
      }
      setSourceOptions(data);
    } catch (error) {
      messageApi.error(t('messages.fetchFailed', { ns: 'common', message: error.response?.data?.message }));
    }
  };

  const getMore = () => {
    setMoreSourceOption(sourceOptions);
  };

  useEffect(() => {
    getHelmRepoListInfo();
  }, []);
  return (
    <div className='application_type'>
      <ToastMsg contextHolder={contextHolder} />
      <div className='application_type_item'>
        <div
          className='application_type_title'
          style={{ display: 'flex', justifyContent: 'space-between' }}
        >
          <div>{t('filter.appType')}</div>
          <div onClick={handleClear} style={{ zIndex: '99999', cursor: 'pointer' }}><ClearOutlined /></div>
        </div>
        <div>
          <Checkbox style={{ marginBottom: '15px' }} checked={computingChecked} onChange={handleComputingEngine}>
            <div style={{ display: 'flex', gap: '8px' }}>
              <span>{t('filter.computeEngine')}</span>
              <span>
                <Tooltip title={t('tooltip.computeEngine')} placement='top'>
                  <QuestionCircleOutlined style={{ color: '#89939bff' }} />
                </Tooltip>
              </span>
            </div>
          </Checkbox>
          <Checkbox checked={extensionChecked} onChange={handleExtensionComponent}>
            <div style={{ display: 'flex', gap: '8px' }}>
              <span>{t('filter.extension')}</span>
              <span>
                <Tooltip title={t('tooltip.extension')} placement='top'>
                  <QuestionCircleOutlined style={{ color: '#89939bff' }} />
                </Tooltip>
              </span>
            </div>
          </Checkbox>
        </div>
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
        <div style={{ display: 'flex', gap: '25px', alignItems: 'center' }}>
          <div className='application_type_title'>
            {t('filter.scene')}
          </div>
          {!isSceneExtend && <div><UpOutlined onClick={() => setIsSceneExtend(true)} style={{ color: '#89939bff' }} /></div>}
          {isSceneExtend && <div><DownOutlined onClick={() => setIsSceneExtend(false)} style={{ color: '#89939bff' }} /></div>}
        </div>
        {isSceneExtend && <CheckboxGroup
          options={plainOptions}
          value={sceneCheckedList}
          onChange={onChange}
        />}
      </div>
      <div className='application_type_item'>
        <div style={{ display: 'flex', gap: '25px', alignItems: 'center' }}>
          <div className='application_type_title'>{t('category.source')}</div>
          {!isSourceExtend && <div><UpOutlined onClick={() => setIsSourceExtend(true)} style={{ color: '#89939bff' }} /></div>}
          {isSourceExtend && <div><DownOutlined onClick={() => setIsSourceExtend(false)} style={{ color: '#89939bff' }} /></div>}
        </div>
        {isSourceExtend && sourceOptions.length <= 10 && <div>
          <CheckboxGroup
            options={sourceOptions}
            value={sourceCheckedList}
            onChange={handleSelectSource}
          />
        </div>
        }
        {isSourceExtend && sourceOptions.length > 10 && <div><CheckboxGroup
          options={moreSourceOption}
          value={sourceCheckedList}
          onChange={handleSelectSource}
          style={{ marginBottom: '18px' }}
        />
        {sourceOptions.length !== moreSourceOption.length && <div style={{ color: '#3f66f5', cursor: 'pointer' }} onClick={getMore}>{t('category.getMore')}</div>}
        </div>
        }
      </div>
    </div>
  );
}
