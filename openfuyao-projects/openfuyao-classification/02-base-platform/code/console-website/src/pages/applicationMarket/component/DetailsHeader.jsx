/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import operatorIcon from '@/assets/images/operatorIcon.png';
import { Button, Select, message } from 'antd';
import { useHistory } from 'inula-router';
import { useEffect, useState, useStore } from 'openinula';
import { useTranslation } from 'react-i18next';
import '@/styles/applicationMarket/index.less';
import { containerRouterPrefix } from '@/constant.js';
import { listNamespaces } from '@/api/containerApi';
import { convertToRawUrl } from '@/tools/utils';
import ToastMsg from '@/components/ToastMsg';

export default function DetailsHeader({
  namespace,
  extensionName,
  version,
  icon,
  repo,
  chart,
  content,
  isOperator,
  operatorTitle,
  operatorDisabled,
  versionList,
  onSelect = () => { },
  onDeployClick,
  isExtension,
  children,
}) {
  const [versionSelect, setVersionSelect] = useState('');
  const [defaultNameSpace, setDefaultNameSpace] = useState('');
  const themeStore = useStore('theme');
  const { t } = useTranslation('common');
  const [messageApi, contextHolder] = message.useMessage();
  const handleSelect = (value) => {
    onSelect(value);
    setVersionSelect(value);
  };
  const history = useHistory();

  const [imgSrc, setImgSrc] = useState(() => {
    const initialUrl = icon || operatorIcon;
    return convertToRawUrl(initialUrl);
  });

  const getNameSpacesListInfo = async () => {
    try {
      const res = await listNamespaces();
      const arr = (res.data?.items ?? []).map((el) => ({
        label: el.metadata?.name,
        value: el.metadata?.name,
      }));
      if (arr.length) {
        setDefaultNameSpace(arr[0].value);
      }
    } catch (error) {
      messageApi.error(t('messages.fetchFailed', { message: error.response?.data?.message }));
    }
  };

  const goToManagementPage = () => {
    if (isExtension) {
      history.push(`/${containerRouterPrefix}/extendManage/${namespace}/${extensionName}`);
    } else {
      history.push(`/${containerRouterPrefix}/applicationManageHelm/${namespace}/${extensionName}`);
    }
  };

  useEffect(() => {
    const initialUrl = icon || operatorIcon;
    setImgSrc(convertToRawUrl(initialUrl));
  }, [icon]);

  useEffect(() => {
    getNameSpacesListInfo();
  }, []);

  useEffect(() => {
    setVersionSelect(version);
  }, [version]);

  return (
    <div className='details_header' style={{ backgroundColor: themeStore.$s.theme === 'light' ? '#fff' : '#2a2d34ff' }}>
      <ToastMsg contextHolder={contextHolder} />
      <div style={{ display: 'flex', gap: '20px', marginRight: '64px' }}>
        <div>
          <img src={imgSrc} className='img_icon' />
        </div>
        <div>
          <div style={{ display: 'flex', gap: '20px', marginBottom: '15px', alignItems: 'center' }}>
            <div className='clamp_1' style={{ fontWeight: 'bold', maxWidth: '517px', color: themeStore.$s.theme === 'light' ? 'black' : '#fff' }}>{chart}</div>
            {versionList && (
              <Select
                style={{ width: '160px', height: '21px' }}
                value={versionSelect}
                options={versionList}
                onChange={handleSelect}
              ></Select>
            )}
          </div>
          <div style={{ fontSize: '14px', color: '#89939b', marginBottom: '15px' }} className='clamp_3'>{content}</div>
          {children}
        </div>
      </div>
      <div>
        {isOperator === '1' && (
          <div>
            <Button
              className='primary_btn'
              onClick={() => goToManagementPage()}
              disabled={operatorDisabled}
            >
              {operatorTitle}
            </Button>
          </div>
        )}
        {isOperator === '2' && (
          <div>
            <Button
              className='primary_btn'
              onClick={() => {
                if (onDeployClick) {
                  onDeployClick(defaultNameSpace);
                  return;
                }
                history.push(`/${containerRouterPrefix}/appMarket/marketCategory/Deploy/${repo}/${chart}/${versionSelect || versionList[0]?.value}/${defaultNameSpace}`);
              }}
              disabled={operatorDisabled}
            >
              {operatorTitle}
            </Button>
          </div>
        )}
      </div>
    </div>
  );
}
