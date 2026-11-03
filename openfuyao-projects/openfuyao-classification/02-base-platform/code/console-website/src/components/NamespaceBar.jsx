/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import '@/styles/components/namespaceBar.less';
import { useCallback, useEffect, useState, useStore } from 'openinula';
import { NameSpaceIcon } from '@/assets/icon';
import { Select, message } from 'antd';
import { listNamespaces } from '@/api/containerApi';
import { ResponseCode } from '@/common/constants';
import { useHistory } from 'inula-router';
import { useTranslation } from 'react-i18next';

export default function NamespaceBar() {
  const namespaceExampleStore = useStore('namespace');
  const { t, i18n } = useTranslation('nav');
  const themeStore = useStore('theme');
  const history = useHistory();
  const [namespace, setNamespace] = useState('');
  const [messageApi, contextHolder] = message.useMessage();
  const [namespaceOptions, setNamespaceOptions] = useState([]);
  const getNameSpaceOptionsList = useCallback(async () => {
    let namespaceList = [{ value: '', label: t('namespace.all') }];
    try {
      const res = await listNamespaces();
      if (res.status === ResponseCode.OK) {
        res.data.items.forEach(item => {
          namespaceList.push({ value: item.metadata.name, label: item.metadata.name });
        });
      }
    } catch {
      messageApi.error(t('namespace.namespaceGetError'));
    }
    setNamespaceOptions([...namespaceList]);
  }, [t, i18n.language]);

  const handleNameSpaceChange = (value) => {
    setNamespace(value);
    // 设置Store存储
    namespaceExampleStore.$a.setNamespace({ name: value });
    // 判断是否二级页面跳转
    const arrList = window.location.pathname.split('/');
    if (arrList.length > 4) {
      // 判断是否为应用管理和扩展管理
      if (arrList.includes('applicationManageHelm') || arrList.includes('extendManage')) {
        setTimeout(() => {
          history.push(arrList.slice(0, 3).join('/'));
        }, 0);
      } else {
        // 异步跳转
        setTimeout(() => {
          history.push(arrList.slice(0, 4).join('/'));
        }, 0);
      }
    }
  };

  const filterOption = (input, option) => (option?.label ?? '').toLowerCase().includes(input.toLowerCase());

  useEffect(() => {
    getNameSpaceOptionsList();
  }, [getNameSpaceOptionsList]);

  return <div className='namespace_total'>
    <div style={{ background: themeStore.$s.theme === 'dark' ? '#2a2d34ff' : '#fff', color: themeStore.$s.theme === 'dark' ? '#fff' : '#333' }}>
      {contextHolder}
    </div>
    <div className='namespace_bar'>
      <NameSpaceIcon width={16} height={16} fill='#477DD8' style={{ marginRight: '8px' }} />
      <p className='label'>{t('namespace.label')}：</p>
      <Select
        value={namespace}
        showSearch
        filterOption={filterOption}
        popupMatchSelectWidth={false}
        popupClassName='namespace_popup_options'
        options={namespaceOptions}
        onChange={handleNameSpaceChange}
      />
    </div>
  </div>;
}