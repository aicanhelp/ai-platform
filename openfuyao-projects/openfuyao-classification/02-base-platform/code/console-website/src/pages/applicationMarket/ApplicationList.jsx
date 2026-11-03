/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import ApplicationRecommendItem from './ApplicationRecommendItem';
import { Select, Input, message } from 'antd';
import { useEffect, useRef, useStore } from 'openinula';
import EmptyData from '@/components/EmptyData';
import ToastMsg from '@/components/ToastMsg';
import { useTranslation } from 'react-i18next';

const limitSearch = 53;
const { Search } = Input;
export function ApplicationList({
  helmChartListData,
  searchName,
  sortType = 'name',
  onSearchFn,
  onSelectFn,
  onInputChartFn,
  total,
}) {
  const themeStore = useStore('theme');
  const [messageApi, contextHolder] = message.useMessage();
  const { t } = useTranslation(['appMarket', 'common']);
  const debounceTimer = useRef(null);
  useEffect(() => {
    return () => {
      if (debounceTimer.current) {
        clearTimeout(debounceTimer.current);
      }
    };
  }, []);

  const handleSearch = (repoName) => {
    onSearchFn(repoName);
  };

  const handleSelect = (e) => {
    onSelectFn(e);
  };

  const showLimitToast = () => {
    if (debounceTimer.current) {
      clearTimeout(debounceTimer.current);
    }
    debounceTimer.current = setTimeout(() => {
      messageApi.error(t('market.searchLimit', { limit: limitSearch }));
    }, 300);
  };

  const handleInputChange = (e) => {
    const value = e.target.value;
    if (value.length >= limitSearch) {
      showLimitToast();
    }
    onInputChartFn(value);
  };

  const handlePaste = (e) => {
    const pastedText = e.clipboardData.getData('text');
    if (pastedText.length > limitSearch) {
      showLimitToast();
    }
  };

  return (
    <div>
      <ToastMsg contextHolder={contextHolder} />
      <div className='pod_search'>
        <div style={{ color: themeStore.$s.theme !== 'light' ? '#fff' : undefined }}>{t('list.total', { total })}</div>
        <div style={{ display: 'flex', gap: '16px' }}>
          <Search
            value={searchName}
            placeholder={t('searchPlaceholder', { ns: 'common', name: t('searchName.app', { ns: 'common' }) })}
            onSearch={handleSearch}
            onChange={handleInputChange}
            onPaste={handlePaste}
            maxLength={limitSearch}
          />
          <Select
            value={sortType}
            style={{ width: '160px' }}
            onChange={handleSelect}
            options={[
              { value: 'name', label: t('list.sortByName') },
              { value: 'time', label: t('list.sortByTime') },
            ]}
          />
        </div>
      </div>
      {helmChartListData?.length > 0 ?
        <div className='application_list'>
          {helmChartListData.map((el) => (
            <ApplicationRecommendItem key={`${el.repo}/${el.metadata?.name}/${el.metadata?.version}`} itemMeta={el.metadata} repo={el.repo} types={true} />
          ))}
        </div>
        :
        <EmptyData />}
    </div>
  );
}
