/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import { Banner } from './Banner';
import { ApplicationList } from './ApplicationList';
import { ApplicationType } from './ApplicationType';
import { Pagination, message } from 'antd';
import '@/styles/applicationMarket/index.less';
import { useState, useEffect, useRef, useStore } from 'openinula';
import { getHelmChartList } from '@/api/applicationMarketApi';
import { useLocation, useParams } from 'inula-router';
import { useTranslation } from 'react-i18next';
import { debounce } from '@/tools/utils';

const defaultPageSize = 50;
function decodeSceneSafely(value) {
  try {
    return decodeURIComponent(value);
  } catch {
    return value;
  }
}
export default function MarketCategory() {
  const location = useLocation();
  const { scene, isFuyaoExtension, isCompute } = useParams();
  const { t } = useTranslation(['appMarket', 'common']);
  const [total, setTotal] = useState(0);
  const [helmChartListData, setHelmChartListData] = useState([]);
  const [chart, setChart] = useState('');
  const [sortType, setSortType] = useState('');
  const [currentPage, setCurrentPage] = useState(1);
  const [limit, setLimit] = useState(defaultPageSize);
  const [sceneList, setSceneList] = useState(scene && scene !== 'null' ? [decodeSceneSafely(scene)] : []);
  const [sourceList, setSourceList] = useState([]);
  const [isComputingEngine, setIsComputingEngine] = useState(isCompute === 'true');
  const [isSelectExtensionComponent, setIsSelectExtensionComponent] = useState(isFuyaoExtension === 'true' ? 'openfuyao-extension' : '');
  const themeStore = useStore('theme');
  const [messageApi, contextHolder] = message.useMessage();
  const requestSeqRef = useRef(0);
  const getHelmChartListData = async () => {
    let updatedSceneList = sceneList.filter(filterItem => filterItem !== 'undefined' && filterItem !== 'null');
    if (isComputingEngine) {
      updatedSceneList.push('compute-power-engine-plugin');
    }
    const seq = ++requestSeqRef.current;
    try {
      const params = {
        chart,
        sortType,
        sourceList,
        isSelectExtensionComponent,
        sceneList: updatedSceneList,
      };
      const res = await getHelmChartList(params, currentPage, limit);
      if (seq !== requestSeqRef.current) {
        return;
      }
      setHelmChartListData(res.data?.data?.items ?? []);
      setTotal(res.data?.data?.totalItems ?? 0);
    } catch (error) {
      if (seq !== requestSeqRef.current) {
        return;
      }
      messageApi.error(t('messages.fetchDataError', { ns: 'application', message: error.response?.data?.msg }));
    }
  };

  // 与来源筛选一致：只更新查询状态，实际拉取由统一的 getHelmChartListData 负责
  const handleSearchFn = (name) => {
    setChart(name || '');
    setCurrentPage(1);
  };

  const handleInputChartFn = useRef(
    debounce((name) => {
      setChart(name || '');
      setCurrentPage(1);
    }, 300),
  ).current;
  const handleSelectFn = (type) => {
    setSortType(type);
    setCurrentPage(1);
  };

  const handleChange = (page, pageSize) => {
    setCurrentPage(page);
    setLimit(pageSize);
  };

  const handleSelectType = (list) => {
    setSceneList(list);
    setCurrentPage(1);
  };

  const handleSelectExtensionComponent = (e) => {
    setIsSelectExtensionComponent(e);
    setCurrentPage(1);
  };

  const handleSelectSource = (list) => {
    setSourceList(list);
    setCurrentPage(1);
  };

  const handleComputingEngine = (bool) => {
    setIsComputingEngine(bool);
    setCurrentPage(1);
  };
  const handleClearFilter = () => {
    setSourceList([]);
    setSceneList([]);
    setIsSelectExtensionComponent('');
    setIsComputingEngine(false);
    setCurrentPage(1);
  };
  useEffect(() => {
    getHelmChartListData();
  }, [
    chart,
    currentPage,
    limit,
    sortType,
    isSelectExtensionComponent,
    sourceList,
    isComputingEngine,
    sceneList,
  ]);

  useEffect(() => {
    if (location.state?.isQuery) {
      setChart(location?.state?.isQuery.trim());
    }
  }, [location.state?.isQuery]);
  return (
    <div style={{ height: 'calc(100vh - 100px)' }}>
      <div style={{ background: themeStore.$s.theme === 'dark' ? '#2a2d34ff' : '#fff', color: themeStore.$s.theme === 'dark' ? '#fff' : '#333' }}>
        {contextHolder}
      </div>
      <Banner />
      <div style={{ display: 'flex' }}>
        <ApplicationType
          sceneCheckedList={sceneList}
          sourceCheckedList={sourceList}
          extensionChecked={isSelectExtensionComponent === 'openfuyao-extension'}
          computingChecked={isComputingEngine}
          onSelectSource={handleSelectSource}
          onSelectType={handleSelectType}
          onExtensionComponent={handleSelectExtensionComponent}
          onComputingEngine={handleComputingEngine}
          onClearFilter={handleClearFilter}
        />
        <div style={{ display: 'flex', flexDirection: 'column', width: 'calc(100vw - 200px)' }}>
          <ApplicationList
            total={total}
            searchName={chart}
            sortType={sortType || 'name'}
            helmChartListData={helmChartListData}
            onSearchFn={handleSearchFn}
            onSelectFn={handleSelectFn}
            onInputChartFn={handleInputChartFn}
          />
          <Pagination
            className='page'
            showTotal={(totalNumber) => t('pagination.total', { ns: 'common', total: totalNumber })}
            defaultPageSize={limit}
            showSizeChanger={false}
            showQuickJumper={true}
            total={total}
            onChange={handleChange}
          ></Pagination>
        </div>
      </div>
    </div >
  );
}

