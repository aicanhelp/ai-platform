/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import BreadCrumbCom from '@/components/BreadCrumbCom';
import { containerRouterPrefix, ResponseCode } from '../../common/constants';
import '../../styles/overview.less';
import { useCallback, useEffect, useState } from 'openinula';
import { useIntl } from 'inula-intl';
import { getNodesAndPodsCount, getResourcesTop5 } from '@/api/colocationApi';
import OverviewEcharts from './OverviewEcharts';
import useLocalStorage from '@/hooks/useLocalStorage';
import { filterEchartsOption } from '../../utils/common';
import EmptyData from '../../components/EmptyData';
import { SyncOutlined, LoadingOutlined } from '@ant-design/icons';
let setTimeoutId2 = 0;

export default function ResourceManage() {
  const intl = useIntl();
  const [currentTheme] = useLocalStorage('theme', 'light');
  const [oversoldCpuDataOption, setOversoldCpuDataOption] = useState();
  const [oversoldMemoryDataOption, setOversoldMemoryDataOption] = useState();
  const [oversoldCpuTop5Count, setOversoldCpuTop5Count] = useState([]); // 资源统计的超卖可用cpu top5的对应个数
  const [oversoldMemoryTop5Count, setOversoldMemoryTop5Count] = useState([]); // 资源统计的超卖可用memory top5的对应个数
  const [oversoldUsedCpuTop5Count, setOversoldUsedCpuTop5Count] = useState([]); // 资源统计的超卖已用cpu top5的对应个数
  const [oversoldUsedMemoryTop5Count, setOversoldUsedMemoryTop5Count] = useState([]); // 资源统计的超卖已用memory top5的对应个数
  const [oversoldCpuTop5Name, setOversoldCpuTop5Name] = useState([]); // 资源统计的超卖cpu top5的对应名称
  const [oversoldMemoryTop5Name, setOversoldMemoryTop5Name] = useState([]); // 资源统计的超卖memory top5的对应名称
  const [oversoldCpuTotal, setOversoldCpuTotal] = useState([]);
  const [oversoldMemoryTotal, setOversoldMemoryTotal] = useState([]);

  // echarts配置项公共部分
  const getEchartsOption = useCallback(
    (series, leftName) => {
      return filterEchartsOption(series, leftName, currentTheme);
    },
    [currentTheme],
  );

  // 获取资源统计部分的图表数据
  const getOverSoldManageInfo = useCallback(async () => {
    // 接口返回筛选数据  排序后过滤前五项
    let cpuCountArr = [];
    let cpuUsedCountArr = [];
    let cpuNameArr = [];
    let cpuTotalArr = [];
    let memoryCountArr = [];
    let memoryUsedCountArr = [];
    let memoryNameArr = [];
    let memoryTotalArr = [];
    try {
      const res = await getResourcesTop5();
      if (res.status === ResponseCode.OK) {
        let cpuArr = res.data.data.beCPUStats ? res.data.data.beCPUStats : [];
        let memoryArr = res.data.data.beMemoryStats ? res.data.data.beMemoryStats : [];
        cpuArr.slice(0, 5).forEach((item) => {
          cpuCountArr.unshift((item.beCPURemained / 1000).toFixed(3));
          cpuUsedCountArr.unshift((item.beCPURequest / 1000).toFixed(3));
          cpuNameArr.unshift(item.nodeName);
          cpuTotalArr.unshift(item.beCPUTotal);
        });
        memoryArr.slice(0, 5).forEach((item) => {
          memoryCountArr.unshift((item.beMemoryRemained / Math.pow(1024, 2) / 1024).toFixed(3));
          memoryUsedCountArr.unshift((item.beMemoryRequest / Math.pow(1024, 2) / 1024).toFixed(3));
          memoryNameArr.unshift(item.nodeName);
          memoryTotalArr.unshift(item.beMemoryTotal);
        });
        setOversoldCpuTotal(cpuTotalArr);
        setOversoldMemoryTotal(memoryTotalArr);
        setOversoldCpuTop5Count(cpuCountArr);
        setOversoldMemoryTop5Count(memoryCountArr);
        setOversoldUsedCpuTop5Count(cpuUsedCountArr);
        setOversoldUsedMemoryTop5Count(memoryUsedCountArr);
        setOversoldCpuTop5Name(cpuNameArr);
        setOversoldMemoryTop5Name(memoryNameArr);
      } else {
        messageApi.error(intl.formatMessage({ id: 'message.apiError' }));
      }
    } catch (e) {
      setOversoldCpuTop5Count([]);
      setOversoldMemoryTop5Count([]);
      setOversoldUsedCpuTop5Count([]);
      setOversoldUsedMemoryTop5Count([]);
      setOversoldCpuTop5Name([]);
      setOversoldMemoryTop5Name([]);
    }
  }, []);

  // 超卖cpu数据总和
  const getOversoldCpuInfo = useCallback(async () => {
    let oversoldCpuSeries = [
      {
        name: intl.formatMessage({ id: 'overview.oversoldAvailable' }),
        type: 'bar',
        stack: 'total',
        data: oversoldCpuTop5Count,
        // 后续使用oversoldCpuData的数据
        itemStyle: { borderRadius: [5, 5, 5, 5], color: '#CEC4FF' },
        barWidth: 10,
      },
      {
        name: intl.formatMessage({ id: 'overview.oversoldUsed' }),
        type: 'bar',
        stack: 'total',
        data: oversoldUsedCpuTop5Count,
        itemStyle: { borderRadius: [5, 5, 5, 5], color: '#DCDCDC' },
        barWidth: 10,
      },
    ];
    // 后续接口获取data数据
    let cpuName = oversoldCpuTop5Name;
    // 后续接口获取top5名称
    let oversoldCpuOption = getEchartsOption(oversoldCpuSeries, cpuName);
    setOversoldCpuDataOption(oversoldCpuOption);
  }, [oversoldCpuTop5Count, oversoldUsedCpuTop5Count, oversoldCpuTop5Name]);

  // 超卖memory数据总和
  const getOversoldMemoryInfo = useCallback(async () => {
    let oversoldMemorySeries = [
      {
        name: intl.formatMessage({ id: 'overview.oversoldAvailable' }),
        type: 'bar',
        stack: 'total',
        data: oversoldMemoryTop5Count,
        itemStyle: { borderRadius: [5, 5, 5, 5], color: '#CEC4FF' },
        barWidth: 10,
      },
      {
        name: intl.formatMessage({ id: 'overview.oversoldUsed' }),
        type: 'bar',
        stack: 'total',
        data: oversoldUsedMemoryTop5Count,
        itemStyle: { borderRadius: [5, 5, 5, 5], color: '#DCDCDC' },
        barWidth: 10,
      },
    ];
    let memoryName = oversoldMemoryTop5Name;
    // 后续接口获取top5名称
    let oversoldMemoryOption = getEchartsOption(oversoldMemorySeries, memoryName);
    setOversoldMemoryDataOption(oversoldMemoryOption);
  }, [oversoldMemoryTop5Count, oversoldUsedMemoryTop5Count, oversoldMemoryTop5Name]);

  const getShowData = (arr) => {
    if (arr.some((item) => item === 0)) {
      return (
        <div className="top5_show_loading_box">
          <div className="top5_show_loading_box_little">
            <LoadingOutlined />
            <p className="top5_show_loading_box_little_title">
              {intl.formatMessage({ id: 'overview.oversoldAgentDeploying' })}
            </p>
          </div>
        </div>
      );
    } else {
      return <EmptyData />;
    }
  };

  const timeSearch = () => {
    // 存入storage
    localStorage.setItem('colocationTime3Id', setTimeoutId2);
    clearTimeout(setTimeoutId2);
    getOverSoldManageInfo();
    const id2 = setTimeout(() => {
      timeSearch();
    }, 60000);
    setTimeoutId2 = id2;
  };

  useEffect(() => {
    if (setTimeoutId2) {
      clearTimeout(setTimeoutId2);
    }
    timeSearch();
    return () => {
      clearTimeout(setTimeoutId2);
    };
  }, [currentTheme]);

  useEffect(() => {
    getOversoldCpuInfo();
    getOversoldMemoryInfo();
  }, [getOversoldCpuInfo, getOversoldMemoryInfo]);

  return (
    <>
      <div className="overview_flex" style={{ marginBottom: '0px' }}>
        <div
          className="overview_flex_resource overview_card_shadow"
          style={{ backgroundColor: currentTheme === 'light' ? '#fff' : '#2a2d34ff' }}
        >
          <p className="resource_big_title" style={{ color: currentTheme !== 'light' && '#f7f7f7' }}>
            {intl.formatMessage({ id: 'overview.resourceStatistics' })}
          </p>
          <div className="overview_flex_resource_col">
            <div className="overview_flex_resource_half">
              <div className="overview_flex_resource_chart">
                <p className="resource_title" style={{ color: currentTheme !== 'light' && '#f7f7f7' }}>
                  {intl.formatMessage({ id: 'overview.oversoldCpuTop5' })}
                </p>
                {oversoldCpuTop5Count.length && oversoldCpuTotal.some((item) => item !== 0) ? (
                  <OverviewEcharts idProps={'oversold-cpu'} overOption={oversoldCpuDataOption} echartHeight={'260px'} />
                ) : (
                  getShowData(oversoldCpuTotal)
                )}
              </div>
            </div>

            <div className="overview_flex_resource_half">
              <div className="overview_flex_resource_chart">
                <p className="resource_title" style={{ color: currentTheme !== 'light' && '#f7f7f7' }}>
                  {intl.formatMessage({ id: 'overview.oversoldMemoryTop5' })}
                </p>
                {oversoldMemoryTop5Count.length && oversoldMemoryTotal.some((item) => item !== 0) ? (
                  <OverviewEcharts
                    idProps={'oversold-memory'}
                    overOption={oversoldMemoryDataOption}
                    echartHeight={'260px'}
                  />
                ) : (
                  getShowData(oversoldMemoryTotal)
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}
