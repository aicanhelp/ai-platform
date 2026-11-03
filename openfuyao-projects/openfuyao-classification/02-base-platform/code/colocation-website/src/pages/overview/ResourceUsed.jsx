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
import { StockOutlined } from '@ant-design/icons';
import { Select } from 'antd';
import { useHistory, Link } from 'inula-router';
import { useCallback, useEffect, useLayoutEffect, useState } from 'openinula';
import { useIntl } from 'inula-intl';
import { getCpuAndMemoryUsage } from '@/api/colocationApi';
import OverviewEcharts from './OverviewEcharts';
import useLocalStorage from '@/hooks/useLocalStorage';
import { generateRandomColorHex, overviewUsedStatus } from '../../utils/common';
import EmptyData from '../../components/EmptyData';
import Dayjs from 'dayjs';
let setTimeoutId1 = 0;
let choseTypeArr = [];
let choseColorArr = [];
export default function ResourceUsed() {
  const intl = useIntl();
  const [currentTheme] = useLocalStorage('theme', 'light');
  const [cpuResourceOption, setCpuResourceOption] = useState();
  const [memoryResourceOption, setMemoryResourceOption] = useState();
  const [choseType, setChoseType] = useState([]);
  const [isCpuLegend, setIsCpuLegend] = useState(false);
  const [isMemoryLegend, setIsMemoryLegend] = useState(false);
  const [resourceOption, setResourceOption] = useState();
  const [resourceColor, setResourceColor] = useState([]);

  // 改变资源使用率的选择
  const handleSelect = (e) => {
    let newColorArr = [];
    e.forEach((element) => {
      const fillterArr = resourceColor.filter((item) => {
        return item.name === element;
      });
      newColorArr.push(...fillterArr);
    });
    setChoseType(e);
    choseTypeArr = [...e];
    choseColorArr = [...newColorArr];
    // 此处待接口获取选中的data echarts数据
    const data = [...e];
    getCpuResourceInfo(data, newColorArr);
  };

  // 获取资源使用部分的下拉框数据并随机生成图例颜色
  const getResourceUsedManageInfo = useCallback(async () => {
    // 接口返回筛选数据
    let backData = [];
    try {
      const res = await getCpuAndMemoryUsage();
      for (let i = 0; i < res.data.data.length; i++) {
        backData.push({
          value: `${res.data.data[i].name}(${overviewUsedStatus(res.data.data[i].colocationType, intl)})`,
          label: `${res.data.data[i].name}(${overviewUsedStatus(res.data.data[i].colocationType, intl)})`,
        });
      }
      setResourceOption(backData);
      const colorsArr = [];
      const defaultTop5Color = ['#77AEF7', '#09AA71', '#FFBF00', '#E7424A', '#808080'];
      for (let i = 0; i < backData.length; i++) {
        if (i < 5) {
          colorsArr.push({ name: backData[i].label, value: defaultTop5Color[i] });
        } else {
          colorsArr.push({ name: backData[i].label, value: generateRandomColorHex() });
        }
      }
      setResourceColor(colorsArr);
      let firstChoseTypeList = [];
      backData.slice(0, 5).forEach((item) => {
        firstChoseTypeList.push(item.value);
      });
      choseTypeArr = [...firstChoseTypeList];
      choseColorArr = [...colorsArr];
      setChoseType(firstChoseTypeList);
      getCpuResourceInfo(firstChoseTypeList, colorsArr);
    } catch (e) {
      setResourceOption([]);
      setResourceColor([]);
    }
  }, []);

  const getResourceEchartsOption = useCallback(
    (series, timeData) => {
      let option = {
        tooltip: {
          trigger: 'axis',
        },
        grid: {
          left: '3%',
          right: '8%',
          bottom: '3%',
          containLabel: true,
        },
        xAxis: [
          {
            type: 'category',
            boundaryGap: false,
            data: timeData,
          },
        ],
        yAxis: [
          {
            type: 'value',
          },
        ],
        series,
      };
      return option;
    },
    [currentTheme],
  );

  // 资源使用率 - cpu使用率数据
  const getCpuResourceInfo = useCallback(async (choseData = choseType, defaultColorArr = resourceColor) => {
    if (choseData) {
      // data为新选择的数据然后给cpuResourceSeries赋值
      // 处理data数据
      let cpuResourceSeries = [];
      let memoryResourceSeries = [];
      let cpuTime = [];
      try {
        const res = await getCpuAndMemoryUsage();
        for (let i = 0; i < res.data.data.length; i++) {
          cpuResourceSeries.push({
            name: `${res.data.data[i].name}(${overviewUsedStatus(res.data.data[i].colocationType, intl)})`,
            type: 'line',
            data: res.data.data[i].cpuValues,
          });
          memoryResourceSeries.push({
            name: `${res.data.data[i].name}(${overviewUsedStatus(res.data.data[i].colocationType, intl)})`,
            type: 'line',
            data: res.data.data[i].memValues,
          });
        }
        cpuResourceSeries[0].data.map((subItem) => {
          cpuTime.push(Dayjs(subItem.timestamp).format('MM/DD HH:mm'));
        });
        cpuResourceSeries = cpuResourceSeries.filter((item) => {
          return choseData.includes(item.name);
        });
        memoryResourceSeries = memoryResourceSeries.filter((item) => {
          return choseData.includes(item.name);
        });
      } catch (e) {
        cpuResourceSeries = [];
        memoryResourceSeries = [];
      }
      const newCpuResourceSeries = cpuResourceSeries.map((item, index) => {
        return {
          ...item,
          itemStyle: {
            color: defaultColorArr.length ? defaultColorArr.filter((i) => i.name === item.name)[0].value : '',
          },
        };
      });

      const newMemoryResourceSeries = memoryResourceSeries.map((item, index) => {
        return {
          ...item,
          itemStyle: {
            color: defaultColorArr.length ? defaultColorArr.filter((i) => i.name === item.name)[0].value : '',
          },
        };
      });

      // 后续接口获取时间
      let timeData = cpuTime;
      let tempCpuResourceOption = getResourceEchartsOption(newCpuResourceSeries, timeData);
      let tempMemoryResourceOption = getResourceEchartsOption(newMemoryResourceSeries, timeData);
      setCpuResourceOption(tempCpuResourceOption);
      setMemoryResourceOption(tempMemoryResourceOption);
    }
  }, []);

  useLayoutEffect(() => {
    getResourceUsedManageInfo();
  }, []);

  const timeSearch = () => {
    // 存入storage
    localStorage.setItem('colocationTime2Id', setTimeoutId1);
    clearTimeout(setTimeoutId1);
    getCpuResourceInfo(choseTypeArr, choseColorArr);
    const id1 = setTimeout(() => {
      timeSearch();
    }, 60000);
    setTimeoutId1 = id1;
  };

  useEffect(() => {
    if (setTimeoutId1) {
      clearTimeout(setTimeoutId1);
    }
    timeSearch();
    return () => {
      clearTimeout(setTimeoutId1);
    };
  }, [currentTheme]);

  return (
    <>
      <div className="overview_flex">
        <div
          className="overview_flex_resource overview_card_shadow"
          style={{ backgroundColor: currentTheme === 'light' ? '#fff' : '#2a2d34ff' }}
        >
          <div className="overview_flex_resource_row">
            <p className="resource_big_title" style={{ color: currentTheme !== 'light' && '#f7f7f7' }}>
              {intl.formatMessage({ id: 'overview.resourceUsage' })}
            </p>
            <Select
              mode="multiple"
              maxTagCount={2}
              value={choseType}
              placeholder={intl.formatMessage({ id: 'overview.selectNode' })}
              style={{ width: '410px', height: '30px' }}
              onChange={(e) => handleSelect(e)}
              options={resourceOption}
            />
          </div>
          <div className="overview_flex_resource_col">
            <div className="overview_flex_resource_half">
              <div className="overview_flex_resource_chart">
                <div className="overview_flex_resource_chart_legend">
                  <p className="resource_title" style={{ color: currentTheme !== 'light' && '#f7f7f7' }}>
                    {intl.formatMessage({ id: 'overview.cpuUsagePercent' })}
                  </p>
                  {choseType.length > 0 ? (
                    <div className="overview_flex_resource_right_top">
                      <div
                        className="overview_flex_resource_right_legendList"
                        style={{ display: isCpuLegend ? 'block' : 'none' }}
                      >
                        {choseType.map((item, index) => {
                          return (
                            <div className="overview_flex_resource_right_legendList_single">
                              <span
                                className="legend_color"
                                style={{ background: `${resourceColor.filter((i) => i.name === item)[0].value}` }}
                              ></span>
                              <span className="legend_text">{item}</span>
                            </div>
                          );
                        })}
                      </div>
                      <div className="overview_flex_resource_right_top">
                        <div className="overview_flex_resource_right_legend">
                          <div className="legend-point" onClick={() => setIsCpuLegend(!isCpuLegend)}>
                            <StockOutlined
                              style={{ color: currentTheme === 'light' ? '#3f66f5' : '#4b8cea', fontSize: '18px' }}
                            />
                            <span style={{ color: currentTheme === 'light' ? '#3f66f5' : '#4b8cea', fontSize: '14px' }}>
                              {intl.formatMessage({ id: 'overview.legend' })}
                            </span>
                          </div>
                    </div>
                  ) : (
                    <></>
                  )}
                </div>
                {choseType.length > 0 ? (
                  <OverviewEcharts overOption={cpuResourceOption} id="cpu-used" echartHeight={'260px'} />
                ) : (
                  <EmptyData />
                )}
              </div>
            </div>

            <div className="overview_flex_resource_half">
              <div className="overview_flex_resource_chart">
                <div className="overview_flex_resource_chart_legend">
                  <p className="resource_title" style={{ color: currentTheme !== 'light' && '#f7f7f7' }}>
                    {intl.formatMessage({ id: 'overview.memoryUsagePercent' })}
                  </p>
                  {choseType.length > 0 ? (
                    <div className="overview_flex_resource_right_top">
                      <div
                        className="overview_flex_resource_right_legendList"
                        style={{ display: isMemoryLegend ? 'block' : 'none' }}
                      >
                        {choseType.map((item, index) => {
                          return (
                            <div className="overview_flex_resource_right_legendList_single">
                              <span
                                className="legend_color"
                                style={{ background: `${resourceColor.filter((i) => i.name === item)[0].value}` }}
                              ></span>
                              <span className="legend_text">{item}</span>
                            </div>
                          );
                        })}
                      </div>
                      <div className="legend-point" onClick={() => setIsMemoryLegend(!isMemoryLegend)}>
                        <StockOutlined
                          style={{ color: currentTheme === 'light' ? '#3f66f5' : '#4b8cea', fontSize: '18px' }}
                        />
                        <span style={{ color: currentTheme === 'light' ? '#3f66f5' : '#4b8cea', fontSize: '14px' }}>
                          {intl.formatMessage({ id: 'overview.legend' })}
                        </span>
                      </div>
                    </div>
                  ) : (
                    <></>
                  )}
                </div>
                {choseType.length > 0 ? (
                  <OverviewEcharts overOption={memoryResourceOption} id="memory-used" echartHeight={'260px'} />
                ) : (
                  <EmptyData />
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}
