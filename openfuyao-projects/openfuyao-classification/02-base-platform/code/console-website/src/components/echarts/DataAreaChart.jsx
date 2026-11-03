/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import { useCallback, useEffect, useRef, useState, useStore } from 'openinula';
import '@/styles/components/datachart.less';
import Dayjs from 'dayjs';
import { WholeScreenIcon } from '@/assets/icon';
import { getMathValueList } from '@/tools/utils';
import { Modal, Table, Tooltip, Checkbox } from 'antd';
import { CloseOutlined, SyncOutlined, QuestionCircleOutlined } from '@ant-design/icons';
import EmptyData from '@/components/EmptyData';
import { getMappingValueArrayOfKey } from '@/tools/utils';
import * as echarts from 'echarts';
import { useTranslation } from 'react-i18next';
import useButtonLoading from '@/hooks/useButtonLoading';

/**
 *  仅限于面积图
 * // 标题
 * // data 数据组
 * // type 类型
 */
export default function DataAreaChart({
  id = '',
  title,
  data = [],
  type = 'area',
  styleProps = { gridRight: '' },
  rangeColorProps = [],
  isShowScreen = true,
  isShowLegend = true,
  tooltip = '',
  refreshFn = () => { },
}) {
  const chartRef = useRef(null);
  const subContainerRef = useRef(null);
  const resizeHandlerRef = useRef(null);
  const { t } = useTranslation('common');
  const [screenOpen, setScreenOpen] = useState(false); // 打开全屏
  const [refreshLoading, setRefreshLoading] = useButtonLoading(); // 刷新 loading
  const [screenTableData, setScreenTableData] = useState([]); // 图表数据
  const [isCheck, setIsCheck] = useState(true);
  const [totalData, setTotalData] = useState([]); // 排序后数据
  const [typeTotal, setTypeTotal] = useState([]); // 类型
  const [typeColorMap, setTypeColorMap] = useState([]); // 颜色和类型对比map不变总的
  const [changeTypeColorMap, setChangeTypeColorMap] = useState([]); // 变化的
  const [selectTypeList, setSelectTypeList] = useState([]); // 颜色
  const [chart, setChart] = useState(); // 图表
  const [subChart, setSubChart] = useState(); // 小型图表
  const themeStore = useStore('theme');

  const solveDataStyle = (list, right) => {
    let gridRight = right || '10%';
    if (right) {
      if (!list.some(item => item.length > 22)) {
        if (list.some(item => item.length > 15)) {
          gridRight = '25%';
        } else {
          gridRight = '15%';
        }
        if (window.innerWidth < 1400) {
          gridRight = '45%';
        }
        if (window.innerWidth < 1540 && window.innerWidth >= 1400) {
          gridRight = '40%';
        }
        if (window.innerWidth >= 1540 && window.innerWidth <= 1920) {
          gridRight = '30%';
        }
      }
      return gridRight;
    }
    const maxStyle = list.some(item => item.length > 22);
    const mendumStyle = list.some(item => item.length > 15);
    if (maxStyle) {
      gridRight = '20%';
    } else if (mendumStyle) {
      gridRight = '15%';
    } else {
      gridRight;
    }
    if (window.innerWidth < 1540) {
      gridRight = '20%';
    }
    return gridRight;
  };

  const renderChart = useCallback((myChart, echartsData, colorMap, { renderType = 'main', smallLegend = true }) => {
    if (!myChart) {
      return;
    }
    let dataSeries = [];
    const typeList = getMappingValueArrayOfKey(colorMap, 'type');
    let temporyType = type;
    if (typeList.length < 3) {
      temporyType = 'area';
    } else {
      temporyType = 'line';
    }

    // 按时间戳升序，保证 x 轴与数据正确对齐
    const sortedTimestampSet = [...new Set(getMappingValueArrayOfKey(echartsData, 'timestamp'))].sort((a, b) => a - b);

    typeList.forEach(item => {
      const [{ color }] = colorMap.filter(filterItem => filterItem.type === item);
      const filteredByType = echartsData.filter(filterItem => filterItem.type === item);
      const timestampToValue = {};
      filteredByType.forEach(d => { timestampToValue[d.timestamp] = d.value; });
      // 按统一的时间戳顺序生成 series 数据，无数据的时刻填 null
      const seriesData = sortedTimestampSet.map(ts => (timestampToValue[ts] !== undefined ? timestampToValue[ts] : null));

      let obj = {};
      obj.name = item;
      obj.type = 'line';
      obj.symbol = 'none';
      obj.smooth = true;
      obj.data = seriesData;
      temporyType === 'area' ? obj.areaStyle = {
        opacity: 0.3,
        color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
          {
            offset: 0,
            color,
          },
          {
            offset: 1,
            color: `${color}00`,
          },
        ]),
      } : '';
      dataSeries.push(obj);
    });
    const timestampList = sortedTimestampSet.map(ts => Dayjs(ts * 1000).format('MM/DD HH:mm:ss'));

    let option = {
      color: getMappingValueArrayOfKey(colorMap, 'color'),
      tooltip: {
        trigger: 'axis',
        confine: true,
        className: 'monitor_tooltip',
        enterable: true,
        extraCssText: `max-width:90%;max-height:83%;overflow:auto;border-radius:10px;
        ${themeStore.$s.theme === 'light' ? 'background:#fff' : 'background:#171A1F;box-shadow:0px 3px 6px rgba(247,247,247,0.16);border:none;color:#f7f7f7'}`,
        textStyle: {
          color: themeStore.$s.theme === 'light' ? '#666' : '#f7f7f7',
        },
      },
      legend: isShowLegend && smallLegend ? {
        type: 'scroll',
        orient: 'vertical',
        right: 0,
        top: 'center',
        icon: 'rect',
        itemHeight: 4,
        itemWidth: 16,
        pageTextStyle: {
          color: themeStore.$s.theme === 'light' ? 'black' : '#fff',
        },
        textStyle: {
          rich: {
            normal: {
              fontSize: 12,
              lineHeight: 16,
              color: themeStore.$s.theme === 'light' ? 'black' : '#fff',
            },
            label: {
              padding: [4, 0, 0, 0],
              color: themeStore.$s.theme === 'light' ? 'black' : '#fff',
            },
            value: {
              fontSize: 12,
              lineHeight: 16,
              color: themeStore.$s.theme === 'light' ? 'black' : '#fff',
            },
          },
        },
        formatter: (legendItem) => {
          let tip1 = '';
          let tip = '';
          let le = legendItem.length;
          if (le > 22) {
            let l = Math.ceil(le / 22);
            for (let i = 1; i <= l; i++) {
              if (i < l) {
                tip1 += `${legendItem.slice((i * 22) - 22, i * 22)}\n`;
              } else {
                tip = tip1 + legendItem.slice((l - 1) * 22, le);
              }
            }
            return `{value|${tip}}`;
          } else {
            tip = legendItem;
            return `{normal|${tip}}`;
          }
        },
        data: typeList,
      } : false,
      grid: {
        left: '3%',
        right: renderType === 'main' ? solveDataStyle(typeList, styleProps.gridRight) : '3%',
        bottom: '3%',
        containLabel: true,
      },
      xAxis: {
        type: 'category',
        boundaryGap: false,
        data: timestampList,
        axisLabel: {
          formatter: e => {
            return e.replace(' ', '\n');
          },
        },
      },
      yAxis: {
        type: 'value',
        axisLabel: {
          textStyle: {
            color: themeStore.$s.theme === 'light' ? 'black' : '#89939b',
          },
        },
      },
      series: dataSeries,
    };
    myChart.clear();
    myChart.setOption(option);
  }, [isShowLegend]);

  const initChart = useCallback(() => {
    if (chartRef.current) {
      let chartDom = chartRef.current;
      echarts.dispose(chartDom);
      let myChart = echarts.init(chartDom);
      setChart(myChart);
    }
    if (isShowScreen) {
      if (subContainerRef.current) {
        let chartDom = subContainerRef.current;
        echarts.dispose(chartDom);
        let myChart = echarts.init(chartDom);
        setSubChart(myChart);
      }
    }
  }, [id, isShowScreen]);

  useEffect(() => {
    initChart();
  }, [initChart]);

  // 改变
  const handleChange = (e) => {
    if (!e.target.checked) {
      setSelectTypeList([]);
      renderChart(subChart, [], [], { renderType: 'sub', smallLegend: false }); // 小图
      setChangeTypeColorMap([]);
    } else {
      setSelectTypeList(typeTotal);
      setChangeTypeColorMap([...typeColorMap]);
      renderChart(subChart, totalData, typeColorMap, { renderType: 'sub', smallLegend: false }); // 小图
    }
    setIsCheck(e.target.checked);
  };

  const handleSelectChange = (typeSelect, color) => {
    let willTypeColor = changeTypeColorMap;
    let temporySelectTypeList = getMappingValueArrayOfKey(willTypeColor, 'type');
    let totalDataTempory = totalData;
    if (temporySelectTypeList.includes(typeSelect)) {
      temporySelectTypeList = temporySelectTypeList.filter(item => item !== typeSelect);
      willTypeColor = willTypeColor.filter(item => item.type !== typeSelect);
    } else {
      willTypeColor.push({ type: typeSelect, color });
      temporySelectTypeList.push(typeSelect);
    }
    totalDataTempory = totalDataTempory.filter(filterItem => temporySelectTypeList.includes(filterItem.type));
    if (temporySelectTypeList.length === typeTotal.length) {
      setIsCheck(true);
    } else {
      setIsCheck(false);
    }
    setSelectTypeList([...temporySelectTypeList]);
    setChangeTypeColorMap([...willTypeColor]);
    renderChart(subChart,
      totalDataTempory,
      willTypeColor,
      {
        renderType: 'sub',
        smallLegend: false,
      },
    ); // 小图
  };

  const sortNaN = (a, b, sorter) => {
    if (isNaN(a) && isNaN(b)) {
      return 0;
    }
    if (isNaN(a)) {
      return sorter === 'ascend' ? 1 : -1;
    }
    if (isNaN(b)) {
      return sorter === 'ascend' ? -1 : 1;
    }

    return a - b;
  };

  // 表格
  const modalScreenColumns = [
    {
      title: <div style={{ display: 'flex', alignItems: 'center' }}>
        <Checkbox
          checked={isCheck}
          onChange={handleChange}
        >
        </Checkbox>
        <span style={{ marginLeft: '24px' }}>{t('chart.legend')}</span>
      </div>,
      key: 'legend',
      fixed: 'left',
      width: 250,
      render: (_, record) => (
        <div className='legend_group'>
          <span
            className='legend_color'
            style={{ background: `${selectTypeList.includes(record.legend.type) ? record.legend.color : '#ccc'}`, cursor: 'pointer' }}
            onClick={() => handleSelectChange(record.legend.type, record.legend.color)}
          >
          </span>
          <span className='legend_text' style={{ color: themeStore.$s.theme === 'light' ? '#333333' : '#fff' }}>{record.legend.type}</span>
        </div>
      ),
    },
    {
      title: t('chart.currentValue'),
      key: 'current',
      sorter: (a, b, sorter) => sortNaN(a.current, b.current, sorter),
      render: (_, record) => isNaN(record.current) ? '--' : record.current,
    },
    {
      title: t('chart.maxValue'),
      key: 'max',
      sorter: (a, b, sorter) => sortNaN(a.max, b.max, sorter),
      render: (_, record) => isNaN(record.max) ? '--' : record.max,
    },
    {
      title: t('chart.minValue'),
      key: 'min',
      sorter: (a, b, sorter) => sortNaN(a.min, b.min, sorter),
      render: (_, record) => isNaN(record.min) ? '--' : record.min,
    },
    {
      title: t('chart.avgValue'),
      key: 'average',
      sorter: (a, b, sorter) => sortNaN(a.average, b.average, sorter),
      render: (_, record) => isNaN(record.average) ? '--' : record.average,
    },
  ];

  const handleClick = () => {
    // 打开全屏弹窗查看图表与表格数据
    setScreenOpen(true);
  };

  const renderTableData = useCallback((passData, typeList, rangeColor) => {
    let tableData = getMathValueList(passData, typeList, rangeColor);
    setScreenTableData([...tableData]);
  }, []);

  const handleRefresh = async () => {
    // 再次请求接口
    setRefreshLoading(true);
    try {
      await refreshFn();
    } finally {
      setRefreshLoading(false);
    }
  };

  const initOptionsChart = useCallback(() => {
    let typeList = [];
    let colorMap = [];
    let stablizeData = [...data].sort((a, b) => a.type.length - b.type.length); // 排序（拷贝后再排，避免原地修改 props 污染父组件数据）
    let rangeColor = rangeColorProps;
    if (stablizeData.length) {
      stablizeData.forEach(item => {
        if (!typeList.includes(item.type)) {
          typeList.push(item.type);
        }
      });
      typeList.forEach((item, index) => {
        colorMap.push({
          type: item,
          color: rangeColorProps[index] || '#5B8FF9',
        });
      });
      // 渲染
      if (chart) {
        renderChart(chart, stablizeData, colorMap, {});
      }
    }
    setTotalData(stablizeData);
    setTypeTotal(typeList);
    setSelectTypeList([...typeList]);
    setTypeColorMap([...colorMap]);
    setChangeTypeColorMap([...colorMap]);
    if (isShowScreen) {
      renderTableData(stablizeData, typeList, rangeColor);
      if (subChart) {
        renderChart(subChart, stablizeData, colorMap, { renderType: 'sub', smallLegend: false }); // 小图
      }
    }
    if (resizeHandlerRef.current) {
      window.removeEventListener('resize', resizeHandlerRef.current);
    }
    resizeHandlerRef.current = () => {
      if (chart) {
        renderChart(chart, stablizeData, colorMap, {});
        chart.resize();
      }
      if (subChart && isShowScreen) {
        renderChart(subChart, stablizeData, colorMap, { renderType: 'sub', smallLegend: false }); // 小图
        subChart.resize();
      }
    };
    window.addEventListener('resize', resizeHandlerRef.current);
  }, [isShowScreen, renderChart, chart, subChart, JSON.stringify(data)]);

  useEffect(() => {
    initOptionsChart();
  }, [initOptionsChart]);

  useEffect(() => {
    return () => {
      if (resizeHandlerRef.current) {
        window.removeEventListener('resize', resizeHandlerRef.current);
        resizeHandlerRef.current = null;
      }
    };
  }, []);

  const listeners = () => {
    initChart();
    initOptionsChart();
  };

  const listenersRef = useRef(null);
  listenersRef.current = listeners;

  useEffect(() => {
    const subscribed = () => {
      listenersRef.current();
    };
    themeStore.$subscribe(subscribed);
    return () => themeStore.$unsubscribe(subscribed);
  }, []);

  return <div className='chart_container' style={{ backgroundColor: themeStore.$s.theme === 'light' ? '#fff' : '#2a2d34ff' }}>
    <div className='chart_container_header'>
      <p style={{ display: 'flex', alignItems: 'center' }}>
        <span style={{ color: themeStore.$s.theme !== 'light' ? '#fff' : undefined }}>{title}</span>
        <Tooltip title={tooltip}>
          {tooltip ? <QuestionCircleOutlined className='data_area_icon' /> : null}
        </Tooltip>
      </p>
      {(!!data.length && isShowScreen) && <span onClick={handleClick}><WholeScreenIcon fill={themeStore.$s.theme === 'light' ? '#243141' : '#fff'} /></span>}
    </div>
    {data.length ? <div id={`${id}_container`} className={`${id} inner_container`} ref={chartRef} style={{ width: '100%', height: 300, backgroundColor: themeStore.$s.theme === 'light' ? '#fff' : '#2a2d34ff', color: themeStore.$s.theme !== 'light' ? '#fff' : undefined }}></div> : <EmptyData />}
    <Modal
      className='screen_modal'
      open={screenOpen}
      forceRender
      getContainer={false}
      closeIcon={null}
      footer={null}
    >
      <div className='screen_box'>
        <div className='screen_header'>
          <p className='chart_title' style={{ color: themeStore.$s.theme === 'light' ? '#333333' : '#fff' }}>{title}</p>
          <div className='screen_icon_right'>
            <SyncOutlined spin={refreshLoading} onClick={handleRefresh} style={{ color: themeStore.$s.theme === 'light' ? '#333333' : '#fff' }} />
            <CloseOutlined onClick={() => { setScreenOpen(false); }} style={{ color: themeStore.$s.theme === 'light' ? '#333333' : '#fff' }} />
          </div>
        </div>
        {data.length ? <div id={`${id}_sub_container`} className={`${id} inner_sub_container`} ref={subContainerRef} style={{ width: '1100px', height: '300px' }}></div> : <EmptyData />}
        <Table
          columns={modalScreenColumns}
          dataSource={screenTableData}
          pagination={{
            className: 'page',
            showTotal: (total) => t('chart.paginationTotal', { total }),
            showSizeChanger: true,
            showQuickJumper: true,
            pageSizeOptions: [5, 10],
          }}
        />
      </div>
    </Modal>
  </div >;
}
