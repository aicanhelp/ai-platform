/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import { useCallback, useEffect, useRef, useState } from 'openinula';
import { useIntl } from 'inula-intl';
import '@/styles/components/datachart.less';
import * as echarts from 'echarts';
import useLocalStorage from '@/hooks/useLocalStorage';
import EmptyData from '@/components/EmptyData';
export default function DountChart({
  id = '',
  title = '',
  total = { text: '', value: 100 },
  className = '',
  styleProps = { height: 206 },
  data = '',
  colorList = [],
}) {
  const intl = useIntl();
  const totalText = total.text || intl.formatMessage({ id: 'common.total' });
  const [chart, setChart] = useState(); // 切换图表
  const chartRef = useRef(null);
  const [currentTheme] = useLocalStorage('theme', 'light');

  const containerStyle = {
    backgroundColor: currentTheme === 'light' ? '#fff' : '#2a2d34ff',
    ...styleProps, // 合并其他样式属性
  };

  const solveNodeLegend = () => {
    const narbar = document.querySelector('.container_nav_bar');
    let size = 10;
    if (!narbar) {
      size = '5%';
      if (window.innerWidth > 1700) {
        size = '20%';
      }
      if (window.innerWidth < 1400) {
        size = '-5%';
      }
    } else {
      if (window.innerWidth > 1700) {
        size = 60;
      }
      if (window.innerWidth > 1400 && window.innerWidth <= 1700) {
        size = 50;
      }
    }
    return size;
  };

  const renderCenter = () => {
    const narbar = document.querySelector('.container_nav_bar');
    if (narbar) {
      if (window.innerWidth > 1500) {
        return ['30%', '50%'];
      } else {
        return ['20%', '50%'];
      }
    }
    return ['30%', '50%'];
  };

  const renderRadius = () => {
    const narbar = document.querySelector('.container_nav_bar');
    if (narbar) {
      if (window.innerWidth > 1450) {
        return ['80%', '90%'];
      } else {
        return ['50%', '60%'];
      }
    }
    return ['80%', '90%'];
  };

  const renderItemWidth = () => {
    const narbar = document.querySelector('.container_nav_bar');
    if (narbar) {
      if (window.innerWidth > 1500) {
        return 90;
      }
      return 20;
    }
    return 90;
  };
  const renderChart = useCallback(() => {
    let option = {
      color: colorList,
      tooltip: {
        trigger: 'item',
        confine: true,
        className: 'monitor_tooltip',
        enterable: true,
        extraCssText: `max-width:90%;max-height:83%;overflow:auto;border-radius:10px;
        ${currentTheme === 'light' ? 'background:#fff' : 'background:#171A1F;box-shadow:0px 3px 6px rgba(247,247,247,0.16);border:none;color:#f7f7f7'}`,
        textStyle: {
          color: currentTheme === 'light' ? '#666' : '#f7f7f7',
        },
      },
      legend: {
        right: solveNodeLegend(),
        top: 'center',
        orient: 'vertical',
        align: 'left',
        icon: 'path://M6,0 L84,0 A6,6 0 0 1 90,6 L90,6 A6,6 0 0 1 84,12 L6,12 A6,6 0 0 1 0,6 L0,6 A6,6 0 0 1 6,0 Z',
        itemGap: 15,
        itemStyle: {
          borderRadius: 6,
        },
        borderRadius: 6,
        itemHeight: 12,
        itemWidth: renderItemWidth(),
        formatter: (name) => {
          const value = data.find((item) => item.name === name)?.value || 0;
          if (name.includes(totalText)) {
            return `{name|${name}} {value|${total.value}}`;
          }
          return `{name|${name}} {value|${value.toLocaleString()}}`;
        },
        textStyle: {
          rich: {
            rect: {
              width: 90,
              height: 12,
              borderRadius: 6,
              backgroundColor: '#DEF4FF',
              align: 'center',
              verticalAlign: 'middle',
            },
            name: {
              padding: [0, 5],
              color: currentTheme === 'light' ? '#333' : '#fff',
            },
            value: {
              color: currentTheme === 'light' ? '#333' : '#fff',
            },
          },
        },
      },
      grid: {
        top: '10%',
        right: window.innerWidth > 1500 ? '0%' : '20%',
        bottom: '10%',
        left: '0%',
      },
      series: [
        {
          name: title,
          type: 'pie',
          center: renderCenter(),
          radius: renderRadius(),
          avoidLabelOverlap: false,
          padAngle: 2,
          itemStyle: {
            borderRadius: 6,
          },
          label: {
            position: 'center',
            formatter: [`{a|${total.value}}`, `{b|${intl.formatMessage({ id: 'common.total' })}}`].join('\n\n'),
            rich: {
              a: {
                fontSize: window.innerWidth > 1450 ? 36 : 18,
                fontWeight: 'bold',
                color: currentTheme === 'light' ? '#333' : '#fff',
                lineHeight: 30,
              },
              b: {
                fontSize: window.innerWidth > 1450 ? 14 : 12,
                color: '#89939b',
              },
            },
          },
          labelLine: {
            show: false,
          },
          data: [...data],
        },
      ],
    };

    option ? chart.setOption(option, true) : null;
    window.addEventListener('resize', () => {
      if (chart) {
        renderChart();
        chart.resize();
      }
    });
  }, [chart, data, colorList, currentTheme]);

  const initChart = () => {
    let chartDom = chartRef.current;
    let myChart = echarts.init(chartDom);
    setChart(myChart);
  };
  useEffect(() => {
    if (chartRef.current) {
      initChart();
    }
  }, [id, data]);

  useEffect(() => {
    renderChart();
  }, [renderChart]);
  return (
    <div className="chart_container" style={{ backgroundColor: currentTheme === 'light' ? '#fff' : '#2a2d34ff' }}>
      {data ? (
        <div
          id={`${id}_container`}
          className={`${id} inner_container dount_container`}
          ref={chartRef}
          style={containerStyle}
        ></div>
      ) : (
        <EmptyData className={className} />
      )}
    </div>
  );
}
