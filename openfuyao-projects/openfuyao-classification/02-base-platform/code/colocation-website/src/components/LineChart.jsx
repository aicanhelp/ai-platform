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
import '@/styles/components/datachart.less';
import * as echarts from 'echarts';
import useLocalStorage from '@/hooks/useLocalStorage';
import EmptyData from '@/components/EmptyData';
import { solveSeriesMap } from '@/tools/utils';
export default function LineChart({
  id = '',
  className = '',
  colorList = [],
  styleProps = { height: 240 },
  data = '',
  type = 'cluster',
  isShowLegend = false,
}) {
  const [chart, setChart] = useState(); // 切换图表
  const chartRef = useRef(null);
  const [currentTheme] = useLocalStorage('theme', 'light');

  const containerStyle = {
    backgroundColor: currentTheme === 'light' ? '#fff' : '#2a2d34ff',
    ...styleProps, // 合并其他样式属性
  };

  const suitResize = () => {
    let space = '10%';
    if (window.innerWidth > 1600) {
      space = '13%';
    }
    if (window.innerWidth > 1400 && window.innerWidth <= 1600) {
      space = '15%';
    }
    if (window.innerWidth <= 1400) {
      space = '20%';
    }
    return space;
  };

  const suitNodeResize = () => {
    let space = '20%';
    if (window.innerWidth > 1600) {
      space = '20%';
    }
    if (window.innerWidth > 1400 && window.innerWidth <= 1600) {
      space = '23%';
    }
    if (window.innerWidth <= 1400) {
      space = '28%';
    }
    return space;
  };

  const judgeRightSpace = (bool) => {
    let space = '4%';
    if (bool) {
      space = type === 'cluster' ? suitResize() : suitNodeResize();
    }
    return space;
  };
  const renderChart = useCallback(() => {
    const series = solveSeriesMap(data);
    let option = {
      color: colorList,
      grid: {
        left: '3%',
        right: judgeRightSpace(isShowLegend),
        top: '3%',
        bottom: '3%',
        containLabel: true,
      },
      tooltip: {
        trigger: 'axis',
        axisPointer: {
          type: 'line',
        },
        confine: true,
        className: 'monitor_tooltip',
        enterable: true,
        extraCssText: `max-width:90%;max-height:83%;overflow:auto;border-radius:10px;
        ${currentTheme === 'light' ? 'background:#fff' : 'background:#171A1F;box-shadow:0px 3px 6px rgba(247,247,247,0.16);border:none;color:#f7f7f7'}`,
        textStyle: {
          color: currentTheme === 'light' ? '#666' : '#f7f7f7',
        },
        formatter: (params) => {
          let result = `${params[0].axisValueLabel}<br/>`;
          params.forEach((item) => {
            result += `<span style="display:inline-block;margin-right:5px;width:20px;height:4px;border-radius:3px;background-color:${item.color};"></span>`;
            result += `${item.seriesName}: ${item.value[1]}<br/>`;
          });
          return result;
        },
      },
      xAxis: {
        type: 'category',
      },
      yAxis: {
        type: 'value',
      },

      legend: isShowLegend
        ? {
            type: 'scroll',
            orient: 'vertical',
            right: 0,
            top: 'center',
            icon: 'rect',
            itemHeight: 4,
            pageIconColor: '#3F66F5',
            pageIconInactiveColor: '#c4c4c4',
            itemGap: 15,
            pageIconSize: 10,
            pageIcon: {
              vertical: [
                'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA0AAAAKCAQAAADF5xxPAAAAIGNIUk0AAHomAACAhAAA+gAAAIDoAAB1MAAA6mAAADqYAAAXcJy6UTwAAAACYktHRAD/h4/MvwAAAAd0SU1FB+kGCwMONBiv9SkAAAB0SURBVBjTY2BAAkf2M2AHh5OO/D+chOAzwhhnWH/cYZBjeMShYvIbIsIEk/pZzyDHwMAg97MeTdcRdYYbcJM0bG4i6WKsR7KjHknXEU+GbSgu8rLZDtNVz4AK6qEGHs5mMEeTMj+czcDAeILvz0ZsvmTxBwCN3xtnf8zJMwAAACV0RVh0ZGF0ZTpjcmVhdGUAMjAyNS0wNi0xMVQwMzoxNDo1MiswMDowMFnXzMoAAAAldEVYdGRhdGU6bW9kaWZ5ADIwMjUtMDYtMTFUMDM6MTQ6NTIrMDA6MDAoinR2AAAAKHRFWHRkYXRlOnRpbWVzdGFtcAAyMDI1LTA2LTExVDAzOjE0OjUyKzAwOjAwf59VqQAAAABJRU5ErkJggg==',
                'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA0AAAAKBAMAAACdwMn3AAAAIGNIUk0AAHomAACAhAAA+gAAAIDoAAB1MAAA6mAAADqYAAAXcJy6UTwAAAAqUExURQAAADdb7T9l9j5m9T5m9j9l9T9l9kFi8kBm9TNmzD9m9j5l9T9m9fB3RJTUUH6QYLAxIvdL1hmAAAAD9JREFUCNdjEFICAQEG5jNAcJiBgcEHSDsA6RggHQCk2c6cOQakGNjPnDkOohl6zjSA6VmnJoBp7tMMEKAAxAC/PBRP0mtWbgAAACV0RVh0ZGF0ZTpjcmVhdGUAMjAyNS0wNi0xMVQwMzoxODo0NyswMDowMN15A30AAAAldEVYdGRhdGU6bW9kaWZ5ADIwMjUtMDYtMTFUMDM6MTg6NDcrMDA6MDCsJLvBAAAAKHRFWHRkYXRlOnRpbWVzdGFtcAAyMDI1LTA2LTExVDAzOjE4OjQ3KzAwOjAw+zGaHgAAAABJRU5ErkJggg==',
              ],
            },
            itemWidth: 16,
            pageTextStyle: {
              color: currentTheme === 'light' ? 'black' : '#fff',
            },
            textStyle: {
              rich: {
                normal: {
                  fontSize: 12,
                  lineHeight: 16,
                  color: currentTheme === 'light' ? 'black' : '#fff',
                },
                label: {
                  padding: [4, 0, 0, 0],
                  color: currentTheme === 'light' ? 'black' : '#fff',
                },
                value: {
                  fontSize: 12,
                  lineHeight: 16,
                  color: currentTheme === 'light' ? 'black' : '#fff',
                },
              },
            },
            formatter: (legendItem) => {
              let tip1 = '';
              let tip = '';
              let le = legendItem.length;
              let breakWord = window.innerWidth > 1500 ? 50 : 22;
              if (le > breakWord) {
                let l = Math.ceil(le / breakWord);
                for (let i = 1; i <= l; i++) {
                  if (i < l) {
                    tip1 += `${legendItem.slice(i * breakWord - breakWord, i * breakWord)}\n`;
                  } else {
                    tip = tip1 + legendItem.slice((l - 1) * breakWord, le);
                  }
                }
                return `{value|${tip}}`;
              } else {
                tip = legendItem;
                return `{normal|${tip}}`;
              }
            },
          }
        : false,
      series,
    };

    option ? chart.setOption(option, true) : null;
    window.addEventListener('resize', () => {
      if (chart) {
        chart.resize();
      }
    });
  }, [chart, data, colorList]);

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
      {data && data.length ? (
        <div
          id={`${id}_container`}
          className={`${id} inner_container line_container`}
          ref={chartRef}
          style={containerStyle}
        ></div>
      ) : (
        <EmptyData className={className} />
      )}
    </div>
  );
}
