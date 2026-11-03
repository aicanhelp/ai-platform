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
import * as echarts from 'echarts';
import EmptyData from '@/components/EmptyData';

/**
 * 环图 
 *
 */
export default function DataDonutChart({
  id = '',
  title,
  className = '',
  styleProps = { height: 206 },
  data = '',
  type,
}) {
  const [chart, setChart] = useState(); // 切换图表
  const chartRef = useRef(null);
  const themeStore = useStore('theme');

  const containerStyle = {
    backgroundColor: themeStore.$s.theme === 'light' ? '#fff' : '#2a2d34ff',
    ...styleProps, // 合并其他样式属性
  };
  const renderChart = useCallback(() => {
    let option = {
      color: ['#dcdcdc', '#77aef7'],
      series: [
        {
          type: 'pie',
          radius: type === 'cluster' ? ['70%', '85%'] : ['50%', '60%'],
          avoidLabelOverlap: false,
          label: {
            show: true,
            formatter: `{numberFont|${(data * 100).toFixed(2)}}%`,
            position: 'center',
            rich: {
              numberFont: {
                fontSize: 16,
                fontWeight: 'bold',
              },
            },
          },
          emphasis: {
            disabled: true,
            focus: 'none',
          },
          data: [
            {
              value: Number((data * 100)),
              name: title,
              itemStyle: { color: '#77aef7' },
            },
            {
              value: 100 - Number((data * 100)),
              name: 'total',
              itemStyle: { color: '#dcdcdc' },
            },
          ],
          cursor: 'default',
        },
      ],
    };

    option ? chart?.setOption(option) : null;
  }, [chart, data]);

  useEffect(() => {
    const handleResize = () => {
      if (chart) {
        chart.resize();
      }
    };
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [chart]);

  const initChart = () => {
    let chartDom = chartRef.current;
    let myChart = echarts.init(chartDom);
    setChart(myChart);
  };

  useEffect(() => {
    if (chartRef.current) {
      const existing = echarts.getInstanceByDom(chartRef.current);
      if (existing) {
        existing.dispose();
      }
      initChart();
    }
    return () => {
      if (chartRef.current) {
        const inst = echarts.getInstanceByDom(chartRef.current);
        if (inst) {
          inst.dispose();
        }
      }
    };
  }, [id, data]);

  useEffect(() => {
    renderChart();
  }, [renderChart]);

  return <div className='chart_container' style={{ backgroundColor: themeStore.$s.theme === 'light' ? '#fff' : '#2a2d34ff' }}>
    <div className='chart_container_header'>
      <p style={{ color: themeStore.$s.theme !== 'light' && '#fff' }}>{title}</p>
    </div>
    {data ? <div id={`${id}_container`} className={`${id} inner_container dount_container`} ref={chartRef} style={containerStyle}></div> : <EmptyData className={className} />}
  </div>;
}