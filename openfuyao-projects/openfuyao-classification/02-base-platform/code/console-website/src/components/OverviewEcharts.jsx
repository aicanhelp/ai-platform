/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import { useEffect, useRef, useStore } from 'openinula';
import * as echarts from 'echarts';
import { useHistory } from 'inula-router';
import { filterOverviewRouter, filterOverviewType } from '@/utils/common';

export default function OverviewEcharts({
  overOption,
  echartHeight,
  idProps,
  type,
}) {
  const chartRef = useRef(null);
  const chartInstanceRef = useRef(null);
  const collapsedStore = useStore('collapsed');
  const idPropsRef = useRef(idProps);
  idPropsRef.current = idProps;
  const typeRef = useRef(type);
  typeRef.current = type;

  const history = useHistory();
  useEffect(() => {
    const chart = echarts.init(chartRef.current);
    chartInstanceRef.current = chart;
    chart.on('click', params => {
      if (params.componentType === 'series') {
        history.push({
          pathname: filterOverviewRouter(filterOverviewType(idPropsRef.current, params), params).url,
          state: { status: filterOverviewRouter(filterOverviewType(idPropsRef.current, params), params).status },
        });
      }
    });

    const handleResize = () => {
      if (chartInstanceRef.current && !chartInstanceRef.current.isDisposed() && typeRef.current !== 'circle') {
        chartInstanceRef.current.resize();
      }
    };

    window.addEventListener('resize', handleResize);
    
    // 监听容器尺寸变化（侧边栏展开/收起等场景）
    let resizeObserver;
    if (typeof ResizeObserver !== 'undefined' && chartRef.current) {
      resizeObserver = new ResizeObserver(handleResize);
      resizeObserver.observe(chartRef.current);
    }

    return () => {
      window.removeEventListener('resize', handleResize);
      if (resizeObserver) {
        resizeObserver.disconnect();
      }
      chart.dispose(); // 清理chart实例
    };
  }, []);

  useEffect(() => {
    if (chartInstanceRef.current) {
      chartInstanceRef.current.resize();
    }
  }, [collapsedStore.$s.collapsed]);

  useEffect(() => {
    const chart = chartInstanceRef.current;
    if (chart && overOption) {
      chart.setOption(overOption, { notMerge: true });
    }
  }, [overOption]);
  return <div className='overviewEcharts' ref={chartRef} style={{ width: '100%', height: echartHeight }}></div>;
}