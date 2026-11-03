/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import { useEffect, useRef } from 'openinula';
import * as echarts from 'echarts';

export default function OverviewEcharts({ overOption, echartHeight, idProps, type }) {
  const chartRef = useRef(null);

  useEffect(() => {
    const chart = echarts.init(chartRef.current);
    chartRef.current = chart;
    window.addEventListener('resize', () => {
      if (chart && type !== 'circle') {
        chart.resize();
      }
    });
    return () => {
      chart.dispose(); // 清理chart实例
    };
  }, []);
  useEffect(() => {
    const chart = chartRef.current;
    if (chart && overOption) {
      chart.setOption(overOption, true);
    }
  }, [overOption]);
  return <div className="overviewEcharts" ref={chartRef} style={{ width: '100%', height: echartHeight }}></div>;
}
