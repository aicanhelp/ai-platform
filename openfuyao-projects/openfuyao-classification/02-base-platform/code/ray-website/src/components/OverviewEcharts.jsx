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
import useLocalStorage from '@/hooks/useLocalStorage';

export default function OverviewEcharts({
  overOption,
  echartHeight = '276px',
}) {
  const [currentTheme, setCurrentTheme] = useLocalStorage('theme', 'light');
  const chartRef = useRef(null);

  function debounce(fn, delay) {
    let timer;
    return function () {
      clearTimeout(timer);
      timer = setTimeout(fn, delay);
    };
  }

  useEffect(() => {
    const chartDom = chartRef.current;
    if (!chartDom) return;

    const chartInstance = echarts.init(chartDom);

    if (chartInstance && overOption) {
      let newOption = { ...overOption };
      if (currentTheme === 'dark') {
        newOption = {
          ...newOption, tooltip: {
            trigger: 'axis',
            backgroundColor: 'rgb(23, 26, 31)',
            borderColor: 'rgb(23, 26, 31)',
            textStyle: {
              color: '#fff',
            }
          }
        }
      }
      chartInstance.setOption(newOption, true);
    }

    const handleResize = () => {
      chartInstance.resize();
    };

    const debouncedResize = debounce(handleResize, 100);
    window.addEventListener('resize', debouncedResize);

    chartInstance.resize();

    return () => {
      window.removeEventListener('resize', debouncedResize);
      chartInstance.dispose();
    };
  }, [JSON.stringify(overOption), currentTheme]);

  return <div className='overviewEcharts' ref={chartRef} style={{ width: '100%', height: echartHeight }}></div>;
}