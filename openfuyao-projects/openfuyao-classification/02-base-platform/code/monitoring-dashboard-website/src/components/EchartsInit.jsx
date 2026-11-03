/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import { Tooltip } from 'antd';
import * as echarts from 'echarts';
import { Fragment, useEffect, useRef, useState } from 'openinula';
import { useIntl } from 'inula-intl';
export default function EchartsInit({ option, id, type, preview, isAimExist = false, typeAim = '' }) {
  const intl = useIntl();
  const chartRef = useRef(null);
  const [chart, setChart] = useState(null);

  useEffect(() => {
    if (chart) {
      chart.resize();
      option ? chart.setOption(option, true) : null;
      if (option && type === 'bar') {
        extension(chart);
      }
    }
    window.addEventListener('resize', () => {
      chart.resize();
    });
  }, [JSON.stringify(option), chart]);

  function extension(mychart) {
    const extensionBox = document.getElementById('extension');
    let html = document.documentElement;
    if (!extensionBox) {
      let node = document.createElement('div');
      node.id = 'extension';
      node.style.display = 'none';
      html.append(node);
    }

    mychart.on('mouseover', (params) => {
      const extensionBoxTempory = document.getElementById('extension');
      if (params.componentType === 'xAxis') {
        extensionBoxTempory.style.cssText = `position: absolute;color: #fff;
        font-family:Arial;display:block;
        word-break:break-word;font-size:14px; background-color: rgba(0, 0, 0, 0.85);
        box-shadow: 0 6px 16px 0 rgba(0, 0, 0, 0.08), 0 3px 6px -4px rgba(0, 0, 0, 0.12), 0 9px 28px 8px rgba(0, 0, 0, 0.05);
        box-sizing: border-box;min-width: 1em;min-height: 32px; padding: 6px 8px;text-align: start;text-decoration: none;border-radius: 6px;
        `;
        extensionBoxTempory.textContent = params.value;

        html.addEventListener('mousemove', (event) => {
          let xx = event.pageX - 30;
          let yy = event.pageY + 20;
          extensionBoxTempory.style.top = `${yy}px`;
          extensionBoxTempory.style.left = `${xx}px`;
        });
      }
    });

    mychart.on('mouseout', (params) => {
      const extensionBoxTempory = document.getElementById('extension');
      if (params.componentType === 'xAxis') {
        extensionBoxTempory.style.display = 'none';
      }
    });
  }

  useEffect(() => {
    if (chartRef.current) {
      const chartDom = chartRef.current;
      let myChart = echarts.init(chartDom);
      setChart(myChart);
    }
    return () => chart.dispose();
  }, []);
  return (
    <div
      className={`${type}_chartBox ${preview ? `preview_chartBox ${type}_preview_chartBox` : `display_chartBox ${type}_display_chartBox`}`}
    >
      <div
        className={`${id} ${type}_${preview ? 'preview' : 'display'}_chart ${preview ? 'preview_chart' : 'dispaly_chart'}`}
        ref={chartRef}
      ></div>
      {type === 'dashboard' && isAimExist && (
        <Tooltip title={typeAim}>
          <p className="aim_word_type">{typeAim}</p>
        </Tooltip>
      )}
      {!isAimExist && (
        <div className={`aim_word_chart ${type}_word`}>
          {type === 'dashboard'
            ? intl.formatMessage({ id: 'monitor.selectMetricAndSetGauge' })
            : intl.formatMessage({ id: 'monitor.selectMetricBelow' })}
        </div>
      )}
    </div>
  );
}
