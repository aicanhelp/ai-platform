
/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import EchartsInit from '@/components/EchartsInit';
import { useCallback, useEffect, useState } from 'openinula';
import bluePoint from '@/assets/images/chart/bluePoint.png';
import grayPoint from '@/assets/images/chart/grayPoint.png';
import { getMappingValueArrayOfKey } from '@/tools/utils';
import { Tooltip } from 'antd';
import useLocalStorage from '@/hooks/useLocalStorage';
export default function NumberAimCom(
  {
    isAimExist,
    number,
    aimType,
    data,
    isDefault = false,
  }
) {
  const [option, setOption] = useState();
  const [currentTheme, setCurrentTheme] = useLocalStorage('theme', 'light');
  const addOptionsToChart = useCallback(() => {
    let options = {
      cursor: 'default',
      zlevel: 1,
      z: 1,
      grid: {
        left: '-20%',
      },
      xAxis: {
        type: 'category',
        data: getMappingValueArrayOfKey(data, 'timeStamp'),
        splitLine: { show: false },
        axisLine: { show: false },
        axisTick: { show: false },
        axisLabel: {
          show: false,
        }
      },
      yAxis: {
        type: 'value',
        splitLine: { show: false },
        axisLine: { show: false },
        axisTick: { show: false },
        axisLabel: {
          show: false,
        }
      },
      series: [
        {
          data: getMappingValueArrayOfKey(data, 'value'),
          type: 'line',
          symbol: (_value, _params) => {
            let img = isDefault ? `image://${grayPoint}` : `image://${bluePoint}`;
            return img;
          },
          symbolSize: (value, params) => {
            // 最后一个数据 突出展示
            let size = 0;
            (params.dataIndex === data.length - 1) ? size = 20 : size = 0;
            return size;
          },
          smooth: true,
          cursor: 'default', // 鼠标样式
          lineStyle: {
            color: isDefault ? '#cccccc' : '#77AEF7',
          }
        },
      ]
    };
    setOption(options);
  }, [JSON.stringify(data)]);

  useEffect(() => {
    addOptionsToChart();
  }, [addOptionsToChart]);

  return <div className="number_aim_box">
    <Tooltip title={number}><p className='bold_aim' style={currentTheme === 'light' ? { color: '#333333' } : { color: '#f7f7f7' }}>{number}</p></Tooltip>
    <Tooltip title={aimType}><p className='aim_word_type'>{aimType}</p></Tooltip>
    <EchartsInit option={option} type={'number_line'} id={`${aimType}_line`} preview={true} isAimExist={isAimExist} />
  </div>
}