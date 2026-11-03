/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import '@/styles/components/displayBigData.less';
import EchartsCom from '@/components/EchartsCom';
import { reserveWidgetType } from '@/common/constants';
import { solveWidgetData } from '@/tools/utils';
import Draggable from 'react-draggable';

const widthStyle = {
  0: 'calc(100% - 64px)',
  1: 'calc(100% - 64px)',
  2: 'auto',
  3: 'auto',
  4: 'auto',
  5: 'calc(100% - 64px)',
};
export default function DisplayBigData({
  data,
  refreshFn,
}) {
  return <div className="display_content_box">
    {data.map(item => (
      <EchartsCom
        data={solveWidgetData(item.instances)}
        type={reserveWidgetType[item.graphType?.graphType || item.widgetType]}
        id='display_chart'
        isPreview={false}
        dashboardMax={item.graphType.boardMax || 100}
        dashboardMin={item.graphType.boardMin || 0}
        isAimExist={true}
        isMore={true}
        name={item.displayName}
        widgets={item}
        refreshFn={refreshFn}
      />
    ))}
  </div>
}