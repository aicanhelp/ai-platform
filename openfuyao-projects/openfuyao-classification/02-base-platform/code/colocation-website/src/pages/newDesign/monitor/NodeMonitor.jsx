/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import useLocalStorage from '@/hooks/useLocalStorage';
import { message, Popover, Select } from 'antd';
import { StockOutlined } from '@ant-design/icons';
import { ResponseCode } from '@/common/constants';
import LineChart from '@/components/LineChart';
import { useEffect, useState, useCallback, useLayoutEffect } from 'openinula';
import { useIntl } from 'inula-intl';
import { getNodeListData, getColocationNode } from '@/api/colocationApi';
import { solveEchartsNodeClean } from '@/tools/utils';

let setTimeoutIdNode = 0;
export default function NodeMonitor({ propsType, propsColor }) {
  const intl = useIntl();
  const [currentTheme] = useLocalStorage('theme', 'light');
  const [messageApi, contextHolder] = message.useMessage();
  const [statisticsNodeData, setStatisticsNodeData] = useState();
  const [selectValue, setSelectValue] = useState();
  const [NodeCpuList, setNodeCpuList] = useState([]);

  const getStaticData = useCallback(async () => {
    let finallyData = {
      cpuNodeUse: [],
      memoryNodeUse: [],
      cpuNodeApply: [],
      memoryNodeApply: [],
      normalNodeCpu: [],
      normalNodeMemory: [],
    };
    if (selectValue && selectValue.length) {
      const res = await getColocationNode(selectValue.join('|'));
      if (res.status === ResponseCode.OK) {
        finallyData = solveEchartsNodeClean(res.data.data);
      }
    }
    setStatisticsNodeData(finallyData);
  }, [selectValue]);

  const handleNodeChange = (value) => {
    setSelectValue(value);
  };
  const getNodeListTotal = useCallback(async () => {
    const res = await getNodeListData();
    if (res.status === ResponseCode.OK) {
      const list = [];
      res.data.data.nodeInfo.map((item) => {
        list.push({
          label: `${item.name}(${item.isColocated ? intl.formatMessage({ id: 'monitor.colocation' }) : intl.formatMessage({ id: 'monitor.normal' })})`,
          value: item.name,
        });
      });
      setNodeCpuList(list);
      setSelectValue(list.map((user) => user.value));
    }
  }, []);

  const timeSearch = () => {
    localStorage.setItem('colocationTime2Id', setTimeoutIdNode);
    clearTimeout(setTimeoutIdNode);
    getStaticData();
    const nodeId = setTimeout(() => {
      timeSearch();
    }, 60000);
    setTimeoutIdNode = nodeId;
  };

  useLayoutEffect(() => {
    getNodeListTotal();
  }, [getNodeListTotal]);

  useEffect(() => {
    if (setTimeoutIdNode) {
      clearTimeout(setTimeoutIdNode);
    }
    timeSearch();
    return () => {
      clearTimeout(setTimeoutIdNode);
    };
  }, [currentTheme, selectValue]);

  return (
    <div className="monitor_namespace">
      <div
        style={{
          background: currentTheme === 'dark' ? '#2a2d34ff' : '#fff',
          color: currentTheme === 'dark' ? '#fff' : '#333',
        }}
      >
        {contextHolder}
      </div>
      <div className="monitor_line_container">
        <div className="select_box">
          <Select
            value={selectValue}
            options={NodeCpuList}
            mode="multiple"
            allowClear
            maxTagCount={2}
            style={{ width: '100%' }}
            placeholder={intl.formatMessage({ id: 'common.select' })}
            onChange={handleNodeChange}
          />
        </div>
        <div className="each_monitor_container">
          <p className="title">{intl.formatMessage({ id: 'monitor.nodeCpuActualUsage' })}</p>
          <LineChart
            id="cpuUse"
            data={statisticsNodeData?.cpuNodeUse}
            colorList={propsColor}
            isShowLegend={true}
            type="node"
          />
        </div>
        <div className="each_monitor_container">
          <p className="title">{intl.formatMessage({ id: 'monitor.nodeMemoryActualUsage' })}</p>
          <LineChart
            id="memoryUse"
            data={statisticsNodeData?.memoryNodeUse}
            colorList={propsColor}
            isShowLegend={true}
            type="node"
          />
        </div>
        <div className="each_monitor_container">
          <p className="title">{intl.formatMessage({ id: 'monitor.nodeOversoldCpuRequest' })}</p>
          <LineChart
            id="cpuApply"
            data={statisticsNodeData?.cpuNodeApply}
            colorList={propsColor}
            isShowLegend={true}
            type="node"
          />
        </div>
        <div className="each_monitor_container">
          <p className="title">{intl.formatMessage({ id: 'monitor.nodeOversoldMemoryRequest' })}</p>
          <LineChart
            id="memoryApply"
            data={statisticsNodeData?.memoryNodeApply}
            colorList={propsColor}
            isShowLegend={true}
            type="node"
          />
        </div>
        <div className="each_monitor_container">
          <p className="title">{intl.formatMessage({ id: 'monitor.nodeNormalCpuRequest' })}</p>
          <LineChart
            id="normalCpu"
            data={statisticsNodeData?.normalNodeCpu}
            colorList={propsColor}
            isShowLegend={true}
            type="node"
          />
        </div>
        <div className="each_monitor_container">
          <p className="title">{intl.formatMessage({ id: 'monitor.nodeNormalMemoryRequest' })}</p>
          <LineChart
            id="normalMemory"
            data={statisticsNodeData?.normalNodeMemory}
            colorList={propsColor}
            isShowLegend={true}
            type="node"
          />
        </div>
      </div>
    </div>
  );
}
