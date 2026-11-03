/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import {
  containerRouterPrefix,
  ResponseCode,
  getTimePeriodOptions,
  promethusExp,
  stepList,
  getChineseChartType,
} from '@/common/constants';
import { useIntl } from 'inula-intl';
import { QuestionCircleOutlined } from '@ant-design/icons';
import { Select, Tooltip, message } from 'antd';
import dayjs from 'dayjs';
import OverviewEcharts from '@/components/OverviewEcharts';
import {
  filterTimeValue,
  birthOptions,
  solveEchartsData,
  solveDataAreaNumPalette,
  birthEmptyDayTime,
} from '@/tools/utils';
import { getExpressQuery } from '@/api/rayApi';
import { useState, useEffect, useCallback } from 'openinula';
import DataDisplayList from '@/components/DataDisplayList';
import EmptyArea from '@/components/EmptyArea';

const colorList = solveDataAreaNumPalette(100);
export default function RayCharts() {
  const intl = useIntl();
  const timePeriodOptions = getTimePeriodOptions(intl);
  const chineseChartType = getChineseChartType(intl);
  const [startTime, setStartTime] = useState(dayjs().subtract(10, 'minute').unix());
  const [endTime, setEndTime] = useState(dayjs().unix());
  const [stepValue, setStepValue] = useState('10m');
  const [messageApi, contextHolder] = message.useMessage();
  const [allOptions, setAllOptions] = useState(); // cpu实例
  const [legendData, setLegendData] = useState();
  const [emptyOptions, setEmptyOptions] = useState({});

  const handleChangeTime = (e) => {
    setStepValue(e);
    setStartTime(filterTimeValue(e)[0]);
    setEndTime(filterTimeValue(e)[1]);
  };

  const getOriginData = async (type, selectStep, step, start, end) => {
    const urls = promethusExp[type];
    try {
      let finallyData = [];
      const responses = await Promise.allSettled(urls.map((url) => getExpressQuery(url, step, start, end)));
      responses.map((item) => {
        if (item.value.status === ResponseCode.OK) {
          if (item.value.data.data.result.length) {
            item.value.data.data.result.map((dataItem) => {
              finallyData.push(dataItem);
            });
          }
        }
      });
      const xData = birthEmptyDayTime(stepList[stepValue], start, end);
      const { series, legend } = solveEchartsData(type, finallyData, colorList);
      // 封装指标
      setLegendData((oldLegendData) => {
        return { ...oldLegendData, [type]: legend };
      });
      setAllOptions((old) => {
        return { ...old, [type]: birthOptions(selectStep, xData, series) };
      });
    } catch (error) {
      messageApi.error(intl.formatMessage({ id: 'monitor.metricFail' }, { name: chineseChartType[type] || type }));
    }
  };

  const getGrafanaData = useCallback(() => {
    const keys = Object.keys(promethusExp);
    keys.forEach((item) => getOriginData(item, stepValue, stepList[stepValue], startTime, endTime));
  }, [stepValue, startTime, endTime]);

  useEffect(() => {
    //todo
    getGrafanaData();
  }, [getGrafanaData]);

  useEffect(() => {
    let xData = birthEmptyDayTime(stepList[stepValue], startTime, endTime);
    setEmptyOptions(birthOptions(stepValue, xData, [], { min: 0, max: 500, interval: 100 }));
  }, [stepValue, startTime, endTime]);

  return (
    <div className="overview_echarts">
      {contextHolder}
      <div className="overview_echarts_top">
        <p className="overview_echarts_top_title">{intl.formatMessage({ id: 'chart.clusterMonitor' })}</p>
        <div>
          <Select
            options={timePeriodOptions}
            onChange={handleChangeTime}
            value={stepValue}
            style={{ minWidth: '144px', marginRight: '16px' }}
          />
          <a className="overview_echarts_top_link" href={`/rest/ray/grafana`} target="_blank">
            View in Grafana
          </a>
        </div>
      </div>
      <div className="overview_echart2">
        <div className="overview_echart2_one">
          <div className="overview_echart2_one_top">
            <p>{intl.formatMessage({ id: 'chart.managedCpu' })}</p>
            <DataDisplayList data={legendData?.cpu} />
          </div>
          {allOptions && allOptions.cpu && allOptions.cpu.series.length ? (
            <OverviewEcharts overOption={allOptions?.cpu} />
          ) : (
            <EmptyArea options={emptyOptions} />
          )}
        </div>
        <div className="overview_echart2_one">
          <div className="overview_echart2_one_top">
            <div className="overview_echart2_one_top_sub">
              <p>{intl.formatMessage({ id: 'chart.cpuUsage' })}</p>
              <Tooltip placement="topLeft" title={intl.formatMessage({ id: 'chart.cpuUsageTooltip' })}>
                <QuestionCircleOutlined className="overview_tootip" />
              </Tooltip>
            </div>
            <DataDisplayList data={legendData?.nodeCpu} />
          </div>
          {allOptions && allOptions.nodeCpu && allOptions.nodeCpu.series.length ? (
            <OverviewEcharts overOption={allOptions?.nodeCpu} />
          ) : (
            <EmptyArea options={emptyOptions} />
          )}
        </div>
      </div>
      <div className="overview_echart2">
        <div className="overview_echart2_one">
          <div className="overview_echart2_one_top">
            <p>{intl.formatMessage({ id: 'chart.managedMemory' })}</p>
            <DataDisplayList data={legendData?.manageMemory} />
          </div>
          {allOptions && allOptions.manageMemory && allOptions.manageMemory.series.length ? (
            <OverviewEcharts overOption={allOptions?.manageMemory} />
          ) : (
            <EmptyArea options={emptyOptions} />
          )}
        </div>
        <div className="overview_echart2_one">
          <div className="overview_echart2_one_top">
            <div className="overview_echart2_one_top_sub">
              <p>{intl.formatMessage({ id: 'chart.memoryUsage' })}</p>
              <Tooltip placement="topLeft" title={intl.formatMessage({ id: 'chart.memoryUsageTooltip' })}>
                <QuestionCircleOutlined className="overview_tootip" />
              </Tooltip>
            </div>
            <DataDisplayList data={legendData?.nodeMemory} />
          </div>
          {allOptions && allOptions.nodeMemory && allOptions.nodeMemory.series.length ? (
            <OverviewEcharts overOption={allOptions?.nodeMemory} />
          ) : (
            <EmptyArea options={emptyOptions} />
          )}
        </div>
      </div>
      <div className="overview_echart2 overview_echart3">
        <div className="overview_echart2_one">
          <div className="overview_echart2_one_top">
            <p>{intl.formatMessage({ id: 'chart.managedGpu' })}</p>
            <DataDisplayList data={legendData?.manageGpu} />
          </div>
          {allOptions && allOptions.manageGpu && allOptions.manageGpu.series.length ? (
            <OverviewEcharts overOption={allOptions?.manageGpu} />
          ) : (
            <EmptyArea options={emptyOptions} />
          )}
        </div>
        <div className="overview_echart2_one">
          <div className="overview_echart2_one_top">
            <div className="overview_echart2_one_top_sub">
              <p>{intl.formatMessage({ id: 'chart.gpuUsage' })}</p>
              <Tooltip placement="topLeft" title={intl.formatMessage({ id: 'chart.gpuUsageTooltip' })}>
                <QuestionCircleOutlined className="overview_tootip" />
              </Tooltip>
            </div>
            <DataDisplayList data={legendData?.nodeGpu} />
          </div>
          {allOptions && allOptions.nodeGpu && allOptions.nodeGpu.series.length ? (
            <OverviewEcharts overOption={allOptions?.nodeGpu} />
          ) : (
            <EmptyArea options={emptyOptions} />
          )}
        </div>
        <div className="overview_echart2_one">
          <div className="overview_echart2_one_top">
            <p>{intl.formatMessage({ id: 'chart.gramUsage' })}</p>
            <DataDisplayList data={legendData?.nodeGpuMemory} />
          </div>
          {allOptions && allOptions.nodeGpuMemory && allOptions.nodeGpuMemory.series.length ? (
            <OverviewEcharts overOption={allOptions?.nodeGpuMemory} />
          ) : (
            <EmptyArea options={emptyOptions} />
          )}
        </div>
      </div>
      <div className="overview_echart2 overview_echart3">
        <div className="overview_echart2_one">
          <div className="overview_echart2_one_top">
            <p>{intl.formatMessage({ id: 'chart.managedNpu' })}</p>
            <DataDisplayList data={legendData?.manageNpu} />
          </div>
          {allOptions && allOptions.manageNpu && allOptions.manageNpu.series.length ? (
            <OverviewEcharts overOption={allOptions?.manageNpu} />
          ) : (
            <EmptyArea options={emptyOptions} />
          )}
        </div>
        <div className="overview_echart2_one">
          <div className="overview_echart2_one_top">
            <div className="overview_echart2_one_top_sub">
              <p>{intl.formatMessage({ id: 'chart.npuUsage' })}</p>
              <Tooltip placement="topLeft" title={intl.formatMessage({ id: 'chart.npuUsageTooltip' })}>
                <QuestionCircleOutlined className="overview_tootip" />
              </Tooltip>
            </div>
            <DataDisplayList data={legendData?.nodeNpu} />
          </div>
          {allOptions && allOptions.nodeNpu && allOptions.nodeNpu.series.length ? (
            <OverviewEcharts overOption={allOptions?.nodeNpu} />
          ) : (
            <EmptyArea options={emptyOptions} />
          )}
        </div>
        <div className="overview_echart2_one">
          <div className="overview_echart2_one_top">
            <p>{intl.formatMessage({ id: 'chart.hbmUsage' })}</p>
            <DataDisplayList data={legendData?.nodeNpuMemory} />
          </div>
          {allOptions && allOptions.nodeNpuMemory && allOptions.nodeNpuMemory.series.length ? (
            <OverviewEcharts overOption={allOptions?.nodeNpuMemory} />
          ) : (
            <EmptyArea options={emptyOptions} />
          )}
        </div>
      </div>
    </div>
  );
}
