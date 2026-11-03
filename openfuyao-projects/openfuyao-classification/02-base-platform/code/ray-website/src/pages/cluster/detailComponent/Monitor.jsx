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
  lokiExp,
  stepList,
  getChineseLokiType,
} from '@/common/constants';
import { useIntl } from 'inula-intl';
import '@/styles/rayOverview.less';
import { Select, message } from 'antd';
import dayjs from 'dayjs';
import OverviewEcharts from '@/components/OverviewEcharts';
import {
  filterTimeValue,
  birthOptions,
  solveLokiEchartsData,
  solveDataAreaNumPalette,
  birthLokiDayTime,
  birthEmptyDayTime,
} from '@/tools/utils';
import { getLokiQuery } from '@/api/rayApi';
import { useState, useEffect, useCallback } from 'openinula';
import DataDisplayList from '@/components/DataDisplayList';
import EmptyArea from '@/components/EmptyArea';

const colorList = solveDataAreaNumPalette(10000);
let errorList = [];
export default function Monitor({ clusterName = '' }) {
  const intl = useIntl();
  const timePeriodOptions = getTimePeriodOptions(intl);
  const chineseLokiType = getChineseLokiType(intl);
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
    const urls = lokiExp[type];
    const insertText = clusterName ? `job =~ "^${clusterName}.*",` : '';
    try {
      let finallyData = []; // s->ns
      let responses = [];
      if (urls.length) {
        responses = await Promise.allSettled(
          urls.map((url) =>
            getLokiQuery(url.replace(/{/, `{${insertText}`), step, start * 1000000000, end * 1000000000, 5000),
          ),
        );
      }
      responses.map((item) => {
        if (item.value.status === ResponseCode.OK) {
          if (item.value.data.data.result.length) {
            item.value.data.data.result.map((dataItem) => {
              finallyData.push(dataItem);
            });
          }
        }
      });
      const xData = birthLokiDayTime(stepList[stepValue], start, end);
      const { series, legend, config } = solveLokiEchartsData(type, finallyData, colorList);
      // 封装指标
      setLegendData((oldLegendData) => {
        return { ...oldLegendData, [type]: legend };
      });
      setAllOptions((old) => {
        return { ...old, [type]: birthOptions(selectStep, xData, series, config, 'loki') };
      });
    } catch (error) {
      errorList.push(chineseLokiType[type] || type);
    }
  };

  const getGrafanaData = useCallback(async () => {
    const keys = Object.keys(lokiExp);
    errorList = [];
    messageApi.open({
      key: 'loading',
      type: 'loading',
      content: 'loading...',
      duration: 0,
    });
    for (const item of keys) {
      await getOriginData(item, stepValue, stepList[stepValue], startTime, endTime);
    }
    setTimeout(messageApi.destroy('loading'));
    if (errorList.length === keys.length) {
      messageApi.error(intl.formatMessage({ id: 'monitor.configMapError' }));
    } else {
      errorList.map((errorItem) => {
        messageApi.error(intl.formatMessage({ id: 'monitor.metricFail' }, { name: errorItem }));
      });
    }
  }, [stepValue, startTime, endTime]);

  const renderDashboard = (legendTotal) => {
    let legend = [];
    if (legendTotal) {
      legend = [legendTotal.dashboardNode, legendTotal.dashboardLog, legendTotal.dashboardJob]
        .flat()
        .filter((arr) => arr && arr.length !== 0);
    }
    return legend;
  };

  const mergeSeries = (node, log, job) => {
    let newOption = node || log || job;
    let series = [];
    if (node && node.series.length) {
      series = series.concat(node.series);
    }
    if (log && log.series.length) {
      series = series.concat(log.series);
    }
    if (job && job.series.length) {
      series = series.concat(job.series);
    }
    series = [...new Map(series.map((item) => [item.name, item])).values()];
    if (newOption) {
      newOption.yAxis = { type: 'value' };
      newOption.series = series;
    }
    return newOption;
  };

  const judgeIsExist = (options) => {
    if (
      options?.dashboardNode?.series?.length ||
      options?.dashboardLog?.series?.length ||
      options?.dashboardJob?.series?.length
    ) {
      return true;
    }
    return false;
  };

  useEffect(() => {
    getGrafanaData();
  }, [getGrafanaData]);

  useEffect(() => {
    let xData = birthEmptyDayTime(stepList[stepValue], startTime, endTime);
    setEmptyOptions(birthOptions(stepValue, xData, [], { min: 0, max: 500, interval: 100 }));
  }, [stepValue, startTime, endTime]);

  return (
    <div className="overview_content">
      <div className="overview_echarts">
        {contextHolder}
        <div className="overview_echarts_top">
          <p className="overview_echarts_top_title">{intl.formatMessage({ id: 'monitor.clusterHealth' })}</p>
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
        <div className="overview_box">
          <div className="overview_blod_title">{intl.formatMessage({ id: 'monitor.gcsPressure' })}</div>
          <div className="overview_echart2 overview_echart3">
            <div className="overview_echart2_one">
              <div className="overview_echart2_one_top">
                <p>{intl.formatMessage({ id: 'monitor.gcsStateTotalDesc' })}</p>
                <DataDisplayList data={legendData?.gcsStateTotal} />
              </div>
              {allOptions?.gcsStateTotal?.series.length ? (
                <OverviewEcharts id={'gcsStateTotal'} overOption={allOptions?.gcsStateTotal} />
              ) : (
                <EmptyArea options={emptyOptions} />
              )}
            </div>
            <div className="overview_echart2_one">
              <div className="overview_echart2_one_top">
                <div className="overview_echart2_one_top_sub">
                  <p>{intl.formatMessage({ id: 'monitor.gcsStateActiveDesc' })}</p>
                </div>
                <DataDisplayList data={legendData?.gcsStateActive} />
              </div>
              {allOptions?.gcsStateActive?.series.length ? (
                <OverviewEcharts id={'gcsStateActive'} overOption={allOptions?.gcsStateActive} />
              ) : (
                <EmptyArea options={emptyOptions} />
              )}
            </div>
            <div className="overview_echart2_one">
              <div className="overview_echart2_one_top">
                <p>{intl.formatMessage({ id: 'monitor.gcsQueueTimeDesc' })}</p>
                <DataDisplayList data={legendData?.gcsQueueTime} />
              </div>
              {allOptions?.gcsQueueTime?.series.length ? (
                <OverviewEcharts id={'gcsQueueTime'} overOption={allOptions?.gcsQueueTime} />
              ) : (
                <EmptyArea options={emptyOptions} />
              )}
            </div>
          </div>
        </div>

        <div className="overview_box">
          <div className="overview_blod_title">{intl.formatMessage({ id: 'monitor.nodeHealth' })}</div>
          <div className="overview_echart2">
            <div className="overview_echart2_one">
              <div className="overview_echart2_one_top">
                <p>{intl.formatMessage({ id: 'monitor.gcsHealthCheckDesc' })}</p>
                <DataDisplayList data={legendData?.gcsHealthyCheck} />
              </div>
              {allOptions?.gcsHealthyCheck?.series.length ? (
                <OverviewEcharts overOption={allOptions?.gcsHealthyCheck} />
              ) : (
                <EmptyArea options={emptyOptions} />
              )}
            </div>
            <div className="overview_echart2_one">
              <div className="overview_echart2_one_top">
                <div className="overview_echart2_one_top_sub">
                  <p>{intl.formatMessage({ id: 'monitor.rayletHealthCheckDesc' })}</p>
                </div>
                <DataDisplayList data={legendData?.gcsRayLet} />
              </div>
              {allOptions?.gcsRayLet?.series.length ? (
                <OverviewEcharts overOption={allOptions?.gcsRayLet} />
              ) : (
                <EmptyArea options={emptyOptions} />
              )}
            </div>
          </div>
        </div>

        <div className="overview_box">
          <div className="overview_blod_title">{intl.formatMessage({ id: 'monitor.jobPressure' })}</div>
          <div className="overview_echart2 overview_echart3">
            <div className="overview_echart2_one">
              <div className="overview_echart2_one_top">
                <p>{intl.formatMessage({ id: 'monitor.jobGetDesc' })}</p>
                <DataDisplayList data={legendData?.jobGet} />
              </div>
              {allOptions?.jobGet?.series.length ? (
                <OverviewEcharts overOption={allOptions?.jobGet} />
              ) : (
                <EmptyArea options={emptyOptions} />
              )}
            </div>
            <div className="overview_echart2_one">
              <div className="overview_echart2_one_top">
                <div className="overview_echart2_one_top_sub">
                  <p>{intl.formatMessage({ id: 'monitor.jobPostDesc' })}</p>
                </div>
                <DataDisplayList data={legendData?.jobPost} />
              </div>
              {allOptions?.jobPost?.series.length ? (
                <OverviewEcharts overOption={allOptions?.jobPost} />
              ) : (
                <EmptyArea options={emptyOptions} />
              )}
            </div>
            <div className="overview_echart2_one">
              <div className="overview_echart2_one_top">
                <p>{intl.formatMessage({ id: 'monitor.jobDeleteDesc' })}</p>
                <DataDisplayList data={legendData?.jobDelete} />
              </div>
              {allOptions?.jobDelete?.series.length ? (
                <OverviewEcharts overOption={allOptions?.jobDelete} />
              ) : (
                <EmptyArea options={emptyOptions} />
              )}
            </div>
            <div className="overview_echart2_one">
              <div className="overview_echart2_one_top">
                <div className="overview_echart2_one_top_sub">
                  <p>{intl.formatMessage({ id: 'monitor.jobLogGetDesc' })}</p>
                </div>
                <DataDisplayList data={legendData?.jobLogGet} />
              </div>
              {allOptions?.jobLogGet?.series.length ? (
                <OverviewEcharts overOption={allOptions?.jobLogGet} />
              ) : (
                <EmptyArea options={emptyOptions} />
              )}
            </div>
            <div className="overview_echart2_one">
              <div className="overview_echart2_one_top">
                <div className="overview_echart2_one_top_sub">
                  <p>{intl.formatMessage({ id: 'monitor.jobLogDeleteDesc' })}</p>
                </div>
                <DataDisplayList data={legendData?.jobLogDelete} />
              </div>
              {allOptions?.jobLogDelete?.series.length ? (
                <OverviewEcharts overOption={allOptions?.jobLogDelete} />
              ) : (
                <EmptyArea options={emptyOptions} />
              )}
            </div>
          </div>
        </div>

        <div className="overview_box">
          <div className="overview_blod_title">{intl.formatMessage({ id: 'monitor.dashboardPressure' })}</div>
          <div className="overview_echart">
            <div className="overview_echart2_one">
              <div className="overview_echart2_one_top">
                <p>{intl.formatMessage({ id: 'monitor.dashboardDesc' })}</p>
                <DataDisplayList data={renderDashboard(legendData)} />
              </div>
              {judgeIsExist(allOptions) ? (
                <OverviewEcharts
                  overOption={mergeSeries(allOptions.dashboardNode, allOptions.dashboardLog, allOptions.dashboardJob)}
                />
              ) : (
                <EmptyArea options={emptyOptions} />
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
