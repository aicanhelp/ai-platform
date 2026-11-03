/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import DataAreaChart from '@/components/echarts/DataAreaChart';
import { solveBytesToMiB, solveUnitMonitor } from '@/tools/utils';
import SkeletonArea from '@/components/SkeletonArea';
import { useTranslation } from 'react-i18next';

export default function KubeApiServer({
  data,
  loaded,
  propsColor,
  refreshFn,
}) {
  const { t } = useTranslation('monitoring');

  return <>
    <div className='monitor_chart_container'>
      <p className='chart_title'>{t('chart.kubeApiServer.qps')}</p>
      <div className='monitor_half'>
        {loaded
          ? <DataAreaChart
            title={t('chart.kubeApiServer.qpsByVerb')}
            data={solveBytesToMiB(data?.kube_apiserver_qps_by_verb, false)}
            type="line" id="qps_verb"
            rangeColorProps={propsColor}
            styleProps={{ gridRight: '40%' }}
            tooltip={<div className='tooltip_box'>
              <p>{t('chart.kubeApiServer.qpsByVerbTooltip1')}</p>
              <p>{t('chart.kubeApiServer.qpsByVerbTooltip2')}</p>
            </div>}
            refreshFn={refreshFn} />
          : <SkeletonArea />
        }
        {loaded
          ? <DataAreaChart
            title={t('chart.kubeApiServer.qpsByVerbResource')}
            data={solveBytesToMiB(data?.kube_apiserver_qps_by_verb_resource, false)}
            type="line" id="qps_verb_resource"
            rangeColorProps={propsColor}
            styleProps={{ gridRight: '40%' }}
            tooltip={<div className='tooltip_box'>
              <p>{t('chart.kubeApiServer.qpsByVerbResourceTooltip1')}</p>
              <p>{t('chart.kubeApiServer.qpsByVerbResourceTooltip2')}</p>
            </div>}
            refreshFn={refreshFn} />
          : <SkeletonArea />
        }
      </div>
    </div>
    <div className='monitor_chart_container'>
      <p className='chart_title'>{t('chart.kubeApiServer.latency')}</p>
      <div className='monitor_half'>
        {loaded
          ? <DataAreaChart
            title={t('chart.kubeApiServer.latencyByVerb')}
            data={solveBytesToMiB(data?.kube_apiserver_request_latency_by_verb, false)}
            type="line"
            styleProps={{ gridRight: '40%' }}
            id="latency_qps_request_verb"
            rangeColorProps={propsColor}
            refreshFn={refreshFn} />
          : <SkeletonArea />
        }
        {loaded
          ? <DataAreaChart
            title={t('chart.kubeApiServer.latencyByVerbResource')}
            data={solveBytesToMiB(data?.kube_apiserver_request_latency_by_verb_resource, false)}
            type="line"
            styleProps={{ gridRight: '40%' }}
            id="qps_request_verb_resource"
            rangeColorProps={propsColor}
            refreshFn={refreshFn}
          />
          : <SkeletonArea />
        }
      </div>
    </div>
    <div className='monitor_chart_container'>
      <p className='chart_title'>{t('chart.kubeApiServer.resource')}</p>
      <div className='monitor_half KubeApiServer'>
        {loaded
          ? <DataAreaChart
            className='KubeApiServer'
            title={t('chart.kubeApiServer.cpuUsage')}
            data={solveBytesToMiB(data?.kube_apiserver_cpu_usage, false)}
            type="line"
            styleProps={{ gridRight: '40%' }}
            id="qps_cpu"
            rangeColorProps={propsColor}
            refreshFn={refreshFn} />
          : <SkeletonArea />
        }
        {loaded
          ? <DataAreaChart
            title={t('chart.kubeApiServer.memoryUsage', { unit: solveUnitMonitor(data?.kube_apiserver_memory_usage) })}
            data={solveBytesToMiB(data?.kube_apiserver_memory_usage)}
            type="line"
            styleProps={{ gridRight: '40%' }}
            id="qps_memory"
            rangeColorProps={propsColor}
            refreshFn={refreshFn}
          />
          : <SkeletonArea />
        }
      </div>
    </div>
  </>;
}
