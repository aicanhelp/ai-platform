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

export default function KubeController({
  data,
  loaded,
  propsColor,
  refreshFn,
}) {
  const { t } = useTranslation('monitoring');

  return <>
    <div className='monitor_chart_container'>
      <p className='chart_title'>{t('chart.kubeController.qps')}</p>
      {loaded
        ? <DataAreaChart
          title={t('chart.kubeController.qpsTitle')}
          data={solveBytesToMiB(data.requestRate, false)}
          type="line"
          rangeColorProps={propsColor}
          id="kube_controller_manager_kube_api"
          refreshFn={refreshFn} />
        : <SkeletonArea />
      }
    </div>
    <div className='monitor_chart_container KubeController'>
      <p className='chart_title'>{t('chart.kubeController.resource')}</p>
      <div className='monitor_half KubeController'>
        {loaded
          ? <DataAreaChart
            className='KubeController'
            title={t('chart.kubeController.cpuUsage')}
            data={solveBytesToMiB(data.kube_controller_manager_cpu_usage, false)}
            type="line"
            styleProps={{ gridRight: '40%' }}
            rangeColorProps={propsColor}
            id="resource_cpu"
            refreshFn={refreshFn} />
          : <SkeletonArea />
        }
        {loaded
          ? <DataAreaChart
            title={t('chart.kubeController.memoryUsage', { unit: solveUnitMonitor(data?.kube_controller_manager_memory_usage) })}
            data={solveBytesToMiB(data?.kube_controller_manager_memory_usage)}
            type="line"
            rangeColorProps={propsColor}
            id="resource_memory"
            styleProps={{ gridRight: '40%' }}
            refreshFn={refreshFn}
            className='KubeController'
          />
          : <SkeletonArea />
        }
      </div>
    </div>
  </>;
}
