/**
 *  Copyright (c) 2024 Huawei Technologies Co., Ltd.
 *  openFuyao is licensed under Mulan PSL v2.
 *  You can use this software according to the terms and conditions of the Mulan PSL v2.
 *  You may obtain a copy of Mulan PSL v2 at:
  
 *       http://license.coscl.org.cn/MulanPSL2
  
 *   THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 *   EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 *   MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 *   See the Mulan PSL v2 for more details.
 */

import { GET, POST, PUT, DELETE } from '@/api/request';
import { solveParams } from '@/tools/utils';
import qs from 'query-string';
function commonApiPrefix() {
  let clusterName = sessionStorage.getItem('cluster');
  let paramObj = {
    monitorUrl: `/clusters/${clusterName}/rest/monitoring/v1`,
    resourceUrl: `/clusters/${clusterName}/api/kubernetes/api/v1`,
    workloadUrl: `/clusters/${clusterName}/api/kubernetes/apis/apps/v1`,
  };
  return paramObj;
};

/**
 * 控制平面组件
 */
export function getComponentControllPlatformMonitor({ key, component, start, end, conditions }, signal) {
  if (component) {
    let prefix = 'control-plane-components';
    switch (key) {
      case 'controll': {
        prefix = 'control-plane-components';
        break;
      }
      case 'node': {
        prefix = 'node-components';
        break;
      }
      case 'addon': {
        prefix = 'add-ons-components';
        break;
      }
      default: {
        prefix = 'control-plane-components';
      }
    }
    const url = qs.stringifyUrl({
      url: `${commonApiPrefix().monitorUrl}/${prefix}/${component}`, query: solveParams({
        start,
        end,
        step: conditions.step,
        // metrics_filter: conditions.metricsFilter,
        verb: conditions.verb,
        resource: conditions.resource,
        instance: conditions.instance,
      }),
    }, {
      skipNull: true,
    },
    );
    return GET(url, { signal });
  } else {
    return new Promise();
  }
}

/**
 * 获取k8s资源 instance/verb/resource
 */
export function getResourceMonitor(expr) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().monitorUrl}/query`, query: solveParams({
      expr,
    }),
  }, {
    skipNull: true,
  },
  );
  return GET(url);
};

// 获取cluster资源
export function getResourceControllMonitor({ cluster, value, start, end, conditions }, signal) {
  if (cluster) {
    let prefix = 'cluster';
    switch (value) {
      case 'cluster': {
        prefix = 'cluster';
        break;
      }
      case 'node': {
        prefix = `nodes/${conditions.instance || '.*'}`;
        break;
      }
      case 'workload': {
        prefix = `namespaces/${conditions.namespace}/${conditions.type}s/${conditions.workload || '.*'}`;
        break;
      }
      case 'pod': {
        prefix = `namespaces/${conditions.namespace}/pods/${conditions.pod || '.*'}`;
        break;
      }
      case 'container': {
        prefix = `namespaces/${conditions.namespace}/pods/${conditions.pod || '.*'}/containers/${conditions.container || '.*'}`;
        break;
      }
      default: {
        prefix = 'cluster';
      }
    }
    const url = qs.stringifyUrl({
      url: `${commonApiPrefix().monitorUrl}/${prefix}/metrics`, query: solveParams({
        start,
        end,
        step: conditions?.step,
        // metrics_filter: conditions.metricsFilter,
      }),
    }, {
      skipNull: true,
    },
    );
    return GET(url, { signal });
  } else {
    return new Promise();
  }
};

// 获取节点下的所有资源
export function getNodeList() {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().resourceUrl}/nodes`,
  }
  );
  return GET(url);
};

// 获取工作负载下的namespace
export function getNamespaceList() {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().resourceUrl}/namespaces`,
  }
  );
  return GET(url);
};

// 获取workload
export function getWorkloadList(namespace, type) {
  let prefixUrl = commonApiPrefix().workloadUrl;
  if (type === 'pod') {
    prefixUrl = commonApiPrefix().resourceUrl;
  }
  const url = qs.stringifyUrl({
    url: `${prefixUrl}/namespaces/${namespace}/${type}s`,
  }
  );
  return GET(url);
};

// 查询自定义面板
export function getCustomizeQuery(expr, start, end, step) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().monitorUrl}/query`,
    query: solveParams({
      expr,
      start,
      end,
      step,
    }),
  }, {
    skipNull: true,
  }
  );
  return GET(url);
};

// 获取告警规则列表
export function getAlarmListData() {
  const url = `${commonApiPrefix().monitorUrl}/alerting-rules`;
  return GET(url);
};

// 获取监控目标列表
export function getMonitorGoalListData() {
  const url = `${commonApiPrefix().monitorUrl}/targets`;
  return GET(url);
};

// 获取集群资源数据
export function getClusterData() {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().monitorUrl}/cluster/metrics`,
  }
  );
  return GET(url);
};

// 集群资源节点cpu使用率、内存使用率、磁盘使用率
export function getClusterCpuMemoryDisk(dataType, start, end, step) {
  const url = `${commonApiPrefix().monitorUrl}/nodes/.*/metrics?metrics_filter=${dataType.cpu}|${dataType.memory}|${dataType.disk}&start=${start}&end=${end}&step=${step}
  `;
  return GET(url);
};