/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import { GET, POST, PUT, DELETE, PATCH } from '@/api/request';
import qs from 'query-string';
import { solveParams } from '@/tools/utils';
const pretty = true; // 规范格式数据
// 统一公共前缀
function commonApiPrefix() {
  let clusterName = sessionStorage.getItem('cluster');
  let paramObj = {
    podUrl: `/clusters/${clusterName}/api/kubernetes/api/v1`,
    workloadUrl: `/clusters/${clusterName}/api/kubernetes/apis/apps/v1`,
    workloadEventsUrl: `/clusters/${clusterName}/api/kubernetes/apis/events.k8s.io/v1`,
    jobUrl: `/clusters/${clusterName}/api/kubernetes/apis/batch/v1`,
    crdUrl: `/clusters/${clusterName}/api/kubernetes/apis/apiextensions.k8s.io/v1`,
    crUrl: `/clusters/${clusterName}/api/kubernetes/apis`,
    serviceUrl: `/clusters/${clusterName}/api/kubernetes/api/v1`,
    ingressUrl: `/clusters/${clusterName}/api/kubernetes/apis/networking.k8s.io/v1`,
    helmUrl: `/clusters/${clusterName}/rest/application-management/v1beta1/helm-releases`,
    helmUpdate: `/clusters/${clusterName}/rest/marketplace/v1beta1/helm-repos`,
    helmLogUrl: `/clusters/${clusterName}/api/kubernetes/api/v1`,
    helmEventUrl: `/clusters/${clusterName}/api/kubernetes/apis/events.k8s.io/v1`,
    dictionaryUrl: `/clusters/${clusterName}/api/kubernetes/api/v1`,
    consolePluginUrl: `/clusters/${clusterName}/rest/plugin-management/v1beta1/consoleplugins`,
    authorUrl: `/clusters/${clusterName}/api/kubernetes/apis/rbac.authorization.k8s.io/v1`,
    manager: `/clusters/${clusterName}/api/kubernetes/apis/rbac.authorization.k8s.io/v1`,
    overviewUrl: `/clusters/${clusterName}/rest/monitoring/v1`,
    roleUrl: `/clusters/${clusterName}/api/kubernetes/apis/rbac.authorization.k8s.io/v1`,
    resourcePvUrl: `/clusters/${clusterName}/api/kubernetes/api/v1`,
    resourceStorageUrl:`/clusters/${clusterName}/api/kubernetes/apis/storage.k8s.io/v1`,
    priorityClassesUrl: `/clusters/${clusterName}/rest/scheduling/apis/scheduling.k8s.io/v1`,
  };
  return paramObj;
};

/**
 * 获取ConsolePlugins列表
 */
export function getConsolePlugins() {
  const url = `${commonApiPrefix().consolePluginUrl}`;
  return GET(url);
}

/**
 * 获取Pod列表数据
 */
export function getPodsData(
  namespace,
  conditions = {},
) {
  let fieldSelector = [];
  if (conditions.nodeName) {
    fieldSelector.push(`spec.nodeName=${conditions.nodeName}`);
  }
  const url = qs.stringifyUrl(
    {
      url: `${commonApiPrefix().podUrl}${namespace ? `/namespaces/${namespace}` : ''}/pods`,
      query: solveParams({
        pretty,
        fieldSelector: fieldSelector.join(','),
      }),
    },
    { skipNull: true }
  );
  return GET(url);
}
/**
 * 获取pod详情
 */
export function getPodDetailDescription(namespace, podName) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().podUrl}${namespace ? `/namespaces/${namespace}` : ''}/pods/${podName}`,
    query: { pretty },
  });
  return GET(url);
}

/**
 * 删除pod容器
 */
export function deletePodContainer(namespace, podName) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().podUrl}${namespace ? `/namespaces/${namespace}` : ''}/pods/${podName}`,
    query: { pretty },
  });
  return DELETE(url);
}

/**
 * 增加podyaml
 */
export function addPodYamlData(namespace, data) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().podUrl}${namespace ? `/namespaces/${namespace}` : ''}/pods`,
  });
  return POST(url, data, { 'Content-Type': 'application/json' });
}

export function addRoleYamlData(namespace, data) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().manager}${namespace ? `/namespaces/${namespace}` : ''}/roles`,
  });
  return POST(url, data, { 'Content-Type': 'application/json' });
}

export function addServiceAccountsYamlData(namespace, data) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().podUrl}${namespace ? `/namespaces/${namespace}` : ''}/serviceaccounts`,
  });
  return POST(url, data, { 'Content-Type': 'application/json' });
}

export function addRoleBindingsYamlData(namespace, data) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().manager}${namespace ? `/namespaces/${namespace}` : ''}/rolebindings`,
  });
  return POST(url, data, { 'Content-Type': 'application/json' });
}
/**
 * 修改pod yMAL
 */
export function updatePodYamlData(namespace, podName, data) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().podUrl}${namespace ? `/namespaces/${namespace}` : ''}/pods/${podName}`,
  });
  return PUT(url, data, { 'Content-Type': 'application/json' });
}
/**
 * 编辑注解
 */
export function editAnnotationsOrLabels(type, namespace, name, annotations) {
  let prefixUrl = '';
  let url = '';
  if (type === 'job' || type === 'cronjob') {
    prefixUrl = `${commonApiPrefix().jobUrl}${namespace ? `/namespaces/${namespace}` : ''}`;
  } else if (type === 'node' || type === 'namespace') {
    prefixUrl = `${commonApiPrefix().podUrl}`;
  } else if (type === 'customresourcedefinition') {
    prefixUrl = `${commonApiPrefix().crdUrl}`;
  } else if (type === 'serviceaccount') {
    prefixUrl = `${commonApiPrefix().podUrl}${namespace ? `/namespaces/${namespace}` : ''}`;
  } else if (type === 'role' || type === 'rolebinding') {
    prefixUrl = `${commonApiPrefix().roleUrl}${namespace ? `/namespaces/${namespace}` : ''}`;
  } else if (type === 'clusterrole' || type === 'clusterrolebinding') {
    prefixUrl = `${commonApiPrefix().roleUrl}`;
  } else {
    prefixUrl = `${(type === 'pod' || type === 'limitrange' || type === 'resourcequota')
      ? commonApiPrefix().podUrl
      : commonApiPrefix().workloadUrl}${namespace ? `/namespaces/${namespace}` : ''}`;
  }
  url = `${prefixUrl}/${type}s/${name}`;
  return PATCH(url, annotations, {
    'Content-Type': 'application/json-patch+json',
  });
}

/**
 * 获取pod日志
 */
export function getPodLog(namespace, podName, containerName) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().podUrl}${namespace ? `/namespaces/${namespace}` : ''}/pods/${podName}/log`,
    query: { container: containerName },
  });
  return GET(url);
}

/** 获取事件列表 */
export function getEventsData(
  { page, limit, continueToken },
  namespace,
  conditions
) {
  let filters = [];
  if (conditions) {
    if (conditions.passName && conditions.isNormal) {
      filters.push(`regarding.name=${conditions.resourceName}`); // 所有普通事件入口
    }
    if (conditions.resourceName && !conditions.passName) {
      filters.push(`metadata.name=${conditions.resourceName}`); // 所有普通事件入口
    }
    if (conditions.resourceType) {
      filters.push(`regarding.kind=${conditions.resourceType}`);
    }
    if (conditions.eventLevel) {
      filters.push(`type=${conditions.eventLevel}`);
    }
    if (conditions.time) {
      filters.push(`metadata.creationTimestamp>=${conditions.timeRoundBefore}`);
      filters.push(`metadata.creationTimestamp<=${conditions.timeRoundEnd}`);
    }
    if (conditions.nodeEvents) {
      filters.push(`involvedObject.name=${conditions.nodeEvents.nodeName}`);
      filters.push(
        `involvedObject.kind=${conditions.nodeEvents.type || 'Node'}`
      );
    }
  }

  const url = qs.stringifyUrl(
    {
      url: `${commonApiPrefix().workloadEventsUrl}/${namespace ? `namespaces/${namespace}/` : ''}events`,
      query: filters.length
        ? solveParams({
          fieldSelector: filters.join(','),
          page,
          limit,
          continue: continueToken,
        })
        : solveParams({
          page,
          limit,
          continue: continueToken,
        }),
    },
    { skipNull: true }
  );
  return GET(url);
}

// deployment
export function getDeploymentsData(namespace) {
  const url = qs.stringifyUrl(
    {
      url: `${commonApiPrefix().workloadUrl}${namespace ? `/namespaces/${namespace}` : ''}/deployments`,
      query: solveParams({
        pretty,
      }),
    },
    { skipNull: true }
  );
  return GET(url);
}

export function addDeploymentYamlData(namespace, data) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().workloadUrl}${namespace ? `/namespaces/${namespace}` : ''}/deployments`,
  });
  return POST(url, data, { 'Content-Type': 'application/json' });
}

export function getDeploymentDetailDescription(namespace, name) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().workloadUrl}${namespace ? `/namespaces/${namespace}` : ''}/deployments/${name}`,
    query: { pretty },
  });
  return GET(url);
}
// 删除
export function deleteDeployment(namespace, deploymentName) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().workloadUrl}${namespace ? `/namespaces/${namespace}` : ''}/deployments/${deploymentName}`,
    query: { pretty },
  });
  return DELETE(url);
}

export function updateDeploymentYamlData(namespace, name, data) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().workloadUrl}${namespace ? `/namespaces/${namespace}` : ''}/deployments/${name}`,
  });
  return PUT(url, data, { 'Content-Type': 'application/json' });
}

// statefulSet
export function getStatefulSetsData(namespace) {
  const url = qs.stringifyUrl(
    {
      url: `${commonApiPrefix().workloadUrl}${namespace ? `/namespaces/${namespace}` : ''}/statefulsets`,
      query: solveParams({
        pretty,
      }),
    },
    { skipNull: true }
  );
  return GET(url);
}

// 删除
export function deleteStatefulSet(namespace, statefulSetName) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().workloadUrl}${namespace ? `/namespaces/${namespace}` : ''}/statefulsets/${statefulSetName}`,
    query: { pretty },
  });
  return DELETE(url);
}

export function addStatefulSetYamlData(namespace, data) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().workloadUrl}${namespace ? `/namespaces/${namespace}` : ''}/statefulsets`,
  });
  return POST(url, data, { 'Content-Type': 'application/json' });
}

export function getStatefulSetDetailDescription(namespace, name) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().workloadUrl}${namespace ? `/namespaces/${namespace}` : ''}/statefulsets/${name}`,
    query: { pretty },
  });
  return GET(url);
}

export function updateStatefulSetYamlData(namespace, name, data) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().workloadUrl}${namespace ? `/namespaces/${namespace}` : ''}/statefulsets/${name}`,
  });
  return PUT(url, data, { 'Content-Type': 'application/json' });
}

// daemonSet
export function getDaemonSetsData(namespace) {
  const url = qs.stringifyUrl(
    {
      url: `${commonApiPrefix().workloadUrl}${namespace ? `/namespaces/${namespace}` : ''}/daemonsets`,
      query: solveParams({
        pretty,
      }),
    },
    { skipNull: true }
  );
  return GET(url);
}

export function deleteDaemonSet(namespace, daemonSetName) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().workloadUrl}${namespace ? `/namespaces/${namespace}` : ''}/daemonsets/${daemonSetName}`,
    query: { pretty },
  });
  return DELETE(url);
}

export function addDaemonSetYamlData(namespace, data) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().workloadUrl}${namespace ? `/namespaces/${namespace}` : ''}/daemonsets`,
  });
  return POST(url, data, { 'Content-Type': 'application/json' });
}

export function getDaemonSetDetailDescription(namespace, name) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().workloadUrl}${namespace ? `/namespaces/${namespace}` : ''}/daemonsets/${name}`,
    query: { pretty },
  });
  return GET(url);
}

export function updateDaemonSetYamlData(namespace, name, data) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().workloadUrl}${namespace ? `/namespaces/${namespace}` : ''}/daemonsets/${name}`,
  });
  return PUT(url, data, { 'Content-Type': 'application/json' });
}

// job
export function getJobsData(namespace) {
  const url = qs.stringifyUrl(
    {
      url: `${commonApiPrefix().jobUrl}${namespace ? `/namespaces/${namespace}` : ''}/jobs`,
      query: solveParams({
        pretty,
      }),
    },
    { skipNull: true }
  );
  return GET(url);
}

export function deleteJob(namespace, jobName) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().jobUrl}${namespace ? `/namespaces/${namespace}` : ''}/jobs/${jobName}`,
    query: { pretty },
  });
  return DELETE(url);
}

export function addJobYamlData(namespace, data) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().jobUrl}${namespace ? `/namespaces/${namespace}` : ''}/jobs`,
  });
  return POST(url, data, { 'Content-Type': 'application/json' });
}

export function getJobDetailDescription(namespace, name) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().jobUrl}${namespace ? `/namespaces/${namespace}` : ''}/jobs/${name}`,
    query: { pretty },
  });
  return GET(url);
}

export function updateJobYamlData(namespace, name, data) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().jobUrl}${namespace ? `/namespaces/${namespace}` : ''}/jobs/${name}`,
  });
  return PUT(url, data, { 'Content-Type': 'application/json' });
}

// CronJob
export function getCronJobsData(namespace) {
  const url = qs.stringifyUrl(
    {
      url: `${commonApiPrefix().jobUrl}${namespace ? `/namespaces/${namespace}` : ''}/cronjobs`,
      query: solveParams({
        pretty,
      }),
    },
    { skipNull: true }
  );
  return GET(url);
}

export function deleteCronJob(namespace, cronJobName) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().jobUrl}${namespace ? `/namespaces/${namespace}` : ''}/cronjobs/${cronJobName}`,
    query: { pretty },
  });
  return DELETE(url);
}

export function addCronJobYamlData(namespace, data) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().jobUrl}${namespace ? `/namespaces/${namespace}` : ''}/cronjobs`,
  });
  return POST(url, data, { 'Content-Type': 'application/json' });
}

export function getCronJobDetailDescription(namespace, name) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().jobUrl}${namespace ? `/namespaces/${namespace}` : ''}/cronjobs/${name}`,
    query: { pretty },
  });
  return GET(url);
}

export function updateCronJobYamlData(namespace, name, data) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().jobUrl}${namespace ? `/namespaces/${namespace}` : ''}/cronjobs/${name}`,
  });
  return PUT(url, data, { 'Content-Type': 'application/json' });
}

/**
 * 节点管理
 */
export function getNodeList() {
  const url = qs.stringifyUrl(
    {
      url: `${commonApiPrefix().podUrl}/nodes`,
      query: solveParams({
        pretty,
      }),
    },
    { skipNull: true }
  );
  return GET(url);
}

export function getNodeDetailDescription(nodeName) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().podUrl}/nodes${nodeName ? `/${nodeName}` : ''}`,
    query: { pretty },
  });
  return GET(url);
}

// 命名空间
export function getNamespaceList() {
  const url = qs.stringifyUrl(
    {
      url: `${commonApiPrefix().podUrl}/namespaces`,
      query: solveParams({
        pretty,
      }),
    },
    { skipNull: true }
  );
  return GET(url);
}

export function deleteNamespace(namespaceName) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().podUrl}/namespaces/${namespaceName}`,
    query: { pretty },
  });
  return DELETE(url);
}

export function updateNamespaceYaml(namespaceName, data) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().podUrl}/namespaces/${namespaceName}`,
  });
  return PUT(url, data, { 'Content-Type': 'application/json' });
}

export function getImagePullSecrets(namespaceName) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().podUrl}/namespaces/${namespaceName}/serviceaccounts/default`,
  });
  return GET(url);
}

export function addNamespaceYamlData(data) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().podUrl}/namespaces`,
  });
  return POST(url, data, { 'Content-Type': 'application/json' });
}

export function getNamespaceDetailDescription(namespaceName) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().podUrl}/namespaces/${namespaceName}`,
    query: { pretty },
  });
  return GET(url);
}

// 限制范围
export function getLimitRangeList(namespace) {
  const url = qs.stringifyUrl(
    {
      url: `${commonApiPrefix().podUrl}${namespace ? `/namespaces/${namespace}` : ''}/limitranges`,
      query: solveParams({
        pretty,
      }),
    },
    { skipNull: true }
  );
  return GET(url);
}

export function addLimitRangeYamlData(namespace, data) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().podUrl}${namespace ? `/namespaces/${namespace}` : ''}/limitranges`,
  });
  return POST(url, data, { 'Content-Type': 'application/json' });
}
export function getLimitRangeDetailDescription(namespace, name) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().podUrl}${namespace ? `/namespaces/${namespace}` : ''}/limitranges/${name}`,
    query: { pretty },
  });
  return GET(url);
}
export function deleteLimitRange(namespace, name) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().podUrl}${namespace ? `/namespaces/${namespace}` : ''}/limitranges/${name}`,
    query: { pretty },
  });
  return DELETE(url);
}

export function updateLimitRangeYamlData(namespace, name, data) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().podUrl}${namespace ? `/namespaces/${namespace}` : ''}/limitranges/${name}`,
  });
  return PUT(url, data, { 'Content-Type': 'application/json' });
}

// 资源配额
export function getResourceQuotaList(namespace) {
  const url = qs.stringifyUrl(
    {
      url: `${commonApiPrefix().podUrl}${namespace ? `/namespaces/${namespace}` : ''}/resourcequotas`,
      query: solveParams({
        pretty,
      }),
    },
    { skipNull: true }
  );
  return GET(url);
}
export function addResourceQuotaYamlData(namespace, data) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().podUrl}${namespace ? `/namespaces/${namespace}` : ''}/resourcequotas`,
  });
  return POST(url, data, { 'Content-Type': 'application/json' });
}
export function getResourceQuotaDetailDescription(namespace, name) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().podUrl}${namespace ? `/namespaces/${namespace}` : ''}/resourcequotas/${name}`,
    query: { pretty },
  });
  return GET(url);
}

export function deleteResourceQuota(namespace, name) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().podUrl}${namespace ? `/namespaces/${namespace}` : ''}/resourcequotas/${name}`,
    query: { pretty },
  });
  return DELETE(url);
}

export function updateResourceQuotaYamlData(namespace, name, data) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().podUrl}${namespace ? `/namespaces/${namespace}` : ''}/resourcequotas/${name}`,
  });
  return PUT(url, data, { 'Content-Type': 'application/json' });
}
// 自定义资源
export function getCustomResourceDefinitionList() {
  const url = qs.stringifyUrl(
    {
      url: `${commonApiPrefix().crdUrl}/customresourcedefinitions`,
      query: solveParams({
        pretty,
      }),
    },
    { skipNull: true }
  );
  return GET(url);
}

export function addCustomResourceDefinitionYamlData(data) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().crdUrl}/customresourcedefinitions`,
  });
  return POST(url, data, { 'Content-Type': 'application/json' });
}

export function deleteCustomResourceDefinition(customResourceName) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().crdUrl}/customresourcedefinitions/${customResourceName}`,
  });
  return DELETE(url);
}

export function updateCustomResourceDefinitionYamlData(
  customResourceName,
  data
) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().crdUrl}/customresourcedefinitions/${customResourceName}`,
  });
  return PUT(url, data, { 'Content-Type': 'application/json' });
}

export function getCustomResourceDefinitionDetailDescription(
  customResourceName
) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().crdUrl}/customresourcedefinitions/${customResourceName}`,
  });
  return GET(url);
}

// 获取自定义资源实例列表
export function getResourceDefineationList(group, version) {
  const url = `${commonApiPrefix().crUrl}/${group}/${version}/${plural}`;
  return GET(url);
}
// cr实例列表
export function getResourceExampleList(
  { group, version, plural },
) {
  const url = qs.stringifyUrl(
    {
      url: `${commonApiPrefix().crUrl}/${group}/${version}/${plural}`,
      query: solveParams({
        pretty,
      }),
    },
    { skipNull: true }
  );
  return GET(url);
}

// 删除
export function deleteResourceExample(
  { group, version, namespace, plural },
  name
) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().crUrl}/${group}/${version}${namespace ? `/namespaces/${namespace}` : ''}/${plural}/${name}`,
  });
  return DELETE(url);
}

// 创建
export function addResourceExampleYamlData(
  { group, version, plural, namespace },
  data
) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().crUrl}/${group}/${version}${namespace ? `/namespaces/${namespace}` : ''}/${plural}`,
  });
  return POST(url, data, { 'Content-Type': 'application/json' });
}

// 更新
export function updateResourceExampleYamlData(
  { group, version, plural, namespace },
  name,
  data
) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().crUrl}/${group}/${version}${namespace ? `/namespaces/${namespace}` : ''}/${plural}/${name}`,
  });
  return PUT(url, data, { 'Content-Type': 'application/json' });
}

export function getResourceExampleDetailDescription(
  { group, version, plural, namespace },
  name
) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().crUrl}/${group}/${version}${namespace ? `/namespaces/${namespace}` : ''}/${plural}/${name}`,
  });
  return GET(url);
}

// 注解更新
export function editResourceExampleAnnotationOrlabels(
  { group, version, plural, namespace },
  name,
  data
) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().crUrl}/${group}/${version}${namespace ? `/namespaces/${namespace}` : ''}/${plural}/${name}`,
  });
  return PATCH(url, data, { 'Content-Type': 'application/json-patch+json' });
}

/**
 * 获取网络-Service列表数据
 */
export function getServicesData(namespace) {
  const url = qs.stringifyUrl(
    {
      url: `${commonApiPrefix().serviceUrl}${namespace ? `/namespaces/${namespace}` : ''}/services`,
      query: solveParams({
        pretty,
      }),
    },
    { skipNull: true }
  );
  return GET(url);
}
/**
 * 创建service yaml
 */
export function addServiceYamlData(namespace, data) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().serviceUrl}${namespace ? `/namespaces/${namespace}` : ''}/services`,
  });
  return POST(url, data);
}
/**
 * 获取网络-Service-详情数据
 */
export function getServiceDetailDescription(namespace, serviceName) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().serviceUrl}${namespace ? `/namespaces/${namespace}` : ''}/services/${serviceName}`,
    query: { pretty },
  });
  return GET(url);
}
/**
 * 修改service yaml
 */
export function updateServiceYamlData(namespace, serviceName, data) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().serviceUrl}${namespace ? `/namespaces/${namespace}` : ''}/services/${serviceName}`,
  });
  return PUT(url, data);
}
/**
 * 删除某个service
 */
export function deleteService(namespace, serviceName) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().serviceUrl}${namespace ? `/namespaces/${namespace}` : ''}/services/${serviceName}`,
  });
  return DELETE(url);
}
/**
 * service增加标签与注解
 */
export function editServiceLabelOrAnnotation(namespace, serviceName, data) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().serviceUrl}${namespace ? `/namespaces/${namespace}` : ''}/services/${serviceName}`,
  });
  return PATCH(url, data, { 'Content-Type': 'application/json-patch+json' });
}
/**
 * 获取网络-Ingress列表数据
 */
export function getIngressData(namespace) {
  const url = qs.stringifyUrl(
    {
      url: `${commonApiPrefix().ingressUrl}${namespace ? `/namespaces/${namespace}` : ''}/ingresses`,
      query: solveParams({
        pretty,
      }),
    },
    { skipNull: true }
  );
  return GET(url);
}
/**
 * 获取网络-Ingress-详情数据
 */
export function getIngressDetailDescription(namespace, ingressName) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().ingressUrl}${namespace ? `/namespaces/${namespace}` : ''}/ingresses/${ingressName}`,
    query: { pretty },
  });
  return GET(url);
}
/**
 * 创建Ingress yaml
 */
export function addIngressYamlData(namespace, data) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().ingressUrl}${namespace ? `/namespaces/${namespace}` : ''}/ingresses`,
  });
  return POST(url, data);
}
/**
 * 修改Ingress yaml
 */
export function updateIngressYamlData(namespace, ingressName, data) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().ingressUrl}${namespace ? `/namespaces/${namespace}` : ''}/ingresses/${ingressName}`,
  });
  return PUT(url, data);
}
/**
 * 删除某个Ingress
 */
export function deleteIngress(namespace, ingressName) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().ingressUrl}${namespace ? `/namespaces/${namespace}` : ''}/ingresses/${ingressName}`,
  });
  return DELETE(url);
}
/**
 * Ingress增加标签与注解
 */
export function editIngressLabelOrAnnotation(namespace, ingressName, data) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().ingressUrl}${namespace ? `/namespaces/${namespace}` : ''}/ingresses/${ingressName}`,
  });
  return PATCH(url, data, { 'Content-Type': 'application/json-patch+json' });
}
/**
 * 获取应用/扩展组件管理-Helm-列表数据
 */
export function getHelmsData(extension) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().helmUrl}`,
    query: solveParams({
      extension,
    }),
  });
  return GET(url);
}
/**
 * 获取应用管理-Helm-详情数据
 */
export function getHelmDetailDescriptionData(namespace, name) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().helmUrl}/namespace/${namespace}/release/${name}`,
  });
  return GET(url);
}
/**
 * 删除指定应用
 */
export function deleteRelease(namespace, releaseName) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().helmUrl}${namespace ? `/namespace/${namespace}` : ''}/release/${releaseName}`,
  });
  return DELETE(url);
}
/**
 * 获取应用管理-Helm-回退获取历史版本
 */
export function getHelmHistoryVersionData(namespace, releaseName) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().helmUrl}/namespace/${namespace}/release/${releaseName}/history `,
  });
  return GET(url);
}
/**
 * 获取应用管理-Helm-回退提交
 */
export function rollBackHelmVersion(namespace, releaseName, version) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().helmUrl}/namespace/${namespace}/release/${releaseName}/rollback?version=${version} `,
  });
  return POST(url);
}
/**
 * 获取应用管理-Helm-详情-Yaml-manifest数据
 */
export function getHelmDetailYaml(namespace, releaseName) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().helmUrl}/namespace/${namespace}/release/${releaseName}/getmanifest `,
  });
  return GET(url);
}
/**
 * 获取应用管理-Helm-详情-Yaml-value数据
 */
export function getHelmDetailYamlOnlyValue(namespace, releaseName) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().helmUrl}/namespace/${namespace}/release/${releaseName}/getvalues `,
  });
  return GET(url);
}
/**
 * 获取应用管理-Helm-升级-yaml数据
 */
export function getHelmUpLevelYamlData(repo, chart, version) {
  // return GET()
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().helmUpdate}/${repo}/charts/${chart}/versions/${version}/files`,
  });
  return GET(url);
}

/**
 * 获取应用管理-Helm-升级-获取指定helm的所有版本信息
 */
export function getHelmTemplateDetailVersion(repo, chart) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().helmUpdate}/${repo}/charts/${chart}`,
  });
  return GET(url);
}
/**
 * 获取应用管理-helm-升级提交
 */
export function updateHelmLevelYaml(namespace, releaseName, data) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().helmUrl}/namespace/${namespace}/release/${releaseName}`,
  });
  return PUT(url, data);
}
/**
 * 获取应用管理-Helm详情-日志的pod与container
 */
export function getHelmDetailLogPodAndContainer(namespace, releaseName) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().helmUrl}/namespace/${namespace}/release/${releaseName}/pods`,
  });
  return GET(url);
}
/**
 * 获取应用管理-Helm详情-日志数据
 */
export function getHelmDetailLogData(namespace, pod, container) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().helmLogUrl}/namespaces/${namespace}${pod ? `/pods/${pod}` : ''}/log`,
    query: { container },
  });
  return GET(url);
}
/**
 * 获取helm事件列表
 */
export function getHelmEventsData(
  {
    page,
    pageSize,
    continueToken,
  },
  namespace,
  conditions,
) {
  let filter = [];
  if (conditions) {
    if (conditions.resourceName) {
      filter.push(`regarding.name=${conditions.resourceName}`);
    }
    if (conditions.resourceType) {
      filter.push(`regarding.kind=${conditions.resourceType}`);
    }
    if (conditions.eventLevel) {
      filter.push(`type=${conditions.eventLevel}`);
    }
  }
  const url = qs.stringifyUrl(
    {
      url: `${commonApiPrefix().helmEventUrl}/${namespace ? `namespaces/${namespace}/` : ''}events`, query: filter.length ? solveParams({
        fieldSelector: filter.join(','),
        page,
        limit: pageSize,
        continue: continueToken,
      }) : solveParams({
        page,
        limit: pageSize,
        continue: continueToken,
      }),
    }, { skipNull: true });
  return GET(url);
}
/**
 * 获取扩展组件管理-展示可扩展界面列表
 */
export function getExtendPageList(pluginName) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().consolePluginUrl}/${pluginName}`,
  });
  return GET(url);
}
/**
 * 扩展组件管理-修改可扩展界面列表的状态
 */
export function updateExtendPageItemState(pluginName, data) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().consolePluginUrl}/${pluginName}/enabled`,
  });
  return POST(url, data);
}

export function getConfigMapsList(namespace) {
  const url = qs.stringifyUrl(
    {
      url: `${commonApiPrefix().dictionaryUrl}${namespace ? `/namespaces/${namespace}` : ''}/configmaps`,
      query: solveParams({
        pretty,
      }),
    },
    { skipNull: true }
  );
  return GET(url);
}

// 获取配置字典具体内容
export function getConfigMapsDetails(namespaces, configMapName) {
  const url = `${commonApiPrefix().dictionaryUrl}/namespaces/${namespaces ? namespaces : ''}/configmaps/${configMapName}`;
  return GET(url);
}

// 创建一个配置字典
export function createConfigMaps(namespaces, data) {
  const url = `${commonApiPrefix().dictionaryUrl}/namespaces/${namespaces}/configmaps`;
  return POST(url, data, { 'Content-Type': 'application/json' });
}
// 删除一个配置字典
export function deleteConfigMaps(namespace, name) {
  const url = `${commonApiPrefix().dictionaryUrl}/namespaces/${namespace}/configmaps/${name}`;
  return DELETE(url);
}

// 更新指定的配置字典
export function updateConfigMaps(namespace, data, name) {
  const url = `${commonApiPrefix().dictionaryUrl}/namespaces/${namespace}/configmaps/${name}`;
  return PUT(url, data, { 'Content-Type': 'application/json' });
}

// 部分更新指定的配置字典
export function updateConfigMapsLabelAnnotation(namespace, data, name) {
  const url = `${commonApiPrefix().dictionaryUrl}/namespaces/${namespace}/configmaps/${name}`;
  return PATCH(url, data, { 'Content-Type': 'application/json-patch+json' });
}

export function getSecretList(namespace) {
  const url = qs.stringifyUrl(
    {
      url: `${commonApiPrefix().dictionaryUrl}${namespace ? `/namespaces/${namespace}` : ''}/secrets`,
      query: solveParams({
        pretty,
      }),
    },
    { skipNull: true }
  );
  return GET(url);
}
// 获取保密字典具体内容
export function getSecretDetails(namespaces, SecretName) {
  const url = `${commonApiPrefix().dictionaryUrl}/namespaces/${namespaces ? namespaces : ''}/secrets/${SecretName}`;
  return GET(url);
}

// 创建一个保密字典
export function createSecret(namespaces, data) {
  const url = `${commonApiPrefix().dictionaryUrl}/namespaces/${namespaces}/secrets`;
  return POST(url, data, { 'Content-Type': 'application/json' });
}
// 删除一个配置保密字典
export function deleteSecret(namespace, name) {
  const url = `${commonApiPrefix().dictionaryUrl}/namespaces/${namespace}/secrets/${name}`;
  return DELETE(url);
}

// 更新指定的保密字典
export function updateSecrets(namespace, data, name) {
  const url = `${commonApiPrefix().dictionaryUrl}/namespaces/${namespace}/secrets/${name}`;
  return PUT(url, data, { 'Content-Type': 'application/json' });
}

// 部分更新指定的保密字典
export function updateSecretLabelAnnotation(namespace, data, name) {
  const url = `${commonApiPrefix().dictionaryUrl}/namespaces/${namespace}/secrets/${name}`;
  return PATCH(url, data, { 'Content-Type': 'application/json-patch+json' });
}

/**
 * 获取角色列表
 */
export function getAuthorList(namespace) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().authorUrl}${namespace ? `/namespaces/${namespace}` : ''}/roles`,
  });
  return GET(url);
}

/**
 * 获取概览pod列表
 */
export function getPodList(namespace, podPage, podPageSize) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().podUrl}${namespace ? `/namespaces/${namespace}` : ''}/pods`,
    query: solveParams({
      page: podPage,
      limit: podPageSize,
    }),
  });
  return GET(url);
}

/**
 * 获取概览deployment列表
 */
export function getDeploymentList(namespace, deploymentPage, pageSize) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().workloadUrl}${namespace ? `/namespaces/${namespace}` : ''}/deployments`,
    query: solveParams({
      page: deploymentPage,
      limit: pageSize,
    }),
  });
  return GET(url);
}

/**
 * 获取概览statefulset列表
 */
export function getStatefulSetList(namespace, statefulsetPage, pageSize) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().workloadUrl}${namespace ? `/namespaces/${namespace}` : ''}/statefulsets`,
    query: solveParams({
      page: statefulsetPage,
      limit: pageSize,
    }),
  });
  return GET(url);
}

/**
 * 获取概览daemonset列表
 */
export function getDaemonSetList(namespace, daemonsetPage, pageSize) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().workloadUrl}${namespace ? `/namespaces/${namespace}` : ''}/daemonsets`,
    query: solveParams({
      page: daemonsetPage,
      limit: pageSize,
    }),
  });
  return GET(url);
}


// 获取平台管理-命名空间-CPU&内存使用量
export function getCPUAndMemUsageFromAllPods() {
  let metricsFilter = 'pod_memory_usage|pod_cpu_usage';
  const usageUrl = qs.stringifyUrl({
    url: `${commonApiPrefix().overviewUrl}/namespaces/.*/pods/.*/metrics`,
    query: solveParams({
      metrics_filter: metricsFilter,
    }),
  }, {
    skipNull: true,
  }
  );
  return GET(usageUrl);
};

// 获取概览-节点-top5资源列表
export function getIndexNodesListData(type) {
  let metricsFilter = 'node_network_bytes_received_by_device';
  switch (type) {
    case 'inflow': {
      metricsFilter = 'node_network_bytes_received_by_device';
      break;
    }
    case 'flowOut': {
      metricsFilter = 'node_network_bytes_transmitted_by_device';
      break;
    }
    default: {
      metricsFilter = 'node_network_bytes_received_by_device';
    }
  }
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().overviewUrl}/nodes/.*/metrics`,
    query: solveParams({
      metrics_filter: metricsFilter,
    }),
  }, {
    skipNull: true,
  }
  );
  return GET(url);
};

// 获取概览-pod-top5资源列表
export function getIndexPodsListData(type) {
  let metricsFilter = 'pod_network_bytes_received';
  switch (type) {
    case 'inflow': {
      metricsFilter = 'pod_network_bytes_received';
      break;
    }
    case 'flowOut': {
      metricsFilter = 'pod_network_bytes_transmitted';
      break;
    }
    default: {
      metricsFilter = 'pod_network_bytes_received';
    }
  }
  const podUrl = qs.stringifyUrl({
    url: `${commonApiPrefix().overviewUrl}/namespaces/.*/pods/.*/metrics`,
    query: solveParams({
      metrics_filter: metricsFilter,
    }),
  }, {
    skipNull: true,
  }
  );
  return GET(podUrl);
};

/**
 * 服务账号
 */
export function getServiceAccountsData(namespace) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().podUrl}${namespace ? `/namespaces/${namespace}` : ''}/serviceaccounts`,
    query: solveParams({
      pretty,
    }),
  }, {
    skipNull: true,
  });
  return GET(url);
}

/**
 * 获取serviceAccount详情
 */
export function getServiceAccountDetailDescription(namespace, serviceAccountName) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().podUrl}${namespace ? `/namespaces/${namespace}` : ''}/serviceaccounts/${serviceAccountName}`,
    query: { pretty },
  });
  return GET(url);
}

/**
 * 删除
 */
export function deleteServiceAccount(namespace, serviceAccountName) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().podUrl}${namespace ? `/namespaces/${namespace}` : ''}/serviceaccounts/${serviceAccountName}`,
    query: { pretty },
  });
  return DELETE(url);
}

/**
 * 修改yaml
 */
export function updateServiceAccountYamlData(namespace, name, data) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().podUrl}${namespace ? `/namespaces/${namespace}` : ''}/serviceaccounts/${name}`,
  });
  return PUT(url, data, { 'Content-Type': 'application/json' });
}

/**
 * 角色
 */
export function getRolesData(
  namespace,
  roleName,
  page,
  pageSize
) {
  let fieldSelectorRole = [];
  if (roleName) {
    fieldSelectorRole.push(`metadata.name=${roleName}`);
  }
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().roleUrl}${namespace ? `/namespaces/${namespace}` : ''}/roles`,
    query: solveParams({
      page,
      pageSize,
      pretty,
      fieldSelector: fieldSelectorRole.join(','),
    }),
  }, {
    skipNull: true,
  });
  return GET(url);
}

/**
 * 获取role详情
 */
export function getRoleDetailDescription(namespace, roleName) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().roleUrl}${namespace ? `/namespaces/${namespace}` : ''}/roles/${roleName}`,
    query: { pretty },
  });
  return GET(url);
}

/**
 * 删除
 */
export function deleteRole(namespace, roleName) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().roleUrl}${namespace ? `/namespaces/${namespace}` : ''}/roles/${roleName}`,
    query: { pretty },
  });
  return DELETE(url);
}

/**
 * 修改yaml
 */
export function updateRoleYamlData(namespace, name, data) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().roleUrl}${namespace ? `/namespaces/${namespace}` : ''}/roles/${name}`,
  });
  return PUT(url, data, { 'Content-Type': 'application/json' });
}

/**
 * 角色绑定
 */
export function getRoleBindsData(
  namespace,
  roleName,
  page,
  pageSize
) {
  let fieldSelectorRoleBind = [];
  if (roleName) {
    fieldSelectorRoleBind.push(`metadata.name=${roleName}`);
  }
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().roleUrl}${namespace ? `/namespaces/${namespace}` : ''}/rolebindings`,
    query: solveParams({
      page,
      pageSize,
      pretty,
      fieldSelector: fieldSelectorRoleBind.join(','),
    }),
  }, {
    skipNull: true,
  });
  return GET(url);
}

/**
 * 获取roleBind详情
 */
export function getRoleBindDetailDescription(namespace, roleBindName) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().roleUrl}${namespace ? `/namespaces/${namespace}` : ''}/rolebindings/${roleBindName}`,
    query: { pretty },
  });
  return GET(url);
}

/**
 * 删除
 */
export function deleteRoleBind(namespace, roleBindName) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().roleUrl}${namespace ? `/namespaces/${namespace}` : ''}/rolebindings/${roleBindName}`,
    query: { pretty },
  });
  return DELETE(url);
}

/**
 * 修改yaml
 */
export function updateRoleBindYamlData(namespace, name, data) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().roleUrl}${namespace ? `/namespaces/${namespace}` : ''}/rolebindings/${name}`,
  });
  return PUT(url, data, { 'Content-Type': 'application/json' });
}

/**
 * 集群角色 clusterRole
 */
export function getClusterRolesData(
  namespace,
  roleName,
  page,
  pageSize
) {
  let fieldSelectorCluster = [];
  if (roleName) {
    fieldSelectorCluster.push(`metadata.name=${roleName}`);
  }
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().roleUrl}/clusterroles`,
    query: solveParams({
      page,
      pageSize,
      pretty,
      fieldSelector: fieldSelectorCluster.join(','),
    }),
  }, {
    skipNull: true,
  });
  return GET(url);
}

/**
 * 集群角色 - 创建
 */
export function addClusterRoleYamlData(data) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().roleUrl}/clusterroles`,
  });
  return POST(url, data, { 'Content-Type': 'application/json' });
}

/**
 * 集群角色 - 修改yaml
 */
export function updateClusterRoleYamlData(name, data) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().roleUrl}/clusterroles/${name}`,
  });
  return PUT(url, data, { 'Content-Type': 'application/json' });
}

/**
 * 集群角色 - 删除
 */
export function deleteClusterRole(roleName) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().roleUrl}/clusterroles/${roleName}`,
    query: { pretty },
  });
  return DELETE(url);
}

/**
 * 获取集群角色详情
 */
export function getClusterRoleDetailDescription(roleName) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().roleUrl}/clusterroles/${roleName}`,
    query: { pretty },
  });
  return GET(url);
}

/**
 * 集群角色绑定 clusterRoleBinding
 */
export function getClusterRoleBindingsData(
  namespace,
  roleName,
  page,
  pageSize
) {
  let fieldSelectorClusterBind = [];
  if (roleName) {
    fieldSelectorClusterBind.push(`metadata.name=${roleName}`);
  }
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().roleUrl}/clusterrolebindings`,
    query: solveParams({
      page,
      pageSize,
      pretty,
      fieldSelector: fieldSelectorClusterBind.join(','),
    }),
  }, {
    skipNull: true,
  });
  return GET(url);
}

/**
 * 集群角色绑定 - 创建
 */
export function addClusterRoleBindingYamlData(data) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().roleUrl}/clusterrolebindings`,
  });
  return POST(url, data, { 'Content-Type': 'application/json' });
}

/**
 * 集群角色绑定 - 删除
 */
export function deleteClusterRoleBinding(roleBindName) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().roleUrl}/clusterrolebindings/${roleBindName}`,
    query: { pretty },
  });
  return DELETE(url);
}

/**
 * 集群角色绑定 - 修改yaml
 */
export function updateClusterRoleBindingYamlData(name, data) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().roleUrl}/clusterrolebindings/${name}`,
  });
  return PUT(url, data, { 'Content-Type': 'application/json' });
}

/**
 * 获取集群角色绑定 - 详情
 */
export function getClusterRoleBindingDetailDescription(roleBindName) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().roleUrl}/clusterrolebindings/${roleBindName}`,
    query: { pretty },
  });
  return GET(url);
}

/*
 * 获取概览-用户列表
 */
export function getOverviewUserList(namespace) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().helmLogUrl}${namespace ? `/namespaces/${namespace}` : ''}/serviceaccounts`,
  });
  return GET(url);
}

/**
 * 获取服务端时区
 */
export function getServiceTimeZoneInterface() {
  return GET('/rest/console/v1beta1/time-offset');
}

/**
 * 获取资源管理存储-Pvc列表数据
 */
export function getPvcData(namespace) {
  const url = qs.stringifyUrl(
    {
      url: `${commonApiPrefix().resourcePvUrl}${namespace ? `/namespaces/${namespace}` : ''}/persistentvolumeclaims`,
      query: solveParams({
        pretty,
      }),
    },
    { skipNull: true }
  );
  return GET(url);
}
/**
 * 获取资源管理存储-Pvc-详情数据
 */
export function getPvcDetailDescription(namespace, pvcName) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().resourcePvUrl}${namespace ? `/namespaces/${namespace}` : ''}/persistentvolumeclaims/${pvcName}`,
    query: { pretty },
  });
  return GET(url);
}
/**
 * 创建Pvc yaml
 */
export function addPvcYamlData(namespace, data) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().resourcePvUrl}${namespace ? `/namespaces/${namespace}` : ''}/persistentvolumeclaims`,
  });
  return POST(url, data);
}
/**
 * 修改Pvc yaml
 */
export function updatePvcYamlData(namespace, pvcName, data) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().resourcePvUrl}${namespace ? `/namespaces/${namespace}` : ''}/persistentvolumeclaims/${pvcName}`,
  });
  return PUT(url, data);
}
/**
 * 删除某个Pvc
 */
export function deletePvc(namespace, pvcName) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().resourcePvUrl}${namespace ? `/namespaces/${namespace}` : ''}/persistentvolumeclaims/${pvcName}`,
  });
  return DELETE(url);
}
/**
 * Pvc增加标签与注解
 */
export function editPvcLabelOrAnnotation(namespace, pvcName, data) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().resourcePvUrl}${namespace ? `/namespaces/${namespace}` : ''}/persistentvolumeclaims/${pvcName}`,
  });
  return PATCH(url, data, { 'Content-Type': 'application/json-patch+json' });
}

/**
 * 获取资源管理存储-Pv列表数据
 */
export function getPvData() {
  const url = qs.stringifyUrl(
    {
      url: `${commonApiPrefix().resourcePvUrl}/persistentvolumes`,
      query: solveParams({
        pretty,
      }),
    },
    { skipNull: true }
  );
  return GET(url);
}
/**
 * 获取资源管理存储-Pv-详情数据
 */
export function getPvDetailDescription(pvName) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().resourcePvUrl}/persistentvolumes/${pvName}`,
    query: { pretty },
  });
  return GET(url);
}
/**
 * 创建Pv yaml
 */
export function addPvYamlData(data) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().resourcePvUrl}/persistentvolumes`,
  });
  return POST(url, data);
}
/**
 * 修改Pv yaml
 */
export function updatePvYamlData(pvName, data) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().resourcePvUrl}/persistentvolumes/${pvName}`,
  });
  return PUT(url, data);
}
/**
 * 删除某个Pv
 */
export function deletePv(pvName) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().resourcePvUrl}/persistentvolumes/${pvName}`,
  });
  return DELETE(url);
}
/**
 * Pv增加标签与注解
 */
export function editPvLabelOrAnnotation(pvName, data) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().resourcePvUrl}/persistentvolumes/${pvName}`,
  });
  return PATCH(url, data, { 'Content-Type': 'application/json-patch+json' });
}

/**
 * 获取资源管理存储-Sc列表数据
 */
export function getScData() {
  const url = qs.stringifyUrl(
    {
      url: `${commonApiPrefix().resourceStorageUrl}/storageclasses`,
      query: solveParams({
        pretty,
      }),
    },
    { skipNull: true }
  );
  return GET(url);
}
/**
 * 获取资源管理存储-Sc-详情数据
 */
export function getScDetailDescription(scName) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().resourceStorageUrl}/storageclasses/${scName}`,
    query: { pretty },
  });
  return GET(url);
}
/**
 * 创建Sc yaml
 */
export function addScYamlData(data) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().resourceStorageUrl}/storageclasses`,
  });
  return POST(url, data);
}
/**
 * 修改Sc yaml
 */
export function updateScYamlData(scName, data) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().resourceStorageUrl}/storageclasses/${scName}`,
  });
  return PUT(url, data);
}
/**
 * 删除某个Sc
 */
export function deleteSc(scName) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().resourceStorageUrl}/storageclasses/${scName}`,
  });
  return DELETE(url);
}
/**
 * Sc增加标签与注解
 */
export function editScLabelOrAnnotation(scName, data) {
  const url = qs.stringifyUrl({
    url: `${commonApiPrefix().resourceStorageUrl}/storageclasses/${scName}`,
  });
  return PATCH(url, data, { 'Content-Type': 'application/json-patch+json' });
}
/**
 * 创建一个优先级队列
 */
export function createSpecifiedPriorityClasses(data) {
  const url = `${commonApiPrefix().priorityClassesUrl}/priorityclasses`;
  return POST(url, data, { 'Content-Type': 'application/json' });
}