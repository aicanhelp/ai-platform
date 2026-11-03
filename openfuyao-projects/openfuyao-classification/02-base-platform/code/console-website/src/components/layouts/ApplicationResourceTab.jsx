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

import { useEffect, useState, useCallback, useRef } from 'openinula';
import { useTranslation } from 'react-i18next';
import { Space, Table, Tooltip, Form, Input } from 'antd';
import { LoadingOutlined, ExclamationCircleFilled, QuestionCircleFilled } from '@ant-design/icons';
import { filterRepeat } from '@/utils/common';
import { ReleaseStatus } from '@/common/constants';
import { useHistory } from 'inula-router';
import { containerRouterPrefix } from '@/constant.js';
import { solveEncodePath } from '@/tools/utils';
import '@/styles/pages/helm.less';

const { Search } = Input;

/** Helm resource status.Status (API) → application.json resource.releaseStatus.* */
const RESOURCE_RELEASE_STATUS_KEYS = {
  Current: 'current',
  Failed: 'failed',
  InProgress: 'inProgress',
  Terminating: 'terminating',
};

/**
 * Application Resource Tab Component
 *
 * @param {Object} props
 * @param {Object} props.dataProps - Resource data
 */
export default function ApplicationResourceTab({ dataProps }) {
  const history = useHistory();
  const { t } = useTranslation('application');

  const [helmResourceList, setHelmResourceList] = useState([]);
  const [helmSearchForm] = Form.useForm();
  const helmFormRef = useRef(null);
  const helmBackListRef = useRef([]);
  const [helmDetailFilterObj, setHelmDetailFilterObj] = useState({});
  const [helmDetailTotal, setHelmDetailTotal] = useState(0);
  const [helmDetailCurrent, setHelmDetailCurrent] = useState(1);
  const [helmDetailPageSize, setHelmDetailPageSize] = useState(10);
  const [kindArr, setKindArr] = useState([]);
  const [currentKind, setCurrentKind] = useState('');
  const [statusArr, setStatusArr] = useState([]);
  const [currentStatus, setCurrentStatus] = useState('');
  const [helmSortObj, setHelmSortObj] = useState({});
  const [helmStatusSortObj, setHelmStatusSortObj] = useState({});
  const [helmNamespaceSortObj, setHelmNamespaceSortObj] = useState({});
  const [helmTypeSortObj, setHelmTypeSortObj] = useState({});

  const linkArr = [
    'Pod', 'Deployment', 'StatefulSet', 'DaemonSet', 'Job', 'CronJob', 'Service', 'Ingress', 'ConfigMap',
    'Secret', 'ServiceAccount', 'Role', 'RoleBinding', 'Namespace', 'ClusterRole', 'ClusterRoleBinding',
    'PersistentVolume', 'PersistentVolumeClaim', 'StorageClass', 'ServiceMonitor', 'CustomResourceDefinition',
    'ResourceQuota', 'LimitRange',
  ];

  const getHelmResource = () => {
    if (helmFormRef.current) {
      let filtertArr;
      if (helmSearchForm.getFieldsValue().name) {
        filtertArr = helmBackListRef.current.filter(item => (item.name).includes(helmSearchForm.getFieldsValue().name));
      } else {
        filtertArr = JSON.parse(JSON.stringify(helmBackListRef.current));
      }
      if (Object.keys(helmDetailFilterObj).length !== 0) {
        if (!helmDetailFilterObj.helmDetailStatus) {
          setCurrentStatus('');
        } else {
          let [temporyStatus] = helmDetailFilterObj.helmDetailStatus;
          setCurrentStatus(temporyStatus);
          filtertArr = filtertArr.filter(item => item.status?.Status === temporyStatus);
        }
        if (!helmDetailFilterObj.kind) {
          setCurrentKind('');
        } else {
          let [kind] = helmDetailFilterObj.kind;
          setCurrentKind(kind);
          filtertArr = filtertArr.filter(item => item.kind === kind);
        }
      }
      // Sorting logic
      if (helmSortObj.order === 'ascend') {
        filtertArr.sort((a, b) => a.name.localeCompare(b.name));
      } else if (helmSortObj.order === 'descend') {
        filtertArr.sort((a, b) => b.name.localeCompare(a.name));
      }
      if (helmStatusSortObj.order === 'ascend') {
        filtertArr.sort((a, b) => {
          const ka = a.status?.Status;
          const kb = b.status?.Status;
          const la = ka && RESOURCE_RELEASE_STATUS_KEYS[ka]
            ? t(`resource.releaseStatus.${RESOURCE_RELEASE_STATUS_KEYS[ka]}`)
            : t('resource.releaseStatus.unknown');
          const lb = kb && RESOURCE_RELEASE_STATUS_KEYS[kb]
            ? t(`resource.releaseStatus.${RESOURCE_RELEASE_STATUS_KEYS[kb]}`)
            : t('resource.releaseStatus.unknown');
          return la.localeCompare(lb);
        });
      } else if (helmStatusSortObj.order === 'descend') {
        filtertArr.sort((a, b) => {
          const ka = a.status?.Status;
          const kb = b.status?.Status;
          const la = ka && RESOURCE_RELEASE_STATUS_KEYS[ka]
            ? t(`resource.releaseStatus.${RESOURCE_RELEASE_STATUS_KEYS[ka]}`)
            : t('resource.releaseStatus.unknown');
          const lb = kb && RESOURCE_RELEASE_STATUS_KEYS[kb]
            ? t(`resource.releaseStatus.${RESOURCE_RELEASE_STATUS_KEYS[kb]}`)
            : t('resource.releaseStatus.unknown');
          return lb.localeCompare(la);
        });
      }
      if (helmNamespaceSortObj.order === 'ascend') {
        filtertArr.sort((a, b) => (a.namespace || '').localeCompare(b.namespace || ''));
      } else if (helmNamespaceSortObj.order === 'descend') {
        filtertArr.sort((a, b) => (b.namespace || '').localeCompare(a.namespace || ''));
      }
      if (helmTypeSortObj.order === 'ascend') {
        filtertArr.sort((a, b) => (a.kind || '').localeCompare(b.kind || ''));
      } else if (helmTypeSortObj.order === 'descend') {
        filtertArr.sort((a, b) => (b.kind || '').localeCompare(a.kind || ''));
      }
      setHelmDetailTotal(filtertArr.length);
      setHelmResourceList([...filtertArr]);
    }
  };

  const handleHelmDetailTableChange = (pagination, filter, _sorter, extra) => {
    if (extra.action === 'paginate') {
      setHelmDetailCurrent(pagination.current || 1);
      setHelmDetailPageSize(pagination.pageSize || 10);
    }
    if (extra.action === 'filter') {
      setHelmDetailFilterObj(filter);
    }
    if (extra.action === 'sort') {
      setHelmSortObj({});
      setHelmNamespaceSortObj({});
      setHelmTypeSortObj({});
      setHelmStatusSortObj({});
      if (_sorter.columnKey === 'helmDetailStatus') {
        setHelmStatusSortObj({ key: _sorter.columnKey, order: _sorter.order });
      } else if (_sorter.columnKey === 'namespace') {
        setHelmNamespaceSortObj({ key: _sorter.columnKey, order: _sorter.order });
      } else if (_sorter.columnKey === 'kind') {
        setHelmTypeSortObj({ key: _sorter.columnKey, order: _sorter.order });
      } else {
        setHelmSortObj({ key: _sorter.columnKey, order: _sorter.order });
      }
    }
  };

  const setFilterStatusData = useCallback(() => {
    const filterStatusArr = Object.keys(ReleaseStatus).map((element) => ({
      text: t(`resource.releaseStatus.${RESOURCE_RELEASE_STATUS_KEYS[element]}`),
      value: element,
    }));
    setStatusArr([...filterStatusArr]);
  }, [t]);

  const helmDetailColumns = [
    {
      title: t('resource.name'),
      dataIndex: 'name',
      sorter: true,
      sortOrder: helmSortObj.order,
      render: (_, record) => (
        <Space>
          {isShowLink(record.kind) ? (
            <p className='resource_name' onClick={() => jumpToDetail(record.kind, record.namespace, record.name)}>
              {record.name}
            </p>
          ) : (
            <p>{record.name}</p>
          )}
        </Space>
      ),
    },
    {
      title: t('resource.namespace'),
      key: 'namespace',
      sorter: true,
      sortOrder: helmNamespaceSortObj.order,
      render: (_, record) => record.namespace || '-',
    },
    {
      title: t('resource.type'),
      dataIndex: 'kind',
      key: 'kind',
      filters: kindArr,
      filteredValue: currentKind ? [currentKind] : null,
      filterMultiple: false,
      sorter: true,
      sortOrder: helmTypeSortObj.order,
    },
    {
      title: t('resource.status'),
      key: 'helmDetailStatus',
      filters: statusArr,
      filteredValue: currentStatus ? [currentStatus] : null,
      filterMultiple: false,
      sorter: true,
      sortOrder: helmStatusSortObj.order,
      render: (_, record) => {
        const raw = record.status?.Status;
        const label = raw && RESOURCE_RELEASE_STATUS_KEYS[raw]
          ? t(`resource.releaseStatus.${RESOURCE_RELEASE_STATUS_KEYS[raw]}`)
          : t('resource.releaseStatus.unknown');
        const tooltipTitle = raw === 'Failed' && record.status?.Message ? record.status.Message : '';
        const known = raw && ['Current', 'Failed', 'InProgress', 'Terminating'].includes(raw);
        return (
          <Space>
            <Tooltip title={tooltipTitle}>
              <div className="status">
                {raw === 'Current' && <div className="already_success" />}
                {raw === 'Failed' && <div className="already_error" />}
                {raw === 'InProgress' && <LoadingOutlined id="wait_moment" />}
                {raw === 'Terminating' && (
                  <ExclamationCircleFilled className="helm_state_unknown" style={{ marginRight: 4 }} />
                )}
                {!known && (
                  <QuestionCircleFilled className="helm_state_unknown" style={{ marginRight: 4 }} />
                )}
                <div className="word">{label}</div>
              </div>
            </Tooltip>
          </Space>
        );
      },
    },
  ];

  const getHelmDetail = useCallback(async () => {
    const resources = dataProps?.resources || [];
    setHelmResourceList(resources);
    helmBackListRef.current = resources;
    setHelmDetailTotal(resources.length);
    if (resources.length) {
      const temporyKindList = resources.map(item => ({ text: item.kind, value: item.kind }));
      setKindArr([...filterRepeat(temporyKindList)]);
    }
  }, [dataProps]);

  const isShowLink = (type) => linkArr.includes(type);

  const createResource = {
    Pod: (namespace, name) => history.push({ pathname: `/${containerRouterPrefix}/workload/pod/${namespace}/${name}` }),
    Deployment: (namespace, name) => history.push({ pathname: `/${containerRouterPrefix}/workload/deployment/${namespace}/${name}` }),
    StatefulSet: (namespace, name) => history.push({ pathname: `/${containerRouterPrefix}/workload/statefulset/${namespace}/${name}` }),
    DaemonSet: (namespace, name) => history.push({ pathname: `/${containerRouterPrefix}/workload/daemonset/${namespace}/${name}` }),
    Job: (namespace, name) => history.push({ pathname: `/${containerRouterPrefix}/workload/job/${namespace}/${name}` }),
    CronJob: (namespace, name) => history.push({ pathname: `/${containerRouterPrefix}/workload/cronjob/${namespace}/${name}` }),
    Service: (namespace, name) => history.push({ pathname: `/${containerRouterPrefix}/network/service/${namespace}/${name}` }),
    Ingress: (namespace, name) => history.push({ pathname: `/${containerRouterPrefix}/network/ingress/${namespace}/${name}` }),
    ConfigMap: (namespace, name) => history.push({ pathname: `/${containerRouterPrefix}/configuration/configMap/${namespace}/${name}` }),
    Secret: (namespace, name) => history.push({ pathname: `/${containerRouterPrefix}/configuration/secret/${namespace}/${name}` }),
    ServiceAccount: (namespace, name) => history.push({ pathname: `/${containerRouterPrefix}/userManage/serviceAccount/${namespace}/${name}` }),
    Role: (namespace, name) => history.push({ pathname: `/${containerRouterPrefix}/userManage/role/${namespace}/${name}` }),
    RoleBinding: (namespace, name) => history.push({ pathname: `/${containerRouterPrefix}/userManage/roleBinding/${namespace}/${name}` }),
    Namespace: (namespace, name) => history.push({ pathname: `/${containerRouterPrefix}/namespace/namespaceManage/${name}` }),
    LimitRange: (namespace, name) => history.push({ pathname: `/${containerRouterPrefix}/namespace/limitRange/${namespace}/${name}` }),
    ResourceQuota: (namespace, name) => history.push({ pathname: `/${containerRouterPrefix}/namespace/resourceQuota/${namespace}/${name}` }),
    ClusterRole: (namespace, name) => history.push({ pathname: `/${containerRouterPrefix}/userManage/clusterRole/${name}`, state: { roleType: 'clusterrole', roleNamespace: '', roleName: name } }),
    ClusterRoleBinding: (namespace, name) => history.push({ pathname: `/${containerRouterPrefix}/userManage/clusterRoleBinding/${name}` }),
    PersistentVolume: (namespace, name) => history.push({ pathname: `/${containerRouterPrefix}/storage/pv/${name}` }),
    PersistentVolumeClaim: (namespace, name) => history.push({ pathname: `/${containerRouterPrefix}/storage/pvc/${namespace}/${name}` }),
    StorageClass: (namespace, name) => history.push({ pathname: `/${containerRouterPrefix}/storage/sc/${name}` }),
    ServiceMonitor: (namespace, name) => history.push({ pathname: `/${containerRouterPrefix}/monitor/monitorGoalManage/serviceMonitor/${namespace}/${name}`, state: { group: 'monitoring.coreos.com', version: 'v1', plural: 'servicemonitors' } }),
    CustomResourceDefinition: (namespace, name) => history.push({ pathname: `/${containerRouterPrefix}/customResourceDefinition/${solveEncodePath(name)}` }),
  };

  const jumpToDetail = (type, namespace, name) => {
    createResource[type]?.(namespace, name);
  };

  useEffect(() => {
    getHelmResource();
  }, [helmDetailFilterObj, helmSortObj, helmNamespaceSortObj, helmTypeSortObj, helmStatusSortObj, t]);

  useEffect(() => {
    setFilterStatusData();
    getHelmDetail();
  }, [getHelmDetail, setFilterStatusData]);

  return (
    <div className="helm_tab_container container_margin_box tooltip_container_height">
      <div className="resource_card">
        <div className="resource_top_search">
          <Form form={helmSearchForm} ref={helmFormRef} className="toolsBox" autoComplete="off">
            <Form.Item name="name" className='search' style={{ marginBottom: '0' }}>
              <Search placeholder={t('searchPlaceholder', { ns: 'common', name: t('searchName.resource', { ns: 'common' }) })} onSearch={() => getHelmResource()} maxLength={53} />
            </Form.Item>
          </Form>
        </div>
        <Table
          className="table"
          rowKey="uid"
          pagination={{
            className: 'page',
            pageSizeOptions: ['10', '20', '50'],
            current: helmDetailCurrent,
            pageSize: helmDetailPageSize,
            total: helmDetailTotal || 0,
            showTotal: (total) => t('list.total', { total }),
            showSizeChanger: true,
          }}
          dataSource={helmResourceList}
          columns={helmDetailColumns}
          onChange={handleHelmDetailTableChange}
        />
      </div>
    </div>
  );
}