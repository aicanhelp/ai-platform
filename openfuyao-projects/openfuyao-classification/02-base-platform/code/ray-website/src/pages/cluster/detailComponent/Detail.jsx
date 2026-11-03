/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import { ResponseCode } from '@/common/constants';
import { Link, useLocation } from 'inula-router';
import { useEffect, useState } from 'openinula';
import { Table, Checkbox, message } from 'antd';
import { InfoCircleOutlined } from '@ant-design/icons';
import useLocalStorage from '@/hooks/useLocalStorage';
import '@/styles/rayServiceDetail.less';
import { getClusterDetail } from '@/api/rayApi.js';
import { yamlTojson, filterClusterStatusValue, filterRepeatVolume, jsonToYaml } from '@/tools/utils';
import CodeMirror from '@uiw/react-codemirror';
import { javascript } from '@codemirror/lang-javascript';
import '@/styles/components/codeMirrorEditor.less';
import { useIntl } from 'inula-intl';
export default function Detail() {
  const intl = useIntl();
  const location = useLocation();
  const [messageApi, contextHolder] = message.useMessage();
  const clusterName = location.state.name;
  const clusterNameSpace = location.state.namespace;
  const [currentTheme, setCurrentTheme] = useLocalStorage('theme', 'light');
  const [detailInfo, setDetailInfo] = useState([]);
  const [clusterImageName, setClusterImageName] = useState();
  const [clusterImageVersion, setClusterImageVersion] = useState();
  const [clusterHeadChangeWork, setClusterHeadChangeWork] = useState(false);
  const [clusterHeadEnv, setClusterHeadEnv] = useState([]);
  const [clusterWorkEnv, setClusterWorkEnv] = useState([]);
  const [clusterWorkTableData, setClusterWorkTableData] = useState([]);
  const [clusterHeadTableData, setClusterHeadTableData] = useState([]);
  const [isHasClusterHeadEnv, setIsHasClusterHeadEnv] = useState(false);
  const [isHasClusterWorkEnv, setIsHasServiceWorkEnv] = useState(false);
  const [clusterHeadVolumesData, setClusterHeadVolumesData] = useState();
  const [clusterHeadVolumeMountsData, setClusterHeadVolumeMountsData] = useState();
  const [clusterWorkVolumesData, setClusterWorkVolumesData] = useState();
  const [clusterWorkVolumeMountsData, setClusterWorkVolumeMountsData] = useState();
  const [isGetClusterDetail, setIsGetClusterDetail] = useState(false);
  const prefixUrl = window.location.hostname;
  const [detailUrl, setDetailUrl] = useState();
  const detailHeadConfigColumns = [
    {
      title: intl.formatMessage({ id: 'table.nodeType' }),
      name: 'clusterDetailType',
      dataIndex: 'type',
      key: 'type',
    },
    {
      title: intl.formatMessage({ id: 'table.nodeConfigName' }),
      name: 'clusterDetailName',
      dataIndex: 'name',
      key: 'name',
      render: (_, record) => (record.name !== '' ? record.name : '--'),
    },
    {
      title: intl.formatMessage({ id: 'table.npuCard' }),
      name: 'clusterDetailNpu',
      dataIndex: 'npu',
      key: 'npu',
      render: (_, record) => (record.npu !== '' ? record.npu : '--'),
    },
    {
      title: 'CPU (Core)',
      name: 'clusterDetailCpu',
      dataIndex: 'cpu',
      key: 'cpu',
      render: (_, record) => record.cpu || '--',
    },
    {
      title: intl.formatMessage({ id: 'table.memory' }),
      name: 'clusterDetailMemory',
      dataIndex: 'memory',
      key: 'memory',
      render: (_, record) => record.memory || '--',
    },
    {
      title: intl.formatMessage({ id: 'table.maxReplicas' }),
      name: 'clusterDetailMaxCopy',
      dataIndex: 'maxCopy',
      key: 'maxCopy',
      render: (_, record) => record.maxCopy || '--',
    },
    {
      title: intl.formatMessage({ id: 'table.minReplicas' }),
      name: 'clusterDetailMinCopy',
      dataIndex: 'minCopy',
      key: 'minCopy',
      render: (_, record) => record.minCopy || '--',
    },
    {
      title: intl.formatMessage({ id: 'table.npuType' }),
      name: 'clusterDetailType',
      dataIndex: 'npuType',
      key: 'npuType',
      render: (_, record) => record.npuType || '--',
    },
  ];
  const getClusterInfo = async () => {
    let detailInfo = [
      {
        label: '集群名称：',
        value: '',
        type: '',
      },
      {
        label: 'Ray版本：',
        value: '',
        type: 'cluster',
      },
      {
        label: '状态：',
        value: '',
        type: '',
      },
      {
        label: '集群IP：',
        value: '',
        type: 'cluster',
      },
      {
        label: 'GCS端口：',
        value: '',
        type: '',
      },
      {
        label: 'Dashboard端口：',
        value: '',
        type: 'cluster',
      },
      {
        label: '创建人：',
        value: '',
        type: '',
      },
      {
        label: '创建时间：',
        value: '',
        type: 'cluster',
      },
      {
        label: '更新时间:',
        value: '',
        type: '',
      },
    ];
    try {
      const res = await getClusterDetail(clusterNameSpace, clusterName);
      if (res.data.code === ResponseCode.OK) {
        let configData = yamlTojson(res.data.data.config_detail);
        const headGroupSpec = configData.spec.headGroupSpec;
        const rayHead = headGroupSpec.template.spec.containers.filter((item) => item.name !== 'fluentbit');
        detailInfo[0].value = res.data.data.name || '-';
        detailInfo[1].value = res.data.data.ray_version || '-';
        detailInfo[2].value = res.data.data.cluster_status_zhcn || '-';
        detailInfo[3].value = res.data.data.cluster_ip || '-';
        detailInfo[4].value =
          rayHead[0].ports.filter((item) => item.name === 'gcs-server' || item.name === 'gcs')?.[0]?.containerPort ||
          res.data.data.gcs_port ||
          '-';
        detailInfo[5].value = res.data.data.dashboard_port || '-';
        detailInfo[6].value = res.data.data.create_by || '-';
        detailInfo[7].value = res.data.data.create_time || '-';
        detailInfo[8].value = res.data.data.update_time || '-';
        setDetailInfo(detailInfo);
        setDetailUrl(res.data.data.dashboard_port ? `http://${prefixUrl}:${res.data.data.dashboard_port}` : '');
        const workerGroupSpecs = configData.spec.workerGroupSpecs;
        setClusterImageName(rayHead[0].image.split(':')[0]);
        setClusterImageVersion(rayHead[0].image.split(':')[1]);
        if (
          Object.prototype.hasOwnProperty.call(headGroupSpec.rayStartParams, 'num-cpus') &&
          headGroupSpec.rayStartParams['num-cpus'] === '0'
        ) {
          setClusterHeadChangeWork(false);
        } else {
          setClusterHeadChangeWork(true);
        }
        if (Object.prototype.hasOwnProperty.call(headGroupSpec.template.spec, 'volumes')) {
          setClusterHeadVolumesData(jsonToYaml(JSON.stringify(headGroupSpec.template.spec.volumes)));
        }
        if (Object.prototype.hasOwnProperty.call(rayHead[0], 'volumeMounts')) {
          setClusterHeadVolumeMountsData(jsonToYaml(JSON.stringify(rayHead[0].volumeMounts)));
        }
        const isHasHeadenv = Object.prototype.hasOwnProperty.call(rayHead[0], 'env');
        if (isHasHeadenv && rayHead[0].env.length > 0) {
          setIsHasClusterHeadEnv(true);
        }
        setClusterHeadEnv(isHasHeadenv ? rayHead[0].resources.env : []);
        let clusterHeadTableSource = [];
        const headLimitsKeys = Object.keys(rayHead[0].resources.limits);
        const headNpuKeyArr = headLimitsKeys.filter((item) => item !== 'npu' && item !== 'cpu' && item !== 'memory');

        clusterHeadTableSource.push({
          type: 'head',
          name: rayHead[0].name,
          npu: headNpuKeyArr.length ? rayHead[0].resources.limits[headNpuKeyArr[0]] : '',
          cpu: rayHead[0].resources.limits.cpu,
          memory: rayHead[0].resources.limits.memory,
          maxCopy: headGroupSpec.maxReplicas,
          minCopy: headGroupSpec.minReplicas,
          npuType: headNpuKeyArr.length ? headNpuKeyArr[0] : '',
        });
        let workEnvData = [];
        let workTableSourece = [];
        workerGroupSpecs.map((item) => {
          if (Object.prototype.hasOwnProperty.call(item.template.spec, 'volumes')) {
            setClusterWorkVolumesData(jsonToYaml(JSON.stringify(item.template.spec.volumes)));
          }
          const rayWorker = item.template.spec.containers.filter((item) => item.name !== 'fluentbit');
          if (Object.prototype.hasOwnProperty.call(rayWorker[0], 'volumeMounts')) {
            setClusterWorkVolumeMountsData(jsonToYaml(JSON.stringify(rayWorker[0].volumeMounts)));
          }
          const limitsKeys = Object.keys(rayWorker[0].resources.limits);
          const npuKeyArr = limitsKeys.filter((item) => item !== 'npu' && item !== 'cpu' && item !== 'memory');
          workTableSourece.push({
            type: 'worker',
            name: rayWorker[0].name,
            npu: npuKeyArr.length ? rayWorker[0].resources.limits[npuKeyArr[0]] : '',
            cpu: rayWorker[0].resources.limits.cpu,
            memory: rayWorker[0].resources.limits.memory,
            maxCopy: item.maxReplicas,
            minCopy: item.minReplicas,
            npuType: npuKeyArr.length ? npuKeyArr[0] : '',
          });
          // 获取workenv
          const isHasWodkenv =
            Object.prototype.hasOwnProperty.call(rayWorker[0], 'resources') &&
            Object.prototype.hasOwnProperty.call(rayWorker[0].resources, 'env');
          if (isHasWodkenv) {
            workEnvData = workEnvData.concat(rayWorker[0].resources.env);
          }
        });
        if (workEnvData.length > 0) {
          setIsHasServiceWorkEnv(true);
        }
        setClusterHeadTableData(clusterHeadTableSource);
        setClusterWorkTableData(workTableSourece);
        setClusterWorkEnv(filterRepeatVolume(workEnvData));
        setIsGetClusterDetail(true);
      } else {
        let msg = '';
        msg = res.data.data ? `${res.data.msg}: ${res.data.data}` : res.data.msg;
        messageApi.error(msg);
      }
    } catch (error) {
      if (error) {
        let msg = '';
        msg =
          JSON.stringify(error.response.data.data) !== '{}'
            ? `${error.response.data.msg + ':' + error.response.data.data}`
            : error.response.data.msg;
        messageApi.error(`获取${clusterName}详情失败,${msg}`);
      }
    }
  };
  useEffect(() => {
    getClusterInfo();
  }, []);
  useEffect(() => {}, [currentTheme]);
  return (
    <div className="detail">
      {contextHolder}
      <p className="detail_dashboard">
        <InfoCircleOutlined style={{ color: '#3f66f5', margin: '0 12px 0 32px' }} />
        {intl.formatMessage({ id: 'detail.monitorGoTo' })}
        <a href={detailUrl} target="_blank" style={{ color: '#3f66f5' }}>
          Ray Dashboard
        </a>
        {intl.formatMessage({ id: 'detail.viewAt' })}
      </p>
      <div className="ray_detail_info detail_common_content">
        <p className="detail_common_title">{intl.formatMessage({ id: 'detail.basicInfo' })}</p>
        <div className="detail_info_row">
          {detailInfo.map((item) => (
            <div className="detail_info_row_single">
              <p className={`detail_info_row_single_label ${item.type.includes('cluster') ? 'cluster_label' : ''}`}>
                {item.label}
              </p>
              <p className="detail_info_row_single_value">
                <span
                  className="detail_info_row_single_status"
                  style={{
                    display: item.label.includes('状态') ? 'flex' : 'none',
                    background: filterClusterStatusValue(item.value),
                  }}
                ></span>
                {item.value}
              </p>
            </div>
          ))}
        </div>
      </div>
      <div className="detail_cluster detail_common_content">
        <p className="detail_common_title">{intl.formatMessage({ id: 'detail.algorithmFramework' })}</p>
        <div className="detail_cluster_image">
          <div className="detail_cluster_image_single">
            <p className="detail_cluster_image_single_lable">{intl.formatMessage({ id: 'detail.imageName' })}</p>
            <p className="detail_cluster_image_single_value">{clusterImageName}</p>
          </div>
          <div className="detail_cluster_image_single">
            <p className="detail_cluster_image_single_lable">{intl.formatMessage({ id: 'detail.imageVersion' })}</p>
            <p className="detail_cluster_image_single_value">{clusterImageVersion}</p>
          </div>
        </div>
      </div>
      <div className="detail_cluster detail_common_content">
        <p className="detail_common_title2">{intl.formatMessage({ id: 'detail.headNodeConfig' })}</p>
        <Table
          locale={{ emptyText: intl.formatMessage({ id: 'common.noData' }) }}
          columns={detailHeadConfigColumns}
          pagination={false}
          dataSource={clusterHeadTableData}
        />
        <Checkbox checked={clusterHeadChangeWork} disabled>
          {intl.formatMessage({ id: 'ray.cluster.allowHeadWorkloadShort' })}
        </Checkbox>
        <div className="detail_cluster_code">
          <p className="detail_config_name">volumes</p>
          <p className="detail_config_name">volumeMounts</p>
        </div>
        <div className="detail_cluster_code">
          {isGetClusterDetail ? (
            <CodeMirror value={clusterHeadVolumesData} extensions={[javascript()]} editable={false} />
          ) : (
            ''
          )}
          {isGetClusterDetail ? (
            <CodeMirror value={clusterHeadVolumeMountsData} extensions={[javascript()]} editable={false} />
          ) : (
            ''
          )}
        </div>
        <p className="detail_common_title3">{intl.formatMessage({ id: 'env.title' })}</p>
        {isHasClusterHeadEnv ? (
          <div className="detail_work_environment">
            {clusterHeadEnv.map((item) => (
              <div className="detail_work_environment_single">
                <p>{item.name}</p>
                <p className="detail_work_environment_single_equal">=</p>
                <p>{item.value}</p>
              </div>
            ))}
          </div>
        ) : (
          <p className="detail_common_title4">{intl.formatMessage({ id: 'detail.noData' })}</p>
        )}
        <p className="split_line"></p>
        <p className="detail_common_title2">{intl.formatMessage({ id: 'detail.workerNodeConfig' })}</p>
        <Table
          locale={{ emptyText: intl.formatMessage({ id: 'common.noData' }) }}
          columns={detailHeadConfigColumns}
          pagination={false}
          dataSource={clusterWorkTableData}
        />
        <div className="detail_cluster_code">
          <p className="detail_config_name">volumes</p>
          <p className="detail_config_name">volumeMounts</p>
        </div>
        <div className="detail_cluster_code">
          {isGetClusterDetail ? (
            <CodeMirror value={clusterWorkVolumesData} extensions={[javascript()]} editable={false} />
          ) : (
            ''
          )}
          {isGetClusterDetail ? (
            <CodeMirror value={clusterWorkVolumeMountsData} extensions={[javascript()]} editable={false} />
          ) : (
            ''
          )}
        </div>
        <p className="detail_common_title3">{intl.formatMessage({ id: 'env.title' })}</p>
        {isHasClusterWorkEnv ? (
          <div className="detail_work_environment">
            {clusterWorkEnv.map((item) => (
              <div className="detail_work_environment_single">
                <p>{item.name}</p>
                <p className="detail_work_environment_single_equal">=</p>
                <p>{item.value}</p>
              </div>
            ))}
          </div>
        ) : (
          <p className="detail_common_title4">{intl.formatMessage({ id: 'detail.noData' })}</p>
        )}
      </div>
    </div>
  );
}
