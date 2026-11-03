/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import { containerRouterPrefix, ResponseCode } from '@/common/constants';
import { useHistory, Link, useLocation } from 'inula-router';
import { useCallback, useEffect, useState, createRef } from 'openinula';
import { Table, Checkbox, message } from 'antd';
import { InfoCircleOutlined } from '@ant-design/icons';
import useLocalStorage from '@/hooks/useLocalStorage';
import '@/styles/rayServiceDetail.less';
import { getServiceDetail } from '@/api/rayApi.js';
import { yamlTojson, jsonToYaml, filterServiceStatusValue, filterRepeatVolume } from '@/tools/utils';
import CodeMirror from '@uiw/react-codemirror';
import { javascript } from '@codemirror/lang-javascript';
import '@/styles/components/codeMirrorEditor.less';
import { useIntl } from 'inula-intl';
export default function Detail() {
  const intl = useIntl();
  const location = useLocation();
  const serviceName = location.state.name;
  const serviceNameSpace = location.state.namespace;
  const [messageApi, contextHolder] = message.useMessage();
  const [currentTheme, setCurrentTheme] = useLocalStorage('theme', 'light');
  const [detailInfo, setDetailInfo] = useState([]);
  const [serviceImageName, setServiceImageName] = useState();
  const [serviceImageVersion, setServiceImageVersion] = useState();
  const [serviceHeadChangeWork, setServiceHeadChangeWork] = useState(false);
  const [serviceHeadEnv, setServiceHeadEnv] = useState([]);
  const [serviceWorkEnv, setServiceWorkEnv] = useState([]);
  const [serviceWorkTableData, setServiceWorkTableData] = useState([]);
  const [serviceHeadTableData, setServiceHeadTableData] = useState([]);
  const [serverConfigData, setServerConfigData] = useState();
  const [headVolumesData, setHeadVolumesData] = useState();
  const [headVolumeMountsData, setHeadVolumeMountsData] = useState();
  const [workVolumesData, setWorkVolumesData] = useState();
  const [workVolumeMountsData, setWorkVolumeMountsData] = useState();
  const [isGetDetail, setIsGetDetail] = useState(false);
  const [isHasServiceHeadEnv, setIsHasServiceHeadEnv] = useState(false);
  const [isHasServiceWorkEnv, setIsHasServiceWorkEnv] = useState(false);
  const [detailUrl, setDetailUrl] = useState('');
  const prefixUrl = window.location.hostname;
  const detailHeadConfigColumns = [
    {
      title: intl.formatMessage({ id: 'table.nodeType' }),
      name: 'serviceDetailType',
      dataIndex: 'type',
      key: 'type',
    },
    {
      title: intl.formatMessage({ id: 'table.nodeConfigName' }),
      name: 'serviceDetailName',
      dataIndex: 'name',
      key: 'name',
      render: (_, record) => (record.name !== '' ? record.name : '--'),
    },
    {
      title: intl.formatMessage({ id: 'table.npuCard' }),
      name: 'serviceDetailNpu',
      dataIndex: 'npu',
      key: 'npu',
      render: (_, record) => (record.npu !== '' ? record.npu : '--'),
    },
    {
      title: 'CPU (Core)',
      name: 'serviceDetailCpu',
      dataIndex: 'cpu',
      key: 'cpu',
      render: (_, record) => record.cpu || '--',
    },
    {
      title: intl.formatMessage({ id: 'table.memory' }),
      name: 'serviceDetailMemory',
      dataIndex: 'memory',
      key: 'memory',
      render: (_, record) => record.memory || '--',
    },
    {
      title: intl.formatMessage({ id: 'table.maxReplicas' }),
      name: 'serviceDetailMaxCopy',
      dataIndex: 'maxCopy',
      key: 'maxCopy',
      render: (_, record) => record.maxCopy || '--',
    },
    {
      title: intl.formatMessage({ id: 'table.minReplicas' }),
      name: 'serviceDetailMinCopy',
      dataIndex: 'minCopy',
      key: 'minCopy',
      render: (_, record) => record.minCopy || '--',
    },
    {
      title: intl.formatMessage({ id: 'table.npuType' }),
      name: 'serviceDetailType',
      dataIndex: 'npuType',
      key: 'npuType',
      render: (_, record) => record.npuType || '--',
    },
  ];
  const getServiceInfo = async () => {
    let detailInfo = [
      {
        label: '服务名称：',
        value: '',
        type: '',
      },
      {
        label: '状态：',
        value: '',
        type: '',
      },
      {
        label: '集群IP地址：',
        value: '',
        type: '',
      },
      {
        label: '创建人：',
        value: '',
        type: '',
      },
      {
        label: '创建时间：',
        value: '',
        type: '',
      },
      {
        label: '更新时间：',
        value: '',
        type: '',
      },
    ];
    try {
      const res = await getServiceDetail(serviceNameSpace, serviceName);
      if (res.data.code === ResponseCode.OK) {
        detailInfo[0].value = res.data.data.name || '-';
        detailInfo[1].value = res.data.data.cluster_status_zhcn || '-';
        detailInfo[2].value = res.data.data.dashboard_address || '-';
        detailInfo[3].value = res.data.data.create_by || '-';
        detailInfo[4].value = res.data.data.create_time || '-';
        detailInfo[5].value = res.data.data.update_time || '-';
        setDetailInfo(detailInfo);
        setDetailUrl(
          res.data.data.dashboard_address ? `http://${prefixUrl}:${res.data.data.dashboard_address.split(':')[1]}` : '',
        );
        let configData = yamlTojson(res.data.data.config_detail);
        setServerConfigData(configData.spec.serveConfigV2);
        const headGroupSpec = configData.spec.rayClusterConfig.headGroupSpec;
        setServiceImageName(headGroupSpec.template.spec.containers[0].image.split(':')[0]);
        setServiceImageVersion(headGroupSpec.template.spec.containers[0].image.split(':')[1]);
        const isHasServiceHeadenv = Object.prototype.hasOwnProperty.call(
          headGroupSpec.template.spec.containers[0],
          'env',
        );
        if (isHasServiceHeadenv && headGroupSpec.template.spec.containers[0].env.length > 0) {
          setIsHasServiceHeadEnv(true);
        }
        setServiceHeadEnv(isHasServiceHeadenv ? headGroupSpec.template.spec.containers[0].env : []);
        if (
          Object.prototype.hasOwnProperty.call(headGroupSpec.rayStartParams, 'num-cpus') &&
          headGroupSpec.rayStartParams['num-cpus'] === '0'
        ) {
          setServiceHeadChangeWork(false);
        } else {
          setServiceHeadChangeWork(true);
        }
        if (Object.prototype.hasOwnProperty.call(headGroupSpec.template.spec, 'volumes')) {
          setHeadVolumesData(jsonToYaml(JSON.stringify(headGroupSpec.template.spec.volumes)));
        }
        if (Object.prototype.hasOwnProperty.call(headGroupSpec.template.spec.containers[0], 'volumeMounts')) {
          setHeadVolumeMountsData(jsonToYaml(JSON.stringify(headGroupSpec.template.spec.containers[0].volumeMounts)));
        }
        let serviceHeadTableSource = [];
        const headLimitsKeys = Object.keys(headGroupSpec.template.spec.containers[0].resources.limits);
        const headNpuKeyArr = headLimitsKeys.filter((item) => item !== 'npu' && item !== 'cpu' && item !== 'memory');
        serviceHeadTableSource.push({
          type: 'head',
          name: headGroupSpec.template.spec.containers[0].name,
          npu: headNpuKeyArr.length ? headGroupSpec.template.spec.containers[0].resources.limits[headNpuKeyArr[0]] : '',
          cpu: headGroupSpec.template.spec.containers[0].resources.limits.cpu,
          memory: headGroupSpec.template.spec.containers[0].resources.limits.memory,
          maxCopy: '',
          minCopy: '',
          npuType: headNpuKeyArr.length ? headNpuKeyArr[0] : '',
        });
        const workerGroupSpecs = configData.spec.rayClusterConfig.workerGroupSpecs;
        let serviceWorkEnvData = [];
        let serviceWorkTableSourece = [];
        workerGroupSpecs.map((item) => {
          if (Object.prototype.hasOwnProperty.call(item.template.spec, 'volumes')) {
            setWorkVolumesData(jsonToYaml(JSON.stringify(item.template.spec.volumes)));
          }
          if (Object.prototype.hasOwnProperty.call(item.template.spec.containers[0], 'volumeMounts')) {
            setWorkVolumeMountsData(jsonToYaml(JSON.stringify(item.template.spec.containers[0].volumeMounts)));
          }
          const limitsKeys = Object.keys(item.template.spec.containers[0].resources.limits);
          const npuKeyArr = limitsKeys.filter((item) => item !== 'npu' && item !== 'cpu' && item !== 'memory');
          serviceWorkTableSourece.push({
            type: 'worker',
            name: item.template.spec.containers[0].name,
            cpu: item.template.spec.containers[0].resources.limits.cpu,
            memory: item.template.spec.containers[0].resources.limits.memory,
            maxCopy: item.maxReplicas,
            minCopy: item.minReplicas,
            npu: npuKeyArr.length ? item.template.spec.containers[0].resources.limits[npuKeyArr[0]] : '',
            npuType: npuKeyArr.length ? npuKeyArr[0] : '',
          });
          // 获取workenv
          if (Object.prototype.hasOwnProperty.call(item.template.spec.containers[0], 'env')) {
            serviceWorkEnvData = serviceWorkEnvData.concat(item.template.spec.containers[0].env);
          }
        });
        if (serviceWorkEnvData.length > 0) {
          setIsHasServiceWorkEnv(true);
        }
        setServiceWorkTableData(serviceWorkTableSourece);
        setServiceWorkEnv(filterRepeatVolume(serviceWorkEnvData));
        setServiceHeadTableData(serviceHeadTableSource);
        setIsGetDetail(true);
      } else {
        let msg = '';
        msg = JSON.stringify(res.data.data) !== '{}' ? `${res.data.msg + ':' + res.data.data}` : res.data.msg;
        messageApi.error(`${serviceName}获取失败,${msg}`);
      }
    } catch (error) {
      if (error) {
        let msg = '';
        msg =
          JSON.stringify(error.response.data.data) !== '{}'
            ? `${error.response.data.msg + ':' + error.response.data.data}`
            : error.response.data.msg;
        messageApi.error(`${serviceName}获取失败,${msg}`);
      }
    }
  };
  useEffect(() => {
    getServiceInfo();
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
                    background: filterServiceStatusValue(item.value),
                  }}
                ></span>
                {item.value}
              </p>
            </div>
          ))}
        </div>
      </div>
      <div className="detail_config detail_common_content">
        <p className="detail_common_title" style={{ marginBottom: '17px' }}>
          {intl.formatMessage({ id: 'ray.service.serveConfig' })}
        </p>
        {isGetDetail ? <CodeMirror value={serverConfigData} extensions={[javascript()]} editable={false} /> : ''}
      </div>
      <div className="detail_cluster detail_common_content">
        <p className="detail_common_title">{intl.formatMessage({ id: 'ray.service.rayClusterConfig' })}</p>
        <div className="detail_cluster_image">
          <div className="detail_cluster_image_single">
            <p className="detail_cluster_image_single_lable">{intl.formatMessage({ id: 'detail.imageName' })}</p>
            <p className="detail_cluster_image_single_value">{serviceImageName}</p>
          </div>
          <div className="detail_cluster_image_single">
            <p className="detail_cluster_image_single_lable">{intl.formatMessage({ id: 'detail.imageVersion' })}</p>
            <p className="detail_cluster_image_single_value">{serviceImageVersion}</p>
          </div>
        </div>
        <p className="split_line"></p>
        <p className="detail_common_title2">{intl.formatMessage({ id: 'detail.headNodeConfig' })}</p>
        <Table
          locale={{ emptyText: intl.formatMessage({ id: 'common.noData' }) }}
          columns={detailHeadConfigColumns}
          pagination={false}
          dataSource={serviceHeadTableData}
        />
        <Checkbox checked={serviceHeadChangeWork} disabled>
          {intl.formatMessage({ id: 'ray.service.allowHeadWorkload' })}
        </Checkbox>
        <div className="detail_cluster_code">
          <p className="detail_config_name">volumes</p>
          <p className="detail_config_name">volumeMounts</p>
        </div>
        <div className="detail_cluster_code">
          {isGetDetail ? <CodeMirror value={headVolumesData} extensions={[javascript()]} editable={false} /> : ''}
          {isGetDetail ? <CodeMirror value={headVolumeMountsData} extensions={[javascript()]} editable={false} /> : ''}
        </div>
        <p className="detail_common_title3">{intl.formatMessage({ id: 'env.title' })}</p>
        {isHasServiceHeadEnv ? (
          <div className="detail_work_environment">
            {serviceHeadEnv.map((item) => (
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
          dataSource={serviceWorkTableData}
        />
        <div className="detail_cluster_code">
          <p className="detail_config_name">volumes</p>
          <p className="detail_config_name">volumeMounts</p>
        </div>
        <div className="detail_cluster_code">
          {isGetDetail ? <CodeMirror value={workVolumesData} extensions={[javascript()]} editable={false} /> : ''}
          {isGetDetail ? <CodeMirror value={workVolumeMountsData} extensions={[javascript()]} editable={false} /> : ''}
        </div>
        <p className="detail_common_title3">{intl.formatMessage({ id: 'env.title' })}</p>
        {isHasServiceWorkEnv ? (
          <div className="detail_work_environment">
            {serviceWorkEnv.map((item) => (
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
