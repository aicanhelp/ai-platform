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
import { useLocation, Link } from 'inula-router';
import { useEffect, useState } from 'openinula';
import { Table, Checkbox, message } from 'antd';
import { InfoCircleOutlined } from '@ant-design/icons';
import useLocalStorage from '@/hooks/useLocalStorage';
import '@/styles/rayServiceDetail.less';
import { getJobDetail } from '@/api/rayApi.js';
import { yamlTojson, filterJobStatusValue, filterRepeatVolume, jsonToYaml } from '@/tools/utils';
import CodeMirror from '@uiw/react-codemirror';
import { javascript } from '@codemirror/lang-javascript';
import '@/styles/components/codeMirrorEditor.less';
import { useIntl } from 'inula-intl';
export default function Detail() {
  const intl = useIntl();
  const [currentTheme, setCurrentTheme] = useLocalStorage('theme', 'light');
  const [jobDetailInfo, setJobDetailInfo] = useState([]);
  const location = useLocation();
  const [messageApi, contextHolder] = message.useMessage();
  const jobName = location.state.name;
  const jobNameSpace = location.state.namespace;
  const [jobImageName, setJobImageName] = useState();
  const [jobImageVersion, setJobImageVersion] = useState();
  const [jobHeadChangeWork, setJobHeadChangeWork] = useState(false);
  const [jobHeadEnv, setJobHeadEnv] = useState([]);
  const [jobWorkEnv, setJobWorkEnv] = useState([]);
  const [jobWorkTableData, setJobWorkTableData] = useState([]);
  const [jobHeadTableData, setJobHeadTableData] = useState([]);
  const [isHasJobHeadEnv, setIsHasJobHeadEnv] = useState(false);
  const [isHasJobWorkEnv, setIsHasJobWorkEnv] = useState(false);
  const [jobHeadVolumesData, setJobHeadVolumesData] = useState();
  const [jobHeadVolumeMountsData, setJobHeadVolumeMountsData] = useState();
  const [jobWorkVolumesData, setJobWorkVolumesData] = useState();
  const [jobWorkVolumeMountsData, setJobWorkVolumeMountsData] = useState();
  const [isGetJobDetail, setIsGetJobDetail] = useState(false);
  const [entrypointResource, setEntrypointResource] = useState([]);
  const [detailUrl, setDetailUrl] = useState('');
  const prefixUrl = window.location.hostname;
  const detailHeadConfigColumns = [
    {
      title: intl.formatMessage({ id: 'table.nodeType' }),
      name: 'jobDetailType',
      dataIndex: 'type',
      key: 'type',
    },
    {
      title: intl.formatMessage({ id: 'table.nodeConfigName' }),
      name: 'jobDetailName',
      dataIndex: 'name',
      key: 'name',
      render: (_, record) => (record.name !== '' ? record.name : '--'),
    },
    {
      title: intl.formatMessage({ id: 'table.npuCard' }),
      name: 'jobDetailNpu',
      dataIndex: 'npu',
      key: 'npu',
      render: (_, record) => (record.npu !== '' ? record.npu : '--'),
    },
    {
      title: 'CPU (Core)',
      name: 'jobDetailCpu',
      dataIndex: 'cpu',
      key: 'cpu',
      render: (_, record) => record.cpu || '--',
    },
    {
      title: intl.formatMessage({ id: 'table.memory' }),
      name: 'jobDetailMemory',
      dataIndex: 'memory',
      key: 'memory',
      render: (_, record) => record.memory || '--',
    },
    {
      title: intl.formatMessage({ id: 'table.maxReplicas' }),
      name: 'jobDetailMaxCopy',
      dataIndex: 'maxCopy',
      key: 'maxCopy',
      render: (_, record) => record.maxCopy || '--',
    },
    {
      title: intl.formatMessage({ id: 'table.minReplicas' }),
      name: 'jobDetailMinCopy',
      dataIndex: 'minCopy',
      key: 'minCopy',
      render: (_, record) => record.minCopy || '--',
    },
    {
      title: intl.formatMessage({ id: 'table.npuType' }),
      name: 'jobDetailType',
      dataIndex: 'npuType',
      key: 'npuType',
      render: (_, record) => record.npuType || '--',
    },
  ];
  const getJobInfo = async () => {
    let jobDetailInfo = [
      {
        label: 'Job ID：',
        value: '',
        type: 'job',
      },
      {
        label: 'Job名称：',
        value: '',
        type: '',
      },
      {
        label: '集群名称：',
        value: '',
        type: 'job',
      },
      {
        label: '集群IP地址：',
        value: '',
        type: '',
      },
      {
        label: '状态：',
        value: '运行中',
        type: 'job',
      },
      {
        label: '创建人：',
        value: '',
        type: '',
      },
      {
        label: '更新时间:',
        value: '',
        type: 'job',
      },
      {
        label: '创建时间：',
        value: '',
        type: '',
      },
      {
        label: 'Job提交模式：',
        value: '',
        type: 'job',
      },
      {
        label: '程序入口：',
        value: '',
        type: '',
      },
      {
        label: '程序入口CPU数量：',
        value: '',
        type: 'job',
      },
    ];
    try {
      const res = await getJobDetail(jobNameSpace, jobName);
      if (res.data.code === ResponseCode.OK) {
        jobDetailInfo[0].value = res.data.data.job_id || '-';
        jobDetailInfo[1].value = res.data.data.name;
        jobDetailInfo[2].value = res.data.data.raycluster_name || '-';
        jobDetailInfo[3].value = res.data.data.dashboard_address || '-';
        jobDetailInfo[4].value = res.data.data.cluster_status_zhcn || '-';
        jobDetailInfo[5].value = res.data.data.create_by || '-';
        jobDetailInfo[6].value = res.data.data.update_time || '-';
        jobDetailInfo[7].value = res.data.data.create_time || '-';
        setDetailUrl(
          res.data.data.dashboard_address ? `http://${prefixUrl}:${res.data.data.dashboard_address.split(':')[1]}` : '',
        );
        if (
          Object.prototype.hasOwnProperty.call(yamlTojson(res.data.data.config_detail).spec, 'submissionMode') &&
          yamlTojson(res.data.data.config_detail).spec.submissionMode
        ) {
          jobDetailInfo[8].value = yamlTojson(res.data.data.config_detail).spec.submissionMode;
        } else {
          jobDetailInfo[8].value = 'K8sJobMode';
        }
        jobDetailInfo[9].value = yamlTojson(res.data.data.config_detail).spec.entrypoint || '--';
        if (
          Object.prototype.hasOwnProperty.call(res.data.data, 'entrypointNumCpus') &&
          res.data.data.entrypointNumCpus
        ) {
          jobDetailInfo[10].value = res.data.data.submissionMode;
        } else {
          jobDetailInfo[10].value = '--';
        }
        if (
          Object.prototype.hasOwnProperty.call(res.data.data, 'entrypointResources') &&
          res.data.data.entrypointResources
        ) {
          let entrypointResourceArr = [];
          Object.keys(yamlTojson(res.data.data.entrypointResources)).forEach((item) => {
            entrypointResourceArr.push({
              key: item,
              value: yamlTojson(res.data.data.entrypointResources)[item],
            });
            return item;
          });
          setEntrypointResource(entrypointResourceArr);
        } else {
          jobDetailInfo[10].value = '--';
        }
        setJobDetailInfo(jobDetailInfo);
        let configData = yamlTojson(res.data.data.config_detail);
        const headGroupSpec = configData.spec.rayClusterSpec.headGroupSpec;
        const workerGroupSpecs = configData.spec.rayClusterSpec.workerGroupSpecs;
        setJobImageName(headGroupSpec.template.spec.containers[0].image.split(':')[0]);
        setJobImageVersion(headGroupSpec.template.spec.containers[0].image.split(':')[1]);
        if (
          Object.prototype.hasOwnProperty.call(headGroupSpec.rayStartParams, 'num-cpus') &&
          headGroupSpec.rayStartParams['num-cpus'] === '0'
        ) {
          setJobHeadChangeWork(false);
        } else {
          setJobHeadChangeWork(true);
        }
        if (Object.prototype.hasOwnProperty.call(headGroupSpec.template.spec, 'volumes')) {
          setJobHeadVolumesData(jsonToYaml(JSON.stringify(headGroupSpec.template.spec.volumes)));
        }
        if (Object.prototype.hasOwnProperty.call(headGroupSpec.template.spec.containers[0], 'volumeMounts')) {
          setJobHeadVolumeMountsData(
            jsonToYaml(JSON.stringify(headGroupSpec.template.spec.containers[0].volumeMounts)),
          );
        }
        const isHasHeadenv = Object.prototype.hasOwnProperty.call(headGroupSpec.template.spec.containers[0], 'env');
        if (isHasHeadenv && headGroupSpec.template.spec.containers[0].env.length > 0) {
          setIsHasJobHeadEnv(true);
        }
        setJobHeadEnv(isHasHeadenv ? headGroupSpec.template.spec.containers[0].resources.env : []);
        let jobHeadTableSource = [];
        const headLimitsKeys = Object.keys(headGroupSpec.template.spec.containers[0].resources.limits);
        const headNpuKeyArr = headLimitsKeys.filter((item) => item !== 'npu' && item !== 'cpu' && item !== 'memory');
        jobHeadTableSource.push({
          type: 'head',
          name: headGroupSpec.template.spec.containers[0].name,
          npu: headNpuKeyArr.length ? headGroupSpec.template.spec.containers[0].resources.limits[headNpuKeyArr[0]] : '',
          maxCopy: headGroupSpec.maxReplicas,
          minCopy: headGroupSpec.minReplicas,
          npuType: headNpuKeyArr.length ? headNpuKeyArr[0] : '',
          cpu: headGroupSpec.template.spec.containers[0].resources.limits.cpu,
          memory: headGroupSpec.template.spec.containers[0].resources.limits.memory,
        });
        let workEnvData = [];
        let workTableSourece = [];
        workerGroupSpecs.map((item) => {
          if (Object.prototype.hasOwnProperty.call(item.template.spec, 'volumes')) {
            setJobWorkVolumesData(jsonToYaml(JSON.stringify(item.template.spec.volumes)));
          }
          if (Object.prototype.hasOwnProperty.call(item.template.spec.containers[0], 'volumeMounts')) {
            setJobWorkVolumeMountsData(jsonToYaml(JSON.stringify(item.template.spec.containers[0].volumeMounts)));
          }
          const limitsKeys = Object.keys(item.template.spec.containers[0].resources.limits);
          const npuKeyArr = limitsKeys.filter((item) => item !== 'npu' && item !== 'cpu' && item !== 'memory');
          workTableSourece.push({
            type: 'worker',
            name: item.template.spec.containers[0].name,
            npu: npuKeyArr.length ? item.template.spec.containers[0].resources.limits[npuKeyArr[0]] : '',
            cpu: item.template.spec.containers[0].resources.limits.cpu,
            memory: item.template.spec.containers[0].resources.limits.memory,
            maxCopy: item.maxReplicas,
            minCopy: item.minReplicas,
            npuType: npuKeyArr.length ? npuKeyArr[0] : '',
          });
          // 获取workenv
          const isHasWodkenv =
            Object.prototype.hasOwnProperty.call(item.template.spec.containers[0], 'resources') &&
            Object.prototype.hasOwnProperty.call(item.template.spec.containers[0].resources, 'env');
          if (isHasWodkenv) {
            workEnvData = workEnvData.concat(item.template.spec.containers[0].resources.env);
          }
        });
        if (workEnvData.length > 0) {
          setIsHasJobWorkEnv(true);
        }
        setJobHeadTableData(jobHeadTableSource);
        setJobWorkTableData(workTableSourece);
        setJobWorkEnv(filterRepeatVolume(workEnvData));
        setIsGetJobDetail(true);
      } else {
        let msg = '';
        msg = res.data.data ? `${res.data.msg}: ${res.data.data}` : res.data.msg;
        messageApi.error(`获取${jobName}详情失败,${msg}`);
      }
    } catch (error) {
      if (error) {
        let msg = '';
        msg =
          JSON.stringify(error.response.data.data) !== '{}'
            ? `${error.response.data.msg + ':' + error.response.data.data}`
            : error.response.data.msg;
        messageApi.error(`获取${jobName}详情失败,${msg}`);
      }
    }
  };
  useEffect(() => {
    getJobInfo();
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
          {jobDetailInfo.map((item) => (
            <div className="detail_info_row_single">
              <p className={`detail_info_row_single_label ${item.type.includes('job') ? 'job_label' : ''}`}>
                {item.label}
              </p>
              <p className="detail_info_row_single_value">
                <span
                  className="detail_info_row_single_status"
                  style={{
                    display: item.label.includes('状态') ? 'flex' : 'none',
                    background: filterJobStatusValue(item.value),
                  }}
                ></span>
                {item.value}
              </p>
            </div>
          ))}
        </div>
        <p
          className="detail_common_title3"
          style={{ fontSize: '14px', display: entrypointResource.length > 0 ? 'block' : 'flex' }}
        >
          {intl.formatMessage({ id: 'detail.entryResource' })}：{' '}
          <span style={{ display: entrypointResource.length > 0 ? 'none' : 'block', marginLeft: '52px' }}>--</span>
        </p>
        <div className="detail_work_environment">
          {entrypointResource.length > 0
            ? entrypointResource.map((item) => (
                <div className="detail_work_environment_single">
                  <p>{item.key}</p>
                  <p className="detail_work_environment_single_equal">=</p>
                  <p>{item.value}</p>
                </div>
              ))
            : ''}
        </div>
      </div>
      <div className="detail_cluster detail_common_content">
        <p className="detail_common_title">{intl.formatMessage({ id: 'ray.job.rayClusterConfig' })}</p>
        <div className="detail_cluster_image">
          <div className="detail_cluster_image_single">
            <p className="detail_cluster_image_single_lable">{intl.formatMessage({ id: 'detail.imageName' })}</p>
            <p className="detail_cluster_image_single_value">{jobImageName}</p>
          </div>
          <div className="detail_cluster_image_single">
            <p className="detail_cluster_image_single_lable">{intl.formatMessage({ id: 'detail.imageVersion' })}</p>
            <p className="detail_cluster_image_single_value">{jobImageVersion}</p>
          </div>
        </div>
        <div className="detail_cluster_image"></div>
        <p className="detail_common_title2">{intl.formatMessage({ id: 'detail.headNodeConfig' })}</p>
        <Table
          locale={{ emptyText: intl.formatMessage({ id: 'common.noData' }) }}
          columns={detailHeadConfigColumns}
          pagination={false}
          dataSource={jobHeadTableData}
        />
        <Checkbox checked={jobHeadChangeWork} disabled>
          {intl.formatMessage({ id: 'ray.job.allowHeadWorkload' })}
        </Checkbox>
        <div className="detail_cluster_code">
          <p className="detail_config_name">volumes</p>
          <p className="detail_config_name">volumeMounts</p>
        </div>
        <div className="detail_cluster_code">
          {isGetJobDetail ? <CodeMirror value={jobHeadVolumesData} extensions={[javascript()]} editable={false} /> : ''}
          {isGetJobDetail ? (
            <CodeMirror value={jobHeadVolumeMountsData} extensions={[javascript()]} editable={false} />
          ) : (
            ''
          )}
        </div>
        <p className="detail_common_title3">{intl.formatMessage({ id: 'env.title' })}</p>
        {isHasJobHeadEnv ? (
          <div className="detail_work_environment">
            {jobHeadEnv.map((item) => (
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
          dataSource={jobWorkTableData}
        />
        <div className="detail_cluster_code">
          <p className="detail_config_name">volumes</p>
          <p className="detail_config_name">volumeMounts</p>
        </div>
        <div className="detail_cluster_code">
          {isGetJobDetail ? <CodeMirror value={jobWorkVolumesData} extensions={[javascript()]} editable={false} /> : ''}
          {isGetJobDetail ? (
            <CodeMirror value={jobWorkVolumeMountsData} extensions={[javascript()]} editable={false} />
          ) : (
            ''
          )}
        </div>
        <p className="detail_common_title3">{intl.formatMessage({ id: 'env.title' })}</p>
        {isHasJobWorkEnv ? (
          <div className="detail_work_environment">
            {jobWorkEnv.map((item) => (
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
