/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */
import BreadCrumbCom from '@/components/BreadCrumbCom';
import { containerRouterPrefix, ResponseCode } from '@/common/constants';
import { useHistory, Link, useLocation } from 'inula-router';
import { useEffect, useState } from 'openinula';
import { Button, Dropdown, Tabs, message } from 'antd';
import { DownOutlined, UpOutlined } from '@ant-design/icons';
import useLocalStorage from '@/hooks/useLocalStorage';
import '@/styles/rayServiceDetail.less';
import Detail from '@/pages/job/detailComponent/Detail';
import Yaml from '@/pages/job/detailComponent/Yaml';
import Log from '@/pages/job/detailComponent/Log';
import { deleteJobList, stopJobList, restartJobList, getJobDetail } from '@/api/rayApi.js';
import DeleteInfoModal from '@/components/DeleteInfoModal';
import { useIntl } from 'inula-intl';
export default function RayJobDetail() {
  const intl = useIntl();
  const location = useLocation();
  const history = useHistory();
  const [messageApi, contextHolder] = message.useMessage();
  const [currentTheme, setCurrentTheme] = useLocalStorage('theme', 'light');
  const [isShowJobDetailOperation, setIsShowJobDetailOperation] = useState(false);
  const jobName = location.state.name;
  const jobNameSpace = location.state.namespace;
  const [jobStatus, setJobStatus] = useState('');
  const [jobDetailDelModalOpen, setJobDetailDelModalOpen] = useState(false);
  const [isJobDetailDelCheck, setIsJobDetailDelCheck] = useState(false); // 是否选中

  const items = [
    {
      key: '1',
      label: (
        <Button
          type="link"
          disabled={jobStatus !== '失败' && jobStatus !== '未运行'}
          onClick={() => handleRestartJobDetail()}
        >
          {intl.formatMessage({ id: 'common.start' })}
        </Button>
      ),
    },
    {
      key: '2',
      label: (
        <Button type="link" disabled={jobStatus === '未运行'} onClick={() => handleStopJobDetail()}>
          {intl.formatMessage({ id: 'common.stop' })}
        </Button>
      ),
    },
    {
      key: '3',
      label: (
        <Button type="link" disabled={jobStatus === '运行中'} onClick={() => handleDeleteJobDetail()}>
          {intl.formatMessage({ id: 'common.delete' })}
        </Button>
      ),
    },
  ];
  const jobDetailTabItems = [
    {
      key: '1',
      label: intl.formatMessage({ id: 'detail.tab.detail' }),
      children: <Detail className="defaultRayJobDetail" />,
    },
    {
      key: '2',
      label: 'YAML',
      children: <Yaml className="defaultRayJobDetail" />,
    },
    {
      key: '3',
      label: intl.formatMessage({ id: 'detail.tab.log' }),
      children: <Log className="defaultRayJobDetail" />,
    },
  ];
  const handleJobDetailOperation = (open) => {
    // 展示创建下拉菜单
    setIsShowJobDetailOperation(open);
  };
  const handleRestartJobDetail = async () => {
    try {
      const res = await restartJobList(jobNameSpace, jobName);
      if (res.data.code === ResponseCode.OK) {
        messageApi.success(intl.formatMessage({ id: 'ray.job.startSuccess' }, { name: jobName }));
        getJobStatus();
      } else {
        let msg = '';
        msg = JSON.stringify(res.data.data) !== '{}' ? `${res.data.msg + ':' + res.data.data}` : res.data.msg;
        messageApi.error(`${intl.formatMessage({ id: 'ray.job.startFail' })},${msg}`);
      }
    } catch (error) {
      if (error) {
        let msg = '';
        msg =
          JSON.stringify(error.response.data.data) !== '{}'
            ? `${error.response.data.msg + ':' + error.response.data.data}`
            : error.response.data.msg;
        messageApi.error(`${intl.formatMessage({ id: 'ray.job.startFail' })},${msg}`);
      }
    }
  };
  const handleStopJobDetail = async () => {
    try {
      const res = await stopJobList(jobNameSpace, jobName);
      if (res.data.code === ResponseCode.OK) {
        messageApi.success(intl.formatMessage({ id: 'ray.job.stopSuccess' }, { name: jobName }));
        getJobStatus();
      } else {
        let msg = '';
        msg = JSON.stringify(res.data.data) !== '{}' ? `${res.data.msg + ':' + res.data.data}` : res.data.msg;
        messageApi.error(`${intl.formatMessage({ id: 'ray.job.stopFail' })},${msg}`);
      }
    } catch (error) {
      if (error) {
        let msg = '';
        msg =
          JSON.stringify(error.response.data.data) !== '{}'
            ? `${error.response.data.msg + ':' + error.response.data.data}`
            : error.response.data.msg;
        messageApi.error(`${intl.formatMessage({ id: 'ray.job.stopFail' })},${msg}`);
      }
    }
  };

  // 打开删除弹窗
  const handleDeleteJobDetail = () => {
    setJobDetailDelModalOpen(true);
  };
  // 取消删除
  const handleDelJobDetailCancel = () => {
    setJobDetailDelModalOpen(false);
    setIsJobDetailDelCheck(false);
  };
  // 确认删除按钮删除
  const handleJobDetailCheckFn = (e) => {
    setIsJobDetailDelCheck(e.target.checked);
  };
  const deleteJobDetail = async () => {
    try {
      const res = await deleteJobList(jobNameSpace, jobName);
      if (res.data.code === ResponseCode.OK) {
        messageApi.success(intl.formatMessage({ id: 'ray.job.deleteSuccess' }, { name: jobName }));
        setTimeout(() => {
          history.push('/RayJob');
          setJobDetailDelModalOpen(false);
        }, 500);
        setIsJobDetailDelCheck(false);
      } else {
        let msg = '';
        msg = JSON.stringify(res.data.data) !== '{}' ? `${res.data.msg + ':' + res.data.data}` : res.data.msg;
        messageApi.error(`${intl.formatMessage({ id: 'ray.job.deleteFail' })},${msg}`);
      }
    } catch (error) {
      if (error) {
        let msg = '';
        msg =
          JSON.stringify(error.response.data.data) !== '{}'
            ? `${error.response.data.msg + ':' + error.response.data.data}`
            : error.response.data.msg;
        messageApi.error(`${intl.formatMessage({ id: 'ray.job.deleteFail' })},${msg}`);
      }
    }
  };
  const getJobStatus = async () => {
    const res = await getJobDetail(jobNameSpace, jobName);
    try {
      if (res.data.code === ResponseCode.OK) {
        setJobStatus(res.data.data.cluster_status_zhcn);
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
        messageApi.error(intl.formatMessage({ id: 'ray.job.detailError' }, { name: jobName }) + ',' + msg);
      }
    }
  };
  useEffect(() => {
    getJobStatus();
  }, []);
  useEffect(() => {}, [currentTheme]);
  return (
    <div className="child_content service_detail">
      <BreadCrumbCom
        className="create_bread"
        items={[
          { title: 'openFuyao Ray', disabled: true },
          { title: 'RayJob', path: `/RayJob` },
          { title: intl.formatMessage({ id: 'detail.title' }), path: `/detail` },
        ]}
      />
      {contextHolder}
      <div>
        <div className="service_detail_top defaultRayJobDetail">
          <div className="service_detail_top_title_btn">
            <p className="service_detail_top_title defaultRayJobDetail">{location.state.name}</p>
            <div>
              <Dropdown
                menu={{ items }}
                open={isShowJobDetailOperation}
                className="defaultRayJobDetail"
                onOpenChange={handleJobDetailOperation}
                placement="bottom"
              >
                <Button className="primary_btn">
                  {intl.formatMessage({ id: 'detail.operation' })}
                  {isShowJobDetailOperation ? <UpOutlined /> : <DownOutlined />}{' '}
                </Button>
              </Dropdown>
            </div>
          </div>
        </div>
        <Tabs defaultActiveKey="1" items={jobDetailTabItems} />
      </div>
      <DeleteInfoModal
        title={intl.formatMessage({ id: 'ray.job.deleteTitle' })}
        open={jobDetailDelModalOpen}
        cancelFn={handleDelJobDetailCancel}
        content={[
          intl.formatMessage({ id: 'ray.job.deleteWarning' }),
          intl.formatMessage({ id: 'ray.job.deleteConfirm' }, { name: jobName, extra: '' }),
        ]}
        showCheck={true}
        isCheck={isJobDetailDelCheck}
        checkFn={handleJobDetailCheckFn}
        confirmFn={deleteJobDetail}
      />
    </div>
  );
}
