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

import '@/styles/affinityPolicyConfiguration.less';
import { QuestionCircleOutlined, ExclamationCircleFilled } from '@ant-design/icons';
import { Radio, Switch, message, Table, Collapse, Space, Modal, Button, Drawer, theme } from 'antd';
import { useEffect, useState } from 'openinula';
import { useIntl } from 'inula-intl';
import noData from '@/assets/images/noData.png';
import PriorityPreemption from '@/pages/SchedulingPolicyConfiguration/PriorityPreemption';
import { getVolcano, getDistance, getRunningPod } from '@/assets/md';
import { ResponseCode } from '@/common/constants';
import {
  getNumaAffinitySchedulingStatus,
  updateNumaAffinitySchedulingStatus,
  getNumaDistanceSchedulingStatus,
  updateNumaDistanceSchedulingStatus,
  getPodStatus,
  updatePodStatus,
  getBalanceStatus,
  updateBalanceStatus,
} from '@/api/schedulingApi';
import { getNodeList, normalizeKeys, getCR } from '@/api/numaApi';
import { sorterFirstAlphabet } from '@/tools/utils';
import { filterRepeat } from '@/tools/utils';

export default function AffinityPolicyConfiguration() {
  const intl = useIntl();
  const [affinityData, setAffinityData] = useState([]);
  const [numaAffKey, setNumaAffKey] = useState();
  const [descData, setDescData] = useState();
  const [className, setClassName] = useState('');
  const [numaChecked, setNumaChecked] = useState(false);
  const [numaDistanceChecked, setNumaDistanceChecked] = useState(false);
  const [numaPartial, setNumaPartial] = useState(false);
  const [numaDistancePartial, setNumaDistancePartial] = useState(false);
  const [podChecked, setPodChecked] = useState(false);
  const [balanceChecked, setBalanceChecked] = useState(false);
  const [afinityFilter, setAfinityFilter] = useState();
  const [modalInfoOpen, setModalInfoOpen] = useState(false);
  const [modalType, setModalType] = useState('');
  const [kubeletVersionWarn, setKubeletVersionWarn] = useState(false);
  const affinityColumns = [
    {
      title: intl.formatMessage({ id: 'numa.nodeName' }),
      dataIndex: 'name',
      key: 'name',
      sorter: (a, b) => sorterFirstAlphabet(a.name, b.name),
      render: (_, record) => record.name,
    },
    {
      title: intl.formatMessage({ id: 'numa.topologyPolicy' }),
      dataIndex: 'topologyPolicy',
      key: 'topologyPolicy',
      filters: afinityFilter,
      filteredValue: affinityData.topologyPolicy,
      onFilter: (value, record) => record.topologyPolicy.includes(value),
      sorter: (a, b) => sorterFirstAlphabet(a.topologyPolicy, b.topologyPolicy),
      render: (_, record) => record.topologyPolicy,
    },
  ];

  const affinityitems = [
    {
      key: 'nodeTopologyStrategy',
      label: intl.formatMessage({ id: 'affinity.nodeTopologyStrategy' }),
      children: (
        <div className="volcano-table" style={{ padding: 0 }}>
          <Table
            locale={{ emptyText: intl.formatMessage({ id: 'common.noData' }) }}
            dataSource={affinityData}
            columns={affinityColumns}
            pagination={{
              className: 'volcano-page',
              pageSizeOptions: [10, 20, 50],
              showSizeChanger: true,
              showQuickJumper: true,
              showTotal: (total) => intl.formatMessage({ id: 'common.totalItems' }, { total }),
            }}
            style={{ marginBottom: '20px' }}
          />
        </div>
      ),
    },
  ];
  const [distanceData, setDistanceData] = useState([
    {
      name: 'numa1',
      num: '2',
      memory: ['pad1', 'pado22'],
      net: ['pad1', 'pado22'],
    },
  ]);
  const [distanceCurrent, setDistanceCurrent] = useState(1);
  const distanceColumns = [
    {
      title: intl.formatMessage({ id: 'numa.numaName' }),
      dataIndex: 'name',
      key: 'name',
      sorter: (a, b) => sorterFirstAlphabet(a.name, b.name),
      render: (_, record) => (
        <Space size="middle" className="">
          {record.name}
        </Space>
      ),
    },
    {
      title: intl.formatMessage({ id: 'affinity.affinityPodPairsCount' }),
      dataIndex: 'num',
      key: 'num',
      sorter: (a, b) => sorterFirstAlphabet(a.num, b.num),
      render: (_, record) => (
        <Space size="middle" className="">
          {record.num}
        </Space>
      ),
    },
    {
      title: intl.formatMessage({ id: 'numa.memoryAffinityPodName' }),
      dataIndex: 'memoryName',
      key: 'memoryName',
      render: (_, record) => (
        <div>
          {record.memory.map((item, index) => (
            <p>{item}</p>
          ))}
        </div>
      ),
    },
    {
      title: intl.formatMessage({ id: 'numa.networkAffinityPodName' }),
      dataIndex: 'netName',
      key: 'netName',
      render: (_, record) => (
        <div>
          {record.net.map((item, index) => (
            <p>{item}</p>
          ))}
        </div>
      ),
    },
  ];

  const handleDistancePageChange = (page, pageSize) => {
    setDistanceCurrent(page); // 更新当前页码
  };
  const distanceItems = [
    {
      key: '1',
      label: intl.formatMessage({ id: 'affinity.nodeTopologyStrategy' }),
      children: (
        <div className="volcano-table">
          <Table
            locale={{ emptyText: intl.formatMessage({ id: 'common.noData' }) }}
            dataSource={distanceData}
            columns={distanceColumns}
            pagination={{
              className: 'volcano-page',
              current: distanceCurrent,
              pageSizeOptions: [10, 20, 50],
              showSizeChanger: true,
              showQuickJumper: true,
              onChange: handleDistancePageChange,
              showTotal: (total) => intl.formatMessage({ id: 'common.totalItems' }, { total: dataSource.length }),
            }}
            style={{ marginBottom: '20px' }}
          />
        </div>
      ),
    },
  ];
  const { token } = theme.useToken();
  const [open, setOpen] = useState(false);
  const showDrawer = (data, temproryClass) => {
    setDescData(data);
    setClassName(temproryClass);
    setOpen(true);
  };
  const onClose = () => {
    setOpen(false);
  };

  const handleSkip = () => {
    setKubeletVersionWarn(false);
    showDrawer(getDistance());
  };

  const handleNumaChange = async (e) => {
    if (e) {
      setModalInfoOpen(true);
      setModalType('numa');
    } else {
      let obj = { policy: '' };
      obj.policy = 'close';
      const res = await updateNumaAffinitySchedulingStatus(obj);
      if (res.status === ResponseCode.OK) {
        setNumaChecked(false);
        setNumaPartial(false);
      }
    }
  };

  const handleConfirmInfo = async () => {
    if (modalType === 'numa') {
      let obj = { policy: '' };
      obj.policy = 'open';
      const res = await updateNumaAffinitySchedulingStatus(obj);
      if (res.status === ResponseCode.OK) {
        setNumaChecked(true);
        setNumaPartial(false);
        setModalInfoOpen(false);
      }
    }
    if (modalType === 'numaDistance') {
      let obj = { policy: '' };
      obj.policy = 'open';
      try {
        const res = await updateNumaDistanceSchedulingStatus(obj);
        if (res.status === ResponseCode.OK) {
          setNumaDistanceChecked(true);
          setNumaDistancePartial(false);
          setModalInfoOpen(false);
        }
      } catch (e) {
        if (e.response.data.code === ResponseCode.NotAccept) {
          setModalInfoOpen(false);
          setKubeletVersionWarn(true);
        }
      }
    }
  };

  const handleNumaDistanceChange = async (e) => {
    if (e) {
      setModalInfoOpen(true);
      setModalType('numaDistance');
    } else {
      let obj = { policy: '' };
      obj.policy = 'close';
      const res = await updateNumaDistanceSchedulingStatus(obj);
      if (res.status === ResponseCode.OK) {
        setNumaDistanceChecked(e);
        setNumaDistancePartial(false);
      }
    }
  };

  const handlePodChange = async (e) => {
    let obj = { policy: '' };
    setModalType('pod');
    e ? (obj.policy = 'open') : (obj.policy = 'close');
    const res = await updatePodStatus(obj);
    if (res.status === ResponseCode.OK) {
      setPodChecked(e);
    }
  };

  const handleBalanceChange = async (e) => {
    let obj = { policy: '' };
    setModalType('balance');
    e ? (obj.policy = 'open') : (obj.policy = 'close');
    const res = await updateBalanceStatus(obj);
    if (res.status === ResponseCode.OK) {
      setBalanceChecked(e);
    }
  };

  const hanldeNumaAff = async (e) => {
    setNumaAffKey(e);
    if (e && e.length && e.includes('nodeTopologyStrategy')) {
      const res = await getNodeList();
      if (res.status === ResponseCode.OK) {
        const finalData = await Promise.all(
          res.data.data.items.map(async (nodeItem) => {
            const crDetails = await getCR(nodeItem.metadata.name);
            if (crDetails?.data) {
              crDetails.data = normalizeKeys(crDetails.data);
            }
            return {
              name: nodeItem.metadata.name,
              topologyPolicy: crDetails?.data?.data?.spec?.policies?.topologymanagerpolicy || 'N/A',
            };
          }),
        );
        let filterData = [];
        finalData.forEach((item) => {
          filterData.push({
            text: item.topologyPolicy,
            value: item.topologyPolicy,
          });
        });
        setAfinityFilter(filterRepeat(filterData));
        setAffinityData(finalData);
      }
    }
  };

  const getNumaStatus = async () => {
    const [{ value: numaRes = null }, { value: distanceRes = null }, { value: podRes = null }] =
      await Promise.allSettled([getNumaAffinitySchedulingStatus(), getNumaDistanceSchedulingStatus(), getPodStatus()]);
    if (distanceRes?.status === ResponseCode.NotAccept) {
      setKubeletVersionWarn(true);
    }
    if (numaRes?.status === ResponseCode.OK) {
      setNumaChecked(numaRes.data.data !== 'close');
      setNumaPartial(numaRes.data.data === 'partial');
    }
    if (distanceRes?.status === ResponseCode.OK) {
      setNumaDistanceChecked(distanceRes.data.data !== 'close');
      setNumaDistancePartial(distanceRes.data.data === 'partial');
    }

    if (podRes?.status === ResponseCode.OK) {
      setPodChecked(podRes.data.data === 'open' ? true : false);
    }
  };

  useEffect(() => {
    getNumaStatus();
  }, []);

  return (
    <div id="affinity_policy" className="affinity_policy">
      <BreadCrumbCom
        items={[
          { title: intl.formatMessage({ id: 'common.numaAffinityScheduling' }), disabled: true },
          { title: intl.formatMessage({ id: 'affinity.affinityPolicyConfig' }), disabled: true },
        ]}
      />
      <div className="affinity_policy_content">
        <div className="affinity_policy_content_one">
          <div className="affinity_policy_content_one_title">
            numa-aware
            <QuestionCircleOutlined
              className="question-icon-hover"
              onClick={() => showDrawer(getVolcano())}
              style={{ marginLeft: '7px' }}
            />
          </div>
          <Switch className="mb_12" checked={numaChecked} onChange={handleNumaChange} />
          {numaPartial && (
            <div className="affinity_policy_content_one_desc" style={{ color: '#d46b08' }}>
              <ExclamationCircleFilled style={{ marginRight: '6px' }} />
              {intl.formatMessage({ id: 'affinity.partialEnabled' })}
            </div>
          )}
          <div className="affinity_policy_content_one_desc">{intl.formatMessage({ id: 'affinity.numaAwareDesc' })}</div>
          <Collapse activeKey={numaAffKey} items={affinityitems} bordered={false} onChange={hanldeNumaAff} />
        </div>
        <div className="affinity_policy_content_one">
          <div className="affinity_policy_content_one_title">
            {intl.formatMessage({ id: 'affinity.optimalNumaDistance' })}
            <QuestionCircleOutlined
              className="question-icon-hover"
              style={{ marginLeft: '7px' }}
              onClick={() => showDrawer(getDistance(), 'break_code_md_body')}
            />
          </div>
          <Switch className="mb_12" checked={numaDistanceChecked} onChange={handleNumaDistanceChange} />
          {numaDistancePartial && (
            <div className="affinity_policy_content_one_desc" style={{ color: '#d46b08' }}>
              <ExclamationCircleFilled style={{ marginRight: '6px' }} />
              {intl.formatMessage({ id: 'affinity.partialEnabled' })}
            </div>
          )}
          <div className="affinity_policy_content_one_desc">
            {intl.formatMessage({ id: 'affinity.numaDistanceDesc' })}
          </div>
        </div>
        <div className="affinity_policy_content_one">
          <div className="affinity_policy_content_one_title">
            {intl.formatMessage({ id: 'affinity.runtimePodAffinity' })}
            <QuestionCircleOutlined
              className="question-icon-hover"
              style={{ marginLeft: '7px' }}
              onClick={() => showDrawer(getRunningPod(), 'break_code_md_body')}
            />
          </div>
          <Switch className="mb_12" checked={podChecked} onChange={handlePodChange} />
          <div className="affinity_policy_content_one_desc">
            {intl.formatMessage({ id: 'affinity.runtimePodAffinityDesc' })}
          </div>
        </div>
        <div className="affinity_policy_content_one" style={{ display: 'none' }}>
          <div className="affinity_policy_content_one_title">
            {intl.formatMessage({ id: 'affinity.balancedScheduling' })}
          </div>
          <Switch className="mb_12" checked={balanceChecked} onChange={handleBalanceChange} />
          <div className="affinity_policy_content_one_desc">
            {intl.formatMessage({ id: 'affinity.balancedSchedulingDesc' })}
          </div>
        </div>
        <Drawer
          title=""
          placement="right"
          closable={false}
          onClose={onClose}
          width={600}
          open={open}
          getContainer={() => document.getElementById('affinity_policy')}
        >
          <PriorityPreemption className={className} onClose={onClose} data={descData} />
        </Drawer>
        <Modal
          className="modal_flex_delete"
          getContainer={false}
          open={modalInfoOpen}
          title={intl.formatMessage({ id: 'affinity.tip' })}
          footer={() => (
            <>
              <Button className="cancel_btn" onClick={handleConfirmInfo}>
                {intl.formatMessage({ id: 'common.confirm' })}
              </Button>
            </>
          )}
          onCancel={() => setModalInfoOpen(false)}
        >
          <div className="modal_delete_content">
            <ExclamationCircleFilled className="warn_icon" />
            <div className="word_tograry">
              <div className="word_tograry_alarm">
                <p className="bread_word_delete">{intl.formatMessage({ id: 'affinity.confirmRestartKubelet' })}</p>
                <p className="bread_word_delete">{intl.formatMessage({ id: 'affinity.impactOnDeployedWorkloads' })}</p>
              </div>
            </div>
          </div>
        </Modal>
        <Modal
          className="modal_flex_delete"
          getContainer={false}
          title={intl.formatMessage({ id: 'affinity.kubeletVersionNotSupported' })}
          open={kubeletVersionWarn}
          footer={() => (
            <>
              <Button className="cancel_btn" onClick={() => setKubeletVersionWarn(false)}>
                {intl.formatMessage({ id: 'common.confirm' })}
              </Button>
            </>
          )}
          onCancel={() => setKubeletVersionWarn(false)}
        >
          <div className="modal_delete_content">
            <ExclamationCircleFilled className="warn_icon" />
            <div className="word_tograry">
              <div className="word_tograry_alarm">
                <p className="bread_word_delete">
                  {intl.formatMessage({ id: 'affinity.kubeletVersionNotSupportedDesc' })}
                </p>
                <p className="bread_word_delete">
                  {intl.formatMessage({ id: 'affinity.kubeletUpgradeHint' })}
                  <a onClick={handleSkip}>{intl.formatMessage({ id: 'affinity.referenceLink' })}</a>。
                </p>
              </div>
            </div>
          </div>
        </Modal>
      </div>
    </div>
  );
}
